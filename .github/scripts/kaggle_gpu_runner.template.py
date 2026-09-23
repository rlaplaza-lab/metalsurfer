#!/usr/bin/env python3
"""Kaggle kernel entry point for metalsurfer GPU CI (rendered by kaggle-gpu.yml).

Modes (__GPU_MODE__):
  smoke   -- gpu_smoke test subset (~7 min)
  full    -- full GPU test suite via run_gpu_tests.sh (~90 min)
  examples -- run_all_examples.sh (bipyridine omitted; ~60 min on T4)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import traceback
from pathlib import Path

GIT_REF = "__GIT_REF__"
GPU_MODE = "__GPU_MODE__"
PYTEST_MARKER = "__PYTEST_MARKER__"
HF_TOKEN = "__HF_TOKEN__"
CONDA_ENV = "metalsurfer-gpu"
# Use /tmp so pytest/pip artifacts are not saved as Kaggle kernel output.
WORKDIR = Path("/tmp/metalsurfer")
DATASET_OWNER = "rlaplaza"
DATASET_SLUG = "metalsurfercisrc"
DATASET_INPUT = Path("/kaggle/input") / DATASET_SLUG
SOURCE_ARCHIVE = "metalsurfer-src.tar.gz"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX = "https://pypi.org/simple"


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None
) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _log_kaggle_inputs() -> None:
    inputs_root = Path("/kaggle/input")
    if not inputs_root.is_dir():
        log("No /kaggle/input directory mounted")
        return
    log("Kaggle input mounts:")
    for path in sorted(inputs_root.rglob("*")):
        if path.is_file():
            log(f"  {path} ({path.stat().st_size} bytes)")


def _conda_exe() -> str | None:
    """Return a usable conda executable, or None when conda is unavailable."""
    for candidate in (
        os.environ.get("CONDA_EXE", ""),
        "/opt/conda/bin/conda",
        shutil.which("conda") or "",
    ):
        if candidate and (candidate == "conda" or os.path.isfile(candidate)):
            return candidate
    return None


def _conda_python() -> list[str]:
    conda = _conda_exe()
    if conda is None:
        raise RuntimeError("conda required for conda Python path but not found")
    conda_env = os.environ.copy()
    conda_env["CONDA_PLUGINS_AUTO_ACCEPT_TOS"] = "yes"
    for tos_cmd in (
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/main",
        ],
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/r",
        ],
    ):
        subprocess.run(tos_cmd, env=conda_env, check=False)
    run([conda, "create", "-y", "-n", CONDA_ENV, "python=3.12"], env=conda_env)
    return [conda, "run", "--no-capture-output", "-n", CONDA_ENV, "python"]


def _resolve_python() -> list[str]:
    """Prefer the conda env interpreter; fall back to system Python when absent."""
    if _conda_exe() is not None:
        return _conda_python()
    log(f"conda not found; using system interpreter {sys.executable}")
    return [sys.executable]


def _safe_extractall(tar: tarfile.TarFile, path: Path) -> None:
    tar.extractall(path=path, filter="data")


def _extract_dataset_archive(archive: Path) -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    log(f"Extracting bundled source from {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extractall(tar, WORKDIR)


def _resolve_dataset_dir() -> Path | None:
    """Locate the mounted CI source dataset regardless of Kaggle's mount layout."""
    candidates = [
        DATASET_INPUT,
        Path("/kaggle/input/datasets") / DATASET_OWNER / DATASET_SLUG,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "pyproject.toml").is_file():
            return cand
    root = Path("/kaggle/input")
    if root.is_dir():
        for match in sorted(root.rglob(DATASET_SLUG)):
            if match.is_dir() and (match / "pyproject.toml").is_file():
                return match
    return None


def _find_dataset_archive() -> Path | None:
    dataset_dir = _resolve_dataset_dir()
    if dataset_dir is None:
        return None
    direct = dataset_dir / SOURCE_ARCHIVE
    if direct.is_file():
        return direct
    matches = sorted(dataset_dir.rglob(SOURCE_ARCHIVE))
    return matches[0] if matches else None


def _dataset_tree_ready() -> bool:
    dataset_dir = _resolve_dataset_dir()
    return dataset_dir is not None and (dataset_dir / "pyproject.toml").is_file()


def _copy_dataset_tree() -> None:
    dataset_dir = _resolve_dataset_dir()
    assert dataset_dir is not None
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    log(f"Copying bundled source tree from {dataset_dir}")
    shutil.copytree(
        dataset_dir,
        WORKDIR,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        dirs_exist_ok=True,
    )


def _fetch_repo_from_dataset() -> bool:
    archive = _find_dataset_archive()
    if archive is not None:
        _extract_dataset_archive(archive)
        return True
    if _dataset_tree_ready():
        _copy_dataset_tree()
        return True
    return False


def _fetch_repo() -> None:
    if not _fetch_repo_from_dataset():
        raise FileNotFoundError(
            "Kaggle dataset bundle 'rlaplaza/metalsurfercisrc' not found in the "
            "kernel input; the kaggle-gpu.yml workflow publishes and polls it to "
            "'complete' before launching the kernel."
        )
    log("Using CI source bundle from Kaggle dataset input")


def _numpy_requirement() -> str:
    """Match pyproject.toml so Kaggle uses the same NumPy pin as CI tests."""
    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        if dep.startswith("numpy"):
            return dep
    raise RuntimeError("numpy requirement missing from pyproject.toml")


def _install_numpy(py: list[str], pip: list[str]) -> None:
    """Install project NumPy before torch (Kaggle base image ships 2.0.x)."""
    del py
    spec = _numpy_requirement()
    run([*pip, "install", "--no-cache-dir", spec])


def _install_torch_stack(py: list[str], pip: list[str]) -> None:
    """Install CUDA torch on Kaggle where cu124 wheels may be 2.4--2.6 only."""
    del py
    attempts = (
        [
            *pip,
            "install",
            "--no-cache-dir",
            "torch>=2.12.0,<2.13",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
        ],
        [
            *pip,
            "install",
            "--no-cache-dir",
            "torch",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
        ],
    )
    for cmd in attempts:
        log("+ " + " ".join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode == 0:
            return
        log(f"Torch install failed (exit {completed.returncode}); trying fallback")
    raise subprocess.CalledProcessError(1, attempts[-1])


def _install_metalsurfer_mlip(py: list[str], pip: list[str]) -> None:
    """Install metalsurfer[mlip,dev]; torch is already installed from the CUDA index.

    IMPORTANT: pip install -e .[mlip,dev] --no-deps only installs the package
    entry point; all dependencies (including torch-sim-atomistic, fairchem-core,
    e3nn, etc.) must be installed explicitly from the combined dependency list.
    The filter must only strip bare torch (the package itself), never
    torch-sim-atomistic or other torch-* packages.
    """
    del py
    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    deps.extend(data["project"]["optional-dependencies"]["mlip"])
    deps.extend(data["project"]["optional-dependencies"]["dev"])
    skip_prefixes = ("ruff", "pre-commit", "mypy", "types-", "coverage")
    install_deps = [
        dep
        for dep in deps
        if not any(dep.startswith(prefix) for prefix in skip_prefixes)
        # Drop bare 'torch' but keep torch-sim-atomistic, torch>=..., torch== ...
        and dep.strip().rstrip(",;") != "torch"
    ]

    run([*pip, "install", "--no-cache-dir", "-e", ".[mlip,dev]", "--no-deps"])
    run([*pip, "install", "--no-cache-dir", *install_deps])
    # FairChem, e3nn, metatensor may pull CPU torch from PyPI as a transitive
    # dep; force-reinstall from the CUDA index so the GPU build wins.
    # Pin <2.13 to match _install_torch_stack: torch 2.13 breaks FairChem's
    # UMA lazy init (model weights stay on CPU while indices land on cuda:0).
    run(
        [
            *pip,
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "torch<2.13",
            "--index-url",
            PYTORCH_CUDA_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
        ]
    )


def _assert_numpy_version(py: list[str]) -> None:
    spec = _numpy_requirement()
    run(
        [
            *py,
            "-c",
            (
                "import re\n"
                "import numpy as np\n"
                f"spec = {spec!r}\n"
                "match = re.fullmatch(r'numpy>=(\\d+)\\.(\\d+)(?:,<(\\d+)\\.(\\d+))?', spec)\n"
                "if match is None:\n"
                "    raise SystemExit(f'Unsupported numpy spec: {spec!r}')\n"
                "lo_major, lo_minor, hi_major, hi_minor = match.groups()\n"
                "lo = (int(lo_major), int(lo_minor))\n"
                "hi = (int(hi_major), int(hi_minor)) if hi_major else None\n"
                "parts = [int(part) for part in np.__version__.split('.')[:2]]\n"
                "version = (parts[0], parts[1])\n"
                "if version < lo or (hi is not None and version >= hi):\n"
                "    raise SystemExit(\n"
                "        f'NumPy {np.__version__} does not satisfy {spec!r}'\n"
                "    )\n"
                "print(f'NumPy {np.__version__} satisfies {spec!r}')\n"
            ),
        ]
    )


def _shadow_broken_system_pkg_resources(py: list[str], pip: list[str]) -> None:
    """Shadow the Ubuntu system pkg_resources with a working pip-installed one.

    The system pkg_resources at /usr/lib/python3/dist-packages/ predates
    Python 3.12: it references ``pkgutil.ImpImporter`` (removed in 3.12)
    and ``FileFinder.find_module`` (removed in 3.12), so importing it
    crashes torchtnt and every other transitive pkg_resources consumer.

    setuptools >= 81 no longer ships pkg_resources, so after all installs
    we force-reinstall ``setuptools<81`` into /usr/local/.../dist-packages,
    which precedes /usr/lib/python3/dist-packages in sys.path.  The healthy
    pkg_resources then shadows the broken system copy for every process,
    with no import-order tricks.
    """
    result = subprocess.run(
        [*py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    pip_site = Path(result.stdout.strip())
    if not pip_site.is_dir():
        log(f"WARNING: pip site-packages not found at {pip_site}")
        return
    log(f"pip site-packages: {pip_site}")

    system_pkg_resources = Path("/usr/lib/python3/dist-packages/pkg_resources")
    if not system_pkg_resources.is_dir():
        log("System pkg_resources not found; no shadowing needed")
        return

    # setuptools 80.9.0 is the last release shipping pkg_resources.
    run([*pip, "install", "--no-cache-dir", "--force-reinstall", "setuptools<81"])
    # Drop the .pth stub from earlier iterations if present.
    for legacy in (
        pip_site / "_0_impimporter_patch.pth",
        pip_site / "sitecustomize.py",
    ):
        legacy.unlink(missing_ok=True)
    log(f"Shadowed {system_pkg_resources} with pkg_resources from setuptools<81")


def _assert_cuda_usable(py: list[str]) -> None:
    run(
        [
            *py,
            "-c",
            (
                "import torch\n"
                "if not torch.cuda.is_available():\n"
                "    raise SystemExit('CUDA required')\n"
                "name = torch.cuda.get_device_name()\n"
                "cap = torch.cuda.get_device_capability()\n"
                "print(f'GPU: {name}, capability sm_{cap[0]}{cap[1]}')\n"
                "if cap[0] < 7:\n"
                "    raise SystemExit(\n"
                "        f'GPU {name} (sm_{cap[0]}{cap[1]}) is incompatible with the '\n"
                "        'installed PyTorch CUDA build; use machine_shape NvidiaTeslaT4'\n"
                "    )\n"
                "torch.ones(1, device='cuda')\n"
                "print('CUDA smoke test passed')\n"
            ),
        ]
    )


def _configure_hf_auth(env: dict[str, str]) -> None:
    token = (HF_TOKEN or "").strip()
    if not token or token.startswith("__HF_"):
        raise SystemExit(
            "HF_TOKEN placeholder was not rendered; gated UMA weights cannot download."
        )
    env["HF_TOKEN"] = token
    env["HUGGING_FACE_HUB_TOKEN"] = token


def _resolve_python_bin(py: list[str]) -> str:
    """Return an absolute interpreter path suitable for run_gpu_tests.sh."""
    if len(py) == 1:
        return py[0]
    completed = subprocess.run(
        [*py, "-c", "import sys; print(sys.executable)"],
        check=True,
        capture_output=True,
        text=True,
    )
    path = completed.stdout.strip()
    if not path:
        raise RuntimeError("failed to resolve Python executable for GPU runner")
    return path


def _is_unexpected_oom_line(line: str) -> bool:
    lowered = line.lower()
    return "out of memory" in lowered or "outofmemory" in lowered


def _run_streaming(cmd: list[str], env: dict[str, str]) -> tuple[int, list[str]]:
    """Run a command, tee output, and collect CUDA OOM lines."""
    oom_lines: list[str] = []
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if len(oom_lines) < 20 and _is_unexpected_oom_line(line):
                oom_lines.append(line.rstrip())
        returncode = proc.wait()
    return returncode, oom_lines


def main() -> int:
    try:
        _log_kaggle_inputs()
        _fetch_repo()
        os.chdir(WORKDIR)

        py = _resolve_python()
        pip = [*py, "-m", "pip"]
        run([*pip, "install", "--upgrade", "pip"])
        run([*pip, "install", "--upgrade", "setuptools"])
        _install_numpy(py, pip)
        _install_torch_stack(py, pip)
        log("Installing metalsurfer[mlip,dev] for GPU suite")
        _install_metalsurfer_mlip(py, pip)
        # After all installs (torch pins setuptools<82 which no longer ships
        # pkg_resources): shadow the Python-3.12-incompatible system
        # pkg_resources with a healthy pip-installed one.
        _shadow_broken_system_pkg_resources(py, pip)
        _assert_numpy_version(py)
        _assert_cuda_usable(py)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["METALSURFER_CLEAR_GPU_PYTHON"] = "0"
        _configure_hf_auth(env)

        mode = (GPU_MODE or "smoke").strip().lower()
        if mode not in ("smoke", "full", "examples"):
            raise SystemExit(
                f"Unsupported GPU_MODE={mode!r}; expected smoke, full, or examples"
            )

        marker_override = (PYTEST_MARKER or "").strip()
        if marker_override:
            pytest_cmd = [
                *py,
                "-m",
                "pytest",
                "tests/",
                "-m",
                marker_override,
                "-v",
                "--tb=short",
                "--capture=tee-sys",
                "--log-cli-level=INFO",
                "-rA",
                "--durations=25",
            ]
            log("+ " + " ".join(pytest_cmd))
            returncode, oom_lines = _run_streaming(pytest_cmd, env)
        elif mode == "examples":
            script = WORKDIR / "scripts" / "run_all_examples.sh"
            if not script.is_file():
                raise FileNotFoundError(f"Missing examples runner: {script}")
            python_bin = _resolve_python_bin(py)
            cmd = ["bash", str(script), python_bin]
            log("+ " + " ".join(cmd))
            returncode, oom_lines = _run_streaming(cmd, env)
        else:
            env["METALSURFER_GPU_MODE"] = mode
            script = WORKDIR / "scripts" / "run_gpu_tests.sh"
            if not script.is_file():
                raise FileNotFoundError(f"Missing GPU runner script: {script}")
            python_bin = _resolve_python_bin(py)
            cmd = ["bash", str(script), python_bin]
            log("+ " + " ".join(cmd))
            returncode, oom_lines = _run_streaming(cmd, env)

        if oom_lines:
            log("")
            log(
                "metalsurfer GPU CI: CUDA out-of-memory observed in kernel log. "
                "Failing the job even if pytest exit code was zero."
            )
            for line in oom_lines:
                log(f"  OOM> {line}")
            return returncode or 1
        return int(returncode)
    except Exception:
        log("metalsurfer Kaggle runner failed:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
