#!/usr/bin/env bash
# Run every test marked @pytest.mark.gpu, each *phase* in a new Python process.
#
# TorchSim / FairChem keep large CUDA allocations for the lifetime of the interpreter.
# Running bayesian + ethene + saturation GPU tests in one pytest session often OOMs;
# splitting into separate subprocesses matches CI-style isolation. Conftest still clears
# autobatchers around individual @gpu tests within a phase.
#
# Usage (from repo root, with your scientific Python env activated):
#   ./scripts/run_gpu_tests.sh
# Or pass an explicit interpreter:
#   bash scripts/run_gpu_tests.sh /path/to/python
# Extra args are forwarded to every pytest invocation (e.g. -q).
#
# Opt out of killing stray GPU Pythons before each phase:
#   METALSURFER_CLEAR_GPU_PYTHON=0 ./scripts/run_gpu_tests.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
# Optional first argument: interpreter path (must not be forwarded to pytest).
if [[ $# -gt 0 ]] && [[ -x "$1" ]] && [[ "$(basename "$1")" == python* ]]; then
  PYTHON="$1"
  shift
fi
if [[ ! -x "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ ! -x "${PYTHON:-}" ]]; then
  echo "No Python interpreter found. Activate your environment or pass the path to python." >&2
  exit 1
fi
# Do not set CUDA_VISIBLE_DEVICES unless you intend to hide the GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTEST_ARGS=(--tb=short -v)
if [[ $# -gt 0 ]]; then
  PYTEST_ARGS+=("$@")
fi

_clear_if_enabled() {
  if [[ "${METALSURFER_CLEAR_GPU_PYTHON:-1}" != "0" ]]; then
    bash "$ROOT/scripts/clear_gpu_python_processes.sh" --yes
  fi
}

_run_phase() {
  _clear_if_enabled
  echo "==> GPU test phase: $*"
  "$PYTHON" -m pytest "$@" "${PYTEST_ARGS[@]}"
}

# One subprocess per module / test so VRAM is released when the interpreter exits.
_run_phase tests/test_bayesian.py -m gpu
_run_phase tests/test_integration_ethene_ru.py
_run_phase tests/test_integration_h2_ru_slab.py
_run_phase tests/test_integration_water_cu_slab.py
_run_phase tests/test_integration_co2_mof.py
_run_phase tests/test_integration_h2_pt12.py
_run_phase tests/test_saturation.py::test_run_saturation_screening_h2_ni111_real_gpu
_run_phase tests/test_saturation.py::test_run_saturation_screening_multi_mol_bo_real_gpu

_clear_if_enabled
echo "All GPU test phases finished."
