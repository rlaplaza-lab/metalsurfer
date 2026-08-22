#!/usr/bin/env bash
# GPU suite: pytest -m gpu, one subprocess per file or test (VRAM isolation).
#
# Requires: pip install -e ".[mlip]", CUDA, HuggingFace access for UMA models.
#
# Usage:
#   ./scripts/run_gpu_tests.sh
#   bash scripts/run_gpu_tests.sh /path/to/python
#
# Opt out of killing stray GPU Pythons before each phase:
#   METALSURFER_CLEAR_GPU_PYTHON=0 ./scripts/run_gpu_tests.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
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
  echo "==> GPU: $*"
  "$PYTHON" -m pytest "$@" "${PYTEST_ARGS[@]}"
}

# Split heavy modules so VRAM is released when each interpreter exits.
_run_phase tests/test_bayesian.py -m gpu
_run_phase "tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[ethene_ru]" -m gpu
_run_phase "tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[h2_ru]" -m gpu
_run_phase tests/test_integration_water_cu_slab.py -m gpu
_run_phase "tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[co2_mof]" -m gpu
_run_phase "tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[h2_pt12]" -m gpu
_run_phase tests/test_saturation.py::test_run_saturation_screening_h2_ni111_real_gpu -m gpu
_run_phase tests/test_saturation.py::test_run_saturation_screening_multi_mol_bo_real_gpu -m gpu

_clear_if_enabled
echo "All GPU tests finished."
