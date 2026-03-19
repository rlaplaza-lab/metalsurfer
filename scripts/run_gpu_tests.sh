#!/usr/bin/env bash
# Run integration tests that need a CUDA GPU (see pytest marker: gpu).
#
# Usage (from repo root, conda env "metalsurfer"):
#   ./scripts/run_gpu_tests.sh
# Or:
#   bash scripts/run_gpu_tests.sh /path/to/conda/envs/metalsurfer/bin/python
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${1:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
if [[ ! -x "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ ! -x "${PYTHON:-}" ]]; then
  echo "No Python interpreter found. Activate conda env metalsurfer or pass path to python." >&2
  exit 1
fi
# Do not set CUDA_VISIBLE_DEVICES unless you intend to hide the GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PYTHON" -m pytest tests/ -m gpu --tb=short -v "$@"
