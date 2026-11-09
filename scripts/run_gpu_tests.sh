#!/usr/bin/env bash
# GPU suite: one subprocess per collected node so VRAM is released on exit.
#
# Node ids come from ``pytest --collect-only -m gpu`` (or ``gpu_smoke``).
# A new ``@gpu_mlip_test`` is picked up without editing this script.
#
# Requires: pip install -e ".[mlip]", CUDA, HuggingFace access for UMA models.
#
# Usage:
#   ./scripts/run_gpu_tests.sh
#   bash scripts/run_gpu_tests.sh /path/to/python
#   METALSURFER_GPU_MODE=smoke ./scripts/run_gpu_tests.sh
#
# Modes (METALSURFER_GPU_MODE):
#   full  (default) — every test marked gpu
#   smoke — gpu_smoke subset
#
# Opt in to killing stray user GPU Pythons around the run (disposable machines):
#   METALSURFER_CLEAR_GPU_PYTHON=1 ./scripts/run_gpu_tests.sh
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
# CUDA workers must not fork after a CUDA context exists.
export METALSURFER_MP_SPAWN=1

MODE="${METALSURFER_GPU_MODE:-full}"
case "${MODE}" in
  full|smoke) ;;
  *)
    echo "METALSURFER_GPU_MODE must be 'full' or 'smoke' (got: ${MODE})" >&2
    exit 1
    ;;
esac

PYTEST_ARGS=(--tb=short -v)
if [[ $# -gt 0 ]]; then
  PYTEST_ARGS+=("$@")
fi

_clear_if_enabled() {
  if [[ "${METALSURFER_CLEAR_GPU_PYTHON:-0}" == "1" ]]; then
    bash "$ROOT/scripts/clear_gpu_python_processes.sh" --yes
  fi
}

if [[ "${MODE}" == "smoke" ]]; then
  MARKER="gpu_smoke"
else
  MARKER="gpu"
fi

COLLECT_OUT="$(mktemp)"
COLLECT_ERR="$(mktemp)"
cleanup_collect() {
  rm -f "$COLLECT_OUT" "$COLLECT_ERR"
}
trap 'cleanup_collect; _clear_if_enabled' EXIT

set +e
"$PYTHON" -m pytest tests/ --collect-only -q -m "$MARKER" \
  --override-ini="addopts=" -p no:cacheprovider \
  >"$COLLECT_OUT" 2>"$COLLECT_ERR"
collect_status=$?
set -e
if [[ "$collect_status" -ne 0 ]]; then
  cat "$COLLECT_ERR" >&2
  echo "GPU collection failed for -m ${MARKER} (exit ${collect_status})" >&2
  exit "$collect_status"
fi

mapfile -t NODES < <(
  "$PYTHON" -c 'import sys; from tests.gpu_nodes import node_ids_from_collect_output
for node in node_ids_from_collect_output(sys.stdin.read()):
    print(node)' <"$COLLECT_OUT"
)

if [[ "${#NODES[@]}" -eq 0 ]]; then
  cat "$COLLECT_ERR" >&2
  echo "No tests collected for -m ${MARKER}" >&2
  exit 1
fi

echo "Collected ${#NODES[@]} GPU test(s) for -m ${MARKER}"
failed=0
declare -a FAILED_NODES=()
for node in "${NODES[@]}"; do
  echo "==> GPU (${MODE}): ${node}"
  set +e
  "$PYTHON" -m pytest "$node" -m "$MARKER" "${PYTEST_ARGS[@]}"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "===== NODE FAILED: ${node} (exit ${status}) ====="
    FAILED_NODES+=("$node")
    failed=1
  else
    echo "===== NODE PASSED: ${node} ====="
  fi
done

echo ""
echo "===== GPU RUN SUMMARY (mode=${MODE}) ====="
if [[ "$failed" -ne 0 ]]; then
  for node in "${FAILED_NODES[@]}"; do
    echo "  FAIL  ${node}"
  done
  exit 1
fi
echo "All GPU tests finished (mode=${MODE}, n=${#NODES[@]})."
