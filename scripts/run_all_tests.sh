#!/usr/bin/env bash
# Run all test suites: quick (CI), remaining CPU, then GPU.
#
# Usage (from repo root, with metalsurfer conda env activated):
#   ./scripts/run_all_tests.sh
#   nohup bash scripts/run_all_tests.sh > logs/test_runs/overnight_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# Optional first argument: python interpreter path.
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
  echo "No Python interpreter found. Activate conda env metalsurfer or pass python path." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${ROOT}/logs/test_runs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PHASE_LOG="${LOG_DIR}/phase_${STAMP}.log"

declare -a PHASE_NAMES=()
declare -a PHASE_STATUS=()

_run_phase() {
  local name="$1"
  shift
  echo "===== PHASE: ${name} =====" | tee -a "$PHASE_LOG"
  set +e
  "$@" 2>&1 | tee -a "$PHASE_LOG"
  local status=${PIPESTATUS[0]}
  set -e
  PHASE_NAMES+=("$name")
  PHASE_STATUS+=("$status")
  if [[ "$status" -ne 0 ]]; then
    echo "===== PHASE FAILED: ${name} (exit ${status}) =====" | tee -a "$PHASE_LOG"
  else
    echo "===== PHASE PASSED: ${name} =====" | tee -a "$PHASE_LOG"
  fi
}

_run_phase "1_quick" \
  bash -c "
    '$PYTHON' -m pytest tests/ -m quick \
      --cov=src/metalsurfer \
      --cov-report=term-missing \
      --tb=short -v && \
    '$PYTHON' -m coverage report --fail-under=85
  "

_run_phase "2_cpu" \
  "$PYTHON" -m pytest tests/ -m "cpu and not quick" --tb=short -v

_run_phase "3_gpu" \
  bash "$ROOT/scripts/run_gpu_tests.sh" "$PYTHON"

echo "" | tee -a "$PHASE_LOG"
echo "===== TEST RUN SUMMARY =====" | tee -a "$PHASE_LOG"
failed=0
for i in "${!PHASE_NAMES[@]}"; do
  name="${PHASE_NAMES[$i]}"
  status="${PHASE_STATUS[$i]}"
  if [[ "$status" -eq 0 ]]; then
    echo "  PASS  ${name}" | tee -a "$PHASE_LOG"
  else
    echo "  FAIL  ${name} (exit ${status})" | tee -a "$PHASE_LOG"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "One or more phases failed. See ${PHASE_LOG}" | tee -a "$PHASE_LOG"
  exit 1
fi

echo "All test phases passed. Log: ${PHASE_LOG}" | tee -a "$PHASE_LOG"
