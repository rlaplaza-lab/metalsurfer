#!/usr/bin/env bash
# Run all official examples except bipyridine (heavy HPC saturation demo).
#
# Usage (from repo root, with metalsurfer conda env and GPU):
#   ./scripts/run_all_examples.sh
#   nohup bash scripts/run_all_examples.sh > logs/example_runs/v0.3_all_$(date +%Y%m%d_%H%M).log 2>&1 &
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

LOG_DIR="${ROOT}/logs/example_runs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

EXAMPLES=(
  examples/h2_pt12_binding_energy.py
  examples/ethene_ru_slab_binding_energy.py
  examples/co2_mof_binding_energy.py
  examples/camphor_cu111_binding_energy.py
)

declare -a EXAMPLE_NAMES=()
declare -a EXAMPLE_STATUS=()

for example in "${EXAMPLES[@]}"; do
  name="$(basename "$example" .py)"
  log_file="${LOG_DIR}/v0.3_${name}_${STAMP}.log"
  echo "===== START ${example} =====" | tee -a "$log_file"
  set +e
  "$PYTHON" "$example" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  set -e
  EXAMPLE_NAMES+=("$name")
  EXAMPLE_STATUS+=("$status")
  if [[ "$status" -eq 0 ]]; then
    echo "===== END ${example} exit=0 =====" | tee -a "$log_file"
  else
    echo "===== END ${example} exit=${status} =====" | tee -a "$log_file"
    echo "FAILED: ${example} (see ${log_file})" >&2
  fi
done

echo ""
echo "===== EXAMPLE RUN SUMMARY ====="
failed=0
for i in "${!EXAMPLE_NAMES[@]}"; do
  name="${EXAMPLE_NAMES[$i]}"
  status="${EXAMPLE_STATUS[$i]}"
  if [[ "$status" -eq 0 ]]; then
    echo "  PASS  ${name}"
  else
    echo "  FAIL  ${name} (exit ${status})"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "All examples passed."
