#!/usr/bin/env bash
# Terminate *this user's* Python-related GPU compute clients so VRAM is released
# after stray pytest / notebooks / Ray workers. Uses nvidia-smi compute-app PIDs.
#
# Usage:
#   ./scripts/clear_gpu_python_processes.sh          # list only, exit 1 if any found
#   ./scripts/clear_gpu_python_processes.sh --yes    # SIGTERM, then SIGKILL if needed
#   ./scripts/clear_gpu_python_processes.sh --dry-run
set -euo pipefail

YES=0
DRY=0
for a in "$@"; do
  case "$a" in
    --yes) YES=1 ;;
    --dry-run) DRY=1 ;;
  esac
done

MYUID=$(id -u)
SELF=$$
PROTECT_MARKER="${METALSURFER_GPU_PROTECT_MARKER:-METALSURFER_KEEP_GPU=1}"

_is_python_like() {
  local pid=$1
  local exe cmd
  exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
  if [[ "$exe" == *python* ]]; then
    return 0
  fi
  cmd=$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)
  if [[ "$cmd" == *[Pp]ython* ]] || [[ "$cmd" == *torchrun* ]] || [[ "$cmd" == *ipython* ]]; then
    return 0
  fi
  return 1
}

_is_protected() {
  local pid=$1
  local environ
  # Skip intentional long GPU jobs that export METALSURFER_KEEP_GPU=1.
  environ=$(tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null || true)
  [[ "$environ" == *"${PROTECT_MARKER}"* ]]
}

found=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line// /}"
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^[0-9]+$ ]] || continue
  pid="$line"
  [[ "$pid" -eq "$SELF" ]] && continue
  [[ -d "/proc/$pid" ]] || continue
  owner=$(stat -c %u "/proc/$pid" 2>/dev/null || echo "")
  [[ "$owner" == "$MYUID" ]] || continue
  _is_python_like "$pid" || continue
  if _is_protected "$pid"; then
    echo "Skipping protected GPU compute PID $pid (METALSURFER_KEEP_GPU)"
    continue
  fi

  found=1
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || echo "(unknown)")
  if [[ "$YES" -eq 0 ]]; then
    if [[ "$DRY" -eq 1 ]]; then
      echo "Would clear GPU compute PID $pid: $cmd"
    else
      echo "GPU compute PID $pid: $cmd"
    fi
    continue
  fi
  echo "Sending SIGTERM to GPU compute PID $pid: $cmd"
  kill -TERM "$pid" 2>/dev/null || true
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

if [[ "$found" -eq 0 ]]; then
  echo "No user-owned Python GPU compute processes found."
  exit 0
fi

if [[ "$DRY" -eq 1 ]]; then
  exit 0
fi

if [[ "$YES" -eq 0 ]]; then
  echo "Run with --yes to terminate them (run_gpu_tests.sh does this unless METALSURFER_CLEAR_GPU_PYTHON=0)." >&2
  exit 1
fi

# Give processes time to release CUDA context
sleep 2

still=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line// /}"
  [[ "$line" =~ ^[0-9]+$ ]] || continue
  pid="$line"
  [[ -d "/proc/$pid" ]] || continue
  owner=$(stat -c %u "/proc/$pid" 2>/dev/null || echo "")
  [[ "$owner" == "$MYUID" ]] || continue
  _is_python_like "$pid" || continue
  if _is_protected "$pid"; then
    continue
  fi
  still=1
  echo "PID $pid still running; SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

if [[ "$still" -eq 1 ]]; then
  sleep 1
fi

echo "GPU Python compute cleanup finished."
