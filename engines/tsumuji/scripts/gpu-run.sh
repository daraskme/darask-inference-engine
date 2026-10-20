#!/usr/bin/env bash
# Runs a command on the GPU through Hayate's GPU slot so Studio's serve is never blocked
# (docs/GPU_SHARING.md in Hayate). Waits politely: no other declaration in gpu.d, serve runs no
# job, and NEED+2 GiB free; gpu_slot gives up after 30 s (exit 124) and we line up again.
#   NEED=12 scripts/gpu-run.sh -- target/release/tsumuji-bench all
set -uo pipefail
NEED=${NEED:-12}
PRIO=${PRIO:-5}
LABEL=${LABEL:-Tsumuji M0 計測}
D=/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.d
LOCK=/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.lock
TOOLS=/mnt/NVME4TB/darask-inference-engine/engines/hayate/python-tools
[ "${1:-}" = "--" ] && shift
idle() {
  local free g extra=0 holder serve_pid=""
  [ "$(find "$D" -name '*.json' 2>/dev/null | wc -l)" -eq 0 ] || return 1
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if g=$(curl -s -m 2 http://127.0.0.1:7880/v1/gpu) && [ -n "$g" ]; then
    [ -z "$(jq -r '.serve.job // empty' <<<"$g")" ] || return 1
    extra=$(jq '[.serve.models[]? | select(.slot) | .vram_bytes] | add // 0 | . / 1048576 | floor' <<<"$g")
    serve_pid=$(jq -r '.serve.pid // empty' <<<"$g")
  fi
  holder=$(grep ":$(stat -c %i "$LOCK") " /proc/locks 2>/dev/null | grep -v -- '->' | awk '{print $5}' | head -1)
  [ -z "$holder" ] || [ "$holder" = "$serve_pid" ] || return 1
  [ $((free + extra)) -ge $(((NEED + 2) * 1024)) ]
}
while true; do
  until idle; do sleep 20; done
  PYTHONPATH="$TOOLS" python3 -m hayate_tools.gpu_slot --need "$NEED" --timeout 30 --priority "$PRIO" \
    --who "tsumuji" --label "$LABEL" -- "$@"
  code=$?
  [ "$code" -ne 124 ] && exit "$code"
  echo "[$(date +%T)] gpu-run: no slot within 30 s, lining up again" >&2
  sleep 30
done
