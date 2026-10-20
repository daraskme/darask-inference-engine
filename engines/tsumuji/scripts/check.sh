#!/usr/bin/env bash
# Tiny end-to-end test, then real-model decode speed, each through the GPU slot.
#   nix develop /mnt/NVME4TB/darask-inference-engine/engines/hayate#default -c scripts/check.sh [NEW_TOKENS]
set -uo pipefail
cd "$(dirname "$0")/.."
NEW=${1:-256}
T=$(ls -t target/release/deps/tiny_e2e-* | grep -v '\.d$' | head -1)
NEED=2 LABEL="Tsumuji tiny e2e" timeout 900 scripts/gpu-run.sh -- "$T" --ignored --nocapture 2>&1 | grep -E "checked|result|panicked|differs|identical"
NEED=80 LABEL="Tsumuji 速度" timeout 1800 scripts/gpu-run.sh -- target/release/tsumuji run \
  --model /mnt/NVME4TB/darask-inference-engine/local/hayate/src/Qwen3.8-Flash-Next-NVFP4 --ids eval/q1.ids --new "$NEW" --ctx 4096 2>&1 |
  grep -E "prompt|generated|^[0-9]|error" | cut -c1-160
