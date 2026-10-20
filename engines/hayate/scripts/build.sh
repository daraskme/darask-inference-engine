#!/usr/bin/env bash
# Optimized GLM workstation build, using the unified workspace and output directory.
set -euo pipefail
if [[ ${1:-} == main ]]; then shift; fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
RUSTFLAGS="-C target-cpu=${DARASK_CPU:-alderlake} ${RUSTFLAGS:-}" \
  cargo build --release -p hayate-cli --features cuda \
  --manifest-path "$root/Cargo.toml" --target-dir "$root/target" "$@"
