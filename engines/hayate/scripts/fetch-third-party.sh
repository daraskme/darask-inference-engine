#!/usr/bin/env bash
# Fetch pinned third-party sources into third_party/ (not committed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/flashinfer ]; then
  # Apache-2.0. Header-only CUDA templates used as attention building blocks.
  git clone --depth 1 --branch v0.7.1 https://github.com/flashinfer-ai/flashinfer.git third_party/flashinfer
  # CUTLASS at the commit FlashInfer pins (needed by its SM120 GEMMs). cccl comes with CUDA.
  git -C third_party/flashinfer submodule update --init --depth 1 3rdparty/cutlass
fi
git -C third_party/flashinfer log -1 --format='flashinfer %H %cd'
