#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir=upstream/gyro-llama
if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo 'Run scripts/fetch-sources.sh first.' >&2
  exit 1
fi
bash scripts/apply-backend-patches.sh
cmake -S "$source_dir" -B build/backend -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_HTTPLIB=ON \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)"
cmake --build build/backend --parallel "${KOGARASHI_BUILD_JOBS:-8}" \
  --target llama-server llama-bench llama-quantize llama-perplexity test-backend-ops
