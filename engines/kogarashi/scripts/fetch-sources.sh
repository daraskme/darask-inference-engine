#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p upstream
fetch() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "upstream/$name" ]]; then
    git clone --no-checkout "$url" "upstream/$name"
    git -C "upstream/$name" checkout --detach "$revision"
  fi
  test "$(git -C "upstream/$name" rev-parse HEAD)" = "$revision" || {
    echo "Unexpected revision in upstream/$name; existing checkout preserved" >&2
    exit 1
  }
}
fetch gyro-llama https://gitlab.com/agentionai-group/llama.cpp.git 9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f
fetch strata https://github.com/Niko1221/Strata.git cbfd5b926b794b356195d3cb600191b0c3567540
fetch heretic https://github.com/p-e-w/heretic.git 09271b620229c469d51fcdcbd57911a0cfdb72bf
fetch deepinfer https://github.com/deepseek-ai/DeepInfer.git 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
fetch qtip https://github.com/Cornell-RelaxML/qtip.git e90c6688c8dfae326a3a81b5eb032db7c6680ec0
fetch exllamav3-fury https://github.com/carpedm20/exllamav3.git 4700920a5d283a561406709ed0e27d9dc3c137f7
