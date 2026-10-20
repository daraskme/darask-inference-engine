#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir=upstream/gyro-llama
expected=9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected" ]]; then
  echo 'Backend revision differs from the pinned version; checkout preserved.' >&2
  exit 1
fi
for patch in "$PWD"/patches/*.patch; do
  if git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: ${patch##*/}"
  elif git -C "$source_dir" apply --check "$patch"; then
    git -C "$source_dir" apply "$patch"
    echo "Applied: ${patch##*/}"
  else
    echo "Conflicting local changes; checkout preserved: ${patch##*/}" >&2
    exit 1
  fi
done
