#!/usr/bin/env bash
# Download one Hugging Face model repo with curl (no Python needed).
# Resumable: files whose size already matches are skipped, partial files resume with -C -.
# Gated repos: the user's token (~/.cache/huggingface/token) is sent to huggingface.co only (curl
# drops it on the redirect to the CDN). EXCLUDE=<regex> skips matching paths.
# usage: [EXCLUDE=regex] hf-download.sh <owner/repo> <dest-dir> [revision]
set -euo pipefail
repo=$1 dest=$2 rev=${3:-main}
mkdir -p "$dest"
auth=()
if [[ -f ~/.cache/huggingface/token ]]; then auth=(-H "Authorization: Bearer $(cat ~/.cache/huggingface/token)"); fi
list=$(curl -sfL --retry 5 "${auth[@]}" "https://huggingface.co/api/models/$repo/tree/$rev?recursive=true" |
       jq -r '.[] | select(.type=="file") | "\(.size)\t\(.path)"')
if [[ -n ${EXCLUDE:-} ]]; then list=$(grep -Ev "$EXCLUDE" <<<"$list" || true); fi
total=$(awk -F'\t' '{s+=$1} END {print s}' <<<"$list")
n=$(wc -l <<<"$list")
echo "$(date '+%F %T') $repo@$rev -> $dest: $n files, $((total / 1000000000)) GB"
i=0
while IFS=$'\t' read -r size path; do
  i=$((i + 1))
  out="$dest/$path"
  mkdir -p "$(dirname "$out")"
  if [[ -f $out && $(stat -c %s "$out") == "$size" ]]; then continue; fi
  for try in 1 2 3 4 5 6; do
    if curl -sfL --retry 5 --retry-delay 5 "${auth[@]}" -C - -o "$out" "https://huggingface.co/$repo/resolve/$rev/$path" &&
       [[ $(stat -c %s "$out") == "$size" ]]; then break; fi
    echo "$(date '+%F %T') retry $try $path"; sleep $((try * 10))
  done
  [[ $(stat -c %s "$out") == "$size" ]] || { echo "FAILED $path"; exit 1; }
  echo "$(date '+%F %T') [$i/$n] $path $((size / 1000000)) MB"
done <<<"$list"
echo "$(date '+%F %T') done $repo"
