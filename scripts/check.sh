#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/engines/kogarashi:$PWD/engines/hayate/python-tools${PYTHONPATH:+:$PYTHONPATH}"
cargo test --offline
cargo test --offline -p tsumuji-server --no-default-features
python3 -m unittest discover -s engines/kogarashi/tests
python3 -m unittest hayate_tools.test_gpu_slot
