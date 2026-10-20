"""Writes the M1 eval token files (eval/*.ids) from local texts with the model tokenizer.

  python3 python/make_eval.py MODEL_DIR

ja / en / code are 512 tokens; long is 4096 tokens, past QSA's dense range (2051), so the
indexer's top-k selection is exercised.
"""

import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "ja": ([ROOT / "docs/m0.md"], 512),
    "en": ([Path(sys.argv[1]) / "README.md"], 512),
    "code": ([ROOT / "crates/tsumuji-engine/src/engine.rs"], 512),
    "long": ([ROOT / "docs/plan.md", ROOT / "crates/tsumuji-engine/src/weights.rs", ROOT / "docs/m0.md"], 4096),
}


def main() -> int:
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    (ROOT / "eval").mkdir(exist_ok=True)
    for name, (paths, n) in SOURCES.items():
        text = "\n\n".join(p.read_text() for p in paths)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < n:
            raise SystemExit(f"{name}: only {len(ids)} tokens")
        (ROOT / "eval" / f"{name}.ids").write_text(" ".join(map(str, ids[:n])) + "\n")
        print(f"{name}: {n} tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
