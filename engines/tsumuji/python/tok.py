"""Token ids for the CLI until Tsumuji has its own tokenizer (M5).

  python3 python/tok.py encode MODEL_DIR OUT.ids [--chat] < text
  python3 python/tok.py decode MODEL_DIR < ids

`--chat` wraps the text as one user turn with the model's chat template (thinking off).
Run in Hayate's tools shell (transformers 5.17).
"""

import sys

from transformers import AutoTokenizer


def main() -> int:
    cmd, model = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model)
    if cmd == "encode":
        text = sys.stdin.read()
        if "--chat" in sys.argv:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
        else:
            ids = tok.encode(text, add_special_tokens=False)
        with open(sys.argv[3], "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        print(f"{len(ids)} tokens", file=sys.stderr)
    elif cmd == "decode":
        ids = [int(x) for x in sys.stdin.read().split()]
        print(tok.decode(ids))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
