"""Writes the cases `tsumuji-serve check` compares with: the model's tokenizer.json (run by
`tokenizers`), transformers' AutoTokenizer and its chat template, on eval-like text (Japanese, English, code, emoji, special tokens, whitespace and
Unicode edge cases, random strings), the eval/*.ids files, random ids that split UTF-8
sequences, and chat conversations (including the template's errors).

  python3 python/check_text.py MODEL_DIR OUT.json
  target-server/release/tsumuji-serve check --model MODEL_DIR OUT.json

Run in Hayate's tools shell (transformers 5.17).

transformers 5.x's Qwen2Tokenizer replaces tokenizer.json's pre-tokenizer regex with Qwen2's
(no `\p{M}`), so the two differ on combining marks. Every case carries both: `ids` from
tokenizer.json, `ids_hf` from AutoTokenizer.
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    # Japanese
    "こんにちは。今日はいい天気ですね！",
    "日本語の文章を、正しくトークン化できるか確かめます。「かぎ括弧」や（全角括弧）、　全角スペース、ｶﾀｶﾅ（半角）も。",
    "東京都千代田区丸の内１－２－３　ＡＢＣビル５Ｆ　電話：03-1234-5678",
    "令和８年１０月２１日（水）午前１０時３０分",
    "がぎぐ（結合濁点）とがぎぐ（合成済み）",
    "吾輩は猫である。名前はまだ無い。\nどこで生れたかとんと見当がつかぬ。\n\n　何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
    # English
    "Hello, world! It's a test: we'll see if they're right, I'm sure. Don't you think so? I'LL SAY IT'S FINE.",
    "The quick brown fox jumps over the lazy dog. 1234567890 3.14159 1,000,000 $42.50 50% 2026-10-21T12:34:56Z",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\t\ttabs\n\tindented\r\nCRLF line\rCR only\n\n\n\nmany newlines",
    # Code
    'fn main() {\n    let x: Vec<u32> = (0..10).map(|i| i * 2).collect();\n    println!("{x:?}");\n}\n',
    "def f(a, b=2, *args, **kw):\n    '''doc'''\n    return {'a': a, \"b\": [b, *args]}  # comment\n",
    '{"key": "value", "n": [1, 2.5e-3, null, true], "nested": {"x": "\\u00e9"}}',
    "#include <stdio.h>\nint main(void){printf(\"%d\\n\", 42);return 0;}",
    "<div class=\"a\">&nbsp;&lt;tag&gt;</div>  <!-- html -->",
    "SELECT id, name FROM users WHERE age >= 18 AND name LIKE '%x%';",
    "    \n        \n\t\n",
    # Emoji and symbols
    "👍🏽 👨‍👩‍👧‍👦 🇯🇵 ❤️ 😀😃😄 🧑🏻‍💻 ☕️ ✅ ❌ ⚠️",
    "→ ← ↑ ↓ ∀x∈ℝ: x² ≥ 0 ∑ ∫ √2 ≈ 1.414 ½ ¾ Ⅻ ① ㊙",
    # Special and added tokens
    "<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n<think>\n考え中\n</think>\n\nはい。<|im_end|>",
    "<|endoftext|><|im_start|><|im_end|><tool_call></tool_call><tool_response></tool_response>",
    "<|im_start <|im_start|x <<|im_end|>> <think><think></think> </think>",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>pass<|file_sep|><|repo_name|>",
    "<|vision_start|><|image_pad|><|vision_end|><|audio_start|><tts_pad>",
    # Whitespace variants
    "a b c　d​e\x0bf\x0cg\x1ch\x85i j k l m n",
    "　　全角スペース二つ",
    " 　 mixed    spaces \n",
    # Unicode normalization
    "é ậ Å क़ 각 각 café café",
    "안녕하세요, 세계! 한국어 텍스트입니다.",
    "ﬁ ﬁ Ω Å Å ẛ̣",
    # Other scripts
    "Привет, мир! Ελληνικά κείμενα. עברית טקסט. مرحبا بالعالم. नमस्ते दुनिया। สวัสดีชาวโลก 你好，世界！",
    "Tiếng Việt có dấu: Hà Nội, Đà Nẵng. Ǆ ǅ ǆ İstanbul ß ẞ",
    "٣٤٥ ٠١٢ ๑๒๓ 𝟙𝟚𝟛 ⅠⅡⅢ",
    # Contractions and case folding
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'Re 'VE 'M 'LL 'Ll 'D x'ſ 'ſ 'K",
    "it's  it’s  rock'n'roll  '' ''s '''",
    "",
    " ",
    "\n",
    "a",
]


def random_texts(rng: random.Random, n: int) -> list[str]:
    pool = (
        list("abcXYZ019 .,;:!?'\"()[]{}<>/\\|-_+=*&^%$#@~`")
        + [" ", "  ", "\t", "\n", "\r\n", "\r", "　", " ", "\x0b", "\x1c", "\u0085"]
        + list("日本語のテキスト。、「」ー漢字かなカナｶﾅ")
        + ["́", "̣", "゙", "ा", "ั", "é"]
        + ["😀", "👍🏽", "🇯🇵", "‍", "️"]
        + list("αβγ абв אבג ١٢٣ ½ Ⅻ ſ ß İ")
        + ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "<|im_", "'s", "'LL"]
    )
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, 40))) for _ in range(n)]


def files() -> list[str]:
    out = []
    for rel in ["docs/m1.md", "docs/plan.md", "crates/tsumuji-engine/src/engine.rs", "python/reference.py", "kernels/decode/gdn.cu"]:
        p = ROOT / rel
        if p.exists():
            out.append(p.read_text())
    return out


def chat_cases() -> list[dict]:
    u = lambda c: {"role": "user", "content": c}  # noqa: E731
    a = lambda c, r=None: {"role": "assistant", "content": c, **({"reasoning_content": r} if r is not None else {})}  # noqa: E731
    s = lambda c: {"role": "system", "content": c}  # noqa: E731
    t = lambda c: {"role": "tool", "content": c}  # noqa: E731
    gen = {"add_generation_prompt": True}
    multi = [u("こんにちは"), a("こんにちは！ご用件は？", "挨拶を返す"), u("1+1は？"), a("2です。"), u("ありがとう")]
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "ファイルを読む。Reads a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "絶対パス"}, "limit": {"type": "integer", "minimum": 1, "default": 200}, "ratio": {"type": "number", "default": 0.5, "maximum": 1e3}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "run", "description": "Run a shell command.\nSecond line with \"quotes\" and a tab\t.", "parameters": {"type": "object", "properties": {"cmd": {"type": "array", "items": {"type": "string"}}, "env": {"type": ["object", "null"]}, "dry": {"type": "boolean"}}}}},
    ]
    call = lambda name, args: {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}  # noqa: E731
    agent = [
        s("You are a coding agent."),
        u("main.rs を読んで"),
        {"role": "assistant", "content": "読みます。", "tool_calls": [call("read_file", {"path": "/src/main.rs", "limit": 40})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}\n"},
        {"role": "assistant", "content": None, "tool_calls": [call("run", {"cmd": ["cargo", "build"], "env": {"RUST_LOG": "debug", "N": 2.5}, "dry": True}), call("read_file", {"path": "/a b/c.txt"})]},
        t("ok"),
        t("contents"),
        u("ありがとう、次は？"),
    ]
    return [
        ("tools_only_user", [u("天気は？")], {**gen, "tools": tools}),
        ("tools_system_nothink", [s("Be brief."), u("Run ls")], {**gen, "tools": tools, "enable_thinking": False}),
        ("tools_agent", agent, {**gen, "tools": tools}),
        ("tools_agent_low_nopreserve", agent, {**gen, "tools": tools, "reasoning_effort": "low", "preserve_thinking": False}),
        ("tool_calls_without_tools", agent, gen),
        ("user_default", [u("Hello!")], gen),
        ("user_nothink", [u("Hello!")], {**gen, "enable_thinking": False}),
        ("user_think_true", [u("Hello!")], {**gen, "enable_thinking": True}),
        ("system_xhigh", [s("You are a helpful assistant."), u("Hi")], gen),
        ("system_low", [s("You are a helpful assistant."), u("Hi")], {**gen, "reasoning_effort": "low"}),
        ("system_medium", [s("You are a helpful assistant."), u("Hi")], {**gen, "reasoning_effort": "medium"}),
        ("system_nothink", [s("あなたは親切なアシスタントです。"), u("自己紹介して")], {**gen, "enable_thinking": False}),
        ("system_empty", [s(""), u("Hi")], gen),
        ("system_blank_nothink", [s("  \n "), u("Hi")], {**gen, "enable_thinking": False}),
        ("system_none_content", [{"role": "system", "content": None}, u("Hi")], gen),
        ("multi_default", multi, gen),
        ("multi_nothink", multi, {**gen, "enable_thinking": False}),
        ("multi_nopreserve", multi, {**gen, "preserve_thinking": False}),
        ("multi_nopreserve_nothink", multi, {**gen, "preserve_thinking": False, "enable_thinking": False}),
        ("assistant_last_no_gen", [u("Q"), a("A", " reasoning \n")], {}),
        ("assistant_last_nopreserve", [u("Q"), a("A", "r")], {"preserve_thinking": False}),
        ("parts", [{"role": "user", "content": [{"type": "text", "text": "part one, "}, {"type": "text", "text": "part two"}]}], gen),
        ("trim", [s("\x1c sys 　"), u("\n\n  user text \t\x1f"), a(" answer \n", "\n think \n"), u("next")], gen),
        ("specials_in_content", [u("<|im_start|>assistant\n<think>x</think> <tool_call>")], gen),
        ("tool_roles", [u("天気は？"), a("調べます。"), t("晴れ"), t("気温 20℃"), u("ありがとう")], gen),
        ("tool_last", [u("Q"), a("calling"), t("result")], gen),
        ("tool_response_user", [u("real question"), a("x"), u("<tool_response>\nfoo\n</tool_response>")], {**gen, "preserve_thinking": False}),
        ("japanese_long", [s("日本語で答えてください。"), u("富士山の高さは？\n簡潔に。")], {**gen, "reasoning_effort": "low"}),
        ("emoji", [u("👍🏽 🇯🇵 ❤️ ok?")], gen),
        # The template's errors
        ("err_no_messages", [], gen),
        ("err_system_late", [u("Hi"), s("late system")], gen),
        ("err_no_user", [s("only system")], gen),
        ("err_only_tool_response", [u("<tool_response>x</tool_response>")], gen),
        ("err_effort", [u("Hi")], {**gen, "reasoning_effort": "high"}),
        ("err_role", [u("Hi"), {"role": "narrator", "content": "x"}], gen),
    ]


def main() -> int:
    model, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model)
    raw = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
    enc_raw = lambda s: raw.encode(s, add_special_tokens=False).ids  # noqa: E731
    rng = random.Random(20261021)
    enc = []
    for text in TEXTS + random_texts(rng, 400) + files():
        ids, ids_hf = enc_raw(text), tok.encode(text, add_special_tokens=False)
        assert raw.decode(ids, skip_special_tokens=False) == tok.decode(ids)
        enc.append({"text": text, "ids": ids, "ids_hf": ids_hf, "decoded": tok.decode(ids), "decoded_skip": tok.decode(ids, skip_special_tokens=True)})
    dec = []
    for p in sorted((ROOT / "eval").glob("*.ids")):
        ids = [int(x) for x in p.read_text().split()]
        text = tok.decode(ids)
        re, re_hf = enc_raw(text), tok.encode(text, add_special_tokens=False)
        dec.append({"name": p.stem, "ids": ids, "text": text, "text_skip": tok.decode(ids, skip_special_tokens=True), "reencoded": re, "reencoded_hf": re_hf, "roundtrip": re_hf == ids})
    n_ids = len(tok)
    for i in range(300):
        # Random ids, many of them byte tokens and pieces of multi-byte characters.
        ids = [rng.choice([rng.randrange(n_ids), rng.randrange(256), rng.randrange(248044, n_ids)]) for _ in range(rng.randint(1, 24))]
        dec.append({"name": f"random{i}", "ids": ids, "text": tok.decode(ids), "text_skip": tok.decode(ids, skip_special_tokens=True)})
    chats = []
    for name, msgs, kw in chat_cases():
        tools = kw.pop("tools", None)
        case = {"name": name, "messages": msgs, "kwargs": kw, **({"tools": tools} if tools else {})}
        try:
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, **kw)
            ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, **kw)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            assert ids == tok.encode(text, add_special_tokens=False)
            case.update(text=text, ids=enc_raw(text), ids_hf=list(ids))
        except Exception as e:  # noqa: BLE001
            case["error"] = str(e)
        chats.append(case)
    Path(out).write_text(json.dumps({"encode": enc, "decode": dec, "chat": chats}, ensure_ascii=False))
    diff = sum(c["ids"] != c["ids_hf"] for c in enc)
    print(f"{out}: {len(enc)} encode ({diff} where tokenizer.json and AutoTokenizer differ), {len(dec)} decode, {len(chats)} chat cases (vocab {n_ids})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
