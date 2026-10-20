"""Talks to tsumuji-serve through the official `openai` client: models, chat (non-streaming and
streaming, thinking on and off, Japanese and English, multi-turn), completions, and a tool call
(streaming and not) with the result sent back. Prints the answers, the speeds and the cached
prompt tokens the server reports.

  python3 python/api_smoke.py [BASE_URL]      (default http://127.0.0.1:8090/v1)

Hayate's tools shell has no `openai`; a Python with it (scratch, not part of the repo):
  nix build --impure --expr '(import (builtins.getFlake "github:NixOS/nixpkgs/<rev from flake.lock>") {}).python3.withPackages (p: [p.openai])' -o /tmp/pyenv
"""

import sys
import time

import openai

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090/v1"


def speeds(extra) -> str:
    t = (extra or {}).get("timings") or {}
    return f"prefill {t.get('prefill_ms')} ms ({t.get('prefill_tok_s')} tok/s), decode {t.get('decode_tok_s')} tok/s ({t.get('sampling')})"


def main() -> int:
    client = openai.OpenAI(base_url=BASE, api_key="none")
    model = client.models.list().data[0].id
    print(f"model: {model}")

    # Non-streaming, thinking off, Japanese, greedy.
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "日本の首都はどこですか？一文で答えてください。"}],
        temperature=0,
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(f"\n[ja, no think, greedy] {r.choices[0].message.content!r} finish={r.choices[0].finish_reason} usage={r.usage.prompt_tokens}+{r.usage.completion_tokens}")
    print("  " + speeds(r.model_extra))

    # Non-streaming, thinking on (low effort), English, sampled.
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are concise."}, {"role": "user", "content": "What is 17 * 23? Answer with the number."}],
        max_tokens=1024,
        seed=1,
        reasoning_effort="low",
    )
    m = r.choices[0].message
    reasoning = (m.model_extra or {}).get("reasoning_content") or ""
    print(f"\n[en, think low, sampled] reasoning {len(reasoning)} chars: {reasoning[:200]!r}...")
    print(f"  answer {m.content!r} finish={r.choices[0].finish_reason} usage={r.usage.prompt_tokens}+{r.usage.completion_tokens}")
    print("  " + speeds(r.model_extra))

    # Streaming, multi-turn, Japanese, thinking off.
    t0 = time.time()
    first = None
    content, n_chunks, usage = "", 0, None
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "好きな果物を一つ挙げて。"},
            {"role": "assistant", "content": "りんごです。"},
            {"role": "user", "content": "その理由を二文で説明して。"},
        ],
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.7,
        max_tokens=200,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for ch in stream:
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            if c.delta.content:
                first = first or time.time()
                content += c.delta.content
                n_chunks += 1
    dt = time.time() - (first or t0)
    print(f"\n[ja, stream, multi-turn] {content!r}")
    print(f"  {n_chunks} chunks, first text after {(first or t0) - t0:.2f} s, usage {usage.prompt_tokens}+{usage.completion_tokens}, client-side {usage.completion_tokens / max(dt, 1e-9):.1f} tok/s")

    # Streaming with reasoning, English.
    reasoning, content = "", ""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name one prime number between 10 and 20. Be brief."}],
        stream=True,
        temperature=0,
        max_tokens=512,
        reasoning_effort="low",
    )
    for ch in stream:
        for c in ch.choices:
            d = c.delta.model_extra or {}
            reasoning += d.get("reasoning_content") or ""
            content += c.delta.content or ""
    print(f"\n[en, stream, think low] reasoning {len(reasoning)} chars, answer {content!r}")

    # Completions.
    r = client.completions.create(model=model, prompt="The three primary colors are", max_tokens=24, temperature=0)
    print(f"\n[completion] {r.choices[0].text!r} finish={r.choices[0].finish_reason}")

    # A tool call, streamed (tool_calls arrive whole after </tool_call>), then the result sent back.
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a text file and return its contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path of the file"}, "limit": {"type": "integer", "description": "Most lines to return"}}, "required": ["path"]}}}]
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools."}, {"role": "user", "content": "README.md の先頭 20 行を読んで要約して。"}]
    calls: dict[int, dict] = {}
    content = ""
    usage = None
    stream = client.chat.completions.create(model=model, messages=msgs, tools=tools, stream=True, temperature=0, max_tokens=1024, extra_body={"enable_thinking": False})
    for ch in stream:
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            content += c.delta.content or ""
            for t in c.delta.tool_calls or []:
                calls.setdefault(t.index, {"id": t.id, "name": t.function.name, "arguments": ""})["arguments"] += t.function.arguments or ""
    print(f"\n[tools, stream] content {content!r}, tool_calls {list(calls.values())}, cached {usage.prompt_tokens_details.cached_tokens if usage else '?'}")
    if not calls:
        print("  (the model answered without a tool call)")
        return 0
    c0 = calls[0]
    msgs.append({"role": "assistant", "content": content or None, "tool_calls": [{"id": c0["id"], "type": "function", "function": {"name": c0["name"], "arguments": c0["arguments"]}}]})
    msgs.append({"role": "tool", "tool_call_id": c0["id"], "content": "# Tsumuji\n\nQwen3.8-Flash-Next 専用の Rust + CUDA 推論エンジンです。"})
    r = client.chat.completions.create(model=model, messages=msgs, tools=tools, temperature=0, max_tokens=256, extra_body={"enable_thinking": False})
    u = r.usage
    print(f"[tools, result back] {r.choices[0].message.content!r} finish={r.choices[0].finish_reason}, prompt {u.prompt_tokens} of which cached {u.prompt_tokens_details.cached_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
