"""Compare the Rust renderer with local transformers apply_chat_template.

PYTHONPATH=python-tools python3 -m hayate_tools.serve_template_check --hayate target/debug/hayate
No downloads; arguments strings are normalized to mappings before HF rendering.
"""
import argparse
import copy
import itertools
import json
import subprocess


def glm_cases():
    tools = [
        {"type": "function", "function": {"name": "read", "description": "ファイルを読む <x> & \"q\"\n", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 2000},
                                             "ratio": {"type": "number", "minimum": 0.5}},
            "required": ["path"]}, "strict": True}},
        {"type": "function", "function": {"name": "lazy", "description": "deferred", "parameters": {}, "defer_loading": True}},
        {"type": "function", "function": {"name": "edit", "description": "置換", "parameters": {
            "type": "object", "properties": {"old_string": {"type": "string"}, "new_string": {"type": "string"},
                                             "flags": {"type": "array", "items": {"type": "boolean"}}}}}},
    ]
    args = {"path": "src/日本.rs", "limit": 3, "ratio": 1.0, "nested": {"x": [1, True, None, 2.5]}, "s": "line\n\"q\""}
    for string_args, effort, clear, generation, thinking, system, reasoning in itertools.product(
        (True, False), (None, "low", "high", "medium"), (None, True, False), (True, False), (None, True, False),
        (None, "  system 日\n", "developer"), (None, "R1 考え"),
    ):
        arguments = json.dumps(args, ensure_ascii=False) if string_args else args
        messages = []
        if system == "developer":
            messages.append({"role": "developer", "content": "dev rules"})
        elif system is not None:
            messages.append({"role": "system", "content": system})
        first = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": arguments}},
            {"id": "b", "type": "function", "function": {"name": "edit", "arguments": json.dumps({"old_string": " x\n", "new_string": "y", "flags": [False]})}},
        ]}
        if reasoning:
            first["reasoning_content"] = reasoning
        messages += [
            {"role": "user", "content": "  go 猫🐈 e\u0301\n"},
            first,
            {"role": "tool", "tool_call_id": "b", "content": "B ok"},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "A1"}, {"type": "text", "text": "A2"}]},
            {"role": "assistant", "content": " done \n", **({"reasoning_content": "R2"} if reasoning else {})},
            {"role": "user", "content": [{"type": "text", "text": "next"}, "raw"]},
            {"role": "assistant", "content": "<think>inline</think> answer", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "zz", "content": "unmatched"},
            {"role": "assistant", "content": "after", **({"reasoning_content": "R3"} if reasoning else {})},
        ]
        case = dict(model="glm53", messages=messages, tools=tools, add_generation_prompt=generation)
        kwargs = {}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if clear is not None:
            kwargs["clear_thinking"] = clear
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        if string_args:
            case["chat_template_kwargs"] = kwargs
        else:
            case.update(kwargs)
        yield case
    yield dict(model="glm53", messages=[{"role": "user", "content": "hi"}])
    yield dict(model="glm53", messages=[{"role": "user", "content": "hi"}], tools=[])
    yield dict(model="glm53", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}],
               reasoning_effort="none")
    yield dict(model="glm53", messages=[{"role": "assistant", "content": "x", "reasoning_content": "r"},
                                        {"role": "tool", "content": "t"}, {"role": "tool", "content": "u"}])


def glm_expected(tokenizer, case):
    """HF rendering plus the serve conventions documented in template/glm.rs."""
    hf = copy.deepcopy(case)
    hf.pop("model")
    hf.update(hf.pop("chat_template_kwargs", {}))
    thinking = hf.pop("enable_thinking", True)
    if hf.get("reasoning_effort") == "none":
        thinking = False
    for message in hf["messages"]:
        if message["role"] == "developer":
            message["role"] = "system"
        if message.get("content", 0) is None:
            del message["content"]
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    generation = hf.setdefault("add_generation_prompt", True)
    text = tokenizer.apply_chat_template(hf.pop("messages"), tokenize=False, **hf)
    return text + ("</think>" if generation and not thinking else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hayate", default="target/debug/hayate")
    parser.add_argument("--model", choices=("glm53",), default="glm53")
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    from transformers import AutoTokenizer, __version__
    directory = args.model_dir or "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8"
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    corpus = list(glm_cases())
    result = subprocess.run([args.hayate, "serve", "--render-template"],
                            input="".join(json.dumps(c, ensure_ascii=False) + "\n" for c in corpus),
                            text=True, capture_output=True, check=True)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(corpus), (len(actual), len(corpus), result.stderr)
    for i, (case, rust) in enumerate(zip(corpus, actual)):
        expected = glm_expected(tokenizer, case)
        if rust.encode() != expected.encode():
            at = next((j for j, (a, b) in enumerate(zip(rust, expected)) if a != b), min(len(rust), len(expected)))
            raise AssertionError(f"case={i} at={at}\ncase={case!r}\nRust={rust[max(0,at-60):at+120]!r}\nHF={expected[max(0,at-60):at+120]!r}")
    print(json.dumps({"transformers": __version__, "model": args.model, "cases": len(corpus), "byte_identical": True}))


if __name__ == "__main__":
    main()
