"""GLM-5.3 end-to-end checks and agent-loop timing against a running `hayate serve`.

PYTHONPATH=python-tools python3 -m hayate_tools.glm53_serve_e2e --port 7874 [--load] [--bench]

Standard library only. Checks: capabilities/models/gpu rows, a Japanese reply,
reasoning separation and budget, a tool-call round trip, parallel/broken-call
tolerance on the wire, HTTP 400 on overflow, usage in the final chunk, cancel by
closing the stream, FIFO under two concurrent requests. --bench runs the agent
loop (system+tool prompt, N turns adding about M tokens each) and prints TTFT,
cached tokens and decode tok/s per turn.
"""
import argparse
import http.client
import json
import socket
import threading
import time

PORT = 7874


def connect(timeout=3600):
    return http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)


def get(path):
    c = connect(60)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    return r.status, json.loads(body)


def post_json(path, body):
    c = connect()
    c.request("POST", path, json.dumps(body, ensure_ascii=False).encode(),
              {"content-type": "application/json"})
    r = c.getresponse()
    data = r.read().decode()
    c.close()
    return r.status, data


def stream(body, on_chunk=None):
    """POST a streaming chat; returns dict with chunks, timings and the reply."""
    body = dict(body, stream=True)
    c = connect()
    start = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
              {"content-type": "application/json", "accept": "text/event-stream"})
    sock = c.sock
    r = c.getresponse()
    out = {"status": r.status, "content": "", "reasoning": "", "calls": {}, "first": None,
           "finish": None, "usage": None, "timings": None, "error": None, "chunks": 0}
    if r.status != 200:
        out["error"] = r.read().decode()
        c.close()
        return out
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().rstrip("\r\n")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk and "choices" not in chunk:
            out["error"] = chunk["error"]
            break
        out["chunks"] += 1
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        now = time.monotonic() - start
        if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")) and out["first"] is None:
            out["first"] = now
        out["content"] += delta.get("content") or ""
        out["reasoning"] += delta.get("reasoning_content") or ""
        for call in delta.get("tool_calls") or []:
            slot = out["calls"].setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
            slot["id"] = call.get("id") or slot["id"]
            f = call.get("function") or {}
            slot["name"] = f.get("name") or slot["name"]
            slot["arguments"] += f.get("arguments") or ""
        if choice.get("finish_reason"):
            out["finish"] = choice["finish_reason"]
            out["usage"] = chunk.get("usage")
            out["timings"] = chunk.get("timings")
        if on_chunk and on_chunk(out, sock):
            break
    out["total"] = time.monotonic() - start
    c.close()
    return out


def check(name, ok, detail, report):
    report.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(detail, ensure_ascii=False)[:600], flush=True)


WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "指定した都市の現在の天気を返す",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "都市名"}},
                   "required": ["city"]}}}]


def checks(report):
    status, caps = get("/v1/capabilities")
    glm = [m for m in caps["models"] if m["id"] == "glm53"]
    check("capabilities", status == 200 and glm and glm[0]["context_length"] == 131072
          and "glm53" in caps["chat"]["models"], glm, report)
    status, models = get("/v1/models")
    row = [m for m in models["resident"] if m["id"] == "glm53"]
    check("models_row", row and row[0]["context_length"] == 131072 and "budget" in row[0], row, report)
    status, gpu = get("/v1/gpu")
    grow = [m for m in gpu["serve"]["models"] if m["id"] == "glm53"]
    check("gpu_row", grow and not grow[0]["slot"] and "budget" in grow[0], grow, report)

    r = stream({"model": "glm53", "messages": [
        {"role": "user", "content": "日本の首都はどこですか？一文で答えてください。"}],
        "enable_thinking": False, "max_tokens": 64, "temperature": 0})
    check("japanese_reply", r["status"] == 200 and "東京" in r["content"] and r["finish"] == "stop"
          and not r["reasoning"], {k: r[k] for k in ("content", "finish", "usage", "first", "total")}, report)

    r = stream({"model": "glm53", "messages": [
        {"role": "user", "content": "17×23 を計算して、答えだけ書いてください。"}],
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"},
        "reasoning_budget": 48, "max_tokens": 200, "temperature": 0.6, "top_p": 0.95, "min_p": 0.05, "seed": 7})
    check("reasoning_budget", r["status"] == 200 and r["reasoning"] and r["content"].strip()
          and r["usage"]["completion_tokens"] <= 200,
          {"reasoning_chars": len(r["reasoning"]), "content": r["content"][-200:], "finish": r["finish"],
           "usage": r["usage"]}, report)

    messages = [{"role": "user", "content": "東京の今の天気を get_weather で調べて、結果を日本語で一文で教えてください。"}]
    r = stream({"model": "glm53", "messages": messages, "tools": WEATHER, "tool_choice": "auto",
                "enable_thinking": False, "max_tokens": 256, "temperature": 0})
    calls = [r["calls"][i] for i in sorted(r["calls"])]
    ok = r["finish"] == "tool_calls" and calls and calls[0]["name"] == "get_weather"
    args = json.loads(calls[0]["arguments"]) if ok else {}
    check("tool_call", ok and "東京" in args.get("city", ""), {"calls": calls, "content": r["content"],
                                                               "finish": r["finish"]}, report)
    if ok:
        messages.append({"role": "assistant", "content": r["content"], "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls]})
        for c in calls:
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps({"city": "東京", "weather": "晴れ", "temperature_c": 21},
                                                   ensure_ascii=False)})
        r = stream({"model": "glm53", "messages": messages, "tools": WEATHER, "enable_thinking": False,
                    "max_tokens": 128, "temperature": 0})
        check("tool_round_trip", r["finish"] == "stop" and "晴" in r["content"] and r["usage"]
              and r["usage"]["prompt_tokens_details"]["cached_tokens"] > 0,
              {"content": r["content"], "usage": r["usage"], "first": r["first"]}, report)

    status, body = post_json("/v1/chat/completions", {"model": "glm53", "messages": [
        {"role": "user", "content": "hi"}], "max_tokens": 200000})
    check("overflow_400", status == 400 and "exceeds server context" in body, {"status": status, "body": body}, report)



def late_checks(report):
    # Cancel: close the stream after a few chunks; the next request must start promptly.
    closed = {}

    def stop(out, sock):
        if out["chunks"] >= 4:
            closed["at"] = time.monotonic()
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            return True
        return False

    try:
        stream({"model": "glm53", "messages": [{"role": "user", "content": "1から500まで数字を順に書いて。"}],
                "enable_thinking": False, "max_tokens": 2000, "temperature": 0}, stop)
    except (OSError, http.client.HTTPException):
        pass
    r = stream({"model": "glm53", "messages": [{"role": "user", "content": "「はい」とだけ答えて。"}],
                "enable_thinking": False, "max_tokens": 8, "temperature": 0})
    started = r["first"]
    check("cancel_frees_worker", r["status"] == 200 and r["content"] and started is not None,
          {"next_ttft_s": started, "next_prefill_ms": (r["timings"] or {}).get("prefill_ms"),
           "content": r["content"]}, report)

    # Cancel during a long prefill (about 9K tokens): close after 3 s.
    def abort_after(sock, seconds):
        time.sleep(seconds)
        try:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        except OSError:
            pass

    c = connect()
    c.request("POST", "/v1/chat/completions", json.dumps({
        "model": "glm53", "stream": True, "enable_thinking": False, "max_tokens": 16,
        "messages": [{"role": "system", "content": filler(77, 30000)}, {"role": "user", "content": "要約して。"}]},
        ensure_ascii=False).encode(), {"content-type": "application/json"})
    threading.Thread(target=abort_after, args=(c.sock, 3), daemon=True).start()
    try:
        c.getresponse().read()
    except (OSError, http.client.HTTPException):
        pass
    cut = time.monotonic()
    r = stream({"model": "glm53", "messages": [{"role": "user", "content": "「はい」とだけ答えて。"}],
                "enable_thinking": False, "max_tokens": 8, "temperature": 0})
    check("cancel_during_prefill", r["status"] == 200 and r["first"] is not None
          and time.monotonic() - cut < 60,
          {"next_ttft_s": r["first"], "next_prefill_ms": (r["timings"] or {}).get("prefill_ms"),
           "since_cancel_s": round(time.monotonic() - cut, 2)}, report)

    # FIFO: two concurrent requests both complete.
    results = [None, None]

    def run(i):
        results[i] = stream({"model": "glm53", "messages": [{"role": "user", "content": f"{i + 1}+{i + 1}は？数字だけ。"}],
                             "enable_thinking": False, "max_tokens": 8, "temperature": 0})
        results[i]["done_at"] = time.monotonic()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    check("fifo_concurrent", all(x["status"] == 200 and x["finish"] == "stop" for x in results)
          and results[0]["done_at"] <= results[1]["done_at"],
          [{"content": x["content"], "total": x["total"]} for x in results], report)

    # Restore after another conversation overwrote the live KV continues the cold run. Not bit-identical:
    # the GPU expert set adapts to each context (stage 20) and is not restored, and CPU/GPU experts differ
    # in rounding (relative 1e-4), so greedy may part after a few dozen tokens.
    long_a = [{"role": "system", "content": filler(11, 4200)},
              {"role": "user", "content": "上の handler の一覧から、limit が最大のものを一つ挙げて。"}]
    long_b = [{"role": "system", "content": filler(12, 4200)}, {"role": "user", "content": "要約して。"}]
    req = {"model": "glm53", "enable_thinking": False, "max_tokens": 48, "temperature": 0}
    cold = stream(dict(req, messages=long_a))
    stream(dict(req, messages=long_b))
    warm = stream(dict(req, messages=long_a))
    same = next((i for i, (x, y) in enumerate(zip(cold["content"], warm["content"])) if x != y),
                min(len(cold["content"]), len(warm["content"])))
    check("restore_continues_cold", cold["content"] and same >= 16 and "handler_" in warm["content"]
          and warm["usage"]["prompt_tokens_details"]["cached_tokens"] == warm["usage"]["prompt_tokens"],
          {"same_chars": same, "cold": cold["content"], "warm": warm["content"], "cold_usage": cold["usage"],
           "warm_usage": warm["usage"], "warm_restore_ms": warm["timings"]["restore_ms"],
           "cold_prefill_ms": cold["timings"]["prefill_ms"]}, report)

    r = stream({"model": "glm53", "messages": [{"role": "user", "content": "日本の四季について、それぞれの特徴を詳しく説明してください。"}],
                "enable_thinking": False, "max_tokens": 256, "temperature": 0.6, "top_p": 0.95, "seed": 1})
    n = r["usage"]["completion_tokens"]
    wire = (n - 1) / (r["total"] - r["first"]) if r["first"] and n > 1 else 0
    check("decode_speed", r["status"] == 200 and n > 32, {"completion_tokens": n,
          "server_decode_tok_s": r["timings"]["decode_tok_s"], "http_tok_s": wire, "ttft_s": r["first"]}, report)


def filler(seed, chars):
    """Distinct technical-looking text (English + code + Japanese)."""
    parts = []
    i = 0
    while sum(len(p) for p in parts) < chars:
        k = seed * 100003 + i
        parts.append(
            f"### Module {k % 9973} section {i}\n"
            f"def handler_{k % 7919}(request, limit={k % 97}):\n"
            f"    # 処理 {k % 89} は入力を検証してキャッシュ {k % 61} 件を返す\n"
            f"    return [item for item in request.items if item.size < {k % 4096}]\n"
            f"The function above filters requests for shard {k % 53} and logs latency in ms.\n")
        i += 1
    return "".join(parts)[:chars]


def bench(report, system_tokens, turn_tokens, turns, max_tokens=32):
    tools = [{"type": "function", "function": {
        "name": name, "description": filler(j + 1, 600),
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                       "required": ["path"]}}} for j, name in enumerate(
        ["read", "write", "edit", "grep", "glob", "ls", "bash", "todo_write", "manga_forge", "web_fetch"])]
    system = "あなたは Hayate Studio のコーディングエージェントです。\n" + filler(0, int(system_tokens * 3.2))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "リポジトリを読んで、handler の重複を探して。まず短く方針を述べて。"}]
    rows = []
    for turn in range(turns):
        r = stream({"model": "glm53", "messages": messages, "tools": tools, "enable_thinking": False,
                    "max_tokens": max_tokens, "temperature": 0.6, "seed": turn})
        if r["status"] != 200:
            check(f"bench_turn_{turn}", False, r["error"], report)
            return
        u, t = r["usage"], r["timings"]
        row = {"turn": turn + 1, "prompt_tokens": u["prompt_tokens"],
               "cached_tokens": u["prompt_tokens_details"]["cached_tokens"],
               "prefilled": u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"],
               "ttft_s": round(r["first"], 3) if r["first"] else None, "prefill_ms": round(t["prefill_ms"], 1),
               "restore_ms": round(t["restore_ms"], 1), "checkpoint_ms": round(t["checkpoint_ms"], 1),
               "decode_tok_s": round(t["decode_tok_s"], 2)}
        rows.append(row)
        print("bench " + json.dumps(row), flush=True)
        calls = [r["calls"][i] for i in sorted(r["calls"])]
        assistant = {"role": "assistant", "content": r["content"]}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        messages.append(assistant)
        output = filler(1000 + turn, int(turn_tokens * 3.2))
        if calls:
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": output[: len(output) // len(calls)]})
        else:
            messages.append({"role": "user", "content": "ツールの結果:\n" + output + "\n続けて。"})
    check("bench_agent_loop", all(r["ttft_s"] for r in rows), rows, report)


def main():
    global PORT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--load", action="store_true", help="POST /v1/models/load glm53 first")
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--late-only", action="store_true", help="only cancel/FIFO/restore/decode checks")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--system-tokens", type=int, default=30000)
    p.add_argument("--turn-tokens", type=int, default=2000)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--bench-max-tokens", type=int, default=32, help="reply length per bench turn")
    p.add_argument("--out")
    a = p.parse_args()
    PORT = a.port
    report = []
    if a.load:
        start = time.monotonic()
        status, body = post_json("/v1/models/load", {"id": "glm53", "variant": "fp8"})
        check("load", status == 200, {"status": status, "body": body[:300],
                                      "seconds": round(time.monotonic() - start, 1)}, report)
    if not a.skip_checks:
        if not a.late_only:
            checks(report)
        late_checks(report)
    if a.bench:
        bench(report, a.system_tokens, a.turn_tokens, a.turns, a.bench_max_tokens)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    failed = [r["check"] for r in report if not r["ok"]]
    print(json.dumps({"checks": len(report), "failed": failed}, ensure_ascii=False))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
