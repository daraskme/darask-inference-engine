"""Preserve a successful Claude result and verify the actual model identity."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("response", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--model", default="claude-opus-5-5")
args = parser.parse_args()
raw = args.response.read_bytes()
try:
    result = json.loads(raw)
except json.JSONDecodeError:
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    results = [event for event in events if event.get("type") == "result"]
    if not results:
        raise SystemExit("No completed result in Claude event stream")
    result = results[-1]
if result.get("is_error") or result.get("terminal_reason") != "completed":
    raise SystemExit(f"Claude did not complete: {result.get('result', result)}")
models = result.get("modelUsage", {})
if args.model not in models:
    raise SystemExit(f"Expected {args.model}, received {list(models)}")
text = result.get("result")
if not isinstance(text, str) or not text.strip():
    raise SystemExit("No review text returned")
args.output.write_text(text + "\n")
record = {
    "model": args.model,
    "canonical_model": models[args.model].get("canonicalModel"),
    "response_sha256": hashlib.sha256(raw).hexdigest(),
    "session_id": result.get("session_id"),
    "duration_ms": result.get("duration_ms"),
    "permission_denials": result.get("permission_denials"),
    "source": str(args.response),
    "text": str(args.output),
}
args.output.with_suffix(".provenance.json").write_text(
    json.dumps(record, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(record, ensure_ascii=False))
