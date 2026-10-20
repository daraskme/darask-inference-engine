"""Compare identical greedy completions with/without the model's own MTP head."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import sys
import atexit
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kogarashi.benchmarks import BenchmarkRun, compare_greedy, draft_acceptance, write_json
PORT = 17831
BASE = f"http://127.0.0.1:{PORT}"
PROMPTS = {
    "japanese": "日本の四季について、春夏秋冬の順にそれぞれ具体例を挙げて詳しく説明してください。\n回答：",
    "code": "Write a Python function that merges overlapping intervals, with type annotations, a docstring, and several test cases.\n```python\n",
    "json": 'List the integers from 1 to 100 as JSON objects with keys "n", "square", "cube".\n[\n',
    "japanese_technical": "投機的デコーディングの仕組みを、初心者向けに日本語で説明してください。ドラフトの提案、検証、棄却の例も示してください。\n回答：",
    "japanese_long": "次は架空の観測記録です。\n" + "春は桜が咲き、夏は雨が多く、秋は紅葉し、冬は雪が降った。\n" * 80 + "\nこの観測記録を踏まえ、日本の四季を順に具体例とともに詳しく説明してください。\n回答：",
}


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--modes", nargs="+", choices=("none", "draft-mtp"), default=["none", "draft-mtp"])
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/mtp-comparison.json")
parser.add_argument("--draft-max", type=int, choices=range(1, 9), default=3)
parser.add_argument("--load-mode", choices=("auto", "none", "mmap", "dio"), default="auto")
parser.add_argument("--prompts", nargs="+", choices=tuple(PROMPTS), default=["japanese", "code", "json"])
parser.add_argument("--n-probs", type=int, default=0)
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--n-predict", type=int, default=256)
parser.add_argument("--bin-dir", type=Path, default=ROOT / "build/backend/bin")
parser.add_argument("--require-match", action="store_true", help="Fail unless both modes agree and MTP accepts drafts")
options = parser.parse_args()
if options.require_match and set(options.modes) != {"none", "draft-mtp"}:
    parser.error("--require-match requires both modes")
if options.trials < 1 or not 1 <= options.n_predict <= 4096:
    parser.error("trials must be positive; n-predict must be between 1 and 4096")
options.bin_dir = options.bin_dir.resolve()
PROMPTS = {name: PROMPTS[name] for name in options.prompts}
if options.output.exists():
    raise SystemExit(f"Refusing to overwrite existing benchmark evidence: {options.output}")
results = []
run = BenchmarkRun(ROOT / "artifacts/benchmarks", {
    "model": "Ornith-1.5-122B-Q4_K_M.gguf",
    "sha256": "729fe18c2491aaa446074a923e12f184f9f2239ffd5ce3dae47b0ce9cf8df40c",
    "backend_revision": "9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f",
    "binary_sha256": {path.name: sha256(path)
                      for path in sorted(options.bin_dir.iterdir())
                      if path.is_file() and not path.is_symlink()},
    "draft_max": options.draft_max, "load_mode": options.load_mode,
    "temperature": 0, "seed": 42, "context": 8192,
    "n_probs": options.n_probs,
    "n_predict": options.n_predict, "trials": options.trials,
    "prompts": PROMPTS, "ordering": "sequential; not an interleaved paired experiment",
}, [f"{mode}/{name}/{trial}" for mode in options.modes for name in PROMPTS for trial in range(options.trials)])
atexit.register(lambda: run.fail(RuntimeError("Benchmark exited before complete coverage")))
print(f"Evidence: {run.path}", flush=True)
for mode in options.modes:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=10)
    free_mib = int(gpu.stdout.splitlines()[0])
    if free_mib < 85000:
        raise RuntimeError(f"Benchmark requires 85000 MiB free VRAM; available {free_mib}. Other processes preserved.")
    args = [str(options.bin_dir / "llama-server"),
            "-m", str(ROOT / "models/Ornith-1.5-122B-Q4_K_M.gguf"),
            "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99",
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-t", "8",
            "-c", "8192", "-np", "1", "-b", "2048", "-ub", "512",
            "--fit", "off", "--no-context-shift", "--spec-type", mode,
            "--load-mode", options.load_mode]
    if mode == "draft-mtp":
        args += ["--spec-draft-n-max", str(options.draft_max), "--spec-draft-n-min", "1",
                 "--no-spec-draft-backend-sampling"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(options.bin_dir) + ":/run/opengl-driver/lib" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    # Do not send benchmark prompts to an unrelated server using our test port.
    import socket
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", PORT))
    with (run.path / f"server-{mode}.log").open("w") as log:
        startup_begin = time.monotonic()
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{mode} server exited: {process.returncode}; see log")
                try:
                    if request("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.HTTPError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{mode} server startup timed out")
                time.sleep(1)
            request("/completion", {"prompt": "1, 2, 3,", "n_predict": 16,
                                    "temperature": 0, "cache_prompt": False})
            for name, prompt in PROMPTS.items():
                for trial in range(options.trials):
                    begin = time.monotonic()
                    response = request("/completion", {
                        "prompt": prompt, "n_predict": options.n_predict, "temperature": 0,
                        "seed": 42, "cache_prompt": False, "stream": False,
                        "return_tokens": True,
                        "n_probs": options.n_probs,
                    })
                    record = {"mode": mode, "prompt": name, "trial": trial,
                              "wall_seconds": time.monotonic() - begin,
                              "response": response}
                    timings = response.get("timings", {})
                    record["draft_acceptance"] = draft_acceptance(
                        timings.get("draft_n_accepted", 0), timings.get("draft_n", 0))
                    record["command"] = args
                    run.record(f"{mode}/{name}/{trial}", record)
                    results.append(record)
                    write_json(options.output, results)
                    print(json.dumps({"mode": mode, "prompt": name, "trial": trial,
                                      "timings": response.get("timings")}), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
summary = {"records": len(results), "raw_output": str(options.output)}
if set(options.modes) == {"none", "draft-mtp"}:
    summary["greedy_comparison"] = compare_greedy(results)
run.finish(summary)
if options.require_match and not summary["greedy_comparison"]["passed"]:
    raise SystemExit("Greedy regression failed; see the run's summary.json for exact token differences")
