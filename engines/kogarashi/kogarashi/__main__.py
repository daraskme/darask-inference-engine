"""Local entrypoint: python -m kogarashi doctor|serve."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(prog="kogarashi")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report local runtime and GPU availability")
    serve = commands.add_parser("serve", help="Serve Ornith through the pinned CUDA backend")
    serve.add_argument("--model", type=Path, default=ROOT / "models/Ornith-1.5-122B-Q4_K_M.gguf")
    serve.add_argument("--context", type=int, default=8192)
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--threads", type=int, default=8)
    serve.add_argument("--mtp", choices=("off", "on"), default="off")
    serve.add_argument("--draft-max", type=int, choices=range(1, 9), default=1)
    serve.add_argument("--load-mode", choices=("auto", "none", "mmap", "dio"), default="dio")
    serve.add_argument("--dry-run", action="store_true", help="Print exact command without loading weights")
    args = parser.parse_args()
    backend = ROOT / "build/backend/bin/llama-server"
    if args.command == "doctor":
        report = {"backend": str(backend), "backend_exists": backend.is_file(),
                  "profile": "RTX PRO 6000 Blackwell (sm_120a), 128 GB host RAM",
                  "custom_quantizer_status": "reviewed TQ2_T CPU prototype; full-model converter and CUDA encoder not implemented"}
        try:
            gpu = subprocess.run([
                "nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            report["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip() or gpu.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            report["gpu_error"] = str(error)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                report[key] = value.strip()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not 256 <= args.context <= 262144:
        parser.error("context must be in 256..262144; larger contexts require separate memory/quality validation")
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("port must be in 1..65535 and threads must be positive")
    if not backend.is_file():
        parser.error("Backend missing. Run nix develop path:./nix --command bash scripts/build.sh")
    model = args.model.expanduser().resolve()
    if not model.is_file() or model.suffix != ".gguf":
        parser.error(f"Complete GGUF file not found: {model}")
    command = [str(backend), "--model", str(model), "--alias", "kogarashi-ornith",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--ctx-size", str(args.context), "--parallel", "1", "--threads", str(args.threads),
               "--gpu-layers", "99", "--flash-attn", "on",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "--batch-size", "2048", "--ubatch-size", "512", "--fit", "off",
               "--no-context-shift", "--load-mode", args.load_mode, "--metrics",
               "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
               "--presence-penalty", "1.5", "--spec-type", "draft-mtp" if args.mtp == "on" else "none"]
    if args.mtp == "on":
        command += ["--spec-draft-n-max", str(args.draft_max), "--spec-draft-n-min", "1",
                    "--no-spec-draft-backend-sampling"]
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    if args.mtp == "on":
        print("MTP is experimental: draft-max=1 passed local greedy token comparisons after the CUDA fix; see research/mtp-fix-validation.json.", flush=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/run/opengl-driver/lib" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    os.execve(backend, command, env)


if __name__ == "__main__":
    main()
