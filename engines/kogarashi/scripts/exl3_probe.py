"""Run upstream EXL3 on small BF16 weight slices; not a full model converter.

Only quantization modules are loaded. Inference dependencies (FlashAttention,
tokenizers, etc.) are not needed for this tensor experiment.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.cpp_extension import load
from safetensors.torch import save_file, load_file
from kogarashi.quant_lab.gguf_sample import read_header
from kogarashi.benchmarks import BenchmarkRun, write_json


def load_quantizer(source):
    package = source / "exllamav3"
    native = package / "exllamav3_ext"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    build = ROOT / "build/exl3-probe"
    build.mkdir(parents=True, exist_ok=True)
    # The Nix toolkit provides core headers matching nvcc. Add only missing
    # library headers from Torch's CUDA wheels, never their cuda_runtime.h.
    extra = build / "library-headers"
    extra.mkdir(exist_ok=True)
    for include in Path(torch.__file__).resolve().parent.parent.glob("nvidia/*/include"):
        for pattern in ("cusparse*.h", "cusolver*.h"):
            for header in include.glob(pattern):
                target = extra / header.name
                if not target.exists():
                    target.symlink_to(header)
    ext = load("kogarashi_exl3_probe", sources=[str(ROOT / "scripts/exl3_probe_bindings.cpp")] +
               [str(native / "quant" / name) for name in
                ("quantize.cu", "pack.cu", "reconstruct.cu", "exl3_devctx.cu")],
               extra_include_paths=[str(native), str(extra)], extra_cuda_cflags=["-O3"],
               build_directory=str(build), verbose=True)
    # Skip package initializers which import the full inference stack.
    for name in ("exllamav3", "exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib"):
        module = types.ModuleType(name)
        module.__path__ = [str(source.joinpath(*name.split(".")))]
        sys.modules[name] = module
    module = types.ModuleType("exllamav3.ext")
    module.exllamav3_ext = ext
    sys.modules[module.__name__] = module
    quant = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
    return quant, ext


def weight_slice(header, tensor, size=128):
    if tensor.kind != 30 or len(tensor.shape) < 2 or min(tensor.shape[:2]) < size:
        raise ValueError("Expected a sufficiently large BF16 matrix")
    width, height = (size, size) if size else tensor.shape[:2]
    rows = []
    with header.path.open("rb") as source:
        for row in range(height):
            source.seek(header.data_offset + tensor.offset + row*tensor.shape[0]*2)
            data = source.read(width*2)
            if len(data) != width*2:
                raise ValueError("Truncated weight slice")
            rows.append(data)
    raw = b"".join(rows)
    values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    # GGUF rows are output features; EXL3 expects input x output.
    return torch.from_numpy(values.reshape(height, width).T.copy()), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "upstream/exllamav3-fury")
    parser.add_argument("--bits", nargs="+", type=int, choices=range(2, 9), default=[2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128, help="Square slice width, or 0 for a whole matrix/expert")
    parser.add_argument("--tensors", nargs="+", default=["blk.0.ffn_gate_exps.weight", "blk.3.attn_q.weight", "blk.47.ffn_down_exps.weight"])
    args = parser.parse_args()
    if args.size < 0 or args.size % 128:
        parser.error("size must be a nonnegative multiple of 128")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    names = args.tensors
    run = BenchmarkRun(args.output, {
        "source": str(source), "revision": revision, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "scope": "whole matrices" if args.size == 0 else f"{args.size}x{args.size} upper-left BF16 slices",
        "expert": "expert 0 only for MoE", "full_model": False,
        "codebook": "mcg (upstream converter default)",
        "calibration": "upstream count=0 fallback, not measured model Hessians",
    }, [f"{name}/{bits}" for name in names for bits in args.bits])
    try:
        quant, ext = load_quantizer(source)
        header = read_header(ROOT / "models/Ornith-1.5-122B-BF16.gguf")
        tensors = {t.name: t for t in header.tensors}
        torch.set_num_threads(8)
        records = []
        for name in names:
            weight, input_hash = weight_slice(header, tensors[name], args.size)
            for bits in args.bits:
                torch.cuda.reset_peak_memory_stats()
                original = weight.to("cuda:0")
                h_data = {"H": torch.zeros((weight.shape[0], weight.shape[0]), device="cuda:0"), "count": 0, "finalized": False}
                settings = {"K": bits, "seed": 42, "devices": ["cuda:0"], "apply_out_scales": True, "mcg": True}
                torch.cuda.synchronize()
                start = time.monotonic()
                with torch.inference_mode():
                    wq, _, packed = quant.quantize_exl3(original.clone(), h_data, settings, True)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-start
                path = args.output / f"{name}-{bits}bpw.safetensors"
                save_file({k: v.cpu().contiguous() for k, v in packed.items()}, str(path))
                saved = load_file(str(path), device="cuda:0")
                indices = torch.empty((*saved["trellis"].shape[:2], 256), dtype=torch.int16, device="cuda:0")
                ext.unpack_trellis(indices, saved["trellis"], bits)
                repacked = quant.pack_trellis(indices, settings)
                if not torch.equal(repacked, saved["trellis"]):
                    raise ValueError("Packed trellis round-trip failed")
                restored = torch.empty_like(original, dtype=torch.half)
                ext.reconstruct(restored, saved["trellis"], bits, True, False)
                restored = quant.preapply_had_l(restored.float(), 128) * saved["suh"].float().unsqueeze(1)
                restored = quant.preapply_had_r(restored, 128) * saved["svh"].float().unsqueeze(0)
                if not torch.isfinite(restored).all() or not torch.isfinite(wq).all():
                    raise ValueError("Nonfinite EXL3 reconstruction")
                denominator = original.double().square().sum()
                nmse = ((restored.double()-original.double()).square().sum()/denominator).item()
                packing_delta = (restored-wq).abs().max().item()
                # Saved scales are FP16; compare with the float encoder output at that precision.
                tolerance = 0.01*wq.abs().max().item()+1e-6
                if packing_delta > tolerance:
                    raise ValueError(f"Packed reconstruction disagrees: {packing_delta} > {tolerance}")
                row = {"tensor": name, "bits": bits, "input_sha256": input_hash,
                       "slice_shape": list(weight.shape), "seconds": elapsed,
                       "stored_weight_nmse": nmse, "max_packing_delta": packing_delta,
                       "tensor_bytes": sum(t.numel()*t.element_size() for t in packed.values()),
                       "file_bytes": path.stat().st_size, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "fallback": settings["q_fallback"], "file": str(path)}
                row["packing_roundtrip_exact"] = True
                records.append(row)
                run.record(f"{name}/{bits}", row)
                write_json(args.output / "results.json", records)
                print(json.dumps(row), flush=True)
        run.finish({"records": len(records), "full_model_converted": False})
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
