"""Small TQ2_T fixed-FP16-scale, all-anchor comparison; never writes a model.

The oracle minimizes left-to-right float64 DP costs, not a real-valued or
scale-global objective. This deliberately uses the CPU DP and is not a CUDA
bitwise oracle. Run only on a complete original floating-point GGUF.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from kogarashi.quant_lab.gguf_sample import read_header, samples
from kogarashi.quant_lab.tq2 import Codec, encode, solve_cycles


def fixed_scale_comparison(codec, x):
    candidate, meta = encode(codec, x, anchors=4)
    scale_bytes = candidate[:2]
    scale = struct.unpack("<e", scale_bytes)[0]
    xd = x.astype(np.float64)
    edges = np.empty((32, 128, 256), dtype=np.float64)
    for t in range(32):
        residual = scale * codec.edges - xd[4*t:4*t+4]
        edges[t] = np.sum(residual * residual, axis=-1)
    high = edges[:, :, 128:] < edges[:, :, :128]
    costs = np.minimum(edges[:, :, :128], edges[:, :, 128:])
    before = time.monotonic()
    paths, losses = solve_cycles(costs, range(128))
    seconds = time.monotonic() - before
    winner = int(np.argmin(losses))
    path = paths[winner]
    packed_path = (path + 128 * high[np.arange(32), np.roll(path, 1), path]).astype(np.uint8)
    oracle = scale_bytes + packed_path.tobytes()
    assert (oracle[-1] & 127) == winner
    np.testing.assert_array_equal(codec.decode(oracle), codec.decode_c(oracle))
    # The sampled-anchor subset is evaluated with the identical saved scale.
    anchors = meta.get("anchors", [0, 32, 64, 96])
    limited_paths, limited_losses = solve_cycles(costs, anchors)
    np.testing.assert_array_equal(limited_losses, losses[anchors])
    np.testing.assert_array_equal(limited_paths, paths[anchors])

    def forward_cost(block):
        q = np.frombuffer(block[2:], dtype=np.uint8)
        total = np.float64(0)
        for t in range(32):
            total = total + edges[t, int(q[t-1]) & 127, q[t]]
        return float(total)

    optimal_cost = float(losses[winner])
    assert forward_cost(oracle) == optimal_cost
    candidate_cost = forward_cost(candidate)
    assert optimal_cost <= candidate_cost
    limited_cost = float(limited_losses.min())
    decoded = codec.decode_c(oracle).astype(np.float64)
    return {
        "saved_scale": scale, "scale_hex": scale_bytes.hex(),
        "input_energy": float(np.dot(xd, xd)), "input_absmax": float(np.max(np.abs(xd))),
        "all_anchor_forward_cost": optimal_cost,
        "limited_anchor_forward_cost": limited_cost,
        "candidate_forward_cost": candidate_cost,
        "limited_anchor_gap_absolute": limited_cost-optimal_cost,
        "limited_anchor_gap_relative": limited_cost/optimal_cost-1 if optimal_cost else None,
        "candidate_gap_absolute": candidate_cost-optimal_cost,
        "oracle_decoded_sse": float(np.dot(decoded-xd, decoded-xd)),
        "all_anchor_seconds": seconds, "oracle_anchor": winner,
        "limited_anchors": anchors, "oracle_hex": oracle.hex(),
        "candidate_hex": candidate.hex(), "global_scale_optimum_claimed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", default=r"blk\.(0|4|8)\.ffn_(gate|up|down)_exps\.weight")
    parser.add_argument("--per-tensor", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.model.suffix != ".gguf":
        parser.error("Use a complete original .gguf file")
    if not 1 <= args.per_tensor <= 32:
        parser.error("per-tensor must be in 1..32")
    codec = Codec()
    header = read_header(args.model)
    sampled = list(itertools.islice(samples(header, args.tensor, args.per_tensor), 33))
    if not 1 <= len(sampled) <= 32:
        parser.error("This CPU oracle accepts only 1..32 sample blocks")
    report = {"model": str(args.model.resolve()), "file_bytes": header.file_bytes,
              "header_sha256": header.sha256, "codebook_sha256": codec.codebook_sha256,
              "library_sha256": codec.library_sha256,
              "numpy": np.__version__, "python": sys.version,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "encoder_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] /
                                                "kogarashi/quant_lab/tq2.py").read_bytes()).hexdigest(),
              "objective": "unweighted, fixed saved FP16 scale, 32 left-to-right float64 edge sums",
              "all_anchor_count": 128, "records": []}
    for info, x in sampled:
        record = {**info, **fixed_scale_comparison(codec, x)}
        report["records"].append(record)
        print(json.dumps({"tensor": info["tensor"], "block": info["block_index"],
                          "fixed_scale_gap": record["limited_anchor_gap_relative"]}), flush=True)
    report["summary"] = {"blocks": len(sampled), "assertion_failures": 0,
                          "all_anchor_seconds": sum(r["all_anchor_seconds"] for r in report["records"])}
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(serialized)


if __name__ == "__main__":
    main()
