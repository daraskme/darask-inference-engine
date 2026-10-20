"""Run a small, reproducible TQ2_T experiment on original GGUF weights."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .gguf_sample import read_header, samples
from .tq2 import Codec, encode, hadamard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", default=r"blk\.(0|4|8)\.ffn_(gate|up|down)_exps\.weight")
    parser.add_argument("--per-tensor", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--basis", choices=("native", "hadamard", "both"), default="both")
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Refusing to overwrite an existing experiment")
    codec = Codec()
    header = read_header(options.model)
    report = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "model": str(options.model.resolve()),
              "model_name": header.metadata.get("general.name"),
              "file_bytes_at_start": header.file_bytes, "header_sha256": header.sha256,
              "allow_partial": options.allow_partial, "full_model_verified": False,
              "zero_importance_policy": "reject; no calibration importance supplied in this experiment",
              "invalid_reference_policy": "null metrics; exclude from paired comparisons",
              "metric": "unweighted weight reconstruction SSE; not perplexity or model accuracy",
              "codebook_sha256": codec.codebook_sha256,
              "decoder_header_sha256": codec.header_sha256,
              "library": codec.library_path,
              "library_sha256": codec.library_sha256,
              "python": platform.python_version(), "numpy": np.__version__,
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob("*.py"))},
              "records": []}
    bases = ("native", "hadamard") if options.basis == "both" else (options.basis,)
    started = time.monotonic()
    for info, values in samples(header, options.tensor, options.per_tensor, allow_partial=options.allow_partial):
        for basis in bases:
            x = values if basis == "native" else hadamard(values).astype(np.float32)
            before = time.monotonic()
            baseline, baseline_status = codec.checked_reference(x)
            baseline_seconds = time.monotonic()-before
            before = time.monotonic()
            block, metrics = encode(codec, x, anchors=options.anchors, refinement=options.refinement)
            seconds = time.monotonic()-before
            decoded = codec.decode_c(block)
            np.testing.assert_array_equal(codec.decode(block), decoded)
            base_decoded = codec.decode_c(baseline).astype(np.float64) if baseline is not None else None
            measured = float(np.sum((decoded.astype(np.float64)-x)**2))
            baseline_sse = float(np.sum((base_decoded-x)**2)) if base_decoded is not None else None
            if baseline_sse is not None and measured > baseline_sse + 1e-12:
                raise RuntimeError("Candidate regressed relative to the upstream encoder")
            original_decoded = decoded.astype(np.float64) if basis == "native" else hadamard(decoded)
            original_base = (base_decoded if basis == "native" else hadamard(base_decoded)) if base_decoded is not None else None
            wide_values = values.astype(np.float64)
            record = {**info, "basis": basis, **metrics, "c_decoder_sse": measured,
                      "c_reference_sse": baseline_sse, "seconds": seconds,
                      "reference_status": baseline_status,
                      "reference_seconds": baseline_seconds,
                      "original_basis_sse": float(np.sum((original_decoded-wide_values)**2)),
                      "reference_original_basis_sse": float(np.sum((original_base-wide_values)**2)) if original_base is not None else None,
                      "energy": float(np.dot(wide_values, wide_values)),
                      "input_absmax": float(np.max(np.abs(wide_values))),
                      "packed_hex": block.hex(), "reference_hex": baseline.hex() if baseline is not None else None}
            report["records"].append(record)
            print(json.dumps({"tensor": info["tensor"], "block": info["block_index"],
                              "basis": basis, "loss_ratio": measured/baseline_sse if baseline_sse else None},
                             allow_nan=False), flush=True)
    if not report["records"]:
        raise ValueError("No complete sample blocks are available")
    grouped = defaultdict(list)
    for record in report["records"]:
        grouped[record["basis"]].append(record)
    report["summary"] = {}
    for basis, rows in grouped.items():
        sse = sum(r["original_basis_sse"] for r in rows)
        paired = [r for r in rows if r["reference_valid"]]
        ref = sum(r["reference_original_basis_sse"] for r in paired) if paired else None
        paired_sse = sum(r["original_basis_sse"] for r in paired) if paired else None
        paired_energy = sum(r["energy"] for r in paired)
        energy = sum(r["energy"] for r in rows)
        report["summary"][basis] = {"blocks": len(rows), "encoded_bytes": len(rows)*34,
                                     "bf16_bytes": len(rows)*256,
                                     "sse": sse, "reference_sse": ref,
                                     "paired_blocks": len(paired),
                                     "invalid_reference_blocks": len(rows)-len(paired),
                                     "paired_sse": paired_sse,
                                     "error_reduction_fraction": 1-paired_sse/ref if ref else None,
                                     "nmse": sse/energy if energy else 0,
                                     "reference_nmse": ref/paired_energy if paired_energy else None,
                                     "encoder_seconds": sum(r["seconds"] for r in rows),
                                     "reference_seconds": sum(r["reference_seconds"] for r in rows)}
    report["wall_seconds"] = time.monotonic()-started
    options.output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the entire JSON before creating a destination. Exclusive creation
    # also prevents a concurrent run from overwriting evidence.
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with options.output.open("x") as file:
        file.write(serialized)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
