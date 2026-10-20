"""CPU-only paired experiment for EXL3-inspired TQ2_T scale search."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kogarashi.quant_lab.gguf_sample import read_header, samples
from kogarashi.quant_lab.scale_search import encode_scale_search
from kogarashi.quant_lab.tq2 import Codec


def save(path, data):
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x") as file:
        file.write(serialized)


def validate_inputs(inputs, expected_keys):
    groups = defaultdict(list)
    for info, _ in inputs:
        groups[info["group"]].append((info["tensor"], info["block_index"]))
    if set(groups) != {"continuity", "holdout"}:
        raise RuntimeError("Expected continuity and holdout groups only")
    if any(len(keys) != 72 or len(set(keys)) != 72 for keys in groups.values()):
        raise RuntimeError("Expected 72 distinct blocks per group")
    continuity, holdout = set(groups["continuity"]), set(groups["holdout"])
    if continuity != set(expected_keys):
        raise RuntimeError("Continuity keys differ from the existing experiment")
    if continuity & holdout:
        raise RuntimeError("Holdout overlaps continuity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    proof_path = ROOT / "research/official-bf16-verification.json"
    proof = json.loads(proof_path.read_text())
    model = ROOT / proof["file"]
    stat = model.stat()
    if (not proof["verified"] or stat.st_size != proof["size"]
            or stat.st_mtime_ns != proof["mtime_ns"]):
        raise RuntimeError("Original model differs from previously verified size/mtime")
    old_path = ROOT / "artifacts/quant-lab-ornith-bf16-integrated.json"
    old = json.loads(old_path.read_text())
    expected = {(r["tensor"], r["block_index"]): r
                for r in old["records"] if r["basis"] == "native"}
    header = read_header(model)
    inputs = []
    for group, layers in (("continuity", "0|4|8"), ("holdout", "20|28|44")):
        pattern = rf"blk\.({layers})\.ffn_(gate|up|down)_exps\.weight"
        for info, values in samples(header, pattern, 8):
            if group == "continuity":
                prior = expected[(info["tensor"], info["block_index"])]
                if prior["source_sha256"] != info["source_sha256"]:
                    raise RuntimeError("Continuity input changed")
            inputs.append(({**info, "group": group}, values))
    validate_inputs(inputs, expected)
    args.output.mkdir(parents=True, exist_ok=False)
    codec = Codec()
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": "CPU only", "metric": "unweighted weight reconstruction SSE; not model quality",
        "model": str(model), "previous_full_file_sha256": proof["sha256"],
        "full_hash_recomputed": False, "prior_size_mtime_match": True,
        "header_sha256": header.sha256,
        "prior_evidence_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "kogarashi/quant_lab/tq2.py",
                                    ROOT / "kogarashi/quant_lab/scale_search.py")},
        "decoder_library_sha256": codec.library_sha256,
        "codebook_sha256": codec.codebook_sha256,
        "python": platform.python_version(), "numpy": np.__version__,
        "parameters": {"anchors": 4, "grid": [0.5, 2.0, 9], "refine_top": 2},
        "adoption_threshold": {"holdout_error_reduction_minimum": 0.01,
                               "holdout_encoder_time_ratio_maximum": 5.0},
        "inputs": [info for info, _ in inputs],
    }
    # Commit the complete sample list and thresholds before any candidate search.
    save(args.output / "manifest.json", manifest)
    rows = []
    continuity_matches = 0
    for info, x in inputs:
        block, meta = encode_scale_search(codec, x)
        if info["group"] == "continuity":
            prior = expected[(info["tensor"], info["block_index"])]
            if meta["baseline_hex"] != prior["packed_hex"]:
                raise RuntimeError("Existing encoder's baseline bytes changed")
            continuity_matches += 1
        if meta["loss"] > meta["baseline_loss"]:
            raise RuntimeError("C-decoded SSE regressed")
        rows.append({**info, **meta, "packed_hex": block.hex()})
        if len(rows) % 12 == 0:
            print(json.dumps({"completed_blocks": len(rows), "total": len(inputs)}), flush=True)
    final_stat = model.stat()
    if (stat.st_size, stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("Original changed during experiment")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    summary = {}
    for group, records in groups.items():
        baseline_sse = sum(r["baseline_loss"] for r in records)
        new_sse = sum(r["loss"] for r in records)
        improvements = [1 - r["loss"] / r["baseline_loss"]
                        for r in records if r["baseline_loss"] > 0]
        baseline_seconds = sum(r["baseline_seconds"] for r in records)
        new_seconds = sum(r["total_seconds"] for r in records)
        summary[group] = {
            "blocks": len(records), "improved_blocks": sum(r["loss"] < r["baseline_loss"] for r in records),
            "baseline_sse": baseline_sse, "new_sse": new_sse,
            "error_reduction_fraction": 1 - new_sse / baseline_sse if baseline_sse else None,
            "relative_improvement_median_all_nonzero_baselines": float(np.median(improvements)) if improvements else None,
            "relative_improvement_p95_all_nonzero_baselines": float(np.percentile(improvements, 95)) if improvements else None,
            "zero_baseline_blocks": len(records) - len(improvements),
            "baseline_seconds": baseline_seconds, "search_seconds_including_baseline": new_seconds,
            "encoder_time_ratio": new_seconds / baseline_seconds,
            "sse_regressions": sum(r["loss"] > r["baseline_loss"] for r in records),
        }
    holdout = summary["holdout"]
    adopted = (holdout["error_reduction_fraction"] is not None
               and holdout["error_reduction_fraction"] >= 0.01 and holdout["encoder_time_ratio"] <= 5)
    save(args.output / "results.json", {
        "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
        "summary": summary, "records": rows,
        "correctness_gates_passed": continuity_matches == 72 and all(s["sse_regressions"] == 0 for s in summary.values()),
        "continuity_input_and_baseline_matches": continuity_matches,
        "all_candidates_c_python_bit_equal": True,
        "experimental_adoption_threshold_met": adopted,
        "default_encoder_changed": False, "gsq_or_rco_implemented": False,
        "model_quality_evaluated": False, "gpu_used": False,
        "timing_limitation": "single paired CPU run; not a controlled inference speed benchmark",
    })
    print(json.dumps({"summary": summary, "experimental_adoption_threshold_met": adopted}, indent=2))


if __name__ == "__main__":
    main()
