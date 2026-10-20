"""Estimate expert-only storage recipes from actual GGUF tensor dimensions.

This produces arithmetic estimates, not converted models or calibrated recipes.
"""
import argparse
import json
from pathlib import Path
import re

from .gguf_sample import read_header


def estimate(header, trunk_layers=48):
    if trunk_layers <= 0:
        raise ValueError("trunk_layers must be positive")
    source_sizes = {0: 4, 1: 2, 30: 2}
    plans = {
        "experts_tq2_t": {"gate": 34, "up": 34, "down": 34},
        "experts_tq2_t_down_tqp6": {"gate": 34, "up": 34, "down": 50},
        "experts_tqp6": {"gate": 50, "up": 50, "down": 50},
    }
    selected, preserved = [], []
    for tensor in header.tensors:
        if tensor.kind not in source_sizes:
            raise ValueError("Storage estimates require original F32/F16/BF16 input")
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if match and int(match[1]) < trunk_layers:
            if tensor.shape[0] % 128:
                raise ValueError(f"Tensor rows are incompatible with TQ: {tensor.name}")
            selected.append((tensor, match[2]))
        else:
            preserved.append(tensor)
    if len(selected) != trunk_layers*3:
        raise ValueError("Expected separate gate/up/down expert tensors in every trunk layer")
    preserved_bytes = sum(t.elements*source_sizes[t.kind] for t in preserved)
    params = sum(t.elements for t in header.tensors)
    expert_params = sum(t.elements for t, _ in selected)
    recipes = {}
    for name, sizes in plans.items():
        expert_bytes = sum(t.elements//128*sizes[role] for t, role in selected)
        total = preserved_bytes+expert_bytes
        recipes[name] = {"payload_bytes": total, "payload_gib": total/2**30,
                         "whole_payload_bits_per_weight": total*8/params,
                         "expert_payload_bytes": expert_bytes,
                         "expert_block_bytes": sizes,
                         "quality_validated": False, "exported": False}
    return {"model": str(header.path.resolve()), "header_sha256": header.sha256,
            "scope": "tensor payload only; excludes metadata, padding, KV/SSM state and workspace",
            "trunk_layers": trunk_layers, "total_parameters": params,
            "routed_parameters": expert_params, "routed_fraction": expert_params/params,
            "routed_tensors": len(selected), "preserved_payload_bytes": preserved_bytes,
            "preserved_policy": "Keep every other tensor in its source dtype, including MTP",
            "recipes": recipes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--trunk-layers", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing estimate")
    result = estimate(read_header(args.model), args.trunk_layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(result["recipes"], indent=2))


if __name__ == "__main__":
    main()
