"""EXL3-inspired scale trials for the existing Gyro-compatible TQ2_T encoder.

This is an independent CPU experiment, not EXL3, GSQ, or RCO implementation.
Each trial runs the unchanged closed-path encoder, including its scale refit.
The trial coordinate therefore need not equal the returned block's scale.
Only the normalized, diagonal weight reconstruction metric is guarded here.
"""
from __future__ import annotations

import math
import struct
import time

import numpy as np

from .tq2 import Codec, _importance, _values, encode


def _trial_scale(rms: float, factor: float) -> float:
    """Match encode's initial DP scale, including underflow/overflow clamps."""
    return float(np.float16(np.clip(
        rms * min(factor, 65504 / rms),
        np.finfo(np.float16).smallest_subnormal, 65504)))


def _local_minima(losses: list[float]) -> list[int]:
    """Include endpoints; an equal-loss plateau is represented by its first index."""
    minima = []
    start = 0
    while start < len(losses):
        end = start
        while end + 1 < len(losses) and losses[end + 1] == losses[start]:
            end += 1
        left = losses[start - 1] if start else math.inf
        right = losses[end + 1] if end + 1 < len(losses) else math.inf
        if losses[start] <= left and losses[start] <= right:
            minima.append(start)
        start = end + 1
    return sorted(minima, key=lambda i: (losses[i], i))


def encode_scale_search(codec: Codec, values, importance=None, *, anchors=4,
                        grid=(0.5, 2.0, 9), refine_top=2) -> tuple[bytes, dict]:
    """Retain baseline on ties; accept only a strict improvement under C decode.

    All calls receive the same anchors and normalized importance in the encoded
    basis. Grid points are deduplicated by their actual first DP scale's FP16
    bytes. One refinement round tests log midpoints beside the best local minima.
    No unimodality, global optimum, or model-quality guarantee is assumed.
    """
    started = time.perf_counter()
    x = _values(values)
    weight = _importance(importance)
    if not isinstance(anchors, int) or not 1 <= anchors <= 128:
        raise ValueError("anchors must be in [1, 128]")
    if len(grid) != 3:
        raise ValueError("grid must be (positive lower factor, upper factor, count)")
    lo, hi, count = grid
    if (not np.isfinite(lo) or not np.isfinite(hi) or not 0 < lo < hi
            or not isinstance(count, int) or not 2 <= count <= 65):
        raise ValueError("grid needs finite 0 < lower < upper and integer count in [2, 65]")
    if not isinstance(refine_top, int) or not 0 <= refine_top <= count:
        raise ValueError("refine_top must be an integer in [0, grid count]")
    xd = x.astype(np.float64)

    def score(block):
        decoded = codec.decode_c(block)
        python_decoded = codec.decode(block)
        if (not np.isfinite(decoded).all()
                or not np.array_equal(decoded.view(np.uint32), python_decoded.view(np.uint32))):
            raise RuntimeError("Nonfinite or bit-inconsistent TQ2_T decoder result")
        loss = float(np.dot(weight, (decoded.astype(np.float64) - xd) ** 2))
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite candidate loss")
        return loss

    before = time.perf_counter()
    base, base_meta = encode(codec, x, weight, anchors=anchors)
    base_seconds = time.perf_counter() - before
    base_loss = score(base)
    best, best_loss, winner = base, base_loss, "baseline"
    visited = set()
    returned = {base}
    trials = []
    energy = float(np.dot(xd, xd))

    if energy:
        rms = float(np.sqrt(energy / 128))

        def trial(factor, stage):
            nonlocal best, best_loss, winner
            scale = _trial_scale(rms, factor)
            key = struct.pack("<e", scale)
            if key in visited:
                return None
            visited.add(key)
            candidate, _ = encode(codec, x, weight, anchors=anchors,
                                  scale_factors=(factor,), refinement=0)
            loss = score(candidate)
            returned.add(candidate)
            row = {"factor": factor, "initial_dp_scale": scale,
                   "initial_dp_scale_hex": key.hex(), "loss": loss, "stage": stage,
                   "returned_scale": struct.unpack("<e", candidate[:2])[0]}
            trials.append(row)
            if loss < best_loss:
                best, best_loss, winner = candidate, loss, stage
            return row

        points = []
        for factor in np.geomspace(lo, hi, count):
            row = trial(float(factor), "grid")
            if row is not None:
                points.append(row)
        if len(points) > 1:
            for index in _local_minima([p["loss"] for p in points])[:refine_top]:
                for neighbor in (index - 1, index + 1):
                    if 0 <= neighbor < len(points):
                        factor = math.exp((math.log(points[index]["factor"])
                                           + math.log(points[neighbor]["factor"])) / 2)
                        trial(factor, "refinement")

    return best, {
        "loss": best_loss, "baseline_loss": base_loss, "baseline_hex": base.hex(),
        "baseline_metadata": base_meta, "winner": winner,
        "anchors": anchors, "grid": list(grid), "refine_top": refine_top,
        "unique_initial_dp_scales": len(visited), "unique_returned_blocks": len(returned),
        "encoder_calls": 1 + len(trials), "trials": trials,
        "baseline_seconds": base_seconds, "total_seconds": time.perf_counter() - started,
        "bits_per_weight": 2.125, "global_optimum_claimed": False,
        "metric": "C-decoded FP64 SSE weighted by importance/max(importance) in encoded basis",
        "model_quality_evaluated": False,
    }
