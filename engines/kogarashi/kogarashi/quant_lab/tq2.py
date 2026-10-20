"""Independent TQ2_T encoder for the pinned Gyro decoder (15 effective bits).

The 34-byte format is unchanged. Forward float64 DP minimizes its accumulated
cost for each supplied boundary at a fixed FP16 scale. Scale refitting and the
limited boundary search are approximate; the returned block is not an oracle.
This is a CPU research implementation, not a full-model conversion pipeline.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import re
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Codec:
    def __init__(self, root: Path = ROOT):
        header = root / "upstream/gyro-llama/ggml/src/ggml-common.h"
        source = header.read_text()
        if not re.search(r"#define TQ_STATE_BITS\s+15\b", source):
            raise ValueError("This experiment requires the 15-bit TQ2_T decoder")
        table = source.split("GGML_TABLE_BEGIN(uint16_t, tq2t_lut_f16, 4096)", 1)[1]
        table = table.split("GGML_TABLE_END()", 1)[0]
        words = re.findall(r"0x([0-9a-fA-F]{4})\b", table)
        if len(words) != 4096:
            raise ValueError("Unexpected upstream TQ2_T codebook")
        raw = np.array([int(w, 16) for w in words], dtype="<u2")
        self.lut = raw.view("<f2").astype(np.float32)
        self.codebook_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()
        self.header_sha256 = hashlib.sha256(header.read_bytes()).hexdigest()
        library = root / "build/backend/bin/libggml-base.so"
        self.library = ctypes.CDLL(str(library))
        self.library_path = str(library)
        self.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        for name in ("quantize_row_tq2_t_ref", "dequantize_row_tq2_t"):
            func = getattr(self.library, name)
            func.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
            func.restype = None
        # Only the low seven bits of the previous byte survive the 15-bit mask.
        states = (np.arange(128, dtype=np.uint64)[:, None] << 8) | np.arange(256, dtype=np.uint64)
        hashed = (states * 0x9E3779B1) & 0xFFFFFFFF
        idx0, idx1 = hashed >> 21, (hashed >> 10) & 2047
        self.edges = np.stack((self.lut[2*idx0], self.lut[2*idx0+1],
                               self.lut[2*idx1], self.lut[2*idx1+1]), axis=-1).astype(np.float64)

    def reference(self, values) -> bytes:
        values = _values(values)
        output = ctypes.create_string_buffer(34)
        self.library.quantize_row_tq2_t_ref(values.ctypes.data, output, 128)
        return output.raw

    def checked_reference(self, values) -> tuple[bytes | None, str]:
        """Return a finite reference, or explain why it cannot be compared.

        Float32 accumulation and the final FP16 scale have separate limits.
        Never invoke the reference when its energy accumulation may overflow.
        """
        values = _values(values)
        wide = values.astype(np.float64)
        if float(np.dot(wide, wide)) >= float(np.finfo(np.float32).max) / 2:
            return None, "float32_energy_overflow"
        block = self.reference(values)
        if not np.isfinite(self.decode_c(block)).all():
            return None, "nonfinite_reference_decode"
        return block, "finite"

    def decode_c(self, block: bytes) -> np.ndarray:
        if len(block) != 34:
            raise ValueError("TQ2_T blocks must contain exactly 34 bytes")
        packed = ctypes.create_string_buffer(block, 34)
        output = np.empty(128, dtype=np.float32)
        self.library.dequantize_row_tq2_t(packed, output.ctypes.data, 128)
        return output

    def decode(self, block: bytes) -> np.ndarray:
        if len(block) != 34:
            raise ValueError("TQ2_T blocks must contain exactly 34 bytes")
        scale = np.float32(struct.unpack("<e", block[:2])[0])
        path = np.frombuffer(block[2:], dtype=np.uint8).astype(np.uint32)
        state = ((np.roll(path, 1) << 8) | path) & 0x7FFF
        hashed = (state.astype(np.uint64) * 0x9E3779B1) & 0xFFFFFFFF
        i, j = hashed >> 21, (hashed >> 10) & 2047
        return (np.stack((self.lut[2*i], self.lut[2*i+1], self.lut[2*j], self.lut[2*j+1]),
                         axis=-1) * scale).reshape(128)


def _values(values):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float32)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError("Expected 128 finite float32 weights")
    return np.ascontiguousarray(values)


def _importance(importance):
    if importance is None:
        return np.ones(128, dtype=np.float64)
    importance = np.asarray(importance, dtype=np.float64)
    if (importance.shape != (128,) or not np.isfinite(importance).all()
            or (importance < 0).any() or not (importance > 0).any()):
        raise ValueError("Importance must be 128 finite nonnegative values with positive mass")
    # Common rescaling preserves the objective and avoids overflow in costs.
    return importance / importance.max()


def solve_cycles(costs: np.ndarray, anchors) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost closed paths for fixed anchors; costs[t, previous, current]."""
    costs = np.asarray(costs, dtype=np.float64)
    if (costs.ndim != 3 or costs.shape[0] == 0 or costs.shape[1] == 0
            or costs.shape[1] != costs.shape[2] or not np.isfinite(costs).all()):
        raise ValueError("Expected finite [steps, states, states] costs")
    anchors = np.asarray(list(anchors), dtype=np.int64)
    steps, states, _ = costs.shape
    if anchors.ndim != 1 or not len(anchors) or ((anchors < 0) | (anchors >= states)).any():
        raise ValueError("Invalid boundary states")
    batch = np.arange(len(anchors))
    dp = np.full((len(anchors), states), np.inf)
    dp[batch, anchors] = 0
    parents = np.empty((steps, len(anchors), states), dtype=np.int32)
    for t in range(steps):
        candidates = dp[:, :, None] + costs[t][None, :, :]
        parents[t] = candidates.argmin(axis=1)
        dp = np.take_along_axis(candidates, parents[t][:, None, :], axis=1)[:, 0, :]
    losses = dp[batch, anchors].copy()
    path = np.empty((len(anchors), steps), dtype=np.int32)
    current = anchors.copy()
    for t in reversed(range(steps)):
        path[:, t] = current
        current = parents[t, batch, current]
    if not np.array_equal(current, anchors):
        raise RuntimeError("Traceback did not close")
    return path, losses


def hadamard(values) -> np.ndarray:
    """Normalized Sylvester transform on the final, power-of-two axis."""
    output = np.array(values, dtype=np.float64, copy=True)
    if output.ndim == 0:
        raise ValueError("Hadamard needs an axis")
    width = output.shape[-1]
    if width == 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    stride = 1
    while stride < width:
        view = output.reshape(*output.shape[:-1], -1, 2, stride)
        left, right = view[..., 0, :].copy(), view[..., 1, :].copy()
        view[..., 0, :], view[..., 1, :] = left + right, left - right
        stride *= 2
    return output / np.sqrt(width)


def encode(codec: Codec, values, importance=None, *, anchors=4,
           scale_factors=(0.8, 1.0, 1.25), refinement=1) -> tuple[bytes, dict]:
    """Search closed paths, refit FP16 scales, retain the reference as fallback.

    importance is a diagonal metric IN THE ENCODED BASIS. A diagonal from the
    original basis cannot simply be reused after a Hadamard transform. All-zero
    importance is rejected; a full-model caller must choose its own policy.
    """
    x = _values(values)
    weight = _importance(importance)
    if not isinstance(anchors, int) or not 1 <= anchors <= 128:
        raise ValueError("anchors must be in [1, 128]")
    if not isinstance(refinement, int) or not 0 <= refinement <= 4:
        raise ValueError("refinement must be in [0, 4]")
    scale_factors = tuple(scale_factors)
    if not scale_factors or any(not np.isfinite(s) or s <= 0 for s in scale_factors):
        raise ValueError("Scale factors must be finite and positive")
    xd = x.astype(np.float64)
    energy = float(np.dot(xd, xd))

    def score(block):
        decoded = codec.decode(block).astype(np.float64)
        if not np.isfinite(decoded).all():
            return float("inf")
        return float(np.dot(weight, (decoded - xd)**2))

    reference, reference_status = codec.checked_reference(x)
    baseline = reference if reference is not None else bytes(34)
    best = baseline
    best_loss = score(baseline)
    baseline_loss = best_loss if reference is not None else None
    winner = "upstream_reference" if reference is not None else "zero"
    reference_meta = {"reference_status": reference_status,
                      "reference_valid": reference is not None}
    if energy == 0:
        return bytes(34), {"loss": 0.0, "reference_loss": 0.0, "winner": "zero",
                           "candidates": 0, **reference_meta,
                           "global_optimum_claimed": False}

    def fit(path):
        unit = codec.decode(struct.pack("<e", 1.0) + path).astype(np.float64)
        denominator = np.dot(weight, unit*unit)
        scale = float(np.dot(weight*xd, unit) / denominator) if denominator > 0 else 0.0
        rounded = np.float16(np.clip(scale, -65504, 65504))
        # Adjacent representable scales cover the FP16 rounding boundary.
        with np.errstate(over="ignore", under="ignore"):
            choices = (rounded, np.nextafter(rounded, np.float16(-np.inf)),
                       np.nextafter(rounded, np.float16(np.inf)))
        candidates = [struct.pack("<e", float(s)) + path for s in choices if np.isfinite(s)]
        return min(candidates, key=score)

    fitted_reference = fit(baseline[2:])
    fitted_loss = score(fitted_reference)
    if fitted_loss < best_loss:
        best, best_loss, winner = fitted_reference, fitted_loss, "reference_scale_refit"
    seed = best[-1] & 127
    boundaries = [(seed + (i * 128) // anchors) % 128 for i in range(anchors)]
    rms = np.sqrt(energy / 128)
    trial_scales = [float(np.float16(np.clip(rms*min(f, 65504/rms),
                                            np.finfo(np.float16).smallest_subnormal, 65504)))
                    for f in scale_factors]
    trial_scales = list(dict.fromkeys(trial_scales))
    candidate_count = 2
    for iteration in range(refinement + 1):
        for scale in trial_scales:
            edge_costs = np.empty((32, 128, 256), dtype=np.float64)
            for t in range(32):
                error = scale*codec.edges - xd[4*t:4*t+4]
                edge_costs[t] = np.sum(error*error*weight[4*t:4*t+4], axis=-1)
            high = edge_costs[:, :, 128:] < edge_costs[:, :, :128]
            costs = np.minimum(edge_costs[:, :, :128], edge_costs[:, :, 128:])
            paths, _ = solve_cycles(costs, boundaries)
            for path in paths:
                previous = np.roll(path, 1)
                bytes_path = (path + 128*high[np.arange(32), previous, path]).astype(np.uint8).tobytes()
                candidate = fit(bytes_path)
                candidate_count += 1
                loss = score(candidate)
                if loss < best_loss:
                    best, best_loss, winner = candidate, loss, "anchored_viterbi"
        trial_scales = [struct.unpack("<e", best[:2])[0]]
        if not trial_scales[0]:
            break
    return best, {"loss": best_loss, "reference_loss": baseline_loss, **reference_meta,
                  "winner": winner, "candidates": candidate_count,
                  "anchors": boundaries, "bits_per_weight": 2.125,
                  "global_optimum_claimed": False}
