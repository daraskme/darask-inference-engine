"""Header-only source catalog and one-tensor-at-a-time readers."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import torch

from .hyt import BF16, E4M3, F16, F32, PLAIN, storage_bytes

CHUNK_BYTES = 1 << 20
SOURCE_DTYPES = {"BF16": BF16, "F16": F16, "F32": F32, "F8_E4M3": E4M3}
I8, U8 = 6, 9  # U8 is source-only, for comfy_quant JSON.
SOURCE_DTYPES.update({"I8": I8, "U8": U8})
TORCH_DTYPES = {BF16: torch.bfloat16, F16: torch.float16, F32: torch.float32,
                E4M3: torch.float8_e4m3fn, I8: torch.int8, U8: torch.uint8}
NGRAM_INTEGERS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")


@dataclass
class SourceTensor:
    name: str
    shape: list[int]
    dtype: int
    path: Path
    start: int
    size: int
    scale: "SourceTensor | None" = None
    block: tuple[int, int] | None = None
    convrot_group: int | None = None


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated source file")
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_safetensors(path, metadata):
    tensors = []
    with path.open("rb") as file:
        file_size = file.seek(0, 2)
        file.seek(0)
        header_len = struct.unpack("<Q", read_exact(file, 8))[0]
        if header_len > file_size - 8:
            raise ValueError("safetensors header exceeds file")
        header = json.loads(read_exact(file, header_len), object_pairs_hook=unique_object)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        data_start, ranges = 8 + header_len, []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict) or info.get("dtype") not in (*SOURCE_DTYPES, "I64"):
                raise ValueError(f"{name}: unsupported source dtype")
            shape = info.get("shape")
            if not isinstance(shape, list):
                raise ValueError(f"{name}: invalid shape")
            dtype = SOURCE_DTYPES.get(info["dtype"], F32)
            size = storage_bytes(shape, (PLAIN, I8 if dtype == U8 else dtype, 0))
            if info["dtype"] == "I64":
                size = math.prod(shape) * 8
            offsets = info.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets)):
                raise ValueError(f"{name}: invalid data_offsets")
            start, end = offsets
            if start < 0 or end - start != size or end > file_size - data_start:
                raise ValueError(f"{name}: invalid data range")
            ranges.append((start, end))
            if info["dtype"] == "I64":
                # These are exact N-gram hash constants, not floating weights.
                if (not name.startswith("model.language_model.layers.1.ple.ple_embedding.")
                        or name.rsplit(".", 1)[-1] not in NGRAM_INTEGERS or size > 1024):
                    raise ValueError(f"{name}: unsupported I64 auxiliary tensor")
                if name in metadata:
                    raise ValueError(f"duplicate tensor name: {name}")
                file.seek(data_start + start)
                metadata[name] = {"shape": shape, "values": list(struct.unpack(
                    "<" + "q" * math.prod(shape), read_exact(file, size)))}
            else:
                tensors.append(SourceTensor(name, shape, dtype, path, data_start + start, size))
        end = 0
        for start, next_end in sorted(ranges):
            if start != end:
                raise ValueError("safetensors data has holes or overlapping tensors")
            end = next_end
        if end != file_size - data_start:
            raise ValueError("safetensors data has unclaimed bytes")
    return tensors


# GGUF v3 metadata types. Arrays are skipped without retaining tokenizer tables.
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_string(file):
    size = struct.unpack("<Q", read_exact(file, 8))[0]
    return read_exact(file, size).decode("utf-8")


def gguf_value(file, kind, file_size, retain=False):
    if kind == 8:
        size = struct.unpack("<Q", read_exact(file, 8))[0]
        if size > file_size - file.tell():
            raise ValueError("truncated GGUF string")
        if retain:
            return read_exact(file, size).decode("utf-8")
        file.seek(size, 1)
    elif kind == 9:
        item_type, count = struct.unpack("<IQ", read_exact(file, 12))
        if item_type in GGUF_SCALARS:
            size = struct.calcsize("<" + GGUF_SCALARS[item_type]) * count
            if size > file_size - file.tell():
                raise ValueError("truncated GGUF array")
            file.seek(size, 1)
        elif item_type == 8:
            for _ in range(count):
                gguf_value(file, item_type, file_size)
        else:
            raise ValueError("unsupported GGUF array type")
    elif kind in GGUF_SCALARS:
        fmt = "<" + GGUF_SCALARS[kind]
        return struct.unpack(fmt, read_exact(file, struct.calcsize(fmt)))[0]
    else:
        raise ValueError("unknown GGUF metadata type")


def read_gguf(path):
    tensors, offsets = [], []
    with path.open("rb") as file:
        file_size = file.seek(0, 2)
        file.seek(0)
        magic, version, count, metadata_count = struct.unpack("<4sIQQ", read_exact(file, 24))
        if magic != b"GGUF" or version != 3:
            raise ValueError("only GGUF v3 is supported")
        alignment, keys = 32, set()
        for _ in range(metadata_count):
            key = gguf_string(file)
            if key in keys:
                raise ValueError("duplicate GGUF metadata key")
            keys.add(key)
            kind = struct.unpack("<I", read_exact(file, 4))[0]
            value = gguf_value(file, kind, file_size)
            if key == "general.alignment":
                if kind != 4:
                    raise ValueError("GGUF alignment must be UINT32")
                alignment = value
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("invalid GGUF alignment")
        for _ in range(count):
            name = gguf_string(file)
            rank = struct.unpack("<I", read_exact(file, 4))[0]
            if rank > 6:
                raise ValueError("invalid GGUF rank")
            # GGML lists fastest-varying dimension first; HYT is row-major.
            shape = list(reversed(struct.unpack("<" + "Q" * rank, read_exact(file, rank * 8))))
            kind, offset = struct.unpack("<IQ", read_exact(file, 12))
            if kind not in (0, 1, 30):
                raise ValueError(f"{name}: quantized GGUF type {kind} is forbidden")
            dtype = {0: F32, 1: F16, 30: BF16}[kind]
            size = storage_bytes(shape, (PLAIN, dtype, 0))
            if offset % alignment:
                raise ValueError("misaligned GGUF tensor")
            tensors.append(SourceTensor(name, shape, dtype, path, offset, size))
            offsets.append((offset, offset + size))
        data_start = (file.tell() + alignment - 1) // alignment * alignment
        end = 0
        for start, next_end in sorted(offsets):
            if start < end or next_end > file_size - data_start:
                raise ValueError("invalid GGUF tensor range")
            end = next_end
        for tensor in tensors:
            tensor.start += data_start
    return tensors


def read_sources(paths, config=None, metadata=None, hash_sources=True):
    """Catalog shapes and scales without loading weight payloads."""
    metadata = {} if metadata is None else metadata
    paths = list(map(Path, paths))
    tensors, names, configs = [], set(), {}
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as file:
            is_gguf = file.read(4) == b"GGUF"
        entries = read_gguf(path) if is_gguf else read_safetensors(path, metadata)
        for tensor in entries:
            if not tensor.name or tensor.name in names or tensor.name in metadata:
                raise ValueError(f"empty or duplicate tensor name: {tensor.name!r}")
            names.add(tensor.name)
        tensors.extend(entries)
        cfg_path = Path(config) if config else path.parent / "config.json"
        if cfg_path not in configs:
            configs[cfg_path] = (json.loads(cfg_path.read_text()).get("quantization_config", {})
                                 if cfg_path.exists() else {}) or {}
        if hash_sources:
            with path.open("rb") as file:
                while chunk := file.read(CHUNK_BYTES):
                    digest.update(chunk)
    by_name, consumed = {t.name: t for t in tensors}, set()
    for tensor in tensors:
        if tensor.dtype != I8:
            continue
        prefix = tensor.name.removesuffix(".weight")
        scale = by_name.get(prefix + ".weight_scale")
        descriptor = by_name.get(prefix + ".comfy_quant")
        if (not tensor.name.endswith(".weight") or len(tensor.shape) != 2
                or scale is None or scale.dtype != F32 or scale.shape != [tensor.shape[0], 1]
                or descriptor is None or descriptor.dtype != U8
                or len(descriptor.shape) != 1 or descriptor.size > 4096):
            raise ValueError(f"{tensor.name}: expected row-wise INT8 weight, F32 [N,1] scale and comfy_quant")
        with descriptor.path.open("rb") as file:
            file.seek(descriptor.start)
            cfg = json.loads(read_exact(file, descriptor.size), object_pairs_hook=unique_object)
        if (not isinstance(cfg, dict) or cfg.get("format") != "int8_tensorwise"
                or cfg.get("per_row") is not True or type(cfg.get("convrot", False)) is not bool):
            raise ValueError(f"{tensor.name}: unsupported comfy_quant")
        group = cfg.get("convrot_groupsize") if cfg.get("convrot") else 0
        if type(group) is not int or group not in (0, 256) or (group and tensor.shape[1] % group):
            raise ValueError(f"{tensor.name}: unsupported ConvRot group")
        tensor.scale, tensor.convrot_group = scale, group
        consumed.update((scale.name, descriptor.name))
    if any(t.dtype == U8 and t.name not in consumed for t in tensors):
        raise ValueError("orphan or unsupported U8 tensor (expected paired comfy_quant)")
    for tensor in tensors:
        if tensor.dtype != E4M3:
            continue
        cfg_path = Path(config) if config else tensor.path.parent / "config.json"
        cfg = configs[cfg_path]
        if cfg.get("quant_method") != "fp8" or cfg.get("fmt", "e4m3") != "e4m3":
            raise ValueError(f"{tensor.name}: expected FP8 quantization_config")
        scale = by_name.get(tensor.name + "_scale_inv")
        if scale is not None:
            block = cfg.get("weight_block_size")
            if (not isinstance(block, list) or len(block) != 2
                    or any(type(v) is not int or v <= 0 for v in block) or len(tensor.shape) < 2):
                raise ValueError(f"{tensor.name}: invalid FP8 block configuration")
            expected = tensor.shape[:-2] + [(d + b - 1) // b for d, b in zip(tensor.shape[-2:], block)]
            if scale.shape != expected:
                raise ValueError(f"{tensor.name}: FP8 scale shape mismatch")
            tensor.block = tuple(block)
        elif ".ngram_embedding.shard_" in tensor.name:
            prefix = tensor.name.split(".ngram_embedding.shard_", 1)[0]
            scale = by_name.get(prefix + ".ngram_embedding.weight_scale")
            if scale is None or math.prod(scale.shape) != 1:
                raise ValueError(f"{tensor.name}: missing shared N-gram FP8 scale")
        else:
            raise ValueError(f"{tensor.name}: missing FP8 block scale")
        if scale.dtype not in (BF16, F16, F32):
            raise ValueError(f"{tensor.name}: invalid FP8 scale dtype")
        tensor.scale = scale
        consumed.add(scale.name)
    return [t for t in tensors if t.name not in consumed], digest.digest()[:16]


def load_raw(tensor, device):
    with tensor.path.open("rb") as file:
        file.seek(tensor.start)
        raw = bytearray(read_exact(file, tensor.size))
    return torch.frombuffer(raw, dtype=TORCH_DTYPES[tensor.dtype]).reshape(tensor.shape).to(device)


def dequantize_block_fp8(weight, scale, block):
    """weight_scale_inv is a decoding multiplier despite its suffix."""
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("FP8 scales must be finite and positive")
    if block is None:
        factors = scale.reshape(())
    else:
        rows, cols = weight.shape[-2:]
        factors = scale.repeat_interleave(block[0], -2).repeat_interleave(block[1], -1)[..., :rows, :cols]
    result = (weight.float() * factors.float()).to(torch.bfloat16)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite FP8 decoded weight")
    return result


def load_tensor(tensor, device="cuda"):
    if tensor.dtype == I8:
        raise ValueError("INT8 ConvRot must retain its payload and row scales")
    value = load_raw(tensor, device)
    if tensor.scale is not None:
        value = dequantize_block_fp8(value, load_raw(tensor.scale, device), tensor.block)
    return value if value.dtype == torch.float32 else value.to(torch.bfloat16)


def iter_tensors(tensors, device="cuda"):
    """Yield (name, shape, BF16/F32 tensor); callers release each before advancing."""
    for tensor in tensors:
        yield tensor.name, tensor.shape, load_tensor(tensor, device)
