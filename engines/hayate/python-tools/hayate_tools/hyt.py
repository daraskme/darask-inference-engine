"""HYT v2 wire format; see docs/ARCHITECTURE.md §5.3."""

import math
import struct

MAGIC = b"HYT0"
FORMAT_VERSION = 2
HEADER_LEN = 64
DATA_ALIGN = 4096
MAX_RANK = 6
U64_MAX = (1 << 64) - 1
HEADER = struct.Struct("<4sIIIQQ16s16x")
DTYPE_BITS = (32, 16, 16, 8, 8, 4, 8, 32, 4)
PLAIN, PER_TENSOR, NVFP4, MXFP8, Q4_BF16, INT8_ROW = range(6)
# Tag 6 (LOWBIT) is written by the quantization track's tools; tag 7 is EXL3
# (dtype word = k2 = twice the bits per weight, group word = codebook).
EXL3 = 7
F32, BF16, F16, E4M3 = range(4)


def exl3_valid(k2, cb):
    """exllamav3 1.6 bitrates: K 1..8 with any codebook, K 1.5/2.5/3.5 with mul1 (2)."""
    return (2 <= k2 <= 16 and k2 % 2 == 0 and cb in (0, 1, 2)) or (k2 in (3, 5, 7) and cb == 2)


def align(value, alignment=DATA_ALIGN):
    return (value + alignment - 1) // alignment * alignment


def storage_bytes(shape, quant):
    """Same sizing and overflow rules as Rust TensorDesc::storage_bytes."""
    if len(shape) > MAX_RANK or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("invalid shape")
    n = math.prod(shape)
    if n > U64_MAX:
        raise ValueError("element count overflows u64")
    tag, dtype, group = quant
    if tag in (PLAIN, PER_TENSOR):
        if dtype not in range(len(DTYPE_BITS)) or group != 0:
            raise ValueError("invalid QuantSpec")
        bits = n * DTYPE_BITS[dtype]
        if bits > U64_MAX:
            raise ValueError("payload bit count overflows u64")
        result = (bits + 7) // 8 + (4 if tag == PER_TENSOR else 0)
    elif tag == EXL3:
        # [rows = outputs, cols = inputs]: the trellis tiles (16 k2 bytes each),
        # then FP16 suh [cols] and svh [rows]; 128-multiples keep each part 256-aligned.
        if len(shape) != 2 or shape[0] % 128 or shape[1] % 128 or not exl3_valid(dtype, group):
            raise ValueError("invalid EXL3 matrix")
        result = n * dtype // 16 + 2 * (shape[0] + shape[1])
    elif tag == INT8_ROW:
        # INT8 payload + one FP32 scale per row; group = ConvRot group (0 = none).
        if dtype != 0 or not 0 <= group <= (1 << 32) - 1:
            raise ValueError("invalid QuantSpec")
        cols = shape[-1] if shape else 1
        result = n + (n // cols) * 4
    else:
        if dtype != 0 or tag not in (NVFP4, MXFP8, Q4_BF16):
            raise ValueError("invalid QuantSpec")
        if (tag == Q4_BF16 and not 0 < group <= (1 << 32) - 1) or (tag != Q4_BF16 and group):
            raise ValueError("invalid block size")
        cols = shape[-1] if shape else 1
        rows = shape[-2] if len(shape) > 1 else 1
        batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
        block = 16 if tag == NVFP4 else 32 if tag == MXFP8 else group
        if cols % block:
            raise ValueError("last dimension is not divisible by block size")
        if tag == Q4_BF16:
            rows, batch = n // cols, 1
            scale_bytes = rows * (cols // block) * 2
        else:
            scale_bytes = align(rows, 128) * align(cols // block, 4)
        bits = rows * cols * (8 if tag == MXFP8 else 4)
        if bits > U64_MAX:
            raise ValueError("payload bit count overflows u64")
        result = batch * ((bits + 7) // 8 + scale_bytes + (4 if tag == NVFP4 else 0))
    if result > U64_MAX:
        raise ValueError("storage size overflows u64")
    return result


def encode_entry(name, shape, quant, offset, size):
    name = name.encode("utf-8")
    if not 0 < len(name) < 1 << 32:
        raise ValueError("invalid tensor name length")
    if offset < 0 or offset % DATA_ALIGN or offset + size > U64_MAX:
        raise ValueError("invalid tensor data range")
    if size != storage_bytes(shape, quant):
        raise ValueError("tensor byte count mismatch")
    return (struct.pack("<I", len(name)) + name + struct.pack("<IIII", *quant, len(shape))
            + b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<QQ", offset, size))
