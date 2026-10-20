"""Torch RTN quantizers. The same operations run on CPU in correctness tests."""

import torch

from .hyt import E4M3, NVFP4, PER_TENSOR, Q4_BF16, align

FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
FP4_MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)


def scale_offset(row, col, padded_cols):
    return ((row // 128) * (padded_cols // 4) + col // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + col % 4


def tile_scales(scales):
    rows, cols = scales.shape[-2:]
    padded = torch.zeros((*scales.shape[:-2], align(rows, 128), align(cols, 4)), dtype=torch.uint8, device=scales.device)
    padded[..., :rows, :cols] = scales.contiguous().view(torch.uint8)
    # [row tile, row quarter, row in quarter, column tile, column in tile]
    return (padded.reshape(-1, align(rows, 128) // 128, 4, 32, align(cols, 4) // 4, 4)
            .permute(0, 1, 4, 3, 2, 5).contiguous()
            .reshape(*scales.shape[:-2], align(rows, 128) * align(cols, 4)))


def untile_scales(data, rows, cols):
    return (data.reshape(align(rows, 128) // 128, align(cols, 4) // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4).contiguous().reshape(align(rows, 128), align(cols, 4))
            [:rows, :cols].contiguous().view(torch.float8_e4m3fn))


def positive_scale(maximum, divisor):
    # Keep the stored divisor nonzero even for the smallest F32 input.
    minimum = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    return torch.where(maximum == 0, 1., (maximum / divisor).clamp_min(minimum))


def pack_nibbles(codes):
    flat = codes.flatten()
    return flat[::2] | (flat[1::2] << 4)


def nvfp4(value):
    rows, cols = value.shape[-2:]
    if cols % 16:
        raise ValueError("NVFP4 K must be divisible by 16")
    x = value.float().reshape(-1, rows, cols // 16, 16)
    matrix_scale = positive_scale(x.abs().amax((-3, -2, -1), keepdim=True), 6. * 448.)
    if value.ndim == 2:
        matrix_scale = matrix_scale.reshape(())
    block_max = x.abs().amax(-1)
    scales = (block_max / matrix_scale.squeeze(-1) / 6.).clamp(0, 448).to(torch.float8_e4m3fn)
    decoded = scales.float()
    normalized = (x / matrix_scale / torch.where(decoded == 0, 1., decoded).unsqueeze(-1)).abs()
    midpoints = torch.tensor(FP4_MIDPOINTS, device=x.device)
    indices = torch.bucketize(normalized, midpoints).to(torch.uint8)
    # Midpoint ties use the even E2M1 code, including the exponent boundary.
    tied_up = (normalized == .75) | (normalized == 1.75) | (normalized == 3.5)
    indices += tied_up.to(torch.uint8)
    indices |= torch.signbit(x).to(torch.uint8) << 3
    batch = x.shape[0]
    # Each matrix owns its payload, padded tiles and FP32 scale (§5.3).
    return torch.cat((pack_nibbles(indices).reshape(batch, -1),
                      tile_scales(scales).reshape(batch, -1),
                      matrix_scale.reshape(batch, 1).view(torch.uint8)), -1).flatten()


def q4_bf16(value):
    cols = value.shape[-1]
    if cols % 32:
        raise ValueError("Q4Bf16 K must be divisible by 32")
    x = value.float().reshape(-1, cols // 32, 32)
    minimum = torch.finfo(torch.bfloat16).tiny * torch.finfo(torch.bfloat16).eps
    scales = positive_scale(x.abs().amax(-1), 7.).clamp_min(minimum).to(torch.bfloat16)
    codes = ((x / scales.float().unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8) + 8).to(torch.uint8)
    return torch.cat((pack_nibbles(codes), scales.contiguous().flatten().view(torch.uint8)))


def fp8(value):
    x = value.float()
    scale = positive_scale(x.abs().amax(), 448.)
    payload = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).flatten().view(torch.uint8)
    return torch.cat((payload, scale.reshape(1).view(torch.uint8)))


def quantize(value, spec):
    if not torch.isfinite(value).all():
        raise ValueError("quantization requires finite weights")
    if not 2 <= value.ndim <= 6 or any(d == 0 for d in value.shape):
        raise ValueError("quantized weights require rank 2..6 and positive dimensions")
    if spec == (NVFP4, 0, 0):
        return nvfp4(value)
    if spec == (Q4_BF16, 0, 32):
        return q4_bf16(value)
    if spec == (PER_TENSOR, E4M3, 0):
        return fp8(value)
    raise ValueError("unsupported quantization spec")


def dequantize_nvfp4(data, shape):
    rows, cols = shape[-2:]
    payload_len = rows * cols // 2
    tile_len = align(rows, 128) * align(cols // 16, 4)
    matrices = data.reshape(-1, payload_len + tile_len + 4)
    payload = matrices[:, :payload_len]
    codes = torch.stack((payload & 15, payload >> 4), -1).reshape(-1, rows, cols // 16, 16)
    lut = torch.tensor(FP4_VALUES, device=data.device)
    values = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
    scales = torch.stack([untile_scales(m[payload_len:-4], rows, cols // 16) for m in matrices]).float()
    matrix_scale = matrices[:, -4:].contiguous().view(torch.float32).reshape(-1, 1, 1, 1)
    return (values * scales.unsqueeze(-1) * matrix_scale).reshape(shape)
