"""Measure GLM routed-expert NVFP4 error before choosing a rotation."""
import argparse
import json
import time
from pathlib import Path
import torch
from .glm53_source import Source, SOURCE, PREFIX
from .hyt import E4M3, NVFP4
from .quant import (quantize, dequantize_nvfp4, positive_scale, tile_scales,
                    untile_scales, pack_nibbles, FP4_VALUES, FP4_MIDPOINTS)


SCALE_SEARCH = {
    'rtn': (1.,),
    # Preserve the original --mse recipe, including candidate order and ties.
    'mse': (1., .75, .8, .85, .9, .95, 1.05, 1.1, 1.15, 1.2),
    # Inspired by Murai's GLM-5.3-Flash experiments: never reduce a block's
    # stored scale below RTN. Wider scales can use the interior E2M1 levels.
    'mse-ge1': (1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75),
}


def scale_search_mode(mse=False, scale_search=None):
    if scale_search is not None and scale_search not in SCALE_SEARCH:
        raise ValueError('scale search must be rtn, mse or mse-ge1')
    if mse and scale_search not in (None, 'mse'):
        raise ValueError('--mse conflicts with --scale-search')
    return scale_search or ('mse' if mse else 'rtn')


def quantize_expert(value, mse=False, *, scale_search=None):
    mode = scale_search_mode(mse, scale_search)
    if (value.ndim != 2 or not value.numel() or value.shape[1] % 16
            or not torch.isfinite(value.float()).all()):
        raise ValueError('GLM NVFP4 requires a nonempty finite matrix with K divisible by 16')
    if mode == 'rtn':
        return quantize(value, (NVFP4, 0, 0))
    x = value.float().reshape(value.shape[0], -1, 16)
    global_scale = positive_scale(x.abs().amax(), 6. * 448.)
    base = x.abs().amax(-1) / global_scale / 6.
    best_error = torch.full(base.shape, float('inf'), device=x.device)
    best_scales = torch.zeros_like(base, dtype=torch.float8_e4m3fn)
    best_codes = torch.zeros_like(x, dtype=torch.uint8)
    midpoints = torch.tensor(FP4_MIDPOINTS, device=x.device)
    lut = torch.tensor(FP4_VALUES, device=x.device)
    # Include RTN first; every block can retain it if the search does not help.
    for ratio in SCALE_SEARCH[mode]:
        scales = (base * ratio).clamp(0, 448).to(torch.float8_e4m3fn)
        decoded = scales.float()
        norm = (x / global_scale / torch.where(decoded == 0, 1., decoded).unsqueeze(-1)).abs()
        codes = torch.bucketize(norm, midpoints).to(torch.uint8)
        codes += ((norm == .75) | (norm == 1.75) | (norm == 3.5)).to(torch.uint8)
        error = ((lut[codes.long()] * decoded.unsqueeze(-1) * global_scale - x.abs()) ** 2).sum(-1)
        better = error < best_error
        best_error = torch.minimum(error, best_error)
        best_scales = torch.where(better, scales.float(), best_scales.float()).to(torch.float8_e4m3fn)
        best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)
    best_codes |= torch.signbit(x).to(torch.uint8) << 3
    return torch.cat((pack_nibbles(best_codes), tile_scales(best_scales),
                      global_scale.reshape(1).view(torch.uint8)))


def reconstruction_metrics(value, image):
    """Weight error and range clipping, not a language-model quality gate.

    GE1 prevents *additional* clipping relative to RTN. E4M3 rounding (and
    underflow for very small blocks) can still put 6 * stored_scale below amax.
    All reductions use F64, including the denominator and clipping energy.
    """
    decoded = dequantize_nvfp4(image, value.shape)
    x = value.double().reshape(value.shape[0], -1, 16)
    energy = x.square().sum().item()
    error = (decoded.double().reshape(x.shape) - x).square().sum().item()
    scales = untile_scales(image[value.numel() // 2:-4], value.shape[0], value.shape[1] // 16).double()
    global_scale = image[-4:].contiguous().view(torch.float32).double().item()
    limits = scales * global_scale * 6.
    excess = (x.abs() - limits.unsqueeze(-1)).clamp_min(0)
    maxima = x.abs().amax(-1)
    live = maxima > 0
    depth = excess.amax(-1) / torch.where(live, maxima, 1.)
    return {
        'payload_bytes': image.numel(), 'weight_energy': energy, 'squared_error': error,
        'relative_l2': (error / energy) ** .5 if energy else 0.,
        'clipped_blocks': int((depth > 0).sum()),
        'deep_clipped_blocks': int((depth > .1).sum()),
        'max_clip_fraction': depth.max().item(),
        'clipped_energy_fraction': excess.square().sum().item() / energy if energy else 0.,
        'zeroed_nonzero_blocks': int((live & (scales == 0)).sum()),
        'blocks': maxima.numel(),
    }


def hadamard(value, block):
    if block == 0:
        return value.float()
    if block not in (32, 64, 128, 256) or value.shape[-1] % block:
        raise ValueError('GLM Hadamard block must be 32/64/128/256 and divide K')
    x = value.float().reshape(*value.shape[:-1], -1, block)
    stride = 1
    while stride < block:
        pairs = x.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        x = torch.stack((a + b, a - b), -2).reshape(x.shape)
        stride *= 2
    return (x * (block ** -.5)).reshape(value.shape)


def restored(source, name, device='cpu'):
    value = source.raw_tensor(name, device).float()
    if source.entries[name].dtype == E4M3:
        scale = source.raw_tensor(name.removesuffix('weight') + 'weight_scale_inv', device).float()
        value *= scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:value.shape[0], :value.shape[1]]
    return value


def measure(source, layers, experts, device='cpu', mse=False, *, scale_search=None):
    start = time.perf_counter()
    source = Source(source)
    mode = scale_search_mode(mse, scale_search)
    if not layers or not experts or any(l < 3 or l > 44 for l in layers) or any(e < 0 or e >= 288 for e in experts):
        raise ValueError('measure layers 3..44 and experts 0..287')
    results = []
    for layer in layers:
        for expert in experts:
            for part in ('gate', 'up', 'down'):
                name = PREFIX + f'layers.{layer}.mlp.experts.{expert}.{part}_proj.weight'
                value = restored(source, name, device)
                energy = value.double().square().sum()
                errors = {}
                for block in (0, 32, 64, 128, 256):
                    rotated = hadamard(value, block)
                    image = quantize_expert(rotated, scale_search=mode)
                    decoded = dequantize_nvfp4(image, value.shape)
                    # Orthogonality makes this equal to error in the original basis.
                    errors[str(block)] = float(((decoded - rotated).double().square().sum() / energy).sqrt())
                entry = dict(layer=layer, expert=expert, part=part, relative_l2=errors)
                results.append(entry)
                print(json.dumps(entry), flush=True)
    means = {str(b): sum(r['relative_l2'][str(b)] for r in results) / len(results)
             for b in (0, 32, 64, 128, 256)}
    return dict(matrices=results, mean_relative_l2=means,
                best_block=int(min(means, key=means.get)), elapsed_s=time.perf_counter() - start,
                device=device, threads=torch.get_num_threads(), scale_search=mode,
                source=str(source.root), torch=torch.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 6, 10])
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 17, 143])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    search = parser.add_mutually_exclusive_group()
    search.add_argument('--mse', action='store_true', help='alias for --scale-search mse')
    search.add_argument('--scale-search', choices=SCALE_SEARCH)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = measure(args.source, args.layers, args.experts, args.device, args.mse, scale_search=args.scale_search)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'matrices'}, indent=2))


if __name__ == '__main__':
    main()
