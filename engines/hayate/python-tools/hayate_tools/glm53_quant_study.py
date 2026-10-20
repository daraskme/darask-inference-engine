"""Compare NVFP4 recipes on real routed experts without converting a checkpoint.

Reports reconstruction and clipping separately; no model-quality claim is made.
One source matrix is loaded at a time. Output includes hashes of restored F32
matrices so results from different FP8/BF16 parents cannot be silently conflated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .glm53_source import Source, SOURCE, PREFIX
from .glm53_nvfp4 import (SCALE_SEARCH, hadamard, quantize_expert,
                          reconstruction_metrics, restored)


def study(source, layers, experts, blocks=(0,), modes=tuple(SCALE_SEARCH), device='cpu'):
    for values, valid, label in (
            (layers, range(3, 45), 'layers'), (experts, range(288), 'experts'),
            (blocks, (0, 32, 64, 128, 256), 'Hadamard blocks'),
            (modes, SCALE_SEARCH, 'scale searches')):
        if not values or len(set(values)) != len(values) or any(v not in valid for v in values):
            raise ValueError(f'invalid or duplicate {label}')
    start = time.perf_counter()
    source = Source(source)
    results = []
    for layer in layers:
        for expert in experts:
            for part in ('gate', 'up', 'down'):
                name = PREFIX + f'layers.{layer}.mlp.experts.{expert}.{part}_proj.weight'
                value = restored(source, name, device)
                digest = hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
                for block in blocks:
                    rotated = hadamard(value, block)
                    recipes = {}
                    for mode in modes:
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        before = time.perf_counter()
                        image = quantize_expert(rotated, scale_search=mode)
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - before
                        recipes[mode] = reconstruction_metrics(rotated, image)
                        recipes[mode]['quantize_s'] = elapsed
                        del image
                    row = dict(layer=layer, expert=expert, part=part, shape=list(value.shape),
                               restored_f32_sha256=digest, hadamard_block=block, recipes=recipes)
                    results.append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                del value, rotated
    summary = {}
    for block in blocks:
        for mode in modes:
            rows = [r['recipes'][mode] for r in results if r['hadamard_block'] == block]
            energy = sum(r['weight_energy'] for r in rows)
            error = sum(r['squared_error'] for r in rows)
            summary[f'{mode}/r{block}'] = {
                'matrices': len(rows), 'relative_l2': (error / energy) ** .5 if energy else 0.,
                'mean_matrix_relative_l2': sum(r['relative_l2'] for r in rows) / len(rows),
                'payload_bytes': sum(r['payload_bytes'] for r in rows),
                'deep_clipped_blocks': sum(r['deep_clipped_blocks'] for r in rows),
                'max_clip_fraction': max(r['max_clip_fraction'] for r in rows),
                'zeroed_nonzero_blocks': sum(r['zeroed_nonzero_blocks'] for r in rows),
                'quantize_s': sum(r['quantize_s'] for r in rows),
            }
    return dict(schema_version=1, source=str(source.root.resolve()),
                source_config_sha256=hashlib.sha256((source.root / 'config.json').read_bytes()).hexdigest(),
                source_basis='F32 restoration of source weights; not a BF16-master comparison',
                layers=list(layers), experts=list(experts), hadamard_blocks=list(blocks),
                scale_search_ratios={m: SCALE_SEARCH[m] for m in modes},
                device=device, torch=torch.__version__, threads=torch.get_num_threads(),
                elapsed_s=time.perf_counter() - start, summary=summary, matrices=results,
                quality_gate=None, quality_status='not_evaluated',
                limitations=['weight reconstruction only; no KLD/NLL/task evaluation',
                             'sampled experts, not a whole-model result',
                             'timings include cold calls; not an inference-speed benchmark'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 23, 44])
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 143, 287])
    parser.add_argument('--hadamard-blocks', type=int, nargs='+', default=[0])
    parser.add_argument('--scale-searches', nargs='+', choices=SCALE_SEARCH, default=list(SCALE_SEARCH))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    report = study(args.source, args.layers, args.experts, args.hadamard_blocks, args.scale_searches, args.device)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
