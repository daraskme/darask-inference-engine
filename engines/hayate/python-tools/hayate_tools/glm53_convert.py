"""Export GLM to HYT; optionally requantize routed experts directly to NVFP4."""
import argparse
import hashlib
import json
import os
import struct
import time
import statistics
import tempfile
from pathlib import Path
from .glm53_source import Source, SOURCE, PREFIX
from .sources import TORCH_DTYPES
import torch
from .hyt import HEADER, FORMAT_VERSION, PLAIN, PER_TENSOR, NVFP4, E4M3, F32, align, encode_entry, storage_bytes
from .glm53_nvfp4 import hadamard, restored, quantize_expert, scale_search_mode, SCALE_SEARCH
from .quant import quantize


DENSE_GROUPS = ('shared', 'kda', 'sparse', 'dense-ffn', 'lm-head')


def dense_options(values):
    result = {}
    for value in values:
        group, sep, fmt = value.partition('=')
        if not sep or group not in DENSE_GROUPS or fmt not in ('bf16', 'fp8', 'nvfp4') or group in result:
            raise ValueError('dense requires unique GROUP=bf16|fp8|nvfp4: ' + ', '.join(DENSE_GROUPS))
        result[group] = fmt
    return result


def dense_group(name, index):
    if index == 45 and name == 'eh_proj.weight':
        return 'sparse'
    if name == 'lm_head.weight':
        return 'lm-head'
    if name.startswith('mlp.shared_experts.') and name.endswith('_proj.weight'):
        return 'shared'
    if name.startswith('mlp.') and not name.startswith(('mlp.experts.', 'mlp.shared_experts.')) and name.endswith('_proj.weight'):
        return 'dense-ffn'
    if name.startswith('self_attn.') and (name.endswith('_proj.weight') or name in (
            'self_attn.kv_a_proj_with_mqa.weight', 'self_attn.indexer.wq_b.weight', 'self_attn.indexer.wk.weight',
            'self_attn.indexer.weights_proj.weight', 'self_attn.indexer.index_kpool_compress_gate')):
        return 'sparse' if index % 4 == 3 or index == 45 else 'kda'
    return None


def dense_spec(fmt):
    return {'bf16': (PLAIN, 1, 0), 'fp8': (PER_TENSOR, E4M3, 0), 'nvfp4': (NVFP4, 0, 0)}[fmt]


def plan(source, index, expert_quant='fp8', hadamard_block=0, down_fp8=False, dense=None):
    dense = dense or {}
    if expert_quant not in ('fp8', 'nvfp4'):
        raise ValueError('expert quant must be fp8 or nvfp4')
    if hadamard_block not in (0, 32, 64, 128, 256) or (expert_quant != 'nvfp4' and (hadamard_block or down_fp8)):
        raise ValueError('rotation/down-fp8 requires NVFP4 experts and block 0/32/64/128/256')
    stem = PREFIX + f'layers.{index}.'
    entries, offset = [], 0
    for full, item in sorted(source.entries.items()):
        if not full.startswith(stem):
            continue
        name = full[len(stem):]
        base = name.removesuffix('weight_scale_inv') + 'weight' if name.endswith('weight_scale_inv') else name
        group = dense_group(base, index)
        if name.endswith('weight_scale_inv') and group in dense:
            continue
        routed = name.startswith('mlp.experts.') and not (down_fp8 and '.down_proj.' in name)
        if routed and expert_quant == 'nvfp4' and name.endswith('weight_scale_inv'):
            continue
        quant = (NVFP4, 0, 0) if routed and expert_quant == 'nvfp4' and name.endswith('.weight') else (PLAIN, item.dtype, 0)
        if group in dense:
            quant = dense_spec(dense[group])
        size = storage_bytes(item.shape, quant)
        entry = (name, item.shape, quant, offset, size)
        encode_entry(*entry)
        entries.append((entry, item))
        offset = align(offset + size)
    if not entries:
        raise ValueError(f'missing layer {index}')
    if index >= 3 and hadamard_block:
        for entry, _ in entries:
            if entry[0].startswith('mlp.experts.') and entry[2] == (NVFP4, 0, 0) and entry[1][-1] % hadamard_block:
                raise ValueError('Hadamard block must divide every rotated K')
        entries.append((('mlp.experts.hadamard_block', [1], (PLAIN, F32, 0), offset, 4), None))
    return entries


def write_layer(dest, entries, digest, source=None, device='cpu', hadamard_block=0, mse=False, *, scale_search=None):
    mode = scale_search_mode(mse, scale_search)
    rotated = hadamard_block != 0 and any(e[0].startswith('mlp.experts.') and e[2] == (NVFP4, 0, 0) for e, _ in entries)
    metadata = any(e[0] == 'mlp.experts.hadamard_block' and item is None for e, item in entries)
    if rotated != metadata:
        raise ValueError('rotation and HYT metadata must be written together')
    dest = Path(dest)
    table = b''.join(encode_entry(*entry) for entry, _ in entries)
    base = align(64 + len(table))
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix('.hyt.partial')
    if dest.exists() or partial.exists():
        raise FileExistsError(dest)
    try:
        with partial.open('xb') as out:
            out.write(HEADER.pack(b'HYT0', FORMAT_VERSION, 0, len(entries), 64, base, digest[:16]))
            out.write(table)
            for entry, item in entries:
                out.seek(base + entry[3])
                if item is None:
                    out.write(struct.pack('<f', hadamard_block))
                    continue
                routed = entry[0].startswith('mlp.experts.')
                if not routed and (entry[2] != (PLAIN, item.dtype, 0) or entry[4] != item.size):
                    if source is None:
                        raise ValueError('dense conversion requires the source')
                    value = restored(source, item.name, device).to(torch.bfloat16).float()
                    payload = quantize(value, entry[2]).cpu().numpy().tobytes() if entry[2][0] != PLAIN else value.to(torch.bfloat16).cpu().view(torch.uint8).numpy().tobytes()
                    if len(payload) != entry[4]:
                        raise ValueError('dense storage size mismatch')
                    out.write(payload)
                    continue
                if entry[2] == (NVFP4, 0, 0):
                    if source is None:
                        raise ValueError('NVFP4 conversion requires the source scales')
                    # F32 FP8*block-scale, without an intermediate BF16 rounding.
                    value = hadamard(restored(source, item.name, device), hadamard_block)
                    payload = quantize_expert(value, scale_search=mode).cpu().numpy().tobytes()
                    if len(payload) != entry[4]:
                        raise ValueError('NVFP4 storage size mismatch')
                    out.write(payload)
                    continue
                with item.path.open('rb') as file:
                    file.seek(item.start)
                    remaining = item.size
                    while remaining:
                        raw = file.read(min(remaining, 4 << 20))
                        if not raw:
                            raise ValueError('short source')
                        if not torch.isfinite(torch.frombuffer(bytearray(raw), dtype=TORCH_DTYPES[item.dtype]).float()).all():
                            raise ValueError(f"{item.name}: nonfinite weight")
                        out.write(raw)
                        remaining -= len(raw)
            out.flush()
            os.fsync(out.fileno())
        partial.rename(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def convert(source, out, layers=5, estimate=False, expert_quant='fp8', device='cpu', hadamard_block=0, mse=False, down_fp8=False, dense=None, mtp=False, *, scale_search=None):
    dense = dense or {}
    start = time.perf_counter()
    if not 5 <= layers <= 45:
        raise ValueError('use 5..45 layers to include all four layer kinds')
    mode = scale_search_mode(mse, scale_search)
    if mode != 'rtn' and expert_quant != 'nvfp4':
        raise ValueError('MSE search requires NVFP4 experts')
    source = Source(source)
    if mtp and (layers != 45 or source.config['text_config']['num_nextn_predict_layers'] != 1):
        raise ValueError('MTP requires 45 target layers and one nextn layer')
    indices = list(range(layers)) + ([45] if mtp else [])
    plans = [plan(source, i, expert_quant, hadamard_block, down_fp8, dense) for i in indices]
    globals_plan, offset = [], 0
    for name, full in [('embed_tokens.weight', PREFIX+'embed_tokens.weight'), ('norm.weight', PREFIX+'norm.weight'), ('lm_head.weight', 'lm_head.weight')]:
        item = source.entries[full]
        spec = dense_spec(dense['lm-head']) if name == 'lm_head.weight' and 'lm-head' in dense else (PLAIN, item.dtype, 0)
        entry = (name, item.shape, spec, offset, storage_bytes(item.shape, spec))
        globals_plan.append((entry, item))
        offset = align(offset + entry[4])
    experts = sum(e[4] for p in plans for e, item in p if item is not None and e[0].startswith('mlp.experts.'))
    def file_bytes(p):
        base = align(64 + sum(len(encode_entry(*e)) for e, _ in p))
        return base + max(e[3] + e[4] for e, _ in p)
    summary = {'mtp_resident_bytes': sum(e[4] for e, _ in plans[-1] if not e[0].startswith('mlp.experts.')) if mtp else 0,
               'mtp_routed_bytes': sum(e[4] for e, _ in plans[-1] if e[0].startswith('mlp.experts.')) if mtp else 0,
               'mtp': mtp, 'mtp_payload_bytes': sum(e[4] for e, _ in plans[-1]) if mtp else 0,
               'dense': dense, 'expert_quant': expert_quant, 'hadamard_block': hadamard_block,
               'scale_search': mode, 'scale_search_ratios': SCALE_SEARCH[mode], 'down_fp8': down_fp8,
               'routed_expert_bytes': experts, 'source_bytes': source.inventory(),
               'disk_bytes': sum(file_bytes(p) for p in plans) + file_bytes(globals_plan),
               'layer_payload_bytes': [sum(e[4] for e, _ in p) for p in plans],
               'globals_payload_bytes': sum(e[4] for e, _ in globals_plan)}
    if not estimate:
        destination = Path(out)
        if destination.joinpath('weights.json').exists() or any(
                (destination / p).exists() or (destination / p).with_suffix('.hyt.partial').exists()
                for p in ['globals.hyt'] + [f'layer{i}/weights.hyt' for i in range(layers)]):
            raise FileExistsError(destination)
        digest = hashlib.sha256()
        for path in sorted(source.root.glob('*.safetensors')):
            with path.open('rb') as file:
                while raw := file.read(4 << 20):
                    digest.update(raw)
        for i, entries in enumerate(plans):
            layer_start = time.perf_counter()
            write_layer(Path(out) / f'layer{i}' / 'weights.hyt', entries, digest.digest(), source, device, hadamard_block, scale_search=mode)
            print(json.dumps({'layer': i, 'elapsed_s': time.perf_counter() - layer_start}), flush=True)
        write_layer(Path(out) / 'globals.hyt', globals_plan, digest.digest(), source, device)
        summary['elapsed_s'] = time.perf_counter() - start
        Path(out, 'weights.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def pilot(source, layers=45, device='cuda', hadamard_block=0, mse=False, down_fp8=False, *, scale_search=None):
    """One real expert, including input I/O, transfers, quantization and fsync.

    Extrapolation excludes the one-time full-source SHA256 and resident copies.
    """
    if not 5 <= layers <= 45:
        raise ValueError('pilot requires 5..45 layers')
    mode = scale_search_mode(mse, scale_search)
    source = Source(source)
    entries, offset = [], 0
    for entry, item in plan(source, 3, 'nvfp4', hadamard_block, down_fp8):
        if entry[0].startswith('mlp.experts.0.') or item is None:
            entry = (*entry[:3], offset, entry[4])
            entries.append((entry, item))
            offset = align(offset + entry[4])
    samples = []
    with tempfile.TemporaryDirectory(prefix='glm53-convert-pilot-') as tmp:
        for run in range(4):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            path = Path(tmp) / f'expert{run}.hyt'
            write_layer(path, entries, bytes(32), source, device, hadamard_block, scale_search=mode)
            if device == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            path.unlink()
            if run:
                samples.append(seconds)
    median = statistics.median(samples)
    return {'device': device, 'layers': layers, 'warmup': 1, 'repeats': 3,
            'hadamard_block': hadamard_block, 'mse': mode == 'mse', 'scale_search': mode,
            'scale_search_ratios': SCALE_SEARCH[mode], 'down_fp8': down_fp8,
            'samples_s': samples, 'median_expert_s': median,
            'estimated_expert_conversion_s': median * (layers - 3) * 288,
            'excluded': ['source SHA256', 'resident/global copies', 'header planning'],
            'pilot_output_directory': tempfile.gettempdir()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=Path('/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/golden/glm53'))
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--estimate', action='store_true')
    parser.add_argument('--mtp', action='store_true', help='include NextN layer45, using the same expert/dense formats')
    parser.add_argument('--pilot', action='store_true', help='time one expert (warmup1/repeat3) and extrapolate; use TMPDIR on the target SSD')
    parser.add_argument('--expert-quant', choices=['fp8', 'nvfp4'], default='fp8')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--hadamard-block', type=int, choices=[0, 32, 64, 128, 256], default=0)
    search = parser.add_mutually_exclusive_group()
    search.add_argument('--mse', action='store_true', help='alias for --scale-search mse')
    search.add_argument('--scale-search', choices=SCALE_SEARCH, help='routed expert scale search; mse-ge1 avoids scale reductions below RTN')
    parser.add_argument('--dense', action='append', default=[], metavar='GROUP=FORMAT', help='shared/kda/sparse/dense-ffn/lm-head=bf16|fp8|nvfp4; repeatable, no rotation')
    parser.add_argument('--down-fp8', action='store_true', help='retain original block-FP8 down projection')
    args = parser.parse_args()
    dense = dense_options(args.dense)
    if args.pilot:
        if dense:
            parser.error("--pilot measures routed experts only; omit --dense")
        if args.estimate or args.expert_quant != 'nvfp4':
            parser.error('--pilot requires --expert-quant nvfp4 without --estimate')
        print(json.dumps(pilot(args.source, args.layers, args.device, args.hadamard_block,
                               args.mse, args.down_fp8, scale_search=args.scale_search), indent=2))
        return
    print(json.dumps(convert(args.source, args.out, args.layers, args.estimate, args.expert_quant,
                             args.device, args.hadamard_block, args.mse, args.down_fp8, dense, args.mtp,
                             scale_search=args.scale_search), indent=2))


if __name__ == '__main__':
    main()
