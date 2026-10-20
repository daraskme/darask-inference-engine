"""Fixed teacher-forced FP8/NVFP4 KLD, top-1 agreement, and next-token NLL."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import torch

# W4A16 tensor-core prefill and GEMV use different reduction orders. The
# 45-layer all-GPU prefix measured 0.01435 on 25f0cd1; retain a per-row gate.
PREFILL_MAX_KLD = .02


def metrics(reference, actual):
    if reference.ndim != 2 or actual.shape != reference.shape or not reference.numel():
        raise ValueError('logits must be matching nonempty [positions,vocab]')
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError('nonfinite logits')
    ref_log = reference.double().log_softmax(-1)
    act_log = actual.double().log_softmax(-1)
    kld = (ref_log.exp() * (ref_log - act_log)).sum(-1).clamp_min(0)
    return kld, reference.argmax(-1) == actual.argmax(-1), ref_log, act_log


def load_meta(folder):
    meta = json.loads((folder / 'eval.json').read_text())
    for key in ('rows', 'vocab', 'prefix', 'layers', 'prefill_chunk'):
        if type(meta.get(key)) is not int or meta[key] <= 0:
            raise ValueError(f'invalid evaluation {key}')
    if meta['rows'] < 2 or not isinstance(meta.get('source_hash'), str) or len(meta['source_hash']) != 32:
        raise ValueError('invalid evaluation rows/source hash')
    bytes.fromhex(meta['source_hash'])
    dtype = meta.get('dtype', 'f32')
    if dtype not in ('bf16', 'f32'):
        raise ValueError('unsupported logits dtype')
    ids = (folder / 'eval_ids.i32.bin').read_bytes()
    if len(ids) != 4 * (meta['rows'] + meta['prefix']):
        raise ValueError('evaluation IDs length mismatch')
    tokens = struct.unpack(f'<{len(ids)//4}I', ids)
    if any(t >= meta['vocab'] for t in tokens):
        raise ValueError('evaluation ID outside vocabulary')
    path = folder / f'eval_logits.{dtype}.bin'
    if path.stat().st_size != meta['rows'] * meta['vocab'] * (2 if dtype == 'bf16' else 4):
        raise ValueError('logits size mismatch')
    return meta, ids, tokens, path, dtype


def compare(reference, actual, chunk=8, mode="quantization", cross_source=False):
    """cross_source: the actual checkpoint comes from another source (e.g. an EXL3 quant of the original
    weights against the dealignai FP8 reference); its source hash is reported instead of required equal."""
    if chunk <= 0 or mode not in ('quantization', 'prefill'):
        raise ValueError('invalid comparison chunk/mode')
    r, rids, tokens, rpath, rdtype = load_meta(Path(reference))
    a, aids, _, apath, adtype = load_meta(Path(actual))
    keys = ('rows', 'vocab', 'prefix', 'layers') + (() if cross_source else ('source_hash',))
    if mode == 'quantization':
        keys += ('prefill_chunk',)
    elif r.get('eval_mode') != 'step' or a.get('eval_mode') != 'prefill':
        raise ValueError('prefill comparison requires step reference and prefill actual')
    if any(r[k] != a[k] for k in keys) or rids != aids:
        raise ValueError('evaluation contexts/layers/vocabulary/source differ')
    rows, vocab, prefix = r['rows'], r['vocab'], r['prefix']
    total, maximum, matches, ref_nll, act_nll = 0., 0., 0, 0., 0.
    with rpath.open('rb') as ref, apath.open('rb') as act:
        for position in range(0, rows, chunk):
            count = min(chunk, rows - position)
            values = []
            for file, dtype in ((ref, rdtype), (act, adtype)):
                width, torch_type = (2, torch.bfloat16) if dtype == 'bf16' else (4, torch.float32)
                raw = bytearray(file.read(count * vocab * width))
                if len(raw) != count * vocab * width:
                    raise ValueError('short logits read')
                values.append(torch.frombuffer(raw, dtype=torch_type).reshape(count, vocab).float())
            kld, equal, ref_log, act_log = metrics(*values)
            total += kld.sum().item()
            maximum = max(maximum, kld.max().item())
            matches += equal.sum().item()
            # Qwen row j consumes ids[prefix+j] and predicts ids[prefix+j+1].
            n = min(count, rows - position - 1)
            if n:
                targets = torch.tensor(tokens[prefix+position+1:prefix+position+1+n])
                ref_nll -= ref_log[torch.arange(n), targets].sum().item()
                act_nll -= act_log[torch.arange(n), targets].sum().item()
    mean, rate = total / rows, matches / rows
    prefix_kld = None
    if mode == 'prefill':
        values = []
        for folder in (Path(reference), Path(actual)):
            raw = bytearray((folder / 'prefix_logits.bf16.bin').read_bytes())
            if len(raw) != vocab * 2:
                raise ValueError('prefix logits size mismatch')
            values.append(torch.frombuffer(raw, dtype=torch.bfloat16).reshape(1, vocab).float())
        prefix_kld = metrics(*values)[0].item()

    return {'mode': mode, 'mean_kld': mean, 'prefix_kld': prefix_kld,
            'prefill_max_kld_limit': PREFILL_MAX_KLD if mode == 'prefill' else None,
            'reference_prefill_chunk': r['prefill_chunk'],
            'actual_prefill_chunk': a['prefill_chunk'], 'positions': rows, 'nll_positions': rows - 1,
            'mean_kld_fp8_to_nvfp4': mean, 'max_kld': maximum,
            'top1_matches': matches, 'top1_rate': rate,
            'reference_mean_nll': ref_nll / (rows - 1),
            'actual_mean_nll': act_nll / (rows - 1),
            'mean_nll_delta': (act_nll - ref_nll) / (rows - 1),
            'input_sha256': hashlib.sha256(rids).hexdigest(),
            'source_hash': r['source_hash'], 'layers': r['layers'], 'prefix': prefix,
            **({'actual_source_hash': a['source_hash']} if cross_source else {}),
            'prefill_chunk': r['prefill_chunk'],
            'quality_gate': (maximum <= PREFILL_MAX_KLD and prefix_kld <= PREFILL_MAX_KLD) if mode == 'prefill' else (mean <= .15 and rate >= .82)}


def corpus(out, tokens=512):
    root = Path(__file__).parent
    text = (root / 'glm53_eval_corpus.txt').read_bytes()
    frozen = json.loads((root / 'glm53_eval_corpus.ids.json').read_text())
    if hashlib.sha256(text).hexdigest() != frozen['text_sha256']:
        raise ValueError('fixed corpus text hash mismatch')
    ids = frozen['ids']
    if not 256 <= tokens <= min(512, len(ids)):
        raise ValueError('choose 256..512 tokens within the frozen corpus')
    raw = struct.pack(f'<{tokens}I', *ids[:tokens])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'input_ids.i32.bin').write_bytes(raw)
    (out / 'corpus.json').write_text(json.dumps({
        'tokens': tokens, 'text_sha256': frozen['text_sha256'],
        'tokenizer_sha256': frozen['tokenizer_sha256'],
        'ids_sha256': hashlib.sha256(raw).hexdigest()}, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path)
    p.add_argument('--actual', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--out', type=Path, help='comparison report JSON')
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--mode', choices=('quantization', 'prefill'), default='quantization')
    p.add_argument('--corpus-out', type=Path)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--cross-source', action='store_true',
                   help='allow an actual checkpoint converted from another source (reported, not required equal)')
    a = p.parse_args()
    if a.corpus_out:
        corpus(a.corpus_out, a.tokens)
        return
    if not a.reference or not a.actual:
        p.error('comparison requires --reference and --actual')
    report = compare(a.reference, a.actual, a.chunk, a.mode, a.cross_source)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if a.out:
        a.out.write_text(text)
    print(text, end='')
    if a.check and not report['quality_gate']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
