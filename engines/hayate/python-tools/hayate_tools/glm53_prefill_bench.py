"""Compare GLM prefill against tokenwise prefixes, then time 2048/8192 tokens."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import torch
from .glm53_eval import compare


def input_ids(length):
    frozen = json.loads(Path(__file__).with_name('glm53_eval_corpus.ids.json').read_text())
    ids = frozen['ids']
    # A deterministic repetition stress test, not a new natural-language quality corpus.
    return struct.pack(f'<{length}I', *(ids * ((length + len(ids)-1)//len(ids)))[:length])


def breakdown(text):
    layers, transfers, lookahead, workspace = [], [], [], []
    for line in text.splitlines():
        if line.startswith(('glm_prefill_layer ', 'glm_prefill_transfer ', 'glm_prefill_lookahead ', 'glm_prefill_workspace ')):
            row = dict(part.split('=', 1) for part in line.split()[1:])
            for key, value in list(row.items()):
                if key.endswith(('_ms', '_bytes')):
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('invalid prefill profile value')
                    row[key] = value
            target = layers if line.startswith('glm_prefill_layer ') else (lookahead if line.startswith('glm_prefill_lookahead ') else (workspace if line.startswith('glm_prefill_workspace ') else transfers))
            target.append(row)
    if not layers:
        raise ValueError('no prefill layer profile')
    for row in transfers + lookahead:
        size = row.get('host_h2d_bytes', 0.)
        ms = row.get('transfer_ms', 0.)
        if size > 0 and ms > 0 and row.get('disk_h2d_bytes', 0.) == 0:
            # Sum of stream intervals: conservative when both streams overlap.
            row['pinned_gb_s'] = size / ms / 1e6
            row['at_least_30_gb_s'] = row['pinned_gb_s'] >= 30
    kinds = {}
    for row in layers:
        group = kinds.setdefault(row['kind'], {})
        for key, value in row.items():
            if key.endswith(('_ms', '_bytes')):
                group[key] = group.get(key, 0.) + value
    return dict(layers=layers, transfers=transfers, lookahead=lookahead, workspace=workspace, by_kind=kinds,
                notes=['CUDA intervals include host submission gaps and pending copies.',
                       'transfer_ms overlaps attention/MoE; never add intervals to wall time.',
                       'clock=upload sums H2D intervals on two copy streams; concurrent intervals must not be added to wall.',
                       'pinned_gb_s excludes disk staging and is a conservative rate when stream intervals overlap.',
                       'scan_attention_ms is the chunked parallel KDA scan (legacy: sequential recurrence), or sparse KV union/projection/attention.',
                       'Profile disables graphs and is separate from throughput runs.'])


def run(args, ids, prefix, out, mode='prefill', profile=False, legacy=False, policy='previous'):
    command = [str(args.binary), 'run', 'glm53', '--hyt-dir', str(args.hyt),
               '--layers', '45', '--context', str(prefix+args.tail),
               '--eval-ids', str(ids), '--eval-prefix', str(prefix), '--eval-mode', mode,
               '--prefill-chunk', str(args.chunk), '--prefill-experts', policy,
               '--maintenance-interval', '0', '--out', str(out),
               '--warmup', str(0 if mode == 'step' else args.warmups)]
    if args.host_gb is not None:
        command += ['--host-gb', str(args.host_gb)]
    if args.frequencies:
        command += ['--frequencies', str(args.frequencies)]
    if profile:
        command += ['--profile']
    if legacy:
        command += ['--prefill-legacy']
    with out.with_suffix('.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    meta = json.loads((out/'eval.json').read_text())
    seconds = meta['prefix_seconds']
    if not isinstance(seconds, (float, int)) or not 0 < seconds < float('inf'):
        raise ValueError('invalid prefill timing')
    report = dict(command=command, seconds=seconds, tok_s=prefix/seconds,
                  prefill_chunk=meta['prefill_chunk'], log=str(out.with_suffix('.log')))
    if profile:
        report['breakdown'] = breakdown(out.with_suffix('.log').read_text())
    return report


def measure(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(target_tok_s=1000, warmups=args.warmups, repeats=args.repeats,
                  input='fixed GLM evaluation IDs, repeated for length stress', lengths={})
    if not torch.cuda.is_available():
        report['skip'] = 'CUDA unavailable; no performance measurements'
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    for length in args.lengths:
        ids = args.out/f'ids-{length}.i32.bin'
        raw = input_ids(length+args.tail)
        ids.write_bytes(raw)
        reference = args.out/f'step-{length}'
        baseline = run(args, ids, length, reference, mode='step')
        quality = args.out/f'quality-{length}'
        run(args, ids, length, quality)
        gate = compare(reference, quality, mode='prefill')
        record = dict(input_sha256=hashlib.sha256(raw).hexdigest(), quality=gate,
                      step_reference=baseline)
        report['lengths'][str(length)] = record
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if not gate['quality_gate']:
            raise ValueError(f'prefill quality gate failed at {length}; timing withheld')
        variants = [('legacy', True, 'previous')]+[('batch' if policy == 'previous' else 'all', False, policy)
                      for policy in args.expert_policies]
        for policy in args.expert_policies:
            if policy != 'previous':
                quality_out = args.out/f'quality-{policy}-{length}'
                run(args, ids, length, quality_out, policy=policy)
                checked = compare(reference, quality_out, mode='prefill')
                record[f'quality_{policy}'] = checked
                if not checked['quality_gate']:
                    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                    raise ValueError(f'{policy} quality gate failed; timing withheld')
        for label, legacy, policy in variants:
            runs = [run(args, ids, length, args.out/f'{label}-{length}-{i}', legacy=legacy, policy=policy)
                    for i in range(args.repeats)]
            # Keep quality assertions on timing runs as well (including auto chunks).
            for i in range(args.repeats):
                checked = compare(reference, args.out/f'{label}-{length}-{i}', mode='prefill')
                if not checked['quality_gate']:
                    raise ValueError(f'{label} timing quality gate failed at {length}')
            profile = run(args, ids, length, args.out/f'{label}-{length}-profile', profile=True, legacy=legacy, policy=policy)
            checked = compare(reference, args.out/f'{label}-{length}-profile', mode='prefill')
            if not checked['quality_gate']:
                raise ValueError('profile output differs from tokenwise reference')
            record[label] = dict(runs=runs, median_tok_s=statistics.median(r['tok_s'] for r in runs),
                                 median_seconds=statistics.median(r['seconds'] for r in runs), profile=profile)
            details = profile['breakdown']
            expert_bytes = sum(r.get('h2d_bytes', 0) for r in details['transfers'] + details['lookahead'])
            restore_bytes = sum(r.get('h2d_bytes', 0) for r in details['workspace'])
            record[label]['expert_h2d_bytes'] = expert_bytes
            record[label]['restore_h2d_bytes'] = restore_bytes
            record[label]['transfer_seconds_at_34_gb_s'] = (expert_bytes + restore_bytes) / 34e9
        for label, legacy, policy in variants[1:]:
            record[label]['speedup'] = record['legacy']['median_seconds']/record[label]['median_seconds']
            record[label]['target_met'] = record[label]['median_tok_s'] >= 1000
        if 'batch' in record:
            record['speedup'] = record['batch']['speedup']
            record['target_met'] = record['batch']['target_met']
        if 'batch' in record and 'all' in record:
            record['all_vs_previous_speedup'] = record['batch']['median_seconds']/record['all']['median_seconds']
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=Path('target/release/hayate'))
    p.add_argument('--hyt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--host-gb', type=int)
    p.add_argument('--frequencies', type=Path)
    p.add_argument('--lengths', nargs='+', type=int, default=[2048])
    p.add_argument('--tail', type=int, default=4)
    p.add_argument('--chunk', type=int, default=0)
    p.add_argument('--expert-policies', nargs='+', choices=['previous', 'all'], default=['previous', 'all'])
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if (any(n < 1 or n + args.tail > 131072 for n in args.lengths) or args.tail < 2
            or args.warmups < 0 or args.repeats < 1 or not 0 <= args.chunk <= 8192
            or len(set(args.expert_policies)) != len(args.expert_policies)):
        p.error('invalid length/tail/warmup/repeats/chunk')
    print(json.dumps(measure(args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
