"""Compare sync/doorbell decode at identical placement; require bit-identical logits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s,}]+)', line))


def parse_log(log):
    def one(prefix):
        rows = [fields(row) for row in log.splitlines() if row.startswith(prefix)]
        if len(rows) != 1:
            raise ValueError(f'expected one {prefix}')
        return rows[0]
    timing = one('load_s=')
    sync = one('glm_decode_sync ')
    placement = one('glm_mtp_cache ')
    result = {k: float(timing[k]) for k in ('decode_tok_s', 'ITL_p50_ms', 'ITL_p95_ms')}
    result.update({k: int(sync[k]) for k in ('steps', 'host_syncs')})
    result['host_syncs_per_step'] = float(sync['host_syncs_per_step'])
    result['main_placement_hash'] = int(placement['main_placement_hash'])
    result['main_expert_GPU_bytes'] = int(placement['main_expert_GPU_bytes'])
    if not result['steps'] or any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('invalid decode measurement')
    if result['host_syncs'] / result['steps'] != result['host_syncs_per_step']:
        if abs(result['host_syncs'] / result['steps'] - result['host_syncs_per_step']) > 1e-6:
            raise ValueError('inconsistent sync counter')
    return result


def exact_outputs(first, second, tokens):
    first, second = Path(first), Path(second)
    if json.loads(first.with_suffix('.ids.json').read_text()) != json.loads(second.with_suffix('.ids.json').read_text()):
        raise ValueError('generated token IDs differ')
    digest = hashlib.sha256()
    for row in range(tokens):
        a = Path(f'{first}.step{row}.bin').read_bytes()
        b = Path(f'{second}.step{row}.bin').read_bytes()
        if not a or a != b:
            raise ValueError(f'logits are not bit identical at row {row}')
        digest.update(a)
    return digest.hexdigest()


def summarize(runs):
    result = {}
    for mode, rows in runs.items():
        result[mode] = {key: statistics.median(row[key] for row in rows)
                        for key in ('decode_tok_s', 'ITL_p50_ms', 'ITL_p95_ms', 'host_syncs_per_step')}
    result['speedup'] = result['async']['decode_tok_s'] / result['sync']['decode_tok_s']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--hyt-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frequencies', type=Path)
    parser.add_argument('--host-gb', type=int)
    parser.add_argument('--prompt', default='Write a Rust function that sorts a list of integers and explain its complexity.')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--n', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--reserve-mtp', action='store_true', help='reserve MTP even for an n=0-only run')
    parser.add_argument('--mtp-experts', choices=('gpu', 'cpu'), default='gpu')
    args = parser.parse_args()
    if args.tokens < 16 or args.warmup < 1 or args.repeats < 1 or any(n not in range(4) for n in args.n):
        parser.error('tokens >=16, warmup/repeats >=1, n in 0..3')
    args.out.mkdir(parents=True, exist_ok=True)
    common = [str(args.binary.resolve()), 'run', 'glm53', '--hyt-dir', str(args.hyt_dir),
              '--model-dir', str(args.model_dir), '--prompt', args.prompt, '--raw',
              '--ignore-eos', '--max-tokens', str(args.tokens), '--context', str(args.context),
              '--warmup', str(args.warmup), '--maintenance-interval', '0', '--pcie-frac', '0',
              '--mtp-experts', args.mtp_experts]
    if args.frequencies is not None:
        common += ['--frequencies', str(args.frequencies)]
    if args.host_gb is not None:
        common += ['--host-gb', str(args.host_gb)]
    report = {'complete': False, 'command_base': common, 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'prompt': args.prompt, 'tokens': args.tokens, 'context': args.context, 'mtp_experts': args.mtp_experts, 'warmup': args.warmup, 'repeats': args.repeats, 'maintenance_interval': 0,
              'timing_includes_logit_dump': False, 'n': {}}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    placement_keys = ('main_placement_hash', 'main_expert_GPU_bytes')
    expected_placement = None
    for n in args.n:
        spec = ['--spec-draft-n', str(n), '--spec-no-adaptive']
        # Keep n=0 at the same main expert capacity as speculative configurations.
        if n == 0 and (args.reserve_mtp or any(args.n) or args.mtp_experts == 'cpu'):
            spec += ['--spec-reserve-mtp']
        quality = {}
        for mode in ('sync', 'async'):
            prefix = args.out / f'n{n}-{mode}-quality'
            command = common + spec + ['--decode-host', mode, '--dump-logits', str(prefix)]
            run = subprocess.run(command, text=True, capture_output=True, check=False)
            prefix.with_suffix('.log').write_text(run.stderr)
            run.check_returncode()
            quality[mode] = parse_log(run.stderr)
        placement = {key: quality['sync'][key] for key in placement_keys}
        if any(placement[key] != quality['async'][key] for key in placement_keys):
            raise ValueError('quality runs used different main expert placement')
        if expected_placement is not None and placement != expected_placement:
            raise ValueError('main expert placement changed between draft lengths')
        expected_placement = placement
        digest = exact_outputs(args.out / f'n{n}-sync-quality', args.out / f'n{n}-async-quality', args.tokens)
        runs = {'sync': [], 'async': []}
        # Alternate order to reduce systematic cold/thermal ordering bias.
        for repeat in range(args.repeats):
            for mode in (('sync', 'async') if repeat % 2 == 0 else ('async', 'sync')):
                command = common + spec + ['--decode-host', mode]
                run = subprocess.run(command, text=True, capture_output=True, check=False)
                (args.out / f'n{n}-{mode}-{repeat}.log').write_text(run.stderr)
                run.check_returncode()
                measured = parse_log(run.stderr)
                if any(measured[key] != placement[key] for key in placement_keys):
                    raise ValueError('placement changed between quality and timing')
                if mode == 'async' and measured['host_syncs'] != measured['steps']:
                    raise ValueError('async decode did not perform exactly one host sync per step')
                runs[mode].append(measured)
        report['n'][n] = {'placement': placement, 'bit_identical_logits_sha256': digest, 'runs': runs, 'summary': summarize(runs)}
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
