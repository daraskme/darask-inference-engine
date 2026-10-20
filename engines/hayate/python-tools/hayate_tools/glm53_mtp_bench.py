"""Diagnose placement drift and gate same-prefix quality before reporting MTP speed."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import struct
import torch
from hayate_tools.glm53_eval import metrics

# Odds of the two leading tokens differ by at most 10%. Fixed before GPU runs.
TIE_GAP = math.log(1.1)
MAX_KLD = .01
MIN_TOP1 = .99

PROMPTS = {
    'code': 'Write a Rust function that sorts a list of integers and explain its complexity.',
    'json': 'Return a JSON array of ten objects with id, name, and an array of three numbers.',
    'prose': 'Describe a quiet morning in a mountain village in detail.',
}


def parse_log(log):
    summary = re.findall(r'^glm_spec_summary (.+)$', log, re.MULTILINE)
    timing = re.findall(r'^load_s=.+$', log, re.MULTILINE)
    if len(summary) != 1 or len(timing) != 1:
        raise ValueError('missing/duplicate GLM measurement')
    def fields(text):
        return dict(re.findall(r'(\w+)=([^\s,}]+)', text))
    info = fields(summary[0])
    measured = fields(timing[0])
    result = {k: float(measured[k]) for k in ('load_s', 'TTFT_s', 'prefill_s', 'decode_tok_s', 'ITL_p50_ms', 'ITL_p95_ms')}
    result.update({k: int(info[k]) for k in ('n', 'steps', 'proposed', 'accepted', 'emitted', 'MTP_added_VRAM_bytes', 'MTP_expert_GPU_bytes', 'peak_VRAM_bytes')})
    result['acceptance'] = float(info['acceptance'])
    result['disabled'] = info['disabled'] == 'true'
    cache = re.findall(r'^glm_mtp_cache (.+)$', log, re.MULTILINE)
    if len(cache) != 1:
        raise ValueError('missing/duplicate main expert cache measurement')
    result.update({k: int(v) for k, v in fields(cache[0]).items()})
    if result['main_decode_hits'] > result['main_decode_routes']:
        raise ValueError('invalid main expert hit count')
    result['step_ms'] = float(info['seconds']) * 1000 / max(1, result['steps'])
    if (not all(math.isfinite(v) and v >= 0 for v in result.values())
            or result['decode_tok_s'] <= 0 or not 0 <= result['acceptance'] <= 1
            or result['accepted'] > result['proposed']):
        raise ValueError('invalid GLM measurement')
    return result


def select_n(runs):
    """Only measured improvements over n=0 are eligible."""
    medians = {n: statistics.median(row['decode_tok_s'] for row in rows) for n, rows in runs.items()}
    baseline = medians[0]
    return max((n for n in medians if n == 0 or medians[n] > baseline), key=lambda n: (medians[n], -n))


def summarize(runs):
    baseline = statistics.median(row['decode_tok_s'] for row in runs[0])
    result = {}
    for n, rows in runs.items():
        rate = statistics.median(row['decode_tok_s'] for row in rows)
        proposed = sum(row['proposed'] for row in rows)
        result[n] = {
            'unprofiled_decode_tok_s_median': rate,
            'speedup_over_n0': rate / baseline,
            'acceptance': sum(row['accepted'] for row in rows) / proposed if proposed else None,
            'main_gpu_hit_rate': sum(row['main_decode_hits'] for row in rows) / max(1, sum(row['main_decode_routes'] for row in rows)),
            'main_expert_GPU_bytes': rows[0]['main_expert_GPU_bytes'],
            'MTP_added_VRAM_bytes': rows[0]['MTP_added_VRAM_bytes'],
            'MTP_expert_GPU_bytes': rows[0]['MTP_expert_GPU_bytes'],
            'mean_step_ms_median': statistics.median(row['step_ms'] for row in rows) if n else None,
        }
    return result


def parse_profile(log):
    """Keep overlapping CUDA/host intervals separate; never sum them as wall time."""
    steps, layers = {}, {}
    for kind, text in re.findall(r'^glm_spec(_layer)? (.+)$', log, re.MULTILINE):
        fields = dict(re.findall(r'(\w+)=([^\s,}]+)', text))
        values = {key: float(value) if key.endswith('_ms') else int(value)
                  for key, value in fields.items()}
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('invalid GLM speculative profile')
        position = values['position']
        if kind:
            layer = values['layer']
            if layer in layers.setdefault(position, {}):
                raise ValueError('duplicate GLM verifier layer')
            layers[position][layer] = values
        else:
            if position in steps or not 1 <= values['n'] <= 3 or values['accepted'] > values['n']:
                raise ValueError('invalid/duplicate GLM speculative step')
            steps[position] = values
    if not steps or set(steps) != set(layers):
        raise ValueError('missing GLM speculative step/layer profile')
    result = []
    for position, step in steps.items():
        if set(layers[position]) != set(range(45)):
            raise ValueError('incomplete GLM verifier layer profile')
        row = dict(step)
        row['verifier_d2h_syncs'] = sum(v['d2h_syncs'] for v in layers[position].values())
        row['layers'] = list(layers[position].values())
        result.append(row)
    return result


def run_command(command, prefix):
    prefix.with_suffix('.command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2) + '\n')
    with prefix.with_suffix('.out').open('w') as out, prefix.with_suffix('.log').open('w') as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    return prefix.with_suffix('.log').read_text()


def first_divergence(reference_ids, actual_ids, reference_logits):
    if len(reference_ids) != len(actual_ids) or len(reference_logits) != len(reference_ids):
        raise ValueError('divergence length mismatch')
    for position, (expected, actual) in enumerate(zip(reference_ids, actual_ids)):
        if expected != actual:
            logits = reference_logits[position].double()
            # Stable token-ID ordering at ties, like GPU greedy.
            top = torch.argsort(logits, descending=True, stable=True)[:2]
            gap = (logits[top[0]] - logits[top[1]]).item()
            regret = (logits[top[0]] - logits[actual]).item()
            if top[0].item() != expected:
                raise ValueError('reference greedy ID does not match logits')
            return dict(position=position, token_number=position+1,
                        reference_top1=expected, reference_top2=top[1].item(), actual=actual,
                        top1_logit=logits[top[0]].item(), top2_logit=logits[top[1]].item(),
                        top1_top2_gap=gap, actual_logit_regret=regret,
                        near_tie=gap <= TIE_GAP and regret <= TIE_GAP)
    return None


def quality(reference, actual):
    """Rows must predict at identical prefixes, including after the first split."""
    kld, equal, _, _ = metrics(reference, actual)
    top = reference.double().topk(2, dim=-1).values
    gaps = top[:, 0] - top[:, 1]
    regret = reference.double().max(-1).values - reference.double().gather(1, actual.argmax(-1)[:, None]).squeeze(1)
    mismatches = ~equal
    near = ((gaps <= TIE_GAP) & (regret <= TIE_GAP) | equal).all().item()
    rate = equal.double().mean().item()
    maximum = kld.max().item()
    return dict(positions=len(kld), top1_matches=equal.sum().item(), top1_rate=rate,
                mean_kld=kld.mean().item(), max_kld=maximum,
                mismatch_max_top1_top2_gap=gaps[mismatches].max().item() if mismatches.any() else None,
                mismatch_max_regret=regret[mismatches].max().item() if mismatches.any() else None,
                all_mismatches_near_ties=near,
                quality_gate=rate >= MIN_TOP1 and maximum <= MAX_KLD and near)


def trace_logits(prefix, tokens):
    rows = []
    for step in range(tokens):
        raw = bytearray(Path(f'{prefix}.step{step}.bin').read_bytes())
        if not raw or len(raw) % 4:
            raise ValueError('invalid diagnostic logits')
        rows.append(torch.frombuffer(raw, dtype=torch.float32).clone())
    result = torch.stack(rows)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite diagnostic logits')
    return result


def eval_logits(folder):
    meta = json.loads((folder / 'eval.json').read_text())
    rows = []
    for name in ('prefix_logits.bf16.bin', 'eval_logits.bf16.bin'):
        raw = bytearray((folder / name).read_bytes())
        rows.append(torch.frombuffer(raw, dtype=torch.bfloat16).float().reshape(-1, meta['vocab']))
    result = torch.cat(rows)
    if rows[0].shape[0] != 1 or result.shape[0] != meta['rows'] + 1:
        raise ValueError('invalid teacher-forced logits')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('target/release/hayate'))
    parser.add_argument('--hyt', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-Uncensored-FP8'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frequencies', type=Path)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--host-gb', type=int, help='host budget in GB (default: automatic within the safe limit)')
    parser.add_argument('--pcie-frac', type=int, choices=(0, 1), default=0)
    parser.add_argument('--maintenance-interval', type=int, default=0,
                        help='0 fixes placement for causal diagnosis; repeat with 4 for operational maintenance')
    parser.add_argument('--mtp-experts', nargs='+', choices=('gpu', 'cpu'), default=['gpu', 'cpu'])
    args = parser.parse_args()
    if (args.tokens < 16 or args.warmups < 0 or args.repeats < 1 or args.maintenance_interval < 0
            or len(set(args.mtp_experts)) != len(args.mtp_experts)
            or args.pcie_frac != 0 and 'cpu' in args.mtp_experts):
        parser.error('tokens >=16, warmups >=0, repeats >=1; CPU MTP requires pcie-frac=0')
    args.out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        (args.out / 'report.json').write_text(json.dumps({'skipped': 'CUDA unavailable; no performance measurements'}) + '\n')
        return
    base = [str(args.binary.resolve()), 'run', 'glm53', '--layers', '45', '--context', '4096',
            '--hyt-dir', str(args.hyt), '--model-dir', str(args.model_dir),
            '--pcie-frac', str(args.pcie_frac), '--prefill-chunk', '128', '--prefill-legacy',
            '--maintenance-interval', str(args.maintenance_interval)]
    if args.frequencies:
        base += ['--frequencies', str(args.frequencies)]
    if args.host_gb is not None:
        base += ['--host-gb', str(args.host_gb)]
    report = {'tokens': args.tokens, 'warmups': args.warmups, 'repeats': args.repeats,
              'maintenance_interval': args.maintenance_interval,
              'quality_thresholds': dict(min_top1=MIN_TOP1, max_row_kld=MAX_KLD, tie_gap=TIE_GAP,
                                         rationale='top-two odds <=1.1, selected-token regret <=ln(1.1); KLD <=0.01 per row as in prefill; top1 >=99%'),
              'cache_counts': 'Capacity, MTP expert bytes and placement hash are sampled after warmup/prefill. Main decode counts include rejected verifier rows. cache_cost compares n=0 only.',
              'profile_intervals': 'Overlapping CUDA/host intervals; verifier_d2h_syncs excludes draft/repair/final waits.',
              'prompts': {}}
    failed = False

    def save():
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def generation(command, prefix, extra=()):
        ids = prefix.with_suffix('.ids.json')
        log = run_command(command + ['--max-tokens', str(args.tokens), '--warmup', str(args.warmups),
                          '--spec-no-adaptive', '--ignore-eos', '--output-ids', str(ids)] + list(extra), prefix)
        output = json.loads(ids.read_text())
        if len(output['generated']) != args.tokens:
            raise ValueError('benchmark generated the wrong number of tokens')
        return log, output

    try:
        for name, prompt in PROMPTS.items():
            command = base + ['--prompt', prompt]
            record = report['prompts'][name] = {'configurations': {}}
            traces, outputs = {}, {}
            # The in-process bit-exact gate stays mandatory for both placements.
            for policy in args.mtp_experts:
                run_command(command + ['--spec-draft-n', '3', '--mtp-experts', policy,
                            '--spec-validate', str(args.tokens)], args.out / f'{name}-{policy}-correctness')
            variants = [('full-n0', 0, [])]
            for policy in args.mtp_experts:
                variants += [(f'{policy}-n{n}', n, ['--mtp-experts', policy] + (['--spec-reserve-mtp'] if n == 0 else []))
                             for n in range(4)]
            for label, n, flags in variants:
                cfg = record['configurations'][label] = {'runs': []}
                cmd = command + flags + ['--spec-draft-n', str(n)]
                expected = None
                for repeat in range(args.repeats):
                    log, ids = generation(cmd, args.out / f'{name}-{label}-r{repeat}')
                    if expected is not None and ids != expected:
                        raise ValueError(f'{name} {label}: repeated generation is nondeterministic')
                    expected = ids
                    cfg['runs'].append(parse_log(log))
                prefix = args.out / f'{name}-{label}-diagnostic'
                dump = args.out / f'{name}-{label}-logits'
                _, ids = generation(cmd, prefix, ['--dump-logits', str(dump)])
                if ids != expected:
                    raise ValueError(f'{name} {label}: diagnostic IDs differ from measured run')
                outputs[label] = ids
                traces[label] = trace_logits(dump, args.tokens)
                if traces[label].argmax(-1).tolist() != ids['generated']:
                    raise ValueError('diagnostic logits do not predict emitted IDs')
                if n:
                    log, ids = generation(cmd, args.out / f'{name}-{label}-profile', ['--profile-spec'])
                    if ids != expected:
                        raise ValueError(f'{name} {label}: profile IDs differ from measured run')
                    cfg['profile'] = parse_profile(log)
                save()

            def compare_to(reference_label, label):
                actual_ids, ref_ids = outputs[label], outputs[reference_label]
                if actual_ids['input'] != ref_ids['input']:
                    raise ValueError('prompt IDs differ')
                divergence = first_divergence(ref_ids['generated'], actual_ids['generated'], traces[reference_label])
                if divergence is None:
                    reference_logits = traces[reference_label]
                else:
                    # Replay the candidate's exact prefixes; free-running logits after a split cannot be compared.
                    prefix = args.out / f'{name}-{label}-against-{reference_label}'
                    input_path = prefix.with_suffix('.i32.bin')
                    tokens = actual_ids['input'] + actual_ids['generated'][:-1]
                    input_path.write_bytes(struct.pack(f'<{len(tokens)}I', *tokens))
                    flags = [] if reference_label == 'full-n0' else ['--spec-reserve-mtp', '--mtp-experts', reference_label.split('-')[0]]
                    run_command(base + flags + ['--eval-ids', str(input_path), '--eval-prefix', str(len(actual_ids['input'])),
                                '--out', str(prefix)], prefix)
                    reference_logits = eval_logits(prefix)
                checked = quality(reference_logits, traces[label])
                if divergence is not None:
                    print(f"{name} {label} vs {reference_label}: token={divergence['token_number']} "
                          f"top1={divergence['reference_top1']} top2={divergence['reference_top2']} "
                          f"gap={divergence['top1_top2_gap']:.9g} regret={divergence['actual_logit_regret']:.9g}", flush=True)
                checked['first_divergence'] = divergence
                checked['exact_tokens'] = divergence is None
                checked['quality_gate'] &= divergence is None or divergence['near_tie']
                return checked

            for policy in args.mtp_experts:
                paired = f'{policy}-n0'
                baseline_bytes = record['configurations'][paired]['runs'][0]['main_expert_GPU_bytes']
                baseline_hash = record['configurations'][paired]['runs'][0]['main_placement_hash']
                for n in range(4):
                    label = f'{policy}-n{n}'
                    cfg = record['configurations'][label]
                    cfg['vs_full_n0'] = compare_to('full-n0', label)
                    cfg['vs_matched_n0'] = compare_to(paired, label)
                    cfg['same_main_cache_capacity'] = all(r['main_expert_GPU_bytes'] == baseline_bytes for r in cfg['runs'])
                    cfg['same_main_placement'] = all(r['main_placement_hash'] == baseline_hash for r in cfg['runs'])
                    cfg['quality_gate'] = (args.maintenance_interval != 0 or cfg['same_main_placement']) and cfg['vs_full_n0']['quality_gate'] and cfg['vs_matched_n0']['quality_gate'] and cfg['same_main_cache_capacity']
                    failed |= not cfg['quality_gate']
                    if cfg['quality_gate']:
                        cfg['summary'] = summarize({0: record['configurations'][paired]['runs'], n: cfg['runs']})[n]
                        full_rate = statistics.median(r['decode_tok_s'] for r in record['configurations']['full-n0']['runs'])
                        cfg['summary']['speedup_over_full_n0'] = cfg['summary']['unprofiled_decode_tok_s_median'] / full_rate
                    save()
                eligible = {n: record['configurations'][f'{policy}-n{n}']['runs'] for n in range(4)
                            if record['configurations'][f'{policy}-n{n}']['quality_gate']}
                record[f'{policy}_selected_n'] = select_n(eligible) if 0 in eligible else None
            full_runs = record['configurations']['full-n0']['runs']
            full_hits = sum(r['main_decode_hits'] for r in full_runs) / max(1, sum(r['main_decode_routes'] for r in full_runs))
            record['cache_cost'] = {}
            for policy in args.mtp_experts:
                paired_runs = record['configurations'][f'{policy}-n0']['runs']
                record['cache_cost'][policy] = dict(
                    main_GPU_bytes_lost=full_runs[0]['main_expert_GPU_bytes'] - paired_runs[0]['main_expert_GPU_bytes'],
                    main_hit_rate_delta=sum(r['main_decode_hits'] for r in paired_runs) / max(1, sum(r['main_decode_routes'] for r in paired_runs)) - full_hits,
                    n0_speed_ratio=statistics.median(r['decode_tok_s'] for r in paired_runs) / statistics.median(r['decode_tok_s'] for r in full_runs))
            record['capacity_hypothesis'] = {
                policy: ('consistent_with_capacity_drift' if args.maintenance_interval == 0
                         and record['configurations'][f'{policy}-n0']['vs_full_n0']['first_divergence'] is not None
                         and all(record['configurations'][f'{policy}-n{n}']['vs_matched_n0']['exact_tokens']
                                 and record['configurations'][f'{policy}-n{n}']['same_main_placement'] for n in range(1, 4))
                         else 'not_established') for policy in args.mtp_experts}
            choices = [(policy, record[f'{policy}_selected_n']) for policy in args.mtp_experts]
            choices = [(policy, n) for policy, n in choices if n and record['configurations'][f'{policy}-n{n}']['summary']['speedup_over_full_n0'] > 1]
            record['measured_recommendation'] = max(choices, key=lambda item: statistics.median(
                r['decode_tok_s'] for r in record['configurations'][f'{item[0]}-n{item[1]}']['runs'])) if choices else None
            save()
        report['quality_gate'] = not failed
        report['default_changed'] = False
        save()
    except Exception as error:
        report['error'] = str(error)
        save()
        raise
    if failed:
        raise SystemExit('MTP quality/capacity gate failed; failed configurations have no speed summary; see report.json')


if __name__ == '__main__':
    main()
