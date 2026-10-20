"""Measure KLD improvement when individual routed-expert layers return to FP8."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .glm53_eval import compare, load_meta


def make_view(fp8, candidate, out, layers, restored):
    fp8, candidate, out = Path(fp8).resolve(), Path(candidate).resolve(), Path(out)
    if not 5 <= layers <= 45 or not restored or len(set(restored)) != len(restored) or any(i < 3 or i >= layers for i in restored):
        raise ValueError('restore distinct routed layers in 3..layers-1')
    targets = [('globals.hyt', candidate / 'globals.hyt')]
    targets += [(f'layer{i}', (fp8 if i in restored else candidate) / f'layer{i}') for i in range(layers)]
    if any(not (p / 'weights.hyt').is_file() if name != 'globals.hyt' else not p.is_file() for name, p in targets):
        raise ValueError('missing HYT globals/layer file')
    out.mkdir(parents=True, exist_ok=False)
    for name, path in targets:
        (out / name).symlink_to(path, target_is_directory=name != 'globals.hyt')
    return sum((fp8 / f'layer{i}/weights.hyt').stat().st_size -
               (candidate / f'layer{i}/weights.hyt').stat().st_size for i in restored)


def measure(args):
    meta, ids, _, _, _ = load_meta(args.reference)
    baseline = compare(args.reference, args.actual)
    # Check alignment with the baseline before spending hours on GPU runs.
    if args.layers != meta['layers'] or min(args.prefill_chunk, len(ids)//4) != meta['prefill_chunk']:
        raise ValueError('--layers/--prefill-chunk differs from the reference')
    restored = args.restore_layers if args.restore_layers is not None else list(range(3, args.layers))
    if len(set(restored)) != len(restored) or not restored or any(i < 3 or i >= args.layers for i in restored):
        raise ValueError('invalid restore layers')
    args.out.mkdir(parents=True, exist_ok=False)
    ids_file = args.out / 'input_ids.i32.bin'
    ids_file.write_bytes(ids)
    results = []
    for selected in ([restored] if args.combine else [[i] for i in restored]):
        label = '-'.join(map(str, selected))
        view, evaluation = args.out / f'hyt-{label}', args.out / f'eval-{label}'
        extra = make_view(args.fp8_hyt, args.candidate_hyt, view, args.layers, selected)
        command = [str(args.hayate.resolve()), 'run', 'glm53', '--hyt-dir', str(view.resolve()),
                   '--layers', str(args.layers), '--context', str(len(ids)//4),
                   '--eval-ids', str(ids_file.resolve()), '--eval-prefix', str(meta['prefix']),
                   '--out', str(evaluation.resolve()), '--allow-disk', '--host-gb', str(args.host_gb),
                   '--prefill-chunk', str(args.prefill_chunk)]
        start = time.perf_counter()
        with (args.out / f'run-{label}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = compare(args.reference, evaluation)
        report.update(restored_layers=selected, additional_weight_bytes=extra,
                      kld_improvement=baseline['mean_kld_fp8_to_nvfp4'] - report['mean_kld_fp8_to_nvfp4'],
                      top1_improvement=report['top1_rate'] - baseline['top1_rate'],
                      elapsed_s=time.perf_counter() - start, command=command)
        results.append(report)
        results.sort(key=lambda r: r['kld_improvement'], reverse=True)
        (args.out / 'sensitivity.json').write_text(json.dumps(
            {'baseline': baseline, 'ranked': results}, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path, help='FP8 evaluation directory')
    p.add_argument('--actual', required=True, type=Path, help='NVFP4 evaluation directory')
    p.add_argument('--fp8-hyt', required=True, type=Path)
    p.add_argument('--candidate-hyt', required=True, type=Path)
    p.add_argument('--hayate', type=Path, default=Path('target/release/hayate'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--layers', type=int, default=45)
    p.add_argument('--restore-layers', type=int, nargs='+', help='default: each MoE layer 3..44')
    p.add_argument('--combine', action='store_true', help='restore the listed layers together and recheck the gate')
    p.add_argument('--host-gb', type=int, default=64)
    p.add_argument('--prefill-chunk', type=int, choices=[1, 128], default=128)
    args = p.parse_args()
    measure(args)


if __name__ == '__main__':
    main()
