"""Compare per-step F32 logits and greedy IDs from glm53_reference and Hayate."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, actual, tolerance=0.01, fp8_baseline=None, greedy_only=False):
    if greedy_only and fp8_baseline is not None:
        raise ValueError('greedy-only cannot replace the FP8-relative quality gate')
    reference, actual = Path(reference), Path(actual)
    ref_ids = json.loads((reference/'ids.json').read_text())
    got_ids = json.loads(actual.with_suffix('.ids.json').read_text())
    if ref_ids['input'] != got_ids['input']:
        raise ValueError('tokenized prompts differ')
    if len(got_ids['generated']) != len(ref_ids['generated']):
        raise ValueError('generated token counts differ')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if fp8_baseline is not None:
        fp8_baseline = Path(fp8_baseline)
        baseline_ids = json.loads(fp8_baseline.with_suffix('.ids.json').read_text())
        if baseline_ids != got_ids or baseline_ids['generated'] != ref_ids['generated']:
            raise ValueError('FP8/NVFP4/reference prompts or greedy IDs differ')
    report = []
    for step, token in enumerate(ref_ids['generated']):
        r = np.fromfile(reference/f'step{step}.bin', dtype='<f4').astype(np.float64)
        a = np.fromfile(f'{actual}.step{step}.bin', dtype='<f4').astype(np.float64)
        if r.size == 0 or r.shape != a.shape or not np.isfinite(r).all() or not np.isfinite(a).all():
            raise ValueError('invalid logits')
        energy = np.linalg.norm(r)
        error = float(np.linalg.norm(a-r)/energy) if energy else (0.0 if np.array_equal(a,r) else float('inf'))
        greedy = int(a.argmax()) == int(r.argmax()) == token == got_ids['generated'][step]
        limit = tolerance
        entry = {'step':step, 'relative_l2':error, 'greedy_match':greedy}
        if fp8_baseline is not None:
            baseline = np.fromfile(f'{fp8_baseline}.step{step}.bin', dtype='<f4').astype(np.float64)
            if baseline.shape != r.shape or not np.isfinite(baseline).all() or int(baseline.argmax()) != token:
                raise ValueError('invalid FP8 baseline logits')
            baseline_error = float(np.linalg.norm(baseline-r)/energy) if energy else (0.0 if np.array_equal(baseline,r) else float('inf'))
            if not np.isfinite(baseline_error):
                raise ValueError('FP8 baseline has infinite relative error')
            limit = 2 * baseline_error
            entry.update(fp8_relative_l2=baseline_error, max_relative_l2=limit)
        entry['passed'] = greedy and np.isfinite(error) and (greedy_only or (error <= limit if fp8_baseline is not None else error < limit))
        if greedy_only:
            entry['gate'] = 'greedy_only'
        report.append(entry)
    print(json.dumps(report, indent=2))
    if any(not r['passed'] for r in report):
        raise ValueError('GLM logits/greedy quality gate failed')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=0.01)
    p.add_argument('--fp8-baseline', type=Path, help='NVFP4 gate: per-step L2 <= 2x this FP8 dump')
    p.add_argument('--greedy-only', action='store_true', help='45-layer greedy check; still reports logits L2')
    a = p.parse_args()
    compare(a.reference, a.actual, a.tolerance, a.fp8_baseline, a.greedy_only)


if __name__ == '__main__':
    main()
