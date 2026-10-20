"""Aggregate eager CUDA-event GLM decode breakdown by layer kind and token."""
import argparse
import json
import math
from pathlib import Path

FIELDS = ('mhc_ms', 'attention_ms', 'dense_ffn_ms', 'routing_ms', 'transfer_ms',
          'gate_up_ms', 'down_ms', 'reduce_ms', 'cpu_tail_wait_ms', 'cpu_merge_ms', 'lm_head_ms', 'greedy_ms',
          'maintenance_ms', 'maintenance_worker_ms', 'host_h2d', 'disk_h2d',
          'h2d_bytes', 'disk_bytes', 'read_wait_ms', 'reader_ms', 'reader_io_ms',
          'reader_bytes', 'reader_pack_ms', 'reader_upload_ms', 'predicted', 'prefetch_hits',
          'maintenance_h2d', 'maintenance_d2h', 'maintenance_disk', 'maintenance_bytes',
          'maintenance_wait_ms', 'cpu_experts', 'cpu_bytes', 'cpu_ms',
          'cpu_worker_ms', 'cpu_dispatch_ms', 'd2h_count', 'd2h_sync_count', 'd2h_wait_ms', 'token_wall_ms')


def summarize(text, skip_tokens=0):
    rows, seen = [], set()
    for line in text.splitlines():
        if not line.startswith('glm_profile '):
            continue
        row = dict(item.split('=', 1) for item in line.split()[1:])
        position, layer, kind = int(row.pop('position')), row.pop('layer'), row.pop('kind')
        key = position, layer
        if key in seen:
            raise ValueError('duplicate position/layer; use one measured run per log')
        seen.add(key)
        metrics = {name: float(row.get(name, 0)) for name in FIELDS}
        if any(not math.isfinite(v) or v < 0 for v in metrics.values()):
            raise ValueError('nonfinite/negative profile metric')
        rows.append((position, layer, kind, metrics))
    positions = sorted({r[0] for r in rows})[skip_tokens:]
    if not positions or skip_tokens < 0:
        raise ValueError('no measured decode tokens')
    expected = {r[1] for r in rows if r[0] == positions[0]}
    if not {'head', 'maintenance'}.issubset(expected):
        raise ValueError('incomplete token profile')
    totals, kinds = dict.fromkeys(FIELDS, 0.), {}
    for position in positions:
        token_rows = [r for r in rows if r[0] == position]
        if {r[1] for r in token_rows} != expected:
            raise ValueError('incomplete token profile')
        for _, _, kind, metrics in token_rows:
            group = kinds.setdefault(kind, dict.fromkeys(FIELDS, 0.))
            for name, value in metrics.items():
                group[name] += value / len(positions)
                totals[name] += value / len(positions)
    bandwidth = dict(
        demand_payload_GBps=(totals['reader_bytes'] / totals['read_wait_ms'] / 1e6
                             if totals['reader_bytes'] and totals['read_wait_ms'] else None),
        io_worker_GBps=(totals['reader_bytes'] / totals['reader_io_ms'] / 1e6
                       if totals['reader_bytes'] and totals['reader_io_ms'] else None))
    return dict(tokens=len(positions), ms_per_token=totals, by_kind=kinds,
                disk_bandwidth=bandwidth,
                cpu_bandwidth_GBps=(totals['cpu_bytes'] / totals['cpu_ms'] / 1e6
                                    if totals['cpu_ms'] else None),
                notes=['CUDA intervals include host submission gaps and pending copies.',
                       'D2H wait is host wall time and overlaps CUDA intervals; do not add it.',
                       'Shared expert is included in gate_up/down/reduce.',
                       'Reference gate_up includes sequential gate/up/down/reduce/shared.',
                       '--profile disables graphs; use unprofiled runs for throughput.',
                       'Transfer/cache counts and bytes are per token, not milliseconds.',
                       'maintenance_worker_ms is background exchange time; maintenance_wait_ms is host blocking.',
                       'cpu_ms includes CPU FFN and waiting; overlaps GPU work, do not add to token wall.',
                       'cpu_worker_ms sums slowest worker wall per GEMV; includes scheduling delays.',
                       'cpu_dispatch_ms includes publication/wakeup and completion wait; overlaps worker_ms.',
                       'reader_ms sums parallel read/upload workers and overlaps compute.',
                       'reader_io_ms/reader_pack_ms/reader_upload_ms split pread, packing/validation and H2D wait.',
                       'demand_payload_GBps uses demand reader bytes / demand wait; not physical NVMe bandwidth.',
                       'io_worker_GBps uses summed reader wall time; parallel workers overlap, so it is not aggregate bandwidth.',
                       'Speculative copies/reads are included; prefetch_hits counts predictions used by exact routing.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--skip-tokens', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text(), args.skip_tokens), indent=2))


if __name__ == '__main__':
    main()
