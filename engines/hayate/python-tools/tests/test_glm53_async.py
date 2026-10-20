"""CPU-only checks for strict async benchmark quality and synchronization gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hayate_tools.glm53_async_bench import exact_outputs, main, parse_log, summarize


class AsyncBench(unittest.TestCase):
    def test_sync_counter_and_latency_parser(self):
        log = ('load_s=1 TTFT_s=2 prefill_s=1 decode_tok_s=12 ITL_p50_ms=80 ITL_p95_ms=90\n'
               'glm_decode_sync host_mode=async steps=8 host_syncs=8 host_syncs_per_step=1.000000\n'
               'glm_mtp_cache main_placement_hash=123 main_expert_GPU_bytes=456\n')
        row = parse_log(log)
        self.assertEqual(row['host_syncs_per_step'], 1)
        self.assertEqual(row['ITL_p95_ms'], 90)
        with self.assertRaises(ValueError):
            parse_log(log + log)
        with self.assertRaises(ValueError):
            parse_log(log.replace('host_syncs=8', 'host_syncs=9'))
        summary = summarize({'sync': [dict(row, decode_tok_s=10)], 'async': [row]})
        self.assertEqual(summary['speedup'], 1.2)

    def test_every_logit_byte_and_id_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = (Path(directory) / name for name in ('sync', 'async'))
            for prefix in (first, second):
                prefix.with_suffix('.ids.json').write_text(json.dumps({'input': [1], 'generated': [2, 3]}))
                for row in range(2):
                    Path(f'{prefix}.step{row}.bin').write_bytes(bytes([row, 0, 1, 0]))
            self.assertEqual(len(exact_outputs(first, second, 2)), 64)
            Path(f'{second}.step1.bin').write_bytes(bytes([1, 0, 1, 1]))
            with self.assertRaisesRegex(ValueError, 'row 1'):
                exact_outputs(first, second, 2)
            second.with_suffix('.ids.json').write_text(json.dumps({'input': [1], 'generated': [2, 4]}))
            with self.assertRaisesRegex(ValueError, 'token IDs'):
                exact_outputs(first, second, 2)

    def test_benchmark_checks_capacity_during_timing_and_across_draft_lengths(self):
        import subprocess
        for changed in ('none', 'timing_bytes', 'draft_hash'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root/'hayate'
                binary.write_bytes(b'binary')
                out = root/'report'
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    if '--dump-logits' in command:
                        prefix = Path(command[command.index('--dump-logits')+1])
                        prefix.with_suffix('.ids.json').write_text(json.dumps({'generated': list(range(16))}))
                        for row in range(16):
                            Path(f'{prefix}.step{row}.bin').write_bytes(b'\0\0')
                    size = 457 if changed == 'timing_bytes' and len(calls) == 3 else 456
                    placement_hash = 124 if changed == 'draft_hash' and len(calls) > 4 else 123
                    log = ('load_s=1 decode_tok_s=12 ITL_p50_ms=80 ITL_p95_ms=90\n'
                           'glm_decode_sync steps=8 host_syncs=8 host_syncs_per_step=1.000000\n'
                           f'glm_mtp_cache main_placement_hash={placement_hash} main_expert_GPU_bytes={size}\n')
                    return subprocess.CompletedProcess(command, 0, stderr=log)

                argv = ['bench', '--binary', str(binary), '--hyt-dir', '/tmp/unused-hyt',
                        '--model-dir', '/tmp/unused-source', '--out', str(out),
                        '--tokens', '16', '--warmup', '1', '--repeats', '1', '--n', '0']
                if changed == 'draft_hash':
                    argv += ['1']
                with patch('sys.argv', argv), patch('subprocess.run', side_effect=run), patch('builtins.print'):
                    if changed == 'none':
                        main()
                    else:
                        with self.assertRaisesRegex(ValueError, 'placement changed'):
                            main()
                report = json.loads((out/'report.json').read_text())
                self.assertEqual(report['complete'], changed == 'none')
                if changed == 'none':
                    self.assertEqual(report['n']['0']['placement'],
                                     {'main_placement_hash': 123, 'main_expert_GPU_bytes': 456})
                if changed == 'draft_hash':
                    self.assertTrue(all('--spec-reserve-mtp' in c for c in calls[:4]))
                    self.assertNotIn('1', report['n'])


class AsyncMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ctypes
        import subprocess
        cls.tmp = tempfile.TemporaryDirectory()
        source = Path(__file__).resolve().parents[2] / 'crates/kernels/cu/glm53/async_cpu_test.cpp'
        library = Path(cls.tmp.name) / 'async.so'
        subprocess.run(['c++', '-shared', '-fPIC', '-O3', '-std=c++20', str(source), '-o', str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.glm53_async_sparse.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint]
        cls.lib.glm53_async_groups.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cuda_sort_and_union_helpers_against_sets(self):
        import ctypes
        import random
        rng = random.Random(5313)
        for rows in range(1, 5):
            for size in (1, 127, 128, 129, 271, 2051):
                queries = [rng.sample(range(9001), size) for _ in range(rows)]
                selected = (ctypes.c_int * (rows * 2051))(*[v for q in queries for v in q + [-1] * (2051 - size)])
                ids = (ctypes.c_uint * (rows * 2051))()
                counts = (ctypes.c_uint * 5)()
                sorted_rows = (ctypes.c_uint * (rows * 4096))()
                count = self.lib.glm53_async_sparse(ids, counts, sorted_rows, selected, rows)
                self.assertEqual(list(ids)[:count], sorted(set(v for q in queries for v in q)))
                self.assertEqual(list(counts)[1:1+rows], [size] * rows)
                for row, query in enumerate(queries):
                    self.assertEqual(list(sorted_rows)[row*4096:row*4096+size], sorted(query))

    def test_grouped_routes_preserve_per_expert_token_order(self):
        import ctypes
        import random
        rng = random.Random(5314)
        for rows in range(1, 5):
            for _ in range(20):
                queries = [rng.sample(range(16), 8) for _ in range(rows)]
                flat = [v for query in queries for v in query]
                ids = (ctypes.c_int * len(flat))(*flat)
                descriptors = (ctypes.c_uint * (32 * 5))()
                count = self.lib.glm53_async_groups(descriptors, ids, rows)
                expected = [[i for i, v in enumerate(flat) if v == expert] for expert in sorted(set(flat))]
                actual = [list(descriptors)[g*5+1:g*5+1+descriptors[g*5]] for g in range(count)]
                self.assertEqual(actual, expected)
