import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from hayate_tools.glm53_prefill_bench import breakdown, input_ids, measure


class GLMPrefillBench(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(out=self.root/'bench', binary=Path('/tmp/hayate'),
                                    hyt=Path('/tmp/weights'), host_gb=None, frequencies=None,
                                    lengths=[17], tail=3, chunk=0, warmups=2, repeats=2, expert_policies=['previous', 'all'])

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, command, stdout, **kwargs):
        def value(flag): return command[command.index(flag)+1]
        prefix = int(value('--eval-prefix'))
        ids = Path(value('--eval-ids')).read_bytes()
        count = len(ids)//4
        out = Path(value('--out'))
        out.mkdir()
        step = value('--eval-mode') == 'step'
        legacy = '--prefill-legacy' in command
        meta = dict(rows=count-prefix, vocab=154880, prefix=prefix, layers=45,
                    source_hash='53'*16, prefill_chunk=128 if legacy else 17, dtype='bf16',
                    eval_mode=value('--eval-mode'), prefix_seconds=10 if step else (2 if legacy else 1))
        # Full shape, sparse finite logits: test the real streaming comparator.
        row = torch.zeros(meta['vocab'], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
        (out/'eval.json').write_text(json.dumps(meta))
        (out/'eval_ids.i32.bin').write_bytes(ids)
        (out/'prefix_logits.bf16.bin').write_bytes(row)
        (out/'eval_logits.bf16.bin').write_bytes(row*(count-prefix))
        if '--profile' in command:
            stdout.write(f'glm_prefill_layer position=0 rows={prefix} layer=0 kind=kda attention_ms=2.5 scan_attention_ms=1.0 wall_ms=4.0\n')
            stdout.write('glm_prefill_workspace stage=borrow loan_bytes=256 elapsed_ms=0.2\n')
            stdout.write('glm_prefill_workspace stage=return elapsed_ms=0.3\n')
            stdout.write('glm_prefill_lookahead position=0 layer=3 rows=17 policy=all predicted=288 transfer_ms=0.1\n')
            stdout.write(f'glm_prefill_transfer position=0 rows={prefix} layer=3 transfer_ms=0.5 h2d_bytes=256 host_h2d_bytes=256 disk_h2d_bytes=0 overlap=true clock=upload\n')
        self.assertEqual(value('--prefill-chunk'), '0')
        self.assertEqual(value('--maintenance-interval'), '0')
        self.assertEqual(value('--warmup'), '0' if step else '2')

    def test_no_gpu_has_no_fabricated_timing(self):
        with patch('torch.cuda.is_available', return_value=False), patch('subprocess.run') as run:
            report = measure(self.args)
        run.assert_not_called()
        self.assertEqual(report['lengths'], {})
        self.assertIn('CUDA unavailable', report['skip'])
        self.assertEqual(json.loads((self.args.out/'report.json').read_text()), report)

    def test_gate_precedes_timing_and_profile_is_separate(self):
        torch.set_num_threads(2)
        with patch('torch.cuda.is_available', return_value=True), patch('subprocess.run', side_effect=self.fake_run) as run:
            report = measure(self.args)
        record = report['lengths']['17']
        self.assertEqual(run.call_count, 12)  # step/quality, 2+profile for each variant, all quality
        self.assertTrue(record['quality']['quality_gate'])
        self.assertEqual(record['speedup'], 2.)
        self.assertEqual(record['batch']['median_tok_s'], 17.)
        self.assertFalse(record['target_met'])
        self.assertEqual(record['all_vs_previous_speedup'], 1.)
        self.assertNotIn('--prefill-legacy', record['all']['runs'][0]['command'])
        command = record['all']['runs'][0]['command']
        self.assertEqual(command[command.index('--prefill-experts')+1], 'all')
        self.assertTrue(record['quality_all']['quality_gate'])
        self.assertIn('breakdown', record['batch']['profile'])
        transfer = record['batch']['profile']['breakdown']['transfers'][0]
        self.assertEqual(transfer['pinned_gb_s'], 256/.5/1e6)
        self.assertFalse(transfer['at_least_30_gb_s'])
        self.assertEqual(record['batch']['profile']['breakdown']['workspace'][1]['elapsed_ms'], .3)
        self.assertEqual(record['all']['profile']['breakdown']['lookahead'][0]['transfer_ms'], .1)
        self.assertTrue(all('--profile' not in r['command'] for r in record['batch']['runs']))
        self.assertIn('--prefill-legacy', record['legacy']['runs'][0]['command'])

    def test_broken_prefix_withholds_timing(self):
        def broken(command, **kwargs):
            self.fake_run(command, **kwargs)
            if command[command.index('--eval-mode')+1] == 'prefill':
                out = Path(command[command.index('--out')+1])
                raw = bytearray((out/'prefix_logits.bf16.bin').read_bytes())
                raw[:2] = struct.pack('<H', 0x4280)  # 64 logit changes the distribution
                (out/'prefix_logits.bf16.bin').write_bytes(raw)
        with patch('torch.cuda.is_available', return_value=True), patch('subprocess.run', side_effect=broken) as run:
            with self.assertRaisesRegex(ValueError, 'quality gate failed'):
                measure(self.args)
        self.assertEqual(run.call_count, 2)
        record = json.loads((self.args.out/'report.json').read_text())['lengths']['17']
        self.assertNotIn('batch', record)

    def test_profile_rejects_invalid_values_and_ids_are_reproducible(self):
        self.assertEqual(input_ids(8196), input_ids(8196))
        self.assertEqual(len(input_ids(2052)), 2052*4)
        self.assertEqual(input_ids(512), input_ids(8196)[:2048])
        for text in ('', 'glm_prefill_layer position=0 rows=1 layer=0 kind=kda attention_ms=nan',
                     'glm_prefill_layer position=0 rows=1 layer=0 kind=kda wall_ms=-1'):
            with self.assertRaises(ValueError): breakdown(text)



class NVFP4PrefillScale(unittest.TestCase):
    def test_unscaled_fp4_times_fp8_is_exact_bf16(self):
        fp4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        bits = torch.tensor([i for i in range(256) if i & 127 != 127], dtype=torch.uint8)
        scales = bits.view(torch.float8_e4m3fn).float()
        unscaled = fp4[:, None]*scales[None, :]
        self.assertTrue(torch.equal(unscaled, unscaled.bfloat16().float()))

    def test_matrix_scale_after_fp32_accumulation_avoids_weight_rounding(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(5312)
        fp4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        weight = fp4[torch.randint(8, (37, 512), generator=generator)]
        weight *= torch.randn(37, 32, generator=generator).to(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        global_scale = .071312345
        worst_new, worst_old = 0., 0.
        for rows in (1, 2, 3, 4, 17, 128, 257):
            x = torch.randn(rows, 512, generator=generator).bfloat16().float()
            reference = (x.double()@weight.double().T)*global_scale
            old = x@(weight*global_scale).bfloat16().float().T
            new = (x@weight.T)*global_scale
            def error(y): return ((y.double()-reference).norm()/reference.norm()).item()
            worst_new = max(worst_new, error(new))
            worst_old = max(worst_old, error(old))
            self.assertLess(error(new), 1e-6)
            self.assertGreater(error(old), 1e-4)
        print(f'NVFP4 scale CPU: FP32 rescale L2={worst_new:.9g}, old weight rounding L2={worst_old:.9g}')


class KDAChunkAlgebra(unittest.TestCase):
    def test_parallel_triangular_scan_against_tokenwise_state(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(5312)
        worst = 0.
        for dim in (8, 128):
            for rows in (1, 15, 16, 17, 33, 129, 257):
                q, k, v = [torch.randn(rows, dim, generator=generator).bfloat16().float()
                           for _ in range(3)]
                q /= (q.square().sum(-1, keepdim=True)+1e-6).sqrt()
                k /= (k.square().sum(-1, keepdim=True)+1e-6).sqrt()
                g = -5*torch.rand(rows, dim, generator=generator)
                g[::7] = 0  # unit decay, including a nonzero incoming state
                beta = torch.rand(rows, generator=generator).bfloat16().float()
                beta[::11] = 0
                beta[::13] = 1
                initial = torch.randn(dim, dim, generator=generator)*.2
                state = initial.clone()
                serial = []
                for t in range(rows):
                    state = g[t].exp()[:, None]*state
                    delta = beta[t]*(v[t]-k[t]@state)
                    state += k[t, :, None]*delta
                    serial.append(q[t]@state/dim**.5)
                parallel = []
                boundary = initial.clone()
                for start in range(0, rows, 16):
                    end = min(rows, start+16)
                    qq, kk = q[start:end], k[start:end]
                    prefix = g[start:end].cumsum(0)
                    n = end-start
                    # Only decay differences are exponentiated; no reciprocal of
                    # exp(prefix), which underflows for real -5 gates.
                    decay = (prefix[:, None]-prefix[None, :]).exp()
                    causal = torch.ones(n, n).tril()
                    dots = (kk[:, None]*decay*kk[None, :]).sum(-1)
                    lower = -beta[start:end, None]*dots.tril(-1)
                    power = lower
                    inv = torch.eye(n)+lower
                    stride = 2
                    while stride < 16:
                        power = power@power
                        inv = inv+power@inv
                        stride *= 2
                    rhs = beta[start:end, None]*(v[start:end]-(kk*prefix.exp())@boundary)
                    delta = inv@rhs
                    output = (qq*prefix.exp())@boundary
                    output += (((qq[:, None]*decay*kk[None, :]).sum(-1))*causal)@delta
                    parallel.extend(output/dim**.5)
                    boundary = prefix[-1].exp()[:, None]*boundary
                    boundary += (kk*(prefix[-1]-prefix).exp()).T@delta
                for actual, expected in ((torch.stack(parallel), torch.stack(serial)), (boundary, state)):
                    relative = ((actual-expected).double().norm()/expected.double().norm().clamp_min(1e-30)).item()
                    worst = max(worst, relative)
                    self.assertLessEqual(relative, 1e-3, (dim, rows, relative))
        print(f'KDA CPU algebra: worst relative L2={worst:.9g}')


if __name__ == '__main__':
    unittest.main()
