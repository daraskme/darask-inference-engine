import json
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from hayate_tools.glm53_source import Source, PREFIX
from hayate_tools.glm53_convert import plan, write_layer, convert
from hayate_tools.glm53_mtp_bench import main as bench_main, parse_log, parse_profile, select_n, summarize, first_divergence, quality, TIE_GAP, trace_logits, eval_logits
from hayate_tools.hyt import NVFP4, PER_TENSOR, E4M3


class GLMMtp(unittest.TestCase):
    def test_bench_host_budget_uses_runtime_default_or_explicit_value(self):
        for host in [None, 0, 100]:
            with self.subTest(host=host), tempfile.TemporaryDirectory() as folder:
                argv = ['glm53_mtp_bench', '--hyt', folder, '--out', str(Path(folder) / 'out')]
                if host is not None:
                    argv += ['--host-gb', str(host)]
                with patch('sys.argv', argv), patch('torch.cuda.is_available', return_value=True), \
                        patch('hayate_tools.glm53_mtp_bench.run_command', side_effect=RuntimeError('command captured')) as run:
                    with self.assertRaisesRegex(RuntimeError, 'command captured'):
                        bench_main()
                command = run.call_args.args[0]
                if host is None:
                    self.assertNotIn('--host-gb', command)
                else:
                    self.assertEqual(command[command.index('--host-gb') + 1], str(host))

    def test_bench_replays_split_prefixes_and_withholds_failed_speed_summaries(self):
        for bad_margin in (False, True):
            with self.subTest(bad_margin=bad_margin), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / 'out'
                calls = []
                reference = torch.tensor([[3., 0.]] * 128)
                reference[44] = torch.tensor([1., 0. if bad_margin else .999])

                def run(command, prefix):
                    calls.append(command)
                    if '--spec-validate' in command:
                        return 'exact=true'
                    if '--eval-ids' in command:
                        tokens = struct.unpack('<129I', Path(command[command.index('--eval-ids')+1]).read_bytes())
                        self.assertEqual(tokens[2+44], 1)
                        target = Path(command[command.index('--out')+1])
                        target.mkdir()
                        (target / 'eval.json').write_text(json.dumps(dict(vocab=2, rows=127)))
                        (target / 'prefix_logits.bf16.bin').write_bytes(reference[:1].bfloat16().view(torch.uint16).numpy().tobytes())
                        (target / 'eval_logits.bf16.bin').write_bytes(reference[1:].bfloat16().view(torch.uint16).numpy().tobytes())
                        return ''
                    n = int(command[command.index('--spec-draft-n')+1])
                    reserved = '--spec-reserve-mtp' in command or n > 0
                    policy = command[command.index('--mtp-experts')+1] if '--mtp-experts' in command else 'gpu'
                    generated = [0] * 128
                    values = reference.clone()
                    if n:
                        generated[44] = 1
                        values[44] = torch.tensor([0. if bad_margin else 1., 1.001])
                    Path(command[command.index('--output-ids')+1]).write_text(json.dumps(dict(input=[1, 2], generated=generated)))
                    if '--dump-logits' in command:
                        dump = command[command.index('--dump-logits')+1]
                        for step, row in enumerate(values):
                            Path(f'{dump}.step{step}.bin').write_bytes(row.numpy().tobytes())
                    gpu_bytes = 300 if not reserved else (100 if policy == 'gpu' else 200)
                    log = (f'glm_mtp_cache main_decode_hits=8 main_decode_routes=10 main_expert_GPU_bytes={gpu_bytes} mtp_expert_GPU_bytes=0 main_placement_hash=18446744073709551615\n'
                           f'glm_spec_summary n={n} steps=8 proposed=24 accepted=16 acceptance=0.666667 emitted=24 seconds=2.4 disabled=false MTP_added_VRAM_bytes=800 MTP_expert_GPU_bytes=0 peak_VRAM_bytes=10000\n'
                           f'load_s=2 TTFT_s=3 prefill_s=1 decode_tok_s={9+n} ITL_p50_ms=100 ITL_p95_ms=120 memory={{}}\n')
                    if '--profile-spec' in command:
                        log += f'glm_spec position=32 n={n} accepted=0 step_ms=1\n'
                        log += ''.join(f'glm_spec_layer position=32 layer={i} d2h_syncs=1\n' for i in range(45))
                    return log

                argv = ['glm53_mtp_bench', '--hyt', folder, '--out', str(out), '--warmups', '0', '--repeats', '1']
                with patch('sys.argv', argv), patch('torch.cuda.is_available', return_value=True), \
                        patch('hayate_tools.glm53_mtp_bench.PROMPTS', {'code': 'test'}), \
                        patch('hayate_tools.glm53_mtp_bench.run_command', side_effect=run), patch('builtins.print'):
                    if bad_margin:
                        with self.assertRaisesRegex(SystemExit, 'quality/capacity gate failed'):
                            bench_main()
                    else:
                        bench_main()
                report = json.loads((out / 'report.json').read_text())
                self.assertEqual(report['quality_gate'], not bad_margin)
                configs = report['prompts']['code']['configurations']
                self.assertEqual('summary' in configs['gpu-n1'], not bad_margin)
                checked = configs['gpu-n1']['vs_matched_n0']
                self.assertEqual(checked['first_divergence']['token_number'], 45)
                self.assertAlmostEqual(checked['top1_rate'], 127/128)
                self.assertTrue(configs['gpu-n1']['same_main_cache_capacity'])
                self.assertEqual(len([c for c in calls if '--spec-validate' in c]), 2)
                self.assertTrue(any('--eval-ids' in c and '--spec-reserve-mtp' in c for c in calls))
                for command in calls:
                    if '--spec-validate' not in command and '--eval-ids' not in command and '--dump-logits' not in command and '--profile-spec' not in command:
                        self.assertIn('--spec-no-adaptive', command)

    def test_layer45_streaming_nvfp4_fp8_and_norms(self):
        torch.manual_seed(539)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'model_type': 'glm5_next', 'text_config': {'num_nextn_predict_layers': 1}, 'quantization_config': {'weight_block_size': [128, 128]}}))
            stem = PREFIX + 'layers.45.'
            value = torch.randn(16, 32).to(torch.float8_e4m3fn)
            tensors = {}
            for name in ['mlp.experts.0.gate_proj.weight', 'mlp.experts.0.up_proj.weight',
                         'mlp.experts.0.down_proj.weight', 'self_attn.o_proj.weight']:
                tensors[stem + name] = value.clone()
                tensors[stem + name.removesuffix('weight') + 'weight_scale_inv'] = torch.tensor([[.37]])
            tensors[stem + 'eh_proj.weight'] = torch.randn(16, 32).bfloat16()
            for name in ['enorm.weight', 'hnorm.weight', 'shared_head.norm.weight']:
                tensors[stem + name] = torch.ones(16).bfloat16()
            save_file(tensors, root / 'model.safetensors')
            source = Source(root)
            entries = plan(source, 45, 'nvfp4', dense={'sparse': 'fp8'})
            table = {e[0]: e for e, _ in entries}
            self.assertEqual(table['mlp.experts.0.gate_proj.weight'][2], (NVFP4, 0, 0))
            for name in ['eh_proj.weight', 'self_attn.o_proj.weight']:
                self.assertEqual(table[name][2], (PER_TENSOR, E4M3, 0))
            self.assertFalse(any('weight_scale_inv' in e[0] for e, _ in entries))
            write_layer(root / 'mtp.hyt', entries, bytes(32), source)
            self.assertTrue((root / 'mtp.hyt').is_file())
            with self.assertRaises(ValueError):
                convert(root, root / 'bad', layers=5, mtp=True, estimate=True)

    def test_input_order_and_post_norm_hidden(self):
        # Small independent formula: learned scales must remain on their own side.
        e = torch.tensor([1., 2., -3., 4.], dtype=torch.float64)
        h = torch.tensor([3., -1., 4., 2.], dtype=torch.float64)
        en = torch.tensor([.3, .7, 1.3, 1.9], dtype=torch.float64)
        hn = en.flip(0)
        w = torch.arange(32, dtype=torch.float64).reshape(4, 8) / 32
        def norm(x, weight):
            return x * torch.rsqrt(x.square().mean() + 1e-5) * weight
        merged = torch.cat([norm(e, en), norm(h, hn)])
        expected = w[:, :4] @ norm(e, en) + w[:, 4:] @ norm(h, hn)
        torch.testing.assert_close(w @ merged, expected)
        self.assertFalse(torch.allclose(w @ torch.cat([norm(h, hn), norm(e, en)]), expected))

    def test_bench_selection_and_log_rejection(self):
        runs = {0: [{'decode_tok_s': 10.}], 1: [{'decode_tok_s': 9.}], 2: [{'decode_tok_s': 12.}], 3: [{'decode_tok_s': 8.}]}
        self.assertEqual(select_n(runs), 2)
        self.assertEqual(select_n({0: runs[0], 1: runs[1]}), 0)
        log = ('glm_mtp_cache main_decode_hits=8 main_decode_routes=10 main_expert_GPU_bytes=9000 mtp_expert_GPU_bytes=0 main_placement_hash=123\n'
               'glm_spec_summary n=3 steps=8 proposed=24 accepted=16 acceptance=0.666667 emitted=24 seconds=2.4 disabled=false MTP_added_VRAM_bytes=800 MTP_expert_GPU_bytes=0 peak_VRAM_bytes=10000\n'
               'load_s=2 TTFT_s=3 prefill_s=1 decode_tok_s=10 ITL_p50_ms=100 ITL_p95_ms=120 memory={}\n')
        self.assertEqual(parse_log(log)['step_ms'], 300.)
        with self.assertRaises(ValueError):
            parse_log(log + log)
        with self.assertRaises(ValueError):
            parse_log(log.replace('decode_tok_s=10', 'decode_tok_s=nan'))
        measured = parse_log(log)
        summary = summarize({0: [dict(measured, proposed=0, accepted=0)],
                             3: [measured, dict(measured, decode_tok_s=12., accepted=24)]})
        self.assertIsNone(summary[0]['acceptance'])
        self.assertIsNone(summary[0]['mean_step_ms_median'])
        self.assertEqual(summary[3]['unprofiled_decode_tok_s_median'], 11.)
        self.assertAlmostEqual(summary[3]['acceptance'], 40 / 48)
        self.assertEqual(summary[3]['speedup_over_n0'], 1.1)

    def test_quality_uses_same_prefix_and_only_tolerates_near_ties(self):
        reference = torch.tensor([[1., .999, -3.]] * 128)
        actual = reference.clone()
        actual[44, 1] = 1.001
        checked = quality(reference, actual)
        self.assertTrue(checked['quality_gate'])
        self.assertEqual(checked['top1_matches'], 127)
        ids = [0] * 128
        generated = ids.copy()
        generated[44] = 1
        first = first_divergence(ids, generated, reference)
        self.assertEqual(first['token_number'], 45)
        self.assertTrue(first['near_tie'])
        self.assertLess(first['top1_top2_gap'], TIE_GAP)
        self.assertIsNone(first_divergence(ids, ids, reference))
        self.assertFalse(quality(reference, actual.flip(0).flip(1))['quality_gate'])
        # A near tie between top two cannot excuse selection of a distant third.
        generated[44] = 2
        self.assertFalse(first_divergence(ids, generated, reference)['near_tie'])
        actual[44, 2] = 1.1
        self.assertFalse(quality(reference, actual)['quality_gate'])
        # Large margins and shifted probability mass must fail independently.
        self.assertFalse(quality(torch.tensor([[2., 0.]]), torch.tensor([[0., 2.]]))['quality_gate'])
        self.assertFalse(quality(reference, reference * 10)['quality_gate'])
        with self.assertRaises(ValueError):
            quality(reference, actual[:1])
        with self.assertRaises(ValueError):
            quality(reference, reference * float('nan'))
        # Exactly tied BF16 logits preserve the GPU's smaller-ID rule.
        tied = torch.zeros(1, 3)
        self.assertEqual(first_divergence([0], [1], tied)['reference_top2'], 1)

    def test_native_teacher_forcing_alignment_includes_prefill_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'eval.json').write_text(json.dumps(dict(vocab=3, rows=2)))
            rows = torch.tensor([[3., 2., 1.], [4., 3., 2.], [5., 4., 3.]])
            for name, selected in [('prefix_logits.bf16.bin', rows[:1]), ('eval_logits.bf16.bin', rows[1:])]:
                (root / name).write_bytes(selected.bfloat16().view(torch.uint16).numpy().tobytes())
            torch.testing.assert_close(eval_logits(root), rows)
            for i, row in enumerate(rows):
                (root / f'trace.step{i}.bin').write_bytes(row.numpy().tobytes())
            torch.testing.assert_close(trace_logits(root / 'trace', 3), rows)
            (root / 'trace.step1.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                trace_logits(root / 'trace', 3)

    def test_profile_keeps_overlapping_intervals_and_union_counts(self):
        step = 'glm_spec position=32 n=3 accepted=2 draft_submit_ms=2 verify_submit_ms=9 accept_wait_ms=1 repair_ms=3 step_ms=15\n'
        layers = ''.join(f'glm_spec_layer position=32 layer={i} hc_attn_ms=1 projections_ms=2 recurrent_sparse_ms=3 output_hc_ms=1 moe_ms=4 expand_ms=1 d2h_syncs=1 d2h_wait_ms=2 kv_selected_rows=8 kv_union_rows=3\n'
                         for i in range(45))
        profile = parse_profile(layers + step)[0]
        self.assertEqual(profile['step_ms'], 15)
        self.assertEqual(profile['verify_submit_ms'], 9)
        self.assertEqual(profile['verifier_d2h_syncs'], 45)
        self.assertEqual(profile['layers'][0]['kv_union_rows'], 3)
        for log in ['', step, layers, layers + step + step,
                    layers.replace('layer=44 ', 'layer=43 ') + step,
                    layers + step.replace('step_ms=15', 'step_ms=nan'),
                    layers + step.replace('accepted=2', 'accepted=4')]:
            with self.assertRaises(ValueError):
                parse_profile(log)
