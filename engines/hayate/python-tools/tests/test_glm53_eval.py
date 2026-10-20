import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from hayate_tools.glm53_eval import compare, corpus, metrics
from hayate_tools.glm53_sensitivity import make_view, measure


def evaluation(path, values, ids=(0, 1, 0, 2, 1, 0), dtype='bf16'):
    path.mkdir()
    meta = dict(prefix=2, rows=4, vocab=3, source_hash='53'*16, layers=5, prefill_chunk=1, dtype=dtype)
    (path / 'eval.json').write_text(json.dumps(meta))
    (path / 'eval_ids.i32.bin').write_bytes(struct.pack('<6I', *ids))
    values = torch.tensor(values, dtype=torch.bfloat16 if dtype == 'bf16' else torch.float32)
    (path / f'eval_logits.{dtype}.bin').write_bytes(values.view(torch.uint8).numpy().tobytes())


class GLMEval(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.values = [[0., 1., 3.], [0., 3., 1.], [3., 1., 0.], [-2., 4., 0.]]

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_distribution_nll_shift_and_chunk_boundaries(self):
        ref, act = self.root / 'ref', self.root / 'act'
        evaluation(ref, self.values)
        evaluation(act, [[v + 100 for v in row] for row in self.values], dtype='f32')
        expected_nll = math.log(sum(math.exp(v) for v in [0, 1, 3])) - 3
        for chunk in (1, 2, 3, 8):
            report = compare(ref, act, chunk)
            self.assertAlmostEqual(report['mean_kld_fp8_to_nvfp4'], 0., places=12)
            self.assertEqual(report['top1_rate'], 1.)
            self.assertTrue(report['quality_gate'])
            self.assertEqual(report['nll_positions'], 3)
            self.assertAlmostEqual(report['reference_mean_nll'], expected_nll)
            self.assertAlmostEqual(report['mean_nll_delta'], 0.)

    def test_asymmetric_kld_and_nll_against_scalar_probability_formula(self):
        ref, act = self.root / 'ref', self.root / 'act'
        evaluation(ref, self.values, dtype='f32')
        actual = [[1., 0., -2.], [0., 1., 2.], [-1., 0., 1.], [1., 1., 1.]]
        evaluation(act, actual, dtype='f32')
        report = compare(ref, act)
        kl, nll_r, nll_a = [], [], []
        for i, (a, b) in enumerate(zip(self.values, actual)):
            p = [math.exp(v) / sum(math.exp(x) for x in a) for v in a]
            q = [math.exp(v) / sum(math.exp(x) for x in b) for v in b]
            kl.append(sum(x * math.log(x/y) for x, y in zip(p, q)))
            if i < 3:
                target = [2, 1, 0][i]
                nll_r.append(-math.log(p[target]))
                nll_a.append(-math.log(q[target]))
        self.assertAlmostEqual(report['mean_kld_fp8_to_nvfp4'], sum(kl)/4)
        self.assertAlmostEqual(report['max_kld'], max(kl))
        self.assertAlmostEqual(report['mean_nll_delta'], (sum(nll_a)-sum(nll_r))/3)
        self.assertFalse(report['quality_gate'])

    def test_top1_gate_fails_even_for_small_kld(self):
        ref, act = self.root / 'ref', self.root / 'act'
        evaluation(ref, [[.01, 0., 0.]]*4, dtype='f32')
        evaluation(act, [[0., .01, 0.]]*4, dtype='f32')
        report = compare(ref, act)
        self.assertLess(report['mean_kld_fp8_to_nvfp4'], .15)
        self.assertEqual(report['top1_rate'], 0.)
        self.assertFalse(report['quality_gate'])

    def test_rejects_misaligned_context_source_shape_and_nonfinite(self):
        ref, act = self.root / 'ref', self.root / 'act'
        evaluation(ref, self.values)
        evaluation(act, self.values)
        for key, value in [('layers', 8), ('prefix', 1), ('prefill_chunk', 128), ('source_hash', '54'*16),
                           ('rows', 0), ('rows', True), ('dtype', 'f16')]:
            meta = json.loads((ref/'eval.json').read_text())
            meta[key] = value
            (act/'eval.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                compare(ref, act)
        # another source (an EXL3 quant of the original weights) compares only when asked, and is reported
        meta = json.loads((ref/'eval.json').read_text())
        meta['source_hash'] = '54'*16
        (act/'eval.json').write_text(json.dumps(meta))
        report = compare(ref, act, cross_source=True)
        self.assertEqual((report['source_hash'], report['actual_source_hash']), ('53'*16, '54'*16))
        meta['layers'] = 8
        (act/'eval.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError): compare(ref, act, cross_source=True)
        (act/'eval.json').write_bytes((ref/'eval.json').read_bytes())
        (act/'eval_ids.i32.bin').write_bytes(struct.pack('<6I', 1, 1, 0, 2, 1, 0))
        with self.assertRaises(ValueError): compare(ref, act)
        (act/'eval_ids.i32.bin').write_bytes((ref/'eval_ids.i32.bin').read_bytes())
        logits = act/'eval_logits.bf16.bin'
        logits.write_bytes(logits.read_bytes()[:-1])
        with self.assertRaises(ValueError): compare(ref, act)
        logits.write_bytes(b'\x80\x7f'*12)
        with self.assertRaises(ValueError): compare(ref, act)
        for bad in (torch.tensor([[math.nan]]), torch.tensor([[math.inf]]), torch.empty(0, 1)):
            with self.assertRaises(ValueError): metrics(bad, bad)
        with self.assertRaises(ValueError): compare(ref, act, 0)

    def test_fixed_corpus_exports_identical_ids_and_refuses_overwrite(self):
        from hayate_tools import glm53_eval
        frozen = json.loads(Path(glm53_eval.__file__).with_name('glm53_eval_corpus.ids.json').read_text())
        for tokens in (256, 512):
            out = self.root / str(tokens)
            corpus(out, tokens)
            raw = (out/'input_ids.i32.bin').read_bytes()
            self.assertEqual(struct.unpack(f'<{tokens}I', raw), tuple(frozen['ids'][:tokens]))
            self.assertEqual(json.loads((out/'corpus.json').read_text())['ids_sha256'], hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError): corpus(out, tokens)
        with self.assertRaises(ValueError): corpus(self.root/'bad', 255)

    def test_prefill_and_tokenwise_prefix_gate(self):
        ref, act = self.root/'step', self.root/'prefill'
        evaluation(ref, self.values)
        evaluation(act, self.values)
        for path, mode, chunk in ((ref, 'step', 1), (act, 'prefill', 8192)):
            meta=json.loads((path/'eval.json').read_text())
            meta.update(eval_mode=mode, prefill_chunk=chunk)
            (path/'eval.json').write_text(json.dumps(meta))
            (path/'prefix_logits.bf16.bin').write_bytes((path/'eval_logits.bf16.bin').read_bytes()[:6])
        report=compare(ref,act,mode='prefill')
        self.assertTrue(report['quality_gate'])
        self.assertEqual(report['prefix_kld'],0.)
        self.assertEqual(report['reference_prefill_chunk'],1)
        self.assertEqual(report['actual_prefill_chunk'],8192)
        with self.assertRaises(ValueError): compare(ref,act)
        with self.assertRaises(ValueError): compare(act,ref,mode='prefill')
        # Subsequent teacher-forced rows can agree while the prefill itself is broken.
        wrong=torch.tensor([5.,0.,0.],dtype=torch.bfloat16)
        (act/'prefix_logits.bf16.bin').write_bytes(wrong.view(torch.uint8).numpy().tobytes())
        report=compare(ref,act,mode='prefill')
        self.assertEqual(report['mean_kld'],0.)
        self.assertGreater(report['prefix_kld'],.01)
        self.assertFalse(report['quality_gate'])
        (act/'prefix_logits.bf16.bin').write_bytes(b'bad')
        with self.assertRaises(ValueError): compare(ref,act,mode='prefill')

    def test_prefill_kld_limit_checks_each_prefix_and_suffix_row(self):
        ref, act = self.root/'step', self.root/'prefill'
        evaluation(ref, [[0., 0., 0.]]*4)
        evaluation(act, [[0., 0., 0.]]*4)
        for path, mode in ((ref, 'step'), (act, 'prefill')):
            meta = json.loads((path/'eval.json').read_text())
            meta['eval_mode'] = mode
            (path/'eval.json').write_text(json.dumps(meta))
            (path/'prefix_logits.bf16.bin').write_bytes(bytes(6))
        for site in ('prefix', 'suffix'):
            for delta, passed in ((.35, True), (.5, False)):
                values = torch.zeros(4, 3, dtype=torch.bfloat16)
                prefix = torch.zeros(3, dtype=torch.bfloat16)
                if site == 'prefix':
                    prefix[0] = delta
                else:
                    values[2, 0] = delta
                (act/'prefix_logits.bf16.bin').write_bytes(prefix.view(torch.uint8).numpy().tobytes())
                (act/'eval_logits.bf16.bin').write_bytes(values.view(torch.uint8).numpy().tobytes())
                report = compare(ref, act, mode='prefill')
                self.assertEqual(report['quality_gate'], passed, (site, delta, report))
                self.assertEqual(report['prefill_max_kld_limit'], .02)
                self.assertLess(report['mean_kld'], .01)
                peak = report['prefix_kld'] if site == 'prefix' else report['max_kld']
                self.assertGreater(peak, .01 if passed else .02)

    def test_layer_sweep_and_combination_do_not_modify_source_weights(self):
        fp8, candidate = self.root/'fp8', self.root/'candidate'
        for root in (fp8, candidate):
            root.mkdir()
            (root/'globals.hyt').write_bytes(b'global')
            for layer in range(5):
                (root/f'layer{layer}').mkdir()
                (root/f'layer{layer}/weights.hyt').write_bytes(b'fp8_data' if root == fp8 else b'nvfp4')
        ref, act = self.root/'ref', self.root/'act'
        evaluation(ref, self.values)
        evaluation(act, [[0., 0., 0.]]*4)
        def fake_run(command, **kwargs):
            shutil.copytree(ref, command[command.index('--out') + 1])
            self.assertEqual(command[command.index('--eval-prefix')+1], '2')
        args = SimpleNamespace(reference=ref, actual=act, layers=5, restore_layers=[3, 4],
            out=self.root/'sweep', combine=False, fp8_hyt=fp8, candidate_hyt=candidate,
            hayate=Path('/tmp/hayate'), host_gb=0, prefill_chunk=1)
        with patch('hayate_tools.glm53_sensitivity.subprocess.run', side_effect=fake_run) as run:
            results = measure(args)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(r['kld_improvement'] > 0 and r['quality_gate'] for r in results))
            self.assertEqual((args.out/'hyt-3/layer3').resolve(), fp8/'layer3')
            self.assertEqual((args.out/'hyt-3/layer4').resolve(), candidate/'layer4')
            args.out, args.combine = self.root/'combined', True
            self.assertEqual(measure(args)[0]['restored_layers'], [3, 4])
        self.assertEqual((candidate/'layer3/weights.hyt').read_bytes(), b'nvfp4')
        with self.assertRaises(ValueError): make_view(fp8, candidate, self.root/'bad', 5, [2])


if __name__ == '__main__':
    unittest.main()
