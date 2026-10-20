import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file
from hayate_tools.glm53_convert import dense_group, dense_options, plan, write_layer
from hayate_tools.glm53_source import Source, PREFIX
from hayate_tools.glm53_profile import summarize
from hayate_tools.hyt import PLAIN, PER_TENSOR, BF16, E4M3, NVFP4
from hayate_tools.quant import quantize, dequantize_nvfp4

ROOT = Path(__file__).resolve().parents[2]


class GLMDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'math.so'
        subprocess.run(['c++', '-shared', '-fPIC', '-O3', '-std=c++20',
                        str(ROOT / 'crates/kernels/cu/glm53/math_cpu_test.cpp'), '-o', str(path)], check=True)
        cls.lib = ctypes.CDLL(str(path))
        cls.lib.glm53_cpu_dense.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 3
        cls.lib.glm53_cpu_swiglu.argtypes = [ctypes.c_float] * 2
        cls.lib.glm53_cpu_swiglu.restype = ctypes.c_float
        cls.lib.glm53_cpu_reduce.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_float]
        cls.lib.glm53_cpu_reduce.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_compressed_dense_partial_tiles_against_torch(self):
        torch.manual_seed(536)
        worst = 0.
        for n, k in ((7, 16), (129, 80), (33, 256)):
            value = torch.randn(n, k).bfloat16().float()
            x = torch.randn(k).bfloat16().float()
            for fmt, spec in ((2, (PER_TENSOR, E4M3, 0)), (3, (NVFP4, 0, 0))):
                image = quantize(value, spec)
                if fmt == 3:
                    restored = dequantize_nvfp4(image, value.shape)
                else:
                    scale = image[-4:].view(torch.float32).item()
                    restored = image[:-4].view(torch.float8_e4m3fn).float().reshape(n, k) * scale
                restored = restored.bfloat16().float()
                expected = restored.double() @ x.double()
                out = np.empty(n, dtype=np.float32)
                self.lib.glm53_cpu_dense(out.ctypes.data, x.data_ptr(), image.data_ptr(), None, n, k, fmt)
                error = float(np.linalg.norm(out.astype(np.float64) - expected.numpy()) / np.linalg.norm(expected.numpy()))
                worst = max(worst, error)
                self.assertLess(error, 1e-6)
            weights = value.to(torch.float8_e4m3fn)
            scales = torch.rand((n+127)//128, (k+127)//128) + .1
            expected = (weights.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]).bfloat16().double() @ x.double()
            out = np.empty(n, dtype=np.float32)
            self.lib.glm53_cpu_dense(out.ctypes.data, x.data_ptr(), weights.data_ptr(), scales.data_ptr(), n, k, 1)
            np.testing.assert_allclose(out, expected.numpy(), rtol=1e-6, atol=1e-6)
        print(f'GLM dense CPU: 9 cases, max relative L2={worst:.9g}')

    def test_clipping_and_ascending_bf16_expert_reduce(self):
        for gate, up in ((20., -20.), (-20., 20.), (.53125, -3.25), (10., 10.), (0., 0.)):
            a = torch.tensor(min(gate, 10.))
            expected = ((a * torch.sigmoid(a)).bfloat16().float() * max(-10., min(10., up))).bfloat16().item()
            self.assertEqual(self.lib.glm53_cpu_swiglu(gate, up), expected)
        # Reduction order and intermediate BF16 rounding matter with cancellation.
        ids = np.array([287, 0, 17, 42, 8, 256, 3, 4], dtype=np.int32)
        vals = np.array([100., -100., .1, -.2, 1., -1., .3, -.4], dtype=np.float32)
        weights = np.array([.17, .13, .49, .53, .73, .11, .19, .23], dtype=np.float32)
        expected = torch.tensor(0.)
        for i in np.argsort(ids):
            term = (torch.tensor(vals[i]) * float(weights[i])).bfloat16().float()
            expected = (expected + term).bfloat16().float()
        expected = (expected + .53125).bfloat16().item()
        self.assertEqual(self.lib.glm53_cpu_reduce(vals.ctypes.data, ids.ctypes.data, weights.ctypes.data, .53125), expected)

    def test_dense_streaming_format_groups_and_no_rotation(self):
        torch.manual_seed(536)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'model_type': 'glm5_next', 'quantization_config': {'weight_block_size': [128, 128]}}))
            value = torch.randn(129, 80).to(torch.float8_e4m3fn)
            scale = torch.tensor([[.173], [.391]])
            name = PREFIX + 'layers.3.mlp.shared_experts.gate_proj.weight'
            router = PREFIX + 'layers.3.mlp.gate.weight'
            tensors = {name: value, name.removesuffix('weight')+'weight_scale_inv': scale,
                       router: torch.ones(8, 80, dtype=torch.bfloat16)}
            save_file(tensors, root / 'model.safetensors')
            source = Source(root)
            baseline = plan(source, 3)
            self.assertEqual(len(baseline), 3)
            expected_value = (value.float() * scale.repeat_interleave(128, 0)[:129]).bfloat16().float()
            for fmt, spec in (('bf16', (PLAIN, BF16, 0)), ('fp8', (PER_TENSOR, E4M3, 0)), ('nvfp4', (NVFP4, 0, 0))):
                entries = plan(source, 3, 'nvfp4', dense={'shared': fmt})
                self.assertEqual(len(entries), 2)
                dest = root / f'{fmt}.hyt'
                write_layer(dest, entries, bytes(32), source)
                raw = dest.read_bytes()
                base = int.from_bytes(raw[24:32], 'little')
                for entry, item in entries:
                    payload = raw[base+entry[3]:base+entry[3]+entry[4]]
                    if item.name == name:
                        self.assertEqual(entry[2], spec)
                        expected = expected_value.bfloat16().view(torch.uint8) if fmt == 'bf16' else quantize(expected_value, spec)
                        self.assertEqual(payload, expected.numpy().tobytes())
                    else:
                        self.assertEqual(payload, tensors[router].view(torch.uint8).numpy().tobytes())
            self.assertEqual(dense_group('self_attn.q_proj.weight', 4), 'kda')
            self.assertEqual(dense_group('self_attn.kv_b_proj.weight', 3), 'sparse')
            self.assertEqual(dense_group('lm_head.weight', 0), 'lm-head')
            self.assertEqual(dense_group('mlp.gate_proj.weight', 0), 'dense-ffn')
            for name in ('mlp.gate.weight', 'hc_attn_fn', 'input_layernorm.weight', 'self_attn.q_conv1d.weight', 'mlp.experts.0.gate_proj.weight'):
                self.assertIsNone(dense_group(name, 3))
            self.assertEqual(dense_options(['shared=nvfp4', 'kda=fp8']), {'shared': 'nvfp4', 'kda': 'fp8'})
            for values in (['x=fp8'], ['kda=int8'], ['kda'], ['kda=fp8', 'kda=nvfp4']):
                with self.assertRaises(ValueError):
                    dense_options(values)

    def test_profile_async_copy_bytes_and_background_waits(self):
        rows = ['glm_profile position=32 layer=head kind=head d2h_count=1 d2h_sync_count=1',
                'glm_profile position=32 layer=maintenance kind=maintenance maintenance_ms=.2 '
                'maintenance_worker_ms=12 maintenance_wait_ms=.3 maintenance_h2d=8 '
                'maintenance_d2h=6 maintenance_disk=2 maintenance_bytes=196000000 '
                'host_h2d=16 disk_h2d=6 h2d_bytes=308000000 disk_bytes=84000000 '
                'reader_ms=24 reader_io_ms=16 reader_bytes=84000000 reader_pack_ms=4 reader_upload_ms=4 read_wait_ms=5 predicted=8 prefetch_hits=3 d2h_count=6 d2h_sync_count=0']
        report = summarize('\n'.join(rows))
        totals = report['ms_per_token']
        self.assertEqual(totals['d2h_sync_count'], 1)
        self.assertEqual(totals['d2h_count'], 7)
        self.assertEqual(totals['h2d_bytes'], 308000000)
        self.assertEqual(totals['maintenance_worker_ms'], 12)
        self.assertEqual(totals['maintenance_wait_ms'], .3)
        self.assertEqual(totals['prefetch_hits'], 3)
        self.assertEqual(totals['reader_pack_ms'], 4)
        self.assertAlmostEqual(report['disk_bandwidth']['demand_payload_GBps'], 16.8)
        self.assertAlmostEqual(report['disk_bandwidth']['io_worker_GBps'], 5.25)
        with self.assertRaises(ValueError):
            summarize('\n'.join(rows).replace('disk_bytes=84000000', 'disk_bytes=-1'))

    def test_profile_aggregation_and_incomplete_logs(self):
        rows = []
        for p in (9, 10):
            rows += [f'glm_profile position={p} layer=0 kind=kda attention_ms=2 mhc_ms=1 d2h_count=1 d2h_sync_count=1 d2h_wait_ms=3',
                     f'glm_profile position={p} layer=3 kind=sparse attention_ms=4 routing_ms=1 transfer_ms=2 gate_up_ms=3 down_ms=2 reduce_ms=1 d2h_count=5 d2h_sync_count=4 d2h_wait_ms=7',
                     f'glm_profile position={p} layer=head kind=head lm_head_ms=5 d2h_count=1 d2h_sync_count=1 d2h_wait_ms=5',
                     f'glm_profile position={p} layer=maintenance kind=maintenance maintenance_ms=1 d2h_count=1 d2h_sync_count=1 d2h_wait_ms=.5 token_wall_ms=24']
        text = '\n'.join(rows)
        report = summarize(text)
        self.assertEqual(report['tokens'], 2)
        self.assertIsNone(report['disk_bandwidth']['demand_payload_GBps'])
        self.assertIsNone(report['disk_bandwidth']['io_worker_GBps'])
        self.assertEqual(report['ms_per_token']['d2h_count'], 8)
        self.assertEqual(report['ms_per_token']['d2h_sync_count'], 7)
        self.assertEqual(report['ms_per_token']['attention_ms'], 6)
        self.assertEqual(report['by_kind']['sparse']['gate_up_ms'], 3)
        self.assertEqual(summarize(text, 1)['tokens'], 1)
        for invalid in (text+'\n'+rows[0], text.replace('lm_head_ms=5', 'lm_head_ms=nan'), '\n'.join(rows[:-1])):
            with self.assertRaises(ValueError):
                summarize(invalid)
        with self.assertRaises(ValueError):
            summarize(text, 2)

    def test_profile_cpu_bandwidth_uses_total_bytes_over_total_wall(self):
        rows = []
        for p, ms, count in [(1, 2, 1), (2, 6, 2)]:
            rows += [f'glm_profile position={p} layer=head kind=head',
                     f'glm_profile position={p} layer=maintenance kind=maintenance '
                     f'cpu_experts={count} cpu_bytes={count * 14000000} cpu_ms={ms} '
                     f'cpu_worker_ms={ms / 2} cpu_dispatch_ms={ms * .75} token_wall_ms=10']
        report = summarize('\n'.join(rows))
        self.assertEqual(report['ms_per_token']['cpu_experts'], 1.5)
        self.assertEqual(report['ms_per_token']['token_wall_ms'], 10)
        self.assertEqual(report['cpu_bandwidth_GBps'], 5.25)
        self.assertEqual(report['ms_per_token']['cpu_worker_ms'], 2)
        self.assertEqual(summarize('\n'.join(rows), 1)['cpu_bandwidth_GBps'], 28 / 6)


if __name__ == '__main__':
    unittest.main()
