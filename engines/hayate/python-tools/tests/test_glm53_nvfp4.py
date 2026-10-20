import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file
from hayate_tools.glm53_convert import plan, write_layer
from hayate_tools.glm53_source import Source, PREFIX
from hayate_tools.hyt import NVFP4, storage_bytes
from hayate_tools.quant import quantize, dequantize_nvfp4, untile_scales
from hayate_tools.glm53_compare import compare
from hayate_tools.glm53_nvfp4 import (hadamard, quantize_expert, reconstruction_metrics,
                                     scale_search_mode, SCALE_SEARCH)
from hayate_tools.glm53_quant_study import study

ROOT = Path(__file__).resolve().parents[2]


class GLMNvfp4(unittest.TestCase):
    def test_ge1_error_and_clipping_bounds_after_serialization(self):
        torch.manual_seed(5303)
        # Include zero blocks, a wide dynamic range, and non-tile-aligned rows.
        value = torch.randn(129, 256) * torch.logspace(-4, 2, 256)
        value[0].zero_()
        for block in (0, 128):
            x = hadamard(value, block)
            rtn_image = quantize_expert(x)
            image = quantize_expert(x, scale_search='mse-ge1')
            self.assertEqual(image.numel(), rtn_image.numel())
            self.assertTrue(torch.equal(image, quantize_expert(x, scale_search='mse-ge1')))
            self.assertTrue(torch.equal(image[-4:], rtn_image[-4:]))
            scales = [untile_scales(b[x.numel()//2:-4], 129, 16).float() for b in (rtn_image, image)]
            self.assertTrue(torch.all(scales[1] >= scales[0]))
            decoded = [dequantize_nvfp4(b, x.shape) for b in (rtn_image, image)]
            errors = [(d.double() - x.double()).reshape(-1, 16).square().sum(-1) for d in decoded]
            self.assertTrue(torch.all(errors[1] <= errors[0] + errors[0] * 1e-5 + 1e-12))
            before, after = [reconstruction_metrics(x, b) for b in (rtn_image, image)]
            self.assertLess(after['squared_error'], before['squared_error'])
            self.assertLessEqual(after['max_clip_fraction'], before['max_clip_fraction'])
            self.assertLessEqual(after['deep_clipped_blocks'], before['deep_clipped_blocks'])
            self.assertLessEqual(after['clipped_energy_fraction'], before['clipped_energy_fraction'])
            self.assertTrue(torch.equal(decoded[1][0], x[0]))

    def test_scale_search_compatibility_and_invalid_inputs(self):
        torch.manual_seed(53)
        value = torch.randn(128, 128)
        self.assertTrue(torch.equal(quantize_expert(value), quantize(value, (NVFP4, 0, 0))))
        self.assertTrue(torch.equal(quantize_expert(value, True), quantize_expert(value, scale_search='mse')))
        for mode in SCALE_SEARCH:
            for bad in (torch.ones(16), torch.ones(1, 17), torch.empty(0, 16),
                        torch.full((1, 16), float('nan')), torch.full((1, 16), float('inf'))):
                with self.subTest(mode=mode, shape=bad.shape), self.assertRaises(ValueError):
                    quantize_expert(bad, scale_search=mode)
            zero = torch.zeros(1, 16)
            report = reconstruction_metrics(zero, quantize_expert(zero, scale_search=mode))
            self.assertEqual(report['relative_l2'], 0.)
            self.assertEqual(report['max_clip_fraction'], 0.)
            self.assertEqual(report['zeroed_nonzero_blocks'], 0)
        for mode in ('bad', 'rtn', 'mse-ge1'):
            with self.assertRaises(ValueError):
                scale_search_mode(True, mode)

    def test_clipping_metrics_distinguish_scale_rounding_from_deep_clips(self):
        # The smaller block's scale rounds down even with GE1. Never call GE1
        # "zero clipping": it only disallows extra clipping compared with RTN.
        x = torch.zeros(1, 32)
        x[0, 0], x[0, 16] = 2688., 6.1
        report = reconstruction_metrics(x, quantize_expert(x))
        self.assertEqual(report['clipped_blocks'], 1)
        self.assertEqual(report['deep_clipped_blocks'], 0)
        self.assertAlmostEqual(report['max_clip_fraction'], (float(x[0, 16]) - 6.) / float(x[0, 16]))
        self.assertGreater(report['clipped_energy_fraction'], 0.)

    def test_ge1_conversion_and_study_of_late_layer(self):
        torch.manual_seed(53)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(model_type='glm5_next',
                quantization_config=dict(weight_block_size=[128, 128]))))
            tensors = {}
            for part in ('gate', 'up', 'down'):
                name = PREFIX + f'layers.44.mlp.experts.287.{part}_proj.weight'
                tensors[name] = torch.randn(129, 256).to(torch.float8_e4m3fn)
                tensors[name.removesuffix('weight') + 'weight_scale_inv'] = torch.rand(2, 2)
            save_file(tensors, root / 'model.safetensors')
            source = Source(root)
            entries = plan(source, 44, 'nvfp4', 128, down_fp8=True)
            path = root / 'weights.hyt'
            write_layer(path, entries, bytes(32), source, hadamard_block=128, scale_search='mse-ge1')
            raw = path.read_bytes()
            base = int.from_bytes(raw[24:32], 'little')
            from hayate_tools.glm53_nvfp4 import restored
            for entry, item in entries:
                if entry[2] == (NVFP4, 0, 0):
                    expected = quantize_expert(hadamard(restored(source, item.name), 128), scale_search='mse-ge1')
                elif item is not None:
                    expected = source.raw_tensor(item.name).contiguous().view(torch.uint8)
                else:
                    continue
                self.assertEqual(raw[base+entry[3]:base+entry[3]+entry[4]], expected.numpy().tobytes())
            report = study(root, [44], [287], [0, 128], ['rtn', 'mse-ge1'])
            self.assertEqual(len(report['matrices']), 6)
            self.assertIsNone(report['quality_gate'])
            self.assertEqual(report['quality_status'], 'not_evaluated')
            for row in report['matrices']:
                self.assertEqual(len(row['restored_f32_sha256']), 64)
            for block in (0, 128):
                a, b = (report['summary'][f'{mode}/r{block}'] for mode in ('rtn', 'mse-ge1'))
                self.assertEqual(a['payload_bytes'], b['payload_bytes'])
                self.assertLess(b['relative_l2'], a['relative_l2'])
            for layers, experts, blocks, modes in [([45], [0], [0], ['rtn']),
                    ([3], [288], [0], ['rtn']), ([3, 3], [0], [0], ['rtn']),
                    ([3], [0], [16], ['rtn']), ([3], [0], [0], ['oops'])]:
                with self.assertRaises(ValueError):
                    study(root, layers, experts, blocks, modes)

    def test_hadamard_basis_and_cpu_cuda_shared_arithmetic(self):
        torch.manual_seed(53)
        with tempfile.TemporaryDirectory() as folder:
            lib = Path(folder) / 'math.so'
            subprocess.run(['c++', '-shared', '-fPIC', '-O3', '-std=c++20',
                str(ROOT / 'crates/kernels/cu/glm53/math_cpu_test.cpp'), '-o', str(lib)], check=True)
            fn = ctypes.CDLL(str(lib)).glm53_cpu_hadamard
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
            for block in (32, 64, 128, 256):
                matrix = torch.ones(1, 1)
                while matrix.shape[0] < block:
                    matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
                matrix /= block ** .5
                x = torch.randn(3, block * 2).bfloat16().float()
                expected = (x.reshape(-1, block) @ matrix).reshape(x.shape)
                rotated = hadamard(x, block)
                torch.testing.assert_close(rotated, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(hadamard(rotated, block), x, rtol=2e-5, atol=2e-6)
                w = torch.randn(7, block * 2)
                torch.testing.assert_close(hadamard(w, block) @ rotated.T, w @ x.T, rtol=2e-5, atol=2e-5)
                output = np.empty(x.shape, dtype=np.float32)
                fn(output.ctypes.data, x.data_ptr(), x.numel(), block)
                np.testing.assert_allclose(output, rotated.bfloat16().float().numpy(), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                hadamard(torch.ones(1, 80), 32)

    def test_mse_search_never_increases_block_error(self):
        torch.manual_seed(53)
        value = torch.randn(129, 256)
        value[0].zero_()
        for block in (0, 32, 64, 128, 256):
            rotated = hadamard(value, block)
            rtn = dequantize_nvfp4(quantize_expert(rotated), value.shape)
            image = quantize_expert(rotated, True)
            mse = dequantize_nvfp4(image, value.shape)
            self.assertEqual(image.numel(), storage_bytes(value.shape, (NVFP4, 0, 0)))
            rtn_error = (rtn - rotated).reshape(-1, 16).square().sum(-1)
            mse_error = (mse - rotated).reshape(-1, 16).square().sum(-1)
            self.assertTrue(torch.all(mse_error <= rtn_error + 1e-7))
            self.assertLess(float(mse_error.sum()), float(rtn_error.sum()))
            self.assertTrue(torch.equal(mse[0], value[0]))
            self.assertTrue(torch.equal(image, quantize_expert(rotated, True)))

    def test_rotated_conversion_metadata_and_original_fp8_down(self):
        torch.manual_seed(53)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(model_type='glm5_next',
                quantization_config=dict(weight_block_size=[128, 128]))))
            weight = torch.randn(128, 256).to(torch.float8_e4m3fn)
            scales = torch.tensor([[.173, .391]])
            tensors = {}
            for part in ('gate', 'up', 'down'):
                name = PREFIX + f'layers.3.mlp.experts.0.{part}_proj.weight'
                tensors[name] = weight.clone()
                tensors[name.removesuffix('weight') + 'weight_scale_inv'] = scales.clone()
            save_file(tensors, root / 'model.safetensors')
            source = Source(root)
            value = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
            for down_fp8 in (False, True):
                entries = plan(source, 3, 'nvfp4', 64, down_fp8)
                path = root / f'{down_fp8}.hyt'
                write_layer(path, entries, bytes(32), source, hadamard_block=64, mse=True)
                raw = path.read_bytes()
                base = int.from_bytes(raw[24:32], 'little')
                for entry, item in entries:
                    payload = raw[base + entry[3]:base + entry[3] + entry[4]]
                    if item is None:
                        self.assertEqual(np.frombuffer(payload, '<f4').tolist(), [64.])
                    elif entry[2] == (NVFP4, 0, 0):
                        self.assertEqual(payload, quantize_expert(hadamard(value, 64), True).numpy().tobytes())
                    else:
                        expected = source.raw_tensor(item.name).contiguous().view(torch.uint8).numpy().tobytes()
                        self.assertEqual(payload, expected)
                with self.assertRaises(ValueError):
                    write_layer(root / 'bad.hyt', entries, bytes(32), source)
                self.assertFalse((root / 'bad.hyt').exists())
            with self.assertRaises(ValueError):
                plan(source, 3, 'fp8', 64)
    def test_quality_gate_uses_each_fp8_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = dict(input=[17], generated=[0])
            (root / 'ids.json').write_text(json.dumps(ids))
            for name in ('fp8', 'nvfp4'):
                (root / f'{name}.ids.json').write_text(json.dumps(ids))
            np.array([4, 2], dtype='<f4').tofile(root / 'step0.bin')
            np.array([4.25, 2], dtype='<f4').tofile(root / 'fp8.step0.bin')
            np.array([4.5, 2], dtype='<f4').tofile(root / 'nvfp4.step0.bin')
            report = compare(root, root / 'nvfp4', fp8_baseline=root / 'fp8')
            self.assertTrue(report[0]['passed'])
            np.array([4.75, 2], dtype='<f4').tofile(root / 'nvfp4.step0.bin')
            with self.assertRaises(ValueError):
                compare(root, root / 'nvfp4', fp8_baseline=root / 'fp8')
            self.assertTrue(compare(root, root / 'nvfp4', greedy_only=True)[0]['passed'])
            with self.assertRaises(ValueError):
                compare(root, root / 'nvfp4', fp8_baseline=root / 'fp8', greedy_only=True)
            np.array([2, 4], dtype='<f4').tofile(root / 'nvfp4.step0.bin')
            with self.assertRaises(ValueError):
                compare(root, root / 'nvfp4', fp8_baseline=root / 'fp8')
            with self.assertRaises(ValueError):
                compare(root, root / 'nvfp4', greedy_only=True)

    def test_direct_fp8_conversion_and_hyt_payload(self):
        torch.manual_seed(53)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(model_type='glm5_next',
                quantization_config=dict(weight_block_size=[128, 128]))))
            weight = torch.randn(129, 256).to(torch.float8_e4m3fn)
            scales = torch.tensor([[.173, .391], [.723, 1.137]])
            name = PREFIX + 'layers.3.mlp.experts.0.gate_proj.weight'
            shared = PREFIX + 'layers.3.mlp.shared_experts.gate_proj.weight'
            tensors = {shared: weight.clone(), shared.removesuffix('weight') + 'weight_scale_inv': scales.clone()}
            for part in ('gate', 'up', 'down'):
                key = name.replace('gate_proj', f'{part}_proj')
                tensors[key] = weight.clone()
                tensors[key.removesuffix('weight') + 'weight_scale_inv'] = scales.clone()
            save_file(tensors, root / 'model.safetensors')
            source = Source(root)
            entries = plan(source, 3, 'nvfp4')
            routed = [e for e, _ in entries if e[0].startswith('mlp.experts.')]
            self.assertEqual(len(routed), 3)
            entry = routed[0]
            self.assertEqual(entry[2], (NVFP4, 0, 0))
            self.assertEqual(entry[4], storage_bytes([129, 256], entry[2]))
            value = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:129]
            expected = quantize(value, entry[2]).numpy().tobytes()
            self.assertNotEqual(expected, quantize(value.bfloat16(), entry[2]).numpy().tobytes())
            dest = root / 'weights.hyt'
            write_layer(dest, entries, bytes(32), source)
            raw = dest.read_bytes()
            base = int.from_bytes(raw[24:32], 'little')
            for entry in routed:
                self.assertEqual((base + entry[3]) % 4096, 0)
                self.assertEqual((base + entry[3] + 129 * 256 // 2) % 2, 0)
                self.assertEqual((base + entry[3] + entry[4] - 4) % 4, 0)
                self.assertEqual(raw[base + entry[3]:base + entry[3] + entry[4]], expected)
            self.assertEqual(len(plan(source, 3)), 8)
            with self.assertRaises(FileExistsError):
                write_layer(dest, entries, bytes(32), source)
            with self.assertRaises(ValueError):
                plan(source, 3, 'q4')

    def test_shared_cuda_nvfp4_layout_on_cpu(self):
        torch.manual_seed(53)
        with tempfile.TemporaryDirectory() as folder:
            lib = Path(folder) / 'math.so'
            subprocess.run(['c++', '-shared', '-fPIC', '-O3', '-std=c++20',
                str(ROOT / 'crates/kernels/cu/glm53/math_cpu_test.cpp'), '-o', str(lib)], check=True)
            fn = ctypes.CDLL(str(lib)).glm53_cpu_nvfp4
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
            for n, k in [(1, 16), (129, 80), (256, 256)]:
                value = torch.randn(n, k)
                value[0].zero_()
                for mode in SCALE_SEARCH:
                    image = quantize_expert(value, scale_search=mode)
                    output = np.empty((n, k), dtype=np.float32)
                    fn(output.ctypes.data, image.data_ptr(), n, k)
                    expected = dequantize_nvfp4(image, [n, k]).numpy()
                    np.testing.assert_allclose(output, expected, rtol=2e-7, atol=1e-7)
                    error = float((torch.from_numpy(output) - value).norm() / value.norm()) if value.norm() else 0.
                    self.assertLess(error, .14)
                    print(f'GLM CPU NVFP4 {mode} {n}x{k}: quantization relative L2={error:.6f}')

    def test_failed_conversion_removes_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(model_type='glm5_next',
                quantization_config=dict(weight_block_size=[128, 128]))))
            name = PREFIX + 'layers.3.mlp.experts.0.gate_proj.weight'
            save_file({name: torch.full((128, 128), float('nan')).to(torch.float8_e4m3fn),
                name.removesuffix('weight') + 'weight_scale_inv': torch.ones(1, 1)}, root / 'model.safetensors')
            source = Source(root)
            with self.assertRaises(ValueError):
                write_layer(root / 'weights.hyt', plan(source, 3, 'nvfp4'), bytes(32), source)
            self.assertFalse((root / 'weights.hyt.partial').exists())


@unittest.skipUnless(torch.cuda.is_available(), 'requires supported CUDA GPU')
class GLMNvfp4GPU(unittest.TestCase):
    def test_cuda_hadamard_in_place_and_separate_output(self):
        with tempfile.TemporaryDirectory() as folder:
            lib = Path(folder) / 'rotate.so'
            subprocess.run(['nvcc', '-shared', '-O3', '--use_fast_math', '-std=c++20',
                '-gencode=arch=compute_120a,code=sm_120a', '-Xcompiler=-fPIC',
                str(ROOT / 'crates/kernels/cu/glm53/rotate.cu'), '-o', str(lib)], check=True)
            fn = ctypes.CDLL(str(lib)).hayate_glm_hadamard
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            for block in (32, 64, 128, 256):
                x = torch.randn(3, block * 2, device='cuda').bfloat16()
                expected = hadamard(x, block).bfloat16()
                out = torch.empty_like(x)
                stream = torch.cuda.current_stream().cuda_stream
                self.assertEqual(fn(out.data_ptr(), x.data_ptr(), 3, block * 2, block, stream), 0)
                self.assertEqual(fn(x.data_ptr(), x.data_ptr(), 3, block * 2, block, stream), 0)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_cuda_mse_rotation_quality_and_repeatability(self):
        torch.manual_seed(53)
        value = torch.randn(129, 256).to(torch.float8_e4m3fn).float()
        scales = torch.rand(2, 2)
        value *= scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:129]
        for block in (0, 64):
            rotated = hadamard(value, block)
            gpu_value = hadamard(value.cuda(), block)
            errors = []
            for mode in SCALE_SEARCH:
                cpu = dequantize_nvfp4(quantize_expert(rotated, scale_search=mode), value.shape)
                image = quantize_expert(gpu_value, scale_search=mode)
                self.assertTrue(torch.equal(image, quantize_expert(gpu_value, scale_search=mode)))
                gpu = dequantize_nvfp4(image, value.shape).cpu()
                self.assertLess(float((cpu - gpu).norm() / cpu.norm()), .01)
                errors.append(float((gpu - rotated).norm()))
            for error in errors[1:]:
                self.assertLessEqual(error, errors[0])


if __name__ == '__main__':
    unittest.main()
