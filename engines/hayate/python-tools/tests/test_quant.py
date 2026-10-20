import unittest

import torch

from hayate_tools.hyt import NVFP4, PER_TENSOR, E4M3, Q4_BF16, align, storage_bytes
from hayate_tools.quant import (FP4_VALUES, dequantize_nvfp4, quantize, scale_offset,
                                tile_scales, untile_scales)


def nearest(values, table):
    distances = (values.abs().unsqueeze(-1) - table).abs()
    code = distances.argmin(-1)
    # Equal-distance ties select the even code.
    following = (code + 1).clamp_max(table.numel() - 1)
    tied = (distances.gather(-1, code.unsqueeze(-1)) == distances.gather(-1, following.unsqueeze(-1))).squeeze(-1)
    code += (code % 2 == 1) & (code < table.numel() - 1) & tied
    return table[code] * torch.where(torch.signbit(values), -1., 1.)


def decoded_and_bound(data, value, spec):
    """Independent F64 decoder and RTN bound, using the actual stored scales."""
    x = value.double()
    rows, cols = x.shape[-2:]
    if spec[0] == NVFP4:
        payload_len = rows * cols // 2
        matrices = data.reshape(-1, storage_bytes([rows, cols], spec))
        payload = matrices[:, :payload_len]
        codes = torch.stack((payload & 15, payload >> 4), -1).reshape(-1, rows, cols // 16, 16)
        lut = torch.tensor(FP4_VALUES, dtype=torch.float64)
        scales = torch.stack([untile_scales(m[payload_len:-4], rows, cols // 16).double() for m in matrices])
        global_scale = matrices[:, -4:].contiguous().view(torch.float32).double().reshape(-1, 1, 1)
        step = scales * global_scale
        maximum = x.reshape(-1, rows, cols // 16, 16).abs().amax(-1)
        bound = step + (maximum - 6 * step).clamp_min(0)
        decoded = lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1., 1.) * step.unsqueeze(-1)
        return decoded.reshape(x.shape), bound.unsqueeze(-1).expand_as(decoded).reshape(x.shape)
    if spec[0] == Q4_BF16:
        payload_len = x.numel() // 2
        codes = torch.stack((data[:payload_len] & 15, data[:payload_len] >> 4), -1).reshape(-1, 32).double() - 8
        scales = data[payload_len:].contiguous().view(torch.bfloat16).double().reshape(-1, 1)
        maximum = x.reshape(-1, 32).abs().amax(-1, keepdim=True)
        bound = .5 * scales + (maximum - 7 * scales).clamp_min(0)
        return (codes * scales).reshape(x.shape), bound.expand_as(codes).reshape(x.shape)
    scale = data[-4:].clone().view(torch.float32).double().item()
    decoded = data[:-4].view(torch.float8_e4m3fn).double().reshape(x.shape) * scale
    return decoded, x.abs() / 16 + scale / 1024


def reference(value, spec):
    """Small F64 RTN reference; no production packing or normalization code."""
    x = value.double()
    e4m3 = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    if spec[0] == NVFP4:
        rows, cols = x.shape[-2:]
        blocks = x.reshape(-1, rows, cols // 16, 16)
        maximum = blocks.abs().amax((-3, -2, -1), keepdim=True)
        global_scale = torch.where(maximum == 0, 1., maximum / 2688).float().double()
        block_max = blocks.abs().amax(-1, keepdim=True)
        step = nearest(block_max / global_scale / 6, e4m3) * global_scale
        normalized = blocks / torch.where(step == 0, 1., step)
        decoded = nearest(normalized, torch.tensor(FP4_VALUES, dtype=torch.float64)) * step
        bound = step + (block_max - 6 * step).clamp_min(0)
        return decoded.reshape(x.shape), bound.expand_as(decoded).reshape(x.shape)
    if spec[0] == Q4_BF16:
        groups = x.reshape(-1, 32)
        maximum = groups.abs().amax(-1, keepdim=True)
        scale = torch.where(maximum == 0, 1., maximum / 7).to(torch.bfloat16).double()
        decoded = (groups / scale).round().clamp(-8, 7) * scale
        bound = .5 * scale + (maximum - 7 * scale).clamp_min(0)
        return decoded.reshape(x.shape), bound.expand_as(decoded).reshape(x.shape)
    maximum = x.abs().amax()
    scale = torch.where(maximum == 0, 1., maximum / 448).float().double()
    return nearest(x / scale, e4m3) * scale, x.abs() / 16 + scale / 1024


class QuantTests(unittest.TestCase):
    def check_reference_error(self, value, data, spec):
        decoded, bound = decoded_and_bound(data, value, spec)
        expected, reference_bound = reference(value, spec)
        tolerance = value.double().abs() * 1e-6 + 1e-12
        self.assertTrue(((decoded - value.double()).abs() <= bound + tolerance).all())
        self.assertTrue(((decoded - expected).abs() <= bound + reference_bound + tolerance).all())

    def test_batched_matrices_have_independent_scale_tiles_and_capacity(self):
        x = torch.randn((2, 3, 129, 80), generator=torch.Generator().manual_seed(31))
        x *= torch.tensor([0., 1e-4, 1., 10., 1e3, 1e4]).reshape(2, 3, 1, 1)
        data = quantize(x, (NVFP4, 0, 0))
        matrices = data.reshape(6, storage_bytes([129, 80], (NVFP4, 0, 0)))
        for matrix, encoded in zip(x.reshape(6, 129, 80), matrices):
            self.assertTrue(torch.equal(encoded, quantize(matrix, (NVFP4, 0, 0))))
            self.assertEqual(encoded[-4:].clone().view(torch.float32).item(),
                             (matrix.abs().max() / 2688).item() if matrix.any() else 1.)
        self.assertEqual(data.numel(), storage_bytes(list(x.shape), (NVFP4, 0, 0)))
        self.check_reference_error(x, data, (NVFP4, 0, 0))
        decoded, _ = decoded_and_bound(data, x, (NVFP4, 0, 0))
        torch.testing.assert_close(dequantize_nvfp4(data, x.shape).double(), decoded, rtol=1e-6, atol=1e-12)

    def test_batched_q4_and_fp8_match_reference(self):
        x = torch.randn((2, 3, 5, 64), generator=torch.Generator().manual_seed(32))
        for spec in ((Q4_BF16, 0, 32), (PER_TENSOR, E4M3, 0)):
            data = quantize(x, spec)
            self.assertEqual(data.numel(), storage_bytes(list(x.shape), spec))
            self.check_reference_error(x, data, spec)
        self.assertTrue(torch.equal(quantize(x, (Q4_BF16, 0, 32)), quantize(x.reshape(-1, 64), (Q4_BF16, 0, 32))))

    def test_division_and_reciprocal_multiply_cross_fp4_midpoints(self):
        # Reproduce the failing seed without CUDA: a one-ULP difference is
        # enough to change RTN codes, even though both have the same error bound.
        x = torch.randn((129, 80), generator=torch.Generator().manual_seed(23)).to(torch.bfloat16).float()
        scale = x.abs().amax() / 2688
        blocks = x.reshape(129, 5, 16)
        scales = (blocks.abs().amax(-1) / scale / 6).to(torch.float8_e4m3fn).float()
        divided = blocks / scale / scales.unsqueeze(-1)
        multiplied = (blocks * scale.reciprocal()) / scales.unsqueeze(-1)
        lut = torch.tensor(FP4_VALUES, dtype=torch.float64)
        self.assertGreater((nearest(divided.double(), lut) != nearest(multiplied.double(), lut)).sum().item(), 0)
        self.assertLessEqual((divided - multiplied).abs().max().item(), 1e-6)

    def test_nvfp4_error_bound_including_rounded_scale_and_clipping(self):
        generator = torch.Generator().manual_seed(17)
        x = torch.randn((129, 80), generator=generator) * torch.logspace(-4, 4, 129)[:, None]
        data = quantize(x, (NVFP4, 0, 0))
        decoded = dequantize_nvfp4(data, x.shape)
        scales = untile_scales(data[x.numel() // 2:-4], 129, 5).float()
        global_scale = data[-4:].clone().view(torch.float32)[0]
        # Largest E2M1 gap is 2 (4..6): RTN <= one decoded scale.
        # A scale rounded downward may additionally clip the block maximum.
        maximum = x.reshape(129, 5, 16).abs().amax(-1)
        block_scale = scales * global_scale
        bound = block_scale + (maximum - 6 * block_scale).clamp_min(0)
        error = (x - decoded).abs().reshape(129, 5, 16)
        self.assertTrue((error <= bound.unsqueeze(-1) + maximum.unsqueeze(-1) * 1e-6).all())
        self.assertEqual(data.numel(), storage_bytes(list(x.shape), (NVFP4, 0, 0)))

    def test_nvfp4_codes_ties_signs_and_nibble_order(self):
        # The final 2688 forces a global scale of 1; this block scale is 1.
        x = torch.tensor([[0., .25, .5, .75, 1., 1.25, 1.5, 1.75,
                           2., 2.5, 3., 3.5, 4., 5., 6., -0., *([2688.] * 16)]])
        data = quantize(x, (NVFP4, 0, 0))
        expected = [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8]
        self.assertEqual(data[:8].tolist(), [a | (b << 4) for a, b in zip(expected[::2], expected[1::2])])
        self.assertEqual(data[-4:].clone().view(torch.float32)[0].item(), 1.)
        self.assertEqual(data[16 + scale_offset(0, 0, 4)].item(), 56)  # E4M3 1
        self.assertEqual(data[16 + scale_offset(0, 1, 4)].item(), 126)  # E4M3 448
        self.assertEqual(tuple(dequantize_nvfp4(data, x.shape)[0, 2:16:2].tolist()), FP4_VALUES[1:])

    def test_scale_bytes_match_formula_and_zero_padding(self):
        rows, cols = 259, 9
        raw = (torch.arange(rows * cols) % 126).to(torch.uint8).reshape(rows, cols)
        tiled = tile_scales(raw.view(torch.float8_e4m3fn))
        used = set()
        for row in range(rows):
            for col in range(cols):
                offset = ((row // 128) * (align(cols, 4) // 4) + col // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + col % 4
                self.assertEqual(scale_offset(row, col, align(cols, 4)), offset)
                self.assertEqual(tiled[offset].item(), raw[row, col].item())
                used.add(offset)
        self.assertEqual(len(used), rows * cols)
        for offset in set(range(tiled.numel())) - used:
            self.assertEqual(tiled[offset].item(), 0)
        self.assertTrue(torch.equal(untile_scales(tiled, rows, cols).view(torch.uint8), raw))

    def test_zero_and_subnormal_inputs_are_finite(self):
        for mode in ((NVFP4, 0, 0), (PER_TENSOR, E4M3, 0), (Q4_BF16, 0, 32)):
            for magnitude in (0., 1e-40):
                x = torch.full((1, 32), magnitude)
                data = quantize(x, mode)
                self.assertEqual(data.numel(), storage_bytes([1, 32], mode))
                if mode[0] == NVFP4:
                    decoded = dequantize_nvfp4(data, x.shape)
                    self.assertTrue(torch.isfinite(decoded).all())
                    self.assertTrue((decoded - x).abs().max() <= 1e-39)

    def test_q4_uses_stored_bf16_scales_and_plus_eight_codes(self):
        x = torch.linspace(-7., 7., 128).reshape(2, 64)
        data = quantize(x, (Q4_BF16, 0, 32))
        codes = torch.stack((data[:64] & 15, data[:64] >> 4), -1).flatten().float() - 8
        scales = data[64:].view(torch.bfloat16).float()
        decoded = (codes.reshape(4, 32) * scales[:, None]).reshape(2, 64)
        maximum = x.reshape(4, 32).abs().amax(-1)
        bound = .5 * scales + (maximum - 7 * scales).clamp_min(0)
        self.assertTrue(((x - decoded).abs().reshape(4, 32) <= bound[:, None] + 1e-6).all())
        self.assertEqual(data.numel(), 72)
        self.assertEqual((data[0] & 15).item(), 1)

    def test_fp8_scale_finite_codes_signed_zero_and_saturation(self):
        x = torch.tensor([[0., -0., 448., -448., 1., -1.]])
        data = quantize(x, (PER_TENSOR, E4M3, 0))
        self.assertEqual(data[:6].tolist(), [0, 128, 126, 254, 56, 184])
        self.assertEqual(data[-4:].clone().view(torch.float32).item(), 1.)
        self.assertEqual(quantize(torch.zeros((1, 32)), (PER_TENSOR, E4M3, 0))[-4:].view(torch.float32).item(), 1.)

    def test_rejects_invalid_inputs(self):
        for spec in ((NVFP4, 0, 0), (PER_TENSOR, E4M3, 0), (Q4_BF16, 0, 32)):
            for bad in (float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    quantize(torch.full((1, 32), bad), spec)
        for shape, spec in [((2, 17), (NVFP4, 0, 0)), ((2, 16), (Q4_BF16, 0, 32)),
                            ((32,), (NVFP4, 0, 0))]:
            with self.assertRaises(ValueError):
                quantize(torch.zeros(shape), spec)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA hardware")
    def test_cuda_repeatable_bytes_and_reference_error(self):
        x = torch.randn((129, 80), generator=torch.Generator().manual_seed(23)).to(torch.bfloat16)
        experts = torch.stack((x, x * .01, x * 100))
        for source in (x, experts):
            for spec in ((NVFP4, 0, 0), (PER_TENSOR, E4M3, 0), (Q4_BF16, 0, 32)):
                value = source if spec[0] != Q4_BF16 else source[..., :64].contiguous()
                gpu = value.cuda()
                data = quantize(gpu, spec).cpu()
                for _ in range(5):
                    self.assertTrue(torch.equal(data, quantize(gpu, spec).cpu()))
                self.check_reference_error(value, data, spec)


if __name__ == "__main__":
    unittest.main()
