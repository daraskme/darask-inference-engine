import tempfile
import unittest
from pathlib import Path

import numpy as np

from hayate_tools import glm53_exl3 as x3
from hayate_tools.hyt import BF16, E4M3, EXL3, F32, PER_TENSOR, PLAIN, encode_entry, storage_bytes

FIXTURES = Path(__file__).parent / 'data' / 'exl3-exllamav3'


def random_matrix(n, k, k2, seed):
    rng = np.random.default_rng(seed)
    trellis = rng.integers(-32768, 32768, size=(k // 16, n // 16, 8 * k2), dtype=np.int64).astype(np.int16)
    scale = lambda size: (rng.uniform(.5, 2., size) * rng.choice([-1., 1.], size)).astype(np.float16)  # noqa: E731
    return trellis, scale(k), scale(n)


class GLMExl3(unittest.TestCase):
    def test_decoder_matches_exllamav3_fixtures_bit_for_bit(self):
        results = x3.check_fixtures(FIXTURES)
        self.assertEqual(len(results), 13)
        self.assertEqual({(r['k2'], r['cb']) for r in results} >= {(5, 2), (7, 2), (4, 2), (6, 0), (6, 1)}, True)
        for r in results:
            self.assertEqual((r['bad_states'], r['bad_weights']), (0, 0), r['file'])

    def test_rows_are_the_rotated_product(self):
        for k2, n, k in [(5, 256, 384), (7, 128, 256)]:
            trellis, suh, svh = random_matrix(n, k, k2, k2)
            w = x3.rows(trellis, suh, svh, k2)                       # [n][k]
            self.assertEqual(w.shape, (n, k))
            x = np.random.default_rng(1).standard_normal(k)
            # y = H(H(x * suh) W') * svh, blockwise H
            xh = (x * suh.astype(np.float64)).reshape(-1, 128) @ x3.HAD
            y = (xh.reshape(k) @ x3.inner(trellis, k2).astype(np.float64)).reshape(-1, 128) @ x3.HAD
            y = y.reshape(n) * svh.astype(np.float64)
            np.testing.assert_allclose(w.astype(np.float64) @ x, y, rtol=1e-4, atol=1e-4 * np.abs(y).max())
            # column chunks agree with the whole matrix
            if n >= 256:
                np.testing.assert_allclose(x3.rows(trellis, suh, svh, k2, 2, 128, 128), w[128:256], rtol=1e-6, atol=1e-7)

    def test_hyt_tag7_sizes(self):
        self.assertEqual(storage_bytes([2048, 4096], (EXL3, 5, 2)), 2048 * 4096 * 5 // 16 + 2 * 6144)
        self.assertEqual(2 * storage_bytes([2048, 4096], (EXL3, 5, 2)) + storage_bytes([4096, 2048], (EXL3, 5, 2)),
                         7_901_184)
        for bad in [([2048, 4000], (EXL3, 5, 2)), ([2048, 4096], (EXL3, 5, 1)), ([2048, 4096], (EXL3, 9, 2)),
                    ([4096], (EXL3, 6, 2))]:
            with self.assertRaises(ValueError):
                storage_bytes(*bad)
        encode_entry('mlp.experts.0.gate_proj.weight', [2048, 4096], (EXL3, 7, 2), 0,
                     storage_bytes([2048, 4096], (EXL3, 7, 2)))

    def test_layer_round_trip_and_encodings(self):
        trellis, suh, svh = random_matrix(128, 256, 5, 3)
        value = np.linspace(-2, 2, 128 * 256, dtype=np.float32).reshape(128, 256)
        fp8, lost = x3.encode(value, (PER_TENSOR, E4M3, 0))
        self.assertEqual((len(fp8), lost), (128 * 256 + 4, 0))
        bf, lost = x3.encode(value, (PLAIN, BF16, 0))
        self.assertEqual(len(bf), 128 * 256 * 2)
        self.assertGreater(lost, 0)
        self.assertEqual(x3.encode(np.array([1.5, -2.], np.float32), (PLAIN, BF16, 0))[1], 0)
        with self.assertRaises(ValueError):
            x3.encode(np.array([np.nan], np.float32), (PLAIN, F32, 0))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'layer3' / 'weights.hyt'
            entries = [('mlp.gate.weight', [128, 256], (PLAIN, BF16, 0), bf),
                       ('mlp.experts.0.down_proj.weight', [128, 256], (EXL3, 5, 2), (trellis, suh, svh))]
            x3.write_hyt(path, entries, b'\x42' * 32)
            table = x3.hyt_table(path)
            self.assertEqual(table, [(n, q, s) for n, s, q, _ in entries])
            with self.assertRaises(FileExistsError):
                x3.write_hyt(path, entries, b'\x42' * 32)
            raw = path.read_bytes()
            self.assertIn(trellis.tobytes() + suh.tobytes() + svh.tobytes(), raw)


if __name__ == '__main__':
    unittest.main()
