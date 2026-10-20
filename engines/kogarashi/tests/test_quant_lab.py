import ctypes
import itertools
from pathlib import Path
import struct
import tempfile
import unittest
from reference_support import REFERENCE_AVAILABLE, REFERENCE_REASON

import numpy as np

from kogarashi.quant_lab.gguf_sample import Header, Tensor, read_header, samples
from kogarashi.quant_lab.storage import estimate
from kogarashi.quant_lab.tq2 import Codec, encode, hadamard, solve_cycles


@unittest.skipUnless(REFERENCE_AVAILABLE, REFERENCE_REASON)
class TrellisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = Codec()

    def test_every_decoder_state_matches_c(self):
        # Every 16-bit raw state, including the masked high bit of the predecessor.
        states = np.arange(65536, dtype=np.uint32)
        blocks = np.zeros((len(states), 34), dtype=np.uint8)
        blocks[:, :2] = np.frombuffer(struct.pack("<e", 0.371), dtype=np.uint8)
        blocks[:, 2], blocks[:, 3] = states >> 8, states & 255
        output = np.empty((len(states), 128), dtype=np.float32)
        self.codec.library.dequantize_row_tq2_t(blocks.ctypes.data, output.ctypes.data, len(states)*128)
        expected = self.codec.edges[(states >> 8) & 127, states & 255].astype(np.float32)
        expected *= np.float32(struct.unpack("<e", blocks[0, :2].tobytes())[0])
        np.testing.assert_array_equal(output[:, 4:8], expected)

    def test_python_decoder_wraparound_and_scales(self):
        rng = np.random.default_rng(10)
        for scale in (0, -0.3, 1, 65504, 2**-24):
            for _ in range(5):
                block = struct.pack("<e", scale) + rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
                np.testing.assert_array_equal(self.codec.decode(block), self.codec.decode_c(block))

    def test_closed_viterbi_against_exhaustive_oracle(self):
        rng = np.random.default_rng(9)
        for steps, states in ((1, 3), (3, 3), (4, 4)):
            costs = rng.uniform(size=(steps, states, states))
            paths, losses = solve_cycles(costs, range(states))
            for anchor in range(states):
                possibilities = [p for p in itertools.product(range(states), repeat=steps) if p[-1] == anchor]
                scores = [sum(costs[t, p[t-1], p[t]] for t in range(steps)) for p in possibilities]
                self.assertAlmostEqual(losses[anchor], min(scores), places=13)
                p = paths[anchor]
                self.assertEqual(p[-1], anchor)
                self.assertAlmostEqual(losses[anchor], sum(costs[t, p[t-1], p[t]] for t in range(steps)), places=13)

    def test_free_start_is_not_a_cyclic_solution(self):
        costs = np.array([[[9, 0], [9, 9]], [[9, 9], [9, 0]]], dtype=float)
        # The zero-cost open path 0 -> 1 -> 1 cannot close at zero cost.
        _, loss = solve_cycles(costs, (0, 1))
        self.assertEqual(min(loss), 9)

    def test_no_regression_against_stored_reference(self):
        rng = np.random.default_rng(42)
        fixtures = [rng.normal(size=128), rng.laplace(size=128), np.zeros(128),
                    np.full(128, 1e-38), np.eye(1, 128, 7).reshape(128)*30]
        for x in fixtures:
            for weight in (None, np.linspace(0, 4, 128)**3):
                block, meta = encode(self.codec, x, weight, anchors=2, refinement=0)
                self.assertEqual(len(block), 34)
                q = self.codec.decode_c(block).astype(float)
                ref = self.codec.decode_c(self.codec.reference(x)).astype(float)
                metric = np.ones(128) if weight is None else weight
                target = np.asarray(x, dtype=np.float32).astype(float)
                self.assertLessEqual(np.dot(metric, (q-target)**2), np.dot(metric, (ref-target)**2)+1e-12)
                np.testing.assert_array_equal(q, self.codec.decode(block))
                self.assertTrue(np.isfinite(q).all())

    def test_weighted_search_and_determinism(self):
        x = np.random.default_rng(87).normal(size=128)
        weight = np.r_[np.ones(64), np.full(64, 100.0)]
        a, ma = encode(self.codec, x, weight, anchors=2, refinement=0)
        b, mb = encode(self.codec, x, weight, anchors=2, refinement=0)
        self.assertEqual((a, ma), (b, mb))
        self.assertLess(ma["loss"], ma["reference_loss"])

    def test_hadamard_forward_identity(self):
        rng = np.random.default_rng(8)
        w, x = rng.normal(size=(5, 128)), rng.normal(size=128)
        signs = rng.choice((-1, 1), size=128)
        np.testing.assert_allclose(hadamard(hadamard(x)), x, atol=1e-14)
        np.testing.assert_allclose(hadamard(w*signs) @ hadamard(x*signs), w @ x, atol=1e-13)

    def test_bad_inputs_and_large_values(self):
        for x in ([0]*127, [np.nan]*128, [np.inf]*128):
            with self.assertRaises(ValueError):
                encode(self.codec, x)
        for weight in (np.zeros(128), -np.ones(128), np.full(128, np.nan), [1]):
            with self.assertRaises(ValueError):
                encode(self.codec, np.ones(128), weight)
        for opts in ({"anchors": 0}, {"anchors": 129}, {"scale_factors": ()},
                     {"scale_factors": (np.nan,)}, {"refinement": -1}):
            with self.assertRaises(ValueError):
                encode(self.codec, np.ones(128), **opts)
        block, meta = encode(self.codec, np.full(128, 1e30), anchors=1, refinement=0)
        self.assertTrue(np.isfinite(self.codec.decode_c(block)).all())
        self.assertIsNone(meta["reference_loss"])


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _fixture(path, kind=30, truncate=False):
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += _string("general.alignment") + struct.pack("<II", 4, 32)
    header += _string("blk.0.ffn_up_exps.weight") + struct.pack("<IQQIQ", 2, 128, 4, kind, 0)
    header += bytes((-len(header)) % 32)
    values = np.linspace(-1, 1, 512, dtype="<f4")
    if kind == 30:
        raw = (values.view("<u4") >> 16).astype("<u2").tobytes()
    elif kind == 1:
        raw = values.astype("<f2").tobytes()
    else:
        raw = values.tobytes()
    path.write_bytes(header + (raw[:256] if truncate else raw))


class SamplingTests(unittest.TestCase):
    def test_storage_includes_scale_and_preserves_other_tensors(self):
        tensors = tuple(Tensor(f"blk.0.ffn_{role}_exps.weight", (128, 2), 30, 0)
                        for role in ("gate", "up", "down"))
        tensors += (Tensor("output.weight", (128, 2), 30, 0),
                    Tensor("blk.1.ffn_up_exps.weight", (128, 2), 30, 0))
        h = Header(Path("sample.gguf"), 0, 0, {}, tensors, "fixture")
        result = estimate(h, trunk_layers=1)
        self.assertEqual(result["preserved_payload_bytes"], 1024)
        self.assertEqual(result["recipes"]["experts_tq2_t"]["payload_bytes"], 1024+6*34)
        self.assertEqual(result["recipes"]["experts_tq2_t_down_tqp6"]["payload_bytes"], 1024+4*34+2*50)
        self.assertEqual(result["recipes"]["experts_tqp6"]["payload_bytes"], 1024+6*50)

    def test_float_types_and_sparse_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"small.gguf"
            for kind in (0, 1, 30):
                _fixture(path, kind)
                header = read_header(path)
                rows = list(samples(header, r"blk\.0\.ffn_up_exps\.weight", 2))
                self.assertEqual([r[0]["block_index"] for r in rows], [0, 3])
                self.assertEqual(rows[0][1][0], -1)
                self.assertEqual(rows[-1][1][-1], 1)

    def test_partial_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"small.gguf.part"
            _fixture(path, truncate=True)
            header = read_header(path)
            with self.assertRaises(ValueError):
                list(samples(header, ".*"))
            rows = list(samples(header, ".*", allow_partial=True))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0][1]), 128)

    def test_invalid_header_and_quantized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.gguf"
            path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 100001, 1))
            with self.assertRaises(ValueError):
                read_header(path)
            _fixture(path, kind=12)
            with self.assertRaises(ValueError):
                list(samples(read_header(path), ".*", allow_partial=True))


if __name__ == "__main__":
    unittest.main()
