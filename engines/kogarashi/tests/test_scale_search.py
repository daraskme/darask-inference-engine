import struct
import unittest
from reference_support import REFERENCE_AVAILABLE, REFERENCE_REASON
from unittest.mock import patch

import numpy as np

from kogarashi.quant_lab import scale_search
from kogarashi.quant_lab.tq2 import Codec, encode


@unittest.skipUnless(REFERENCE_AVAILABLE, REFERENCE_REASON)
class ScaleSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = Codec()

    def test_same_metric_nonregression_and_stored_bytes(self):
        rng = np.random.default_rng(73)
        for amplitude in (1e-30, 0.03, 1e20):
            x = (rng.normal(size=128) * amplitude).astype(np.float32)
            h = np.geomspace(1e-300, 1e300, 128)
            h[:17] = 0
            block, meta = scale_search.encode_scale_search(
                self.codec, x, h, anchors=1, grid=(0.5, 2.0, 3), refine_top=1)
            base, _ = encode(self.codec, x, h, anchors=1)
            self.assertEqual(base.hex(), meta["baseline_hex"])
            self.assertEqual(len(block), 34)
            self.assertLessEqual(meta["loss"], meta["baseline_loss"])
            w = h / h.max()
            err = self.codec.decode_c(block).astype(np.float64) - x.astype(np.float64)
            self.assertEqual(float(np.dot(w, err * err)), meta["loss"])
            keys = [t["initial_dp_scale_hex"] for t in meta["trials"]]
            self.assertEqual(len(keys), len(set(keys)))
            if amplitude == 1e-30:
                self.assertEqual(len(keys), 1)
                self.assertEqual(meta["trials"][0]["initial_dp_scale"], 2 ** -24)

    def test_zero_has_no_search_and_exact_fallback(self):
        block, meta = scale_search.encode_scale_search(self.codec, np.zeros(128))
        self.assertEqual(block, bytes(34))
        self.assertEqual(meta["encoder_calls"], 1)
        self.assertEqual(meta["trials"], [])
        self.assertEqual(meta["winner"], "baseline")

    def test_all_calls_propagate_anchors_and_ties_keep_baseline(self):
        x = np.linspace(-1, 1, 128).astype(np.float32)
        original = self.codec.reference(x)
        for anchors in (1, 128):
            with patch.object(scale_search, "encode", return_value=(original, {})) as mocked:
                block, _ = scale_search.encode_scale_search(self.codec, x, anchors=anchors)
            self.assertEqual(block, original)
            self.assertGreater(mocked.call_count, 1)
            for call in mocked.call_args_list:
                self.assertEqual(call.kwargs["anchors"], anchors)

    def test_decoder_mismatch_fails_closed(self):
        with patch.object(self.codec, "decode_c", return_value=np.ones(128, dtype=np.float32)):
            with self.assertRaisesRegex(RuntimeError, "bit-inconsistent"):
                scale_search.encode_scale_search(self.codec, np.zeros(128))

    def test_validation(self):
        for kw in ({"anchors": 0}, {"grid": (0, 2, 9)}, {"grid": (1, np.inf, 9)},
                   {"grid": (1, 2, 2.5)}, {"refine_top": -1}, {"refine_top": 10},
                   {"grid": (2, 1, 9)}):
            with self.assertRaises(ValueError):
                scale_search.encode_scale_search(self.codec, np.zeros(128), **kw)
        for h in (np.zeros(128), np.full(128, np.nan), np.full(128, -1)):
            with self.assertRaises(ValueError):
                scale_search.encode_scale_search(self.codec, np.zeros(128), h)

    def test_plateau_rules_and_fp16_limits(self):
        self.assertEqual(scale_search._local_minima([1, 1, 2, 1, 1]), [0, 3])
        self.assertEqual(scale_search._local_minima([2, 1, 1, 2]), [1])
        self.assertEqual(scale_search._local_minima([1]), [0])
        self.assertEqual(scale_search._trial_scale(1e-30, 0.5), 2 ** -24)
        self.assertEqual(scale_search._trial_scale(1e38, 1e300), 65504)
        self.assertEqual(struct.pack("<e", scale_search._trial_scale(1e-30, 2)), b"\x01\x00")


if __name__ == "__main__":
    unittest.main()
