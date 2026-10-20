"""Regression coverage for nonfinite upstream scales and experiment reports."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from reference_support import REFERENCE_AVAILABLE, REFERENCE_REASON

import numpy as np

from kogarashi.quant_lab.tq2 import Codec, encode


@unittest.skipUnless(REFERENCE_AVAILABLE, REFERENCE_REASON)
class ReferenceRangeTests(unittest.TestCase):
    def test_invalid_reference_is_explicit_and_json_safe(self):
        codec = Codec()
        for value, reason in ((1e6, "nonfinite_reference_decode"),
                              (1e30, "float32_energy_overflow")):
            block, metrics = encode(codec, np.full(128, value), anchors=1,
                                    scale_factors=(1.0, 1e308), refinement=0)
            self.assertTrue(np.isfinite(codec.decode_c(block)).all())
            self.assertFalse(metrics["reference_valid"])
            self.assertEqual(metrics["reference_status"], reason)
            self.assertIsNone(metrics["reference_loss"])
            json.dumps(metrics, allow_nan=False)

    def test_cli_keeps_comparable_coverage_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mixed.gguf"
            output = Path(directory) / "report.json"
            name = b"blk.0.ffn_up_exps.weight"
            header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
            header += struct.pack("<Q", len(name)) + name
            header += struct.pack("<IQQIQ", 2, 128, 3, 0, 0)
            header += bytes((-len(header)) % 32)
            values = np.concatenate([np.linspace(-1, 1, 128),
                                     np.full(128, 1e6), np.full(128, 1e30)])
            source.write_bytes(header + values.astype("<f4").tobytes())
            command = [sys.executable, "-Werror", "-m", "kogarashi.quant_lab", str(source),
                       "--output", str(output), "--basis", "native", "--per-tensor", "3",
                       "--anchors", "1", "--refinement", "0"]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text(), parse_constant=lambda x: self.fail(x))
            summary = report["summary"]["native"]
            self.assertEqual(summary["blocks"], 3)
            self.assertEqual(summary["paired_blocks"], 1)
            self.assertEqual(summary["invalid_reference_blocks"], 2)
            valid, *invalid = report["records"]
            self.assertEqual(summary["paired_sse"], valid["original_basis_sse"])
            self.assertEqual(summary["reference_sse"], valid["reference_original_basis_sse"])
            self.assertGreater(summary["sse"], summary["paired_sse"])
            for record in invalid:
                self.assertIsNone(record["c_reference_sse"])
                self.assertIsNone(record["reference_hex"])
            original = output.read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
