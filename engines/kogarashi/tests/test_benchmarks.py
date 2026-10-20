import json
from pathlib import Path
import tempfile
import unittest

from kogarashi.benchmarks import BenchmarkRun, compare_greedy, draft_acceptance, observed_rate


class EvidenceTests(unittest.TestCase):
    def test_failed_repeat_keeps_old_success_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = BenchmarkRun(root, {"model": "same"}, ["a"])
            first.record("a", {"seconds": 1})
            first.finish({"tokens_per_second": 1})
            second = BenchmarkRun(root, {"model": "same"}, ["a"])
            second.fail(RuntimeError("GPU failed before generation"))
            self.assertNotEqual(first.path, second.path)
            self.assertFalse((second.path / "summary.json").exists())
            self.assertEqual(json.loads((first.path / "status.json").read_text())["status"], "complete")
            self.assertFalse(json.loads((second.path / "status.json").read_text())["coverage_complete"])

    def test_partial_coverage_and_nonfinite_measurement_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run = BenchmarkRun(Path(directory), {}, ["a", "b"])
            with self.assertRaises(ValueError):
                run.record("a", {"speed": float("nan")})
            run.record("a", {"speed": 2})
            with self.assertRaises(ValueError):
                run.record("a", {"speed": 3})
            with self.assertRaises(ValueError):
                run.finish({})
            self.assertFalse((run.path / "summary.json").exists())

    def test_impossible_draft_counts_and_rates_are_rejected(self):
        for accepted, proposed in [(6, 5), (-1, 2), (0, -1), (float("nan"), 1), (True, 1)]:
            with self.assertRaises(ValueError):
                draft_acceptance(accepted, proposed)
        self.assertIsNone(draft_acceptance(0, 0))
        self.assertEqual(draft_acceptance(2, 4), 0.5)
        for elapsed in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                observed_rate(256, elapsed)
        self.assertEqual(observed_rate(256, 2), 128)

    def test_greedy_requires_exact_tokens_and_active_drafts(self):
        rows = [{"mode": mode, "prompt": "ja", "trial": 0,
                 "response": {"tokens": [1, 2, 3], "timings": {"draft_n_accepted": 2}}}
                for mode in ("none", "draft-mtp")]
        self.assertTrue(compare_greedy(rows)["passed"])
        rows[1]["response"]["tokens"] = [1, 2]
        result = compare_greedy(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["comparisons"][1]["first_mismatch"], 2)
        rows[1]["response"]["tokens"] = [1, 2, 4]
        self.assertFalse(compare_greedy(rows)["passed"])
        rows[1]["response"]["tokens"] = [1, 2, 3]
        rows[1]["response"]["timings"]["draft_n_accepted"] = 0
        self.assertFalse(compare_greedy(rows)["passed"])
        self.assertFalse(compare_greedy(rows[:1])["passed"])
        rows[1]["response"]["tokens"] = []
        with self.assertRaises(ValueError):
            compare_greedy(rows)
        with self.assertRaises(ValueError):
            compare_greedy([rows[1]])


if __name__ == "__main__":
    unittest.main()
