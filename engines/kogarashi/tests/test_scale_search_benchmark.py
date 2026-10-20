import unittest

from scripts.benchmark_scale_search import validate_inputs


class SampleCoverageTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [({"group": group, "tensor": tensor, "block_index": i}, None)
                       for group, tensor in (("continuity", "old"), ("holdout", "new"))
                       for i in range(72)]
        self.expected = {("old", i) for i in range(72)}

    def test_exact_disjoint_groups(self):
        validate_inputs(self.inputs, self.expected)

    def test_same_total_with_unequal_groups_is_rejected(self):
        self.inputs[-1][0]["group"] = "continuity"
        with self.assertRaisesRegex(RuntimeError, "72 distinct"):
            validate_inputs(self.inputs, self.expected)

    def test_duplicate_and_wrong_continuity_are_rejected(self):
        self.inputs[0][0]["block_index"] = 1
        with self.assertRaisesRegex(RuntimeError, "72 distinct"):
            validate_inputs(self.inputs, self.expected)
        self.inputs[0][0]["block_index"] = 72
        with self.assertRaisesRegex(RuntimeError, "Continuity keys"):
            validate_inputs(self.inputs, self.expected)

    def test_holdout_overlap_is_rejected(self):
        self.inputs[-1][0]["tensor"] = "old"
        with self.assertRaisesRegex(RuntimeError, "overlaps"):
            validate_inputs(self.inputs, self.expected)


if __name__ == "__main__":
    unittest.main()
