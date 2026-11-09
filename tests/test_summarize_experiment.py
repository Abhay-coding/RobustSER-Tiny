import unittest

from scripts.summarize_experiment import aggregate


class SummarizeExperimentTests(unittest.TestCase):
    def test_single_run_does_not_fabricate_variation(self):
        result = aggregate([0.4])
        self.assertEqual(1, result["n"])
        self.assertIsNone(result["sample_std"])
        self.assertIsNone(result["ci95_half_width"])

    def test_multiple_runs_have_sample_variation(self):
        result = aggregate([0.3, 0.4, 0.5])
        self.assertEqual(3, result["n"])
        self.assertGreater(result["sample_std"], 0)
        self.assertGreater(result["ci95_half_width"], 0)

    def test_interval_shrinks_with_more_identical_observations(self):
        result = aggregate([0.4, 0.4, 0.4])
        self.assertEqual(0.0, result["sample_std"])
        self.assertEqual(0.0, result["ci95_half_width"])
