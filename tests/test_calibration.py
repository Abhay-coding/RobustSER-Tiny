import unittest
import torch
from robustser.evaluation import expected_calibration_error, fit_temperature


class CalibrationTests(unittest.TestCase):
    def test_temperature_is_positive_and_ece_is_bounded(self):
        logits = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
        labels = torch.tensor([0, 1, 1, 1])
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        self.assertGreaterEqual(expected_calibration_error(logits, labels, temperature=temperature), 0.0)
        self.assertLessEqual(expected_calibration_error(logits, labels, temperature=temperature), 1.0)
