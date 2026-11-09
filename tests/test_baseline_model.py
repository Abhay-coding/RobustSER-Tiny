import unittest

import torch

from robustser.models import DepthwiseSeparableCNN, PaperInspiredSmallCNN


class PaperInspiredSmallCNNTests(unittest.TestCase):
    def test_output_shape_matches_number_of_classes(self):
        model = PaperInspiredSmallCNN(num_classes=6)
        logits = model(torch.zeros((3, 1, 32, 31)))

        self.assertEqual((3, 6), tuple(logits.shape))

    def test_rejects_invalid_feature_shape(self):
        model = PaperInspiredSmallCNN(num_classes=6)
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.zeros((1, 32, 31)))

    def test_depthwise_separable_model_is_compact_and_has_valid_shape(self):
        model = DepthwiseSeparableCNN(num_classes=6)
        logits = model(torch.zeros((3, 1, 32, 31)))
        parameters = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual((3, 6), tuple(logits.shape))
        self.assertLess(parameters, 1_000)
