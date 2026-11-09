import unittest

import torch
from torch import nn

from robustser.training.engine import train_one_epoch
from robustser.training.metrics import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_scores(self):
        metrics = classification_metrics(
            total_loss=0.0,
            targets=torch.tensor([0, 1, 2]),
            predictions=torch.tensor([0, 1, 2]),
            num_classes=3,
        )
        self.assertEqual(1.0, metrics.accuracy)
        self.assertEqual(1.0, metrics.macro_f1)
        self.assertEqual(1.0, metrics.uar)

    def test_missing_prediction_class_has_zero_precision_not_nan(self):
        metrics = classification_metrics(
            total_loss=3.0,
            targets=torch.tensor([0, 1, 2]),
            predictions=torch.tensor([0, 0, 0]),
            num_classes=3,
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(metrics.per_class_precision))))
        self.assertEqual(0.0, metrics.per_class_precision[1])

    def test_epoch_runner_updates_a_small_model(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))]
        before = model.weight.detach().clone()

        metrics = train_one_epoch(
            model=model,
            batches=batches,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            optimizer=optimizer,
            num_classes=2,
        )

        self.assertNotEqual(before.tolist(), model.weight.detach().tolist())
        self.assertGreaterEqual(metrics.macro_f1, 0.0)
