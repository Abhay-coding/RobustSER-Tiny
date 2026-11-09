"""Dependency-light classification metrics from an integer confusion matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    uar: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(*, total_loss: float, targets: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> ClassificationMetrics:
    """Calculate metrics without external evaluation dependencies."""
    if targets.numel() == 0:
        raise ValueError("Cannot calculate metrics for zero examples")
    targets = targets.to(torch.int64).cpu()
    predictions = predictions.to(torch.int64).cpu()
    bins = torch.bincount(targets * num_classes + predictions, minlength=num_classes**2)
    matrix = bins.reshape(num_classes, num_classes).to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(true_positive))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(true_positive))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    examples = int(targets.numel())
    return ClassificationMetrics(
        loss=total_loss / examples,
        accuracy=float(true_positive.sum() / examples),
        macro_f1=float(f1.mean()),
        balanced_accuracy=float(recall.mean()),
        uar=float(recall.mean()),
        per_class_precision=[float(value) for value in precision],
        per_class_recall=[float(value) for value in recall],
        per_class_f1=[float(value) for value in f1],
        confusion_matrix=[[int(value) for value in row] for row in matrix.to(torch.int64).tolist()],
    )
