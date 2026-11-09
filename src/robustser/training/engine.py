"""One-epoch training and evaluation primitives with explicit metric output."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import ClassificationMetrics, classification_metrics


def set_global_seed(seed: int, *, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _run_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
) -> ClassificationMetrics:
    training = optimizer is not None
    model.train(training)
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in batches:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.size(0)
            targets.append(labels.detach().cpu())
            predictions.append(logits.argmax(dim=1).detach().cpu())
    return classification_metrics(
        total_loss=total_loss,
        targets=torch.cat(targets),
        predictions=torch.cat(predictions),
        num_classes=num_classes,
    )


def train_one_epoch(**kwargs: object) -> ClassificationMetrics:
    return _run_epoch(optimizer=kwargs.pop("optimizer"), **kwargs)  # type: ignore[arg-type]


def evaluate(**kwargs: object) -> ClassificationMetrics:
    return _run_epoch(optimizer=None, **kwargs)  # type: ignore[arg-type]
