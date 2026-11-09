"""Temperature scaling and expected calibration error for fixed classifiers."""

from __future__ import annotations

import torch
from torch import nn


def softmax_confidence(logits: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    return probabilities.max(dim=1)


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, *, temperature: float = 1.0, bins: int = 15) -> float:
    confidence, predictions = softmax_confidence(logits, temperature)
    correct = predictions.eq(labels)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1 else (confidence <= high))
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].float().mean() - correct[mask].float().mean()).abs()
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature on validation logits without changing model weights."""
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    loss_function = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_function(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp())
