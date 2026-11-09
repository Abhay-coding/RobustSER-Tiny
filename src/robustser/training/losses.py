"""Loss functions used by pre-registered RobustSER experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class FocalCrossEntropy(nn.Module):
    """Multi-class focal loss with optional fixed class weights."""

    def __init__(self, *, gamma: float = 1.5, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_item = functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        probability = torch.exp(-per_item)
        return ((1.0 - probability).pow(self.gamma) * per_item).mean()
