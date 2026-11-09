"""Explicit E0 CNN assumption; this is not claimed to be the paper's exact model."""

from __future__ import annotations

import torch
from torch import nn


class PaperInspiredSmallCNN(nn.Module):
    """Two-block CNN for one-channel 32x31 log-Mel feature images."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two output classes are required")
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 7, 16),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(16, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (1, 32, 31):
            raise ValueError(f"Expected [batch, 1, 32, 31], received {tuple(features.shape)}")
        return self.classifier(self.features(features))
