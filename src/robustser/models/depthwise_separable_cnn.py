"""Compact depthwise-separable CNN for the E4 TinyML architecture ablation."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableCNN(nn.Module):
    """A sub-1k-parameter CNN for 1x32x31 acoustic feature maps."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two output classes are required")
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 20, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.20), nn.Linear(20, num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1 or features.shape[2] != 32 or features.shape[3] < 4:
            raise ValueError(f"Expected [batch, 1, 32, time>=4], received {tuple(features.shape)}")
        return self.classifier(self.features(features))
