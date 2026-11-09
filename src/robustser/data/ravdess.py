"""External-only RAVDESS evaluation dataset."""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from robustser.data.cremad import TASK_LABELS, extract_features
from robustser.features.log_mel import SAMPLE_RATE, fit_audio_window


def read_resampled_pcm16_mono(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM WAV: {path}")
        rate = source.getframerate()
        channels = source.getnchannels()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate == SAMPLE_RATE:
        return audio
    divisor = math.gcd(rate, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)


class RAVDESSExternalDataset(Dataset[tuple[torch.Tensor, int]]):
    """Never-train external corpus evaluation for the shared emotion labels."""

    def __init__(self, manifest_path: str | Path, *, task: str, feature_type: str, window_seconds: float = 1.0) -> None:
        if task not in TASK_LABELS:
            raise ValueError(f"Unknown task: {task}")
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        labels = set(TASK_LABELS[task])
        self.rows = [row for row in rows if row["emotion"] in labels]
        if not self.rows:
            raise ValueError("No RAVDESS rows match the requested task")
        self.label_to_index = {label: index for index, label in enumerate(TASK_LABELS[task])}
        self.feature_type = feature_type
        self.window_samples = round(window_seconds * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        audio = fit_audio_window(read_resampled_pcm16_mono(row["path"]), target_samples=self.window_samples, policy="center_crop_or_zero_pad")
        return torch.from_numpy(extract_features(audio, self.feature_type)).unsqueeze(0), self.label_to_index[row["emotion"]]
