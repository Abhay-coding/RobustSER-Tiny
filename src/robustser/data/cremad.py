"""Leakage-safe CREMA-D access based only on the frozen split manifest."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from robustser.augmentation.noise import mix_at_snr, noise_rows_from_manifest
from robustser.features.log_mel import SAMPLE_RATE, extract_log_mel, extract_mfcc, fit_audio_window


TASK_LABELS = {
    "five_emotion_common": ("Angry", "Disgust", "Fear", "Happy", "Neutral"),
    "crema_d_six_emotion": ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"),
}
VALID_SPLITS = frozenset(("train", "validation", "test"))
FEATURE_EXTRACTORS = {
    "log_mel_filterbank_energy": extract_log_mel,
    "mfcc": extract_mfcc,
}


def extract_features(audio: np.ndarray, feature_type: str) -> np.ndarray:
    try:
        return FEATURE_EXTRACTORS[feature_type](audio)
    except KeyError as error:
        raise ValueError(f"Unsupported feature type: {feature_type}") from error


def load_split_rows(manifest_path: str | Path, *, split: str, task: str) -> list[dict[str, str]]:
    """Read only one frozen split and filter it to the requested label task."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task}")
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "speaker_id", "emotion", "split", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Split manifest is empty or lacks required columns")
    labels = set(TASK_LABELS[task])
    selected = [row for row in rows if row["split"] == split and row["emotion"] in labels]
    if not selected:
        raise ValueError(f"No rows for split={split!r}, task={task!r}")
    missing_labels = labels - {row["emotion"] for row in selected}
    if missing_labels:
        raise ValueError(f"Split {split!r} is missing task labels: {sorted(missing_labels)}")
    return selected


def read_pcm16_mono(path: str | Path) -> np.ndarray:
    """Read a CREMA-D PCM16 mono WAV into normalized float samples."""
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV: {path}")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit mono PCM WAV: {path}")
        raw = source.readframes(source.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def read_pcm16_mono_segment(path: str | Path, *, length: int, rng: np.random.Generator) -> np.ndarray:
    """Read only a random fixed-length PCM segment, avoiding full MUSAN loads."""
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != SAMPLE_RATE or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected 16 kHz 16-bit mono PCM WAV: {path}")
        frames = source.getnframes()
        if frames >= length:
            source.setpos(int(rng.integers(0, frames - length + 1)))
            raw = source.readframes(length)
        else:
            raw = source.readframes(frames)
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if not values.size:
        raise ValueError(f"Noise source is empty: {path}")
    return np.tile(values, int(np.ceil(length / values.size)))[:length]


class CREMADSERDataset(Dataset[tuple[torch.Tensor, int]]):
    """Map-style PyTorch dataset for one frozen CREMA-D split and task.

    Training crops are deterministic per ``(seed, epoch, row index)``. This
    offers controlled crop variation without changing the manifest or allowing
    random state from a DataLoader worker to affect reproducibility.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        task: str,
        seed: int = 20_261_108,
        noise_manifest_path: str | Path | None = None,
        noise_probability: float = 0.0,
        snr_db_range: tuple[float, float] = (5.0, 20.0),
        feature_type: str = "log_mel_filterbank_energy",
        window_seconds: float = 1.0,
    ) -> None:
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be between zero and one")
        if snr_db_range[0] > snr_db_range[1]:
            raise ValueError("snr_db_range must be ordered low to high")
        if noise_probability and split != "train":
            raise ValueError("Noise augmentation is allowed only for the training split")
        if noise_probability and noise_manifest_path is None:
            raise ValueError("Noise augmentation requires a MUSAN split manifest")
        if feature_type not in FEATURE_EXTRACTORS:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        if window_seconds < 0.032:
            raise ValueError("window_seconds must contain at least one 32 ms frame")
        self.rows = load_split_rows(manifest_path, split=split, task=task)
        self.split = split
        self.task = task
        self.seed = seed
        self.epoch = 0
        self.label_to_index = {label: index for index, label in enumerate(TASK_LABELS[task])}
        self.noise_probability = noise_probability
        self.snr_db_range = snr_db_range
        self.noise_rows = noise_rows_from_manifest(noise_manifest_path, split="augmentation_train") if noise_probability else []
        self.feature_type = feature_type
        self.window_samples = round(window_seconds * SAMPLE_RATE)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _window_policy(self) -> tuple[str, np.random.Generator | None]:
        if self.split != "train":
            return "center_crop_or_zero_pad", None
        return "seeded_random_crop_or_zero_pad", None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        audio = read_pcm16_mono(row["path"])
        policy, _ = self._window_policy()
        rng = None
        if self.split == "train":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        window = fit_audio_window(audio, target_samples=self.window_samples, policy=policy, rng=rng)
        if self.noise_probability and rng is not None and float(rng.random()) < self.noise_probability:
            noise_row = self.noise_rows[int(rng.integers(0, len(self.noise_rows)))]
            noise = read_pcm16_mono_segment(noise_row["relative_path"], length=window.size, rng=rng)
            snr_db = float(rng.uniform(*self.snr_db_range))
            window = mix_at_snr(window, noise, snr_db=snr_db, rng=rng)
        features = extract_features(window, self.feature_type)
        # Conv2d expects [channels, mel_bins, frames] for a single example.
        tensor = torch.from_numpy(features).unsqueeze(0)
        return tensor, self.label_to_index[row["emotion"]]


class NoisyCREMADSERTestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic noisy held-out test set using MUSAN evaluation-only audio."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        task: str,
        noise_manifest_path: str | Path,
        snr_db: float,
        seed: int,
        feature_type: str = "log_mel_filterbank_energy",
        window_seconds: float = 1.0,
    ) -> None:
        if task not in TASK_LABELS:
            raise ValueError(f"Unknown task: {task}")
        self.rows = load_split_rows(manifest_path, split="test", task=task)
        self.noise_rows = noise_rows_from_manifest(noise_manifest_path, split="evaluation_holdout")
        self.label_to_index = {label: index for index, label in enumerate(TASK_LABELS[task])}
        self.snr_db = snr_db
        self.seed = seed
        self.feature_type = feature_type
        self.window_samples = round(window_seconds * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        clean = fit_audio_window(read_pcm16_mono(row["path"]), target_samples=self.window_samples, policy="center_crop_or_zero_pad")
        # Scaling preserves an exact seed identity even for a fractional SNR.
        snr_key = int(round(self.snr_db * 1_000))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, snr_key, index]))
        noise_row = self.noise_rows[int(rng.integers(0, len(self.noise_rows)))]
        noise = read_pcm16_mono_segment(noise_row["relative_path"], length=clean.size, rng=rng)
        noisy = mix_at_snr(clean, noise, snr_db=self.snr_db, rng=rng)
        return torch.from_numpy(extract_features(noisy, self.feature_type)).unsqueeze(0), self.label_to_index[row["emotion"]]
