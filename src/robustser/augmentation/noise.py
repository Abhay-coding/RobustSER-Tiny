"""MUSAN-backed SNR-controlled waveform mixing for training augmentation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def noise_rows_from_manifest(manifest_path: str | Path, *, split: str) -> list[dict[str, str]]:
    """Return only valid MUSAN files assigned to the requested noise partition."""
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["noise_split"] == split
        and row["valid"].lower() == "true"
        and row["category"] in {"music", "noise", "speech"}
    ]
    if not selected:
        raise ValueError(f"No valid MUSAN rows for noise split {split!r}")
    return selected


def noise_segment(noise: np.ndarray, *, length: int, rng: np.random.Generator) -> np.ndarray:
    """Select a random noise region, repeating short sources when necessary."""
    values = np.asarray(noise, dtype=np.float32).reshape(-1)
    if not values.size:
        raise ValueError("Noise source is empty")
    if values.size >= length:
        start = int(rng.integers(0, values.size - length + 1))
        return values[start : start + length]
    repeats = int(np.ceil(length / values.size))
    return np.tile(values, repeats)[:length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, *, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add a random noise segment at the specified RMS signal-to-noise ratio.

    A higher positive SNR retains more clean speech. The clean waveform is not
    normalized; output is clipped only to the valid PCM float range.
    """
    clean_values = np.asarray(clean, dtype=np.float32).reshape(-1)
    segment = noise_segment(noise, length=clean_values.size, rng=rng)
    clean_rms = float(np.sqrt(np.mean(np.square(clean_values))))
    noise_rms = float(np.sqrt(np.mean(np.square(segment))))
    if clean_rms == 0.0 or noise_rms == 0.0:
        return clean_values.copy()
    scale = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    return np.clip(clean_values + segment * scale, -1.0, 1.0).astype(np.float32)
