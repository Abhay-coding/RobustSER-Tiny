"""Dependency-light log-Mel extraction for the frozen E0 baseline contract."""

from __future__ import annotations

import math

import numpy as np


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
FRAME_SAMPLES = 512
HOP_SAMPLES = 512
FFT_SIZE = 512
MEL_BINS = 32
LOG_FLOOR = 1e-6


def fit_audio_window(
    audio: np.ndarray,
    *,
    target_samples: int = WINDOW_SAMPLES,
    policy: str = "center_crop_or_zero_pad",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a fixed-length mono window without changing the raw recording.

    Evaluation uses the center crop. Training may use a supplied seeded RNG for
    a reproducible random crop. Short recordings are right-padded with zeros.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size <= target_samples:
        return np.pad(samples, (0, target_samples - samples.size))
    if policy == "center_crop_or_zero_pad":
        start = (samples.size - target_samples) // 2
    elif policy == "seeded_random_crop_or_zero_pad":
        if rng is None:
            raise ValueError("A NumPy random Generator is required for seeded random cropping")
        start = int(rng.integers(0, samples.size - target_samples + 1))
    else:
        raise ValueError(f"Unknown window policy: {policy}")
    return samples[start : start + target_samples]


def hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int = SAMPLE_RATE,
    fft_size: int = FFT_SIZE,
    mel_bins: int = MEL_BINS,
    min_hz: float = 20.0,
    max_hz: float | None = None,
) -> np.ndarray:
    """Create a triangular HTK Mel filterbank with shape ``[mel_bins, fft_bins]``."""
    if max_hz is None:
        max_hz = sample_rate / 2
    if not 0 <= min_hz < max_hz <= sample_rate / 2:
        raise ValueError("Mel frequency limits must lie within the Nyquist range")
    mel_points = np.linspace(hz_to_mel(np.array([min_hz]))[0], hz_to_mel(np.array([max_hz]))[0], mel_bins + 2)
    hz_points = mel_to_hz(mel_points)
    fft_frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    filters = np.zeros((mel_bins, fft_frequencies.size), dtype=np.float32)
    for index in range(mel_bins):
        left, center, right = hz_points[index : index + 3]
        rising = (fft_frequencies - left) / (center - left)
        falling = (right - fft_frequencies) / (right - center)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def extract_log_mel(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
    filterbank: np.ndarray | None = None,
) -> np.ndarray:
    """Extract E0's normalized log-Mel feature matrix of shape ``[32, 31]``."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"E0 requires {SAMPLE_RATE} Hz audio, received {sample_rate} Hz")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < FRAME_SAMPLES:
        raise ValueError(f"Audio must contain at least {FRAME_SAMPLES} samples, received {samples.size}")
    frame_count = 1 + (samples.size - FRAME_SAMPLES) // HOP_SAMPLES
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME_SAMPLES)[::HOP_SAMPLES]
    if frames.shape[0] != frame_count:
        raise RuntimeError("Unexpected frame construction result")
    window = np.hanning(FRAME_SAMPLES).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=FFT_SIZE, axis=1)
    power = (np.abs(spectrum) ** 2 / FFT_SIZE).astype(np.float32)
    filters = mel_filterbank() if filterbank is None else np.asarray(filterbank, dtype=np.float32)
    if filters.shape != (MEL_BINS, FFT_SIZE // 2 + 1):
        raise ValueError(f"Expected filterbank shape {(MEL_BINS, FFT_SIZE // 2 + 1)}, got {filters.shape}")
    features = np.log(np.maximum(filters @ power.T, LOG_FLOOR)).astype(np.float32)
    if normalize:
        features = normalize_features(features)
    if features.shape != (MEL_BINS, frame_count):
        raise RuntimeError(f"Feature-shape contract failed: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise RuntimeError("Feature matrix contains non-finite values")
    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Apply E0/E3 per-utterance mean/standard-deviation normalization."""
    values = np.asarray(features, dtype=np.float32)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def extract_mfcc(audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE, normalize: bool = True) -> np.ndarray:
    """Extract 32 MFCCs from E0's fixed 32x31 log-Mel frame grid.

    Keeping all 32 DCT coefficients preserves E0's 32x31 input shape, so E3
    changes only feature representation and not the CNN architecture.
    """
    log_mel = extract_log_mel(audio, sample_rate=sample_rate, normalize=False)
    index = np.arange(MEL_BINS, dtype=np.float32)
    coefficients = np.arange(MEL_BINS, dtype=np.float32)[:, None]
    dct = np.cos(np.pi / MEL_BINS * (index + 0.5) * coefficients).astype(np.float32)
    dct[0] *= 1.0 / np.sqrt(MEL_BINS)
    dct[1:] *= np.sqrt(2.0 / MEL_BINS)
    features = (dct @ log_mel).astype(np.float32)
    if normalize:
        features = normalize_features(features)
    if features.shape != log_mel.shape or not np.all(np.isfinite(features)):
        raise RuntimeError("MFCC feature-shape or finiteness contract failed")
    return features
