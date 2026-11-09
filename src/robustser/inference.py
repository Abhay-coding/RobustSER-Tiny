"""Local, reproducible inference helpers for the RobustSER-Tiny demo."""

from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import signal
from scipy.io import wavfile

from robustser.data.cremad import FEATURE_EXTRACTORS, TASK_LABELS
from robustser.features.log_mel import SAMPLE_RATE, fit_audio_window
from robustser.models import build_model

MAX_DEMO_DURATION_SECONDS = 60.0


def decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV recording, downmix it, and return normalized float samples."""
    if not audio_bytes:
        raise ValueError("The audio file is empty.")
    try:
        sample_rate, values = wavfile.read(io.BytesIO(audio_bytes))
    except (ValueError, OSError) as error:
        raise ValueError("Please provide a valid WAV audio recording.") from error
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.unsignedinteger):
        scale = float(np.iinfo(values.dtype).max + 1) / 2.0
        audio = (values.astype(np.float32) - scale) / scale
    elif np.issubdtype(values.dtype, np.integer):
        audio = values.astype(np.float32) / float(np.iinfo(values.dtype).max)
    elif np.issubdtype(values.dtype, np.floating):
        audio = values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample type: {values.dtype}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError("Audio must have one or two channels.")
    if sample_rate <= 0 or not np.isfinite(audio).all():
        raise ValueError("The WAV file has an invalid sample rate or samples.")
    if audio.size / sample_rate > MAX_DEMO_DURATION_SECONDS:
        raise ValueError(f"Please provide a recording no longer than {MAX_DEMO_DURATION_SECONDS:g} seconds.")
    return np.clip(audio, -1.0, 1.0), int(sample_rate)


def resample_to_model_rate(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert arbitrary WAV sample rate to the model's fixed 16 kHz rate."""
    if source_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return signal.resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor).astype(np.float32)


@lru_cache(maxsize=4)
def load_selected_model(run_dir_text: str, device_text: str) -> tuple[torch.nn.Module, dict, tuple[str, ...]]:
    """Load one validation-selected checkpoint and its frozen configuration."""
    run_dir = Path(run_dir_text)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best_validation_macro_f1.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected config and selected checkpoint in {run_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    labels = TASK_LABELS[task]
    device = torch.device(device_text)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(config["model"]["name"], num_classes=len(labels)).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, labels


def predict_wav_bytes(audio_bytes: bytes, *, run_dir: str | Path, device: str = "cpu") -> dict[str, object]:
    """Classify one WAV recording with a validation-selected experiment checkpoint.

    This is a demonstration utility, not a diagnostic or a measurement of a
    person's emotional state. It reports the model's probabilities only.
    """
    audio, source_rate = decode_wav_bytes(audio_bytes)
    duration_seconds = float(audio.size / source_rate)
    model_audio = resample_to_model_rate(audio, source_rate)
    model, config, labels = load_selected_model(str(Path(run_dir)), device)
    window_seconds = float(config["data"]["train_window_seconds"])
    window = fit_audio_window(
        model_audio,
        target_samples=round(window_seconds * SAMPLE_RATE),
        policy="center_crop_or_zero_pad",
    )
    extractor = FEATURE_EXTRACTORS[config["features"]["type"]]
    features = extractor(window)
    with torch.no_grad():
        logits = model(torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    ranking = np.argsort(probabilities)[::-1]
    confidence = float(probabilities[ranking[0]])
    margin = float(probabilities[ranking[0]] - probabilities[ranking[1]])
    rms_dbfs = 20.0 * math.log10(max(float(np.sqrt(np.mean(np.square(audio)))), 1e-8))
    warnings: list[str] = []
    if duration_seconds < window_seconds:
        warnings.append(f"Recording is shorter than the {window_seconds:g}-second model window and was zero-padded.")
    if rms_dbfs < -45.0:
        warnings.append("Audio is quiet; speak closer to the microphone or use a clearer recording.")
    if confidence < 0.50:
        warnings.append("Low model confidence: treat this label as uncertain.")
    if margin < 0.10:
        warnings.append("The top two classes are close, so this prediction is ambiguous.")
    return {
        "predicted_label": labels[int(ranking[0])],
        "confidence": confidence,
        "margin_to_second_choice": margin,
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(labels)},
        "audio": {
            "source_sample_rate_hz": source_rate,
            "model_sample_rate_hz": SAMPLE_RATE,
            "duration_seconds": duration_seconds,
            "rms_dbfs": rms_dbfs,
            "model_window_seconds": window_seconds,
        },
        "model": {
            "experiment_id": config["experiment"]["id"],
            "checkpoint": "best_validation_macro_f1.pt",
            "feature_type": config["features"]["type"],
        },
        "warnings": warnings,
    }


def prediction_json(result: dict[str, object]) -> str:
    """Provide a stable, human-readable downloadable representation."""
    return json.dumps(result, indent=2) + "\n"
