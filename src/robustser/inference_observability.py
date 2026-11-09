"""
Local, reproducible inference helpers with observability for RobustSER-Tiny demo and edge deployment.
"""

from __future__ import annotations

import io
import json
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from scipy import signal
from scipy.io import wavfile

from robustser.data.cremad import FEATURE_EXTRACTORS, TASK_LABELS
from robustser.features.log_mel import SAMPLE_RATE, fit_audio_window
from robustser.models import build_model
from robustser.evaluation.calibration import softmax_confidence
from robustser.observability import (
    get_logger,
    record_metric,
    record_histogram,
    trace_span,
    LogContext,
    CorrelationContext,
)

MAX_DEMO_DURATION_SECONDS = 60.0

# Global model cache for serving
_model_cache: dict[str, tuple[torch.nn.Module, dict, tuple[str, ...]]] = {}


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


def get_model_serving(run_dir: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict, tuple[str, ...]]:
    """Get model for serving (with caching)."""
    cache_key = f"{run_dir}:{device}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = load_selected_model(str(Path(run_dir)), device)
    return _model_cache[cache_key]


def predict_wav_bytes_with_observability(
    audio_bytes: bytes,
    *,
    run_dir: str | Path,
    device: str = "cpu",
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Classify one WAV recording with full observability.

    This is a demonstration utility, not a diagnostic or a measurement of a
    person's emotional state. It reports the model's probabilities only.
    """
    request_id = request_id or uuid.uuid4().hex[:16]
    run_dir_path = Path(run_dir)
    logger = get_logger("robustser.inference")

    # Correlation context for this request
    corr_context = LogContext(
        experiment_id=run_dir_path.name,
        run_id=run_dir_path.name,
        split="inference",
        device=device,
        request_id=request_id,
    )

    with CorrelationContext(corr_context):
        with trace_span("inference_request", baggage={"request_id": request_id}) as span:
            # 1. Decode
            with trace_span("decode_wav") as decode_span:
                t0 = time.time()
                audio, source_rate = decode_wav_bytes(audio_bytes)
                decode_time = (time.time() - t0) * 1000
                decode_span.set_attribute("duration_ms", decode_time)
                decode_span.set_attribute("source_sample_rate", source_rate)
                decode_span.set_attribute("audio_duration_seconds", float(audio.size / source_rate))

            duration_seconds = float(audio.size / source_rate)

            # 2. Resample
            with trace_span("resample") as resample_span:
                t0 = time.time()
                model_audio = resample_to_model_rate(audio, source_rate)
                resample_time = (time.time() - t0) * 1000
                resample_span.set_attribute("duration_ms", resample_time)

            # 3. Load model
            with trace_span("load_model") as load_span:
                t0 = time.time()
                model, config, labels = get_model_serving(run_dir, device)
                load_time = (time.time() - t0) * 1000
                load_span.set_attribute("duration_ms", load_time)
                load_span.set_attribute("model_name", config["model"]["name"])

            window_seconds = float(config["data"]["train_window_seconds"])

            # 4. Window
            with trace_span("window_audio") as window_span:
                t0 = time.time()
                window = fit_audio_window(
                    model_audio,
                    target_samples=round(window_seconds * SAMPLE_RATE),
                    policy="center_crop_or_zero_pad",
                )
                window_time = (time.time() - t0) * 1000
                window_span.set_attribute("duration_ms", window_time)
                window_span.set_attribute("target_samples", round(window_seconds * SAMPLE_RATE))

            # 5. Features
            with trace_span("extract_features") as feat_span:
                t0 = time.time()
                extractor = FEATURE_EXTRACTORS[config["features"]["type"]]
                features = extractor(window)
                feat_time = (time.time() - t0) * 1000
                feat_span.set_attribute("duration_ms", feat_time)
                feat_span.set_attribute("feature_type", config["features"]["type"])
                feat_span.set_attribute("feature_shape", str(features.shape))

            # 6. Model inference
            with trace_span("model_forward") as forward_span:
                t0 = time.time()
                with torch.no_grad():
                    logits = model(torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device))
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                forward_time = (time.time() - t0) * 1000
                forward_span.set_attribute("duration_ms", forward_time)

            total_time = decode_time + resample_time + load_time + window_time + feat_time + forward_time

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

            # Record serving metrics
            record_metric("inference_latency_ms", total_time, unit="milliseconds", labels={"stage": "total"})
            record_metric("inference_latency_ms", decode_time, unit="milliseconds", labels={"stage": "decode"})
            record_metric("inference_latency_ms", resample_time, unit="milliseconds", labels={"stage": "resample"})
            record_metric("inference_latency_ms", load_time, unit="milliseconds", labels={"stage": "load_model"})
            record_metric("inference_latency_ms", window_time, unit="milliseconds", labels={"stage": "window"})
            record_metric("inference_latency_ms", feat_time, unit="milliseconds", labels={"stage": "features"})
            record_metric("inference_latency_ms", forward_time, unit="milliseconds", labels={"stage": "forward"})

            record_metric("model_confidence", confidence, unit="probability")
            record_metric("prediction_margin", margin, unit="probability")

            # OOD detection
            ood_conf = 1.0 if confidence < 0.5 else 0.0
            ood_margin = 1.0 if margin < 0.1 else 0.0
            record_metric("ood_detection_rate", ood_conf, labels={"method": "confidence"}, unit="rate")
            record_metric("ood_detection_rate", ood_margin, labels={"method": "margin"}, unit="rate")

            record_metric("audio_duration_seconds", duration_seconds, unit="seconds")
            record_metric("audio_rms_dbfs", rms_dbfs, unit="dBFS")

            # Alert checks
            if confidence < 0.35:
                logger.warning(f"Very low confidence prediction: {confidence:.3f}", extra={
                    "request_id": request_id,
                    "confidence": confidence,
                    "predicted_label": labels[int(ranking[0])],
                })

            if total_time > 200:
                logger.warning(f"High inference latency: {total_time:.1f}ms", extra={
                    "request_id": request_id,
                    "latency_ms": total_time,
                })

            span.set_attribute("confidence", confidence)
            span.set_attribute("margin", margin)
            span.set_attribute("predicted_label", labels[int(ranking[0])])
            span.set_attribute("total_latency_ms", total_time)
            span.set_attribute("ood_detected", ood_conf or ood_margin)

            result = {
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
                "timing": {
                    "total_ms": total_time,
                    "decode_ms": decode_time,
                    "resample_ms": resample_time,
                    "load_model_ms": load_time,
                    "window_ms": window_time,
                    "features_ms": feat_time,
                    "forward_ms": forward_time,
                },
                "warnings": warnings,
                "request_id": request_id,
            }

            logger.info("Inference completed", extra={
                "request_id": request_id,
                "predicted_label": labels[int(ranking[0])],
                "confidence": confidence,
                "latency_ms": total_time,
            })

            return result


def predict_wav_bytes_edge(
    audio_bytes: bytes,
    *,
    run_dir: str | Path,
    device: str = "cpu",
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Edge-optimized inference with telemetry for MCU deployment.

    Returns additional edge telemetry fields:
    - edge_inference_latency_ms: model forward pass only
    - edge_peak_ram_kb: estimated peak RAM (if available)
    - edge_energy_per_inference_mj: estimated energy (if available)
    """
    # Run standard inference with observability
    result = predict_wav_bytes_with_observability(audio_bytes, run_dir=run_dir, device=device, request_id=request_id)

    # Add edge-specific telemetry
    result["edge_telemetry"] = {
        "inference_latency_ms": result["timing"]["forward_ms"],
        "model_load_time_ms": result["timing"]["load_model_ms"],
        "peak_ram_kb": None,  # Would be filled by MCU runtime
        "energy_per_inference_mj": None,  # Would be filled by MCU runtime
        "device": device,
    }

    # Record edge metrics
    record_metric("edge_inference_latency_ms", result["timing"]["forward_ms"], unit="milliseconds")
    record_metric("edge_model_load_time_ms", result["timing"]["load_model_ms"], unit="milliseconds")

    return result


def prediction_json(result: dict[str, Any]) -> str:
    """Provide a stable, human-readable downloadable representation."""
    return json.dumps(result, indent=2) + "\n"


def warmup_model(run_dir: str | Path, device: str = "cpu") -> None:
    """Warm up model for serving (compile, cache, etc.)."""
    logger = get_logger("robustser.inference")
    logger.info("Warming up model...")

    model, config, labels = get_model_serving(run_dir, device)

    # Dummy forward pass
    dummy_features = np.random.randn(32, 31).astype(np.float32)
    with torch.no_grad():
        _ = model(torch.from_numpy(dummy_features).unsqueeze(0).unsqueeze(0).to(device))

    # PyTorch 2.0 compilation (optional)
    try:
        model = torch.compile(model)
        _ = model(torch.from_numpy(dummy_features).unsqueeze(0).unsqueeze(0).to(device))
        logger.info("Model compiled with torch.compile")
    except Exception as e:
        logger.debug(f"torch.compile not available: {e}")

    logger.info("Model warmup complete")


def clear_model_cache() -> None:
    """Clear the model cache (useful for testing or model updates)."""
    global _model_cache
    _model_cache.clear()