"""College demonstration UI for the selected RobustSER-Tiny E5 checkpoint with observability."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.inference_observability import (
    predict_wav_bytes_with_observability,
    prediction_json,
    warmup_model,
)
from robustser.observability import setup_structured_logging, get_logger, get_metrics_collector

# Setup observability
LOG_DIR = PROJECT_ROOT / "logs" / "demo"
LOG_DIR.mkdir(parents=True, exist_ok=True)

setup_structured_logging(
    level=20,  # INFO
    log_file=LOG_DIR / "demo.jsonl",
    json_output=True,
)

logger = get_logger("robustser.demo")
metrics_collector = get_metrics_collector()
metrics_collector._jsonl_path = LOG_DIR / "demo_metrics.jsonl"

RUN_DIR = PROJECT_ROOT / "results" / "E5_seed20261108"

st.set_page_config(page_title="RobustSER-Tiny Demo", page_icon="🎙️", layout="wide")
st.title("RobustSER-Tiny: Speech Emotion Recognition")
st.caption("Software-only college demonstration • E5 validation-selected checkpoint • local inference")
st.warning(
    "This prototype classifies patterns in acted speech into six dataset labels. "
    "It is not a mental-health, truthfulness, or emotion assessment tool."
)

# Sidebar with metrics
with st.sidebar:
    st.header("Research snapshot")
    st.metric("Clean test macro-F1", "47.41%")
    st.metric("Model parameters", "930")
    st.metric("Input context", "2 seconds")
    st.caption("Speaker-disjoint CREMA-D test protocol; results are three-seed means.")
    st.markdown("**Compared with the paper**")
    st.write("Stricter speaker-disjoint evaluation, external-corpus testing, and held-out MUSAN noise testing.")

    # Live metrics from observability
    st.markdown("---")
    st.header("Live Metrics")

    # Get recent metrics
    recent_points = metrics_collector.get_points(since=time.time() - 300)  # last 5 min
    if recent_points:
        # Inference latency
        latency_points = [p for p in recent_points if p.name == "inference_latency_ms" and p.labels.get("stage") == "total"]
        if latency_points:
            latencies = [p.value for p in latency_points]
            st.metric("Recent avg latency", f"{sum(latencies)/len(latencies):.1f} ms")
            st.metric("Recent p99 latency", f"{sorted(latencies)[int(0.99*len(latencies))]:.1f} ms")

        # Confidence
        conf_points = [p for p in recent_points if p.name == "model_confidence"]
        if conf_points:
            confs = [p.value for p in conf_points]
            st.metric("Recent avg confidence", f"{sum(confs)/len(confs):.2%}")

        # OOD rate
        ood_points = [p for p in recent_points if p.name == "ood_detection_rate"]
        if ood_points:
            ood_rate = sum(p.value for p in ood_points) / len(ood_points)
            st.metric("Recent OOD rate", f"{ood_rate:.1%}")

    if st.button("Warm up model"):
        with st.spinner("Warming up model..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            warmup_model(RUN_DIR, device)
        st.success("Model warmed up!")

st.subheader("1. Provide a short WAV recording")
uploaded = st.file_uploader("Upload a WAV file", type=["wav"], help="Any WAV sample rate is accepted and converted locally to 16 kHz.")
recorded = st.audio_input("Or record from your microphone")
audio_file = recorded if recorded is not None else uploaded

if audio_file is None:
    st.info("Record or upload a WAV clip. For the clearest demonstration, speak one acted sentence for about two seconds.")
    st.stop()

audio_bytes = audio_file.getvalue()
st.audio(audio_bytes, format="audio/wav")

if st.button("Classify recording", type="primary", use_container_width=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with st.spinner("Extracting MFCC features and running the selected E5 model..."):
            result = predict_wav_bytes_with_observability(audio_bytes, run_dir=RUN_DIR, device=device)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        st.stop()

    left, right = st.columns(2)
    with left:
        st.success(f"Predicted label: {result['predicted_label']}")
        st.metric("Model confidence", f"{100 * result['confidence']:.1f}%")
        st.metric("Margin over second choice", f"{100 * result['margin_to_second_choice']:.1f} points")
    with right:
        audio_info = result["audio"]
        st.metric("Recording duration", f"{audio_info['duration_seconds']:.2f} s")
        st.metric("Input sample rate", f"{audio_info['source_sample_rate_hz']} Hz")
        st.metric("Audio level", f"{audio_info['rms_dbfs']:.1f} dBFS")

    # Timing breakdown
    st.subheader("⏱️ Inference Timing Breakdown")
    timing = result.get("timing", {})
    if timing:
        timing_df = pd.DataFrame({
            "Stage": ["Total", "Decode", "Resample", "Load Model", "Window", "Features", "Forward Pass"],
            "Time (ms)": [
                timing.get("total_ms", 0),
                timing.get("decode_ms", 0),
                timing.get("resample_ms", 0),
                timing.get("load_model_ms", 0),
                timing.get("window_ms", 0),
                timing.get("features_ms", 0),
                timing.get("forward_ms", 0),
            ],
        })
        st.bar_chart(timing_df.set_index("Stage"))

    probabilities = pd.DataFrame(
        {"Emotion label": list(result["probabilities"]), "Probability": list(result["probabilities"].values())}
    ).sort_values("Probability", ascending=False)
    st.subheader("All class probabilities")
    st.bar_chart(probabilities, x="Emotion label", y="Probability", height=320)

    if result["warnings"]:
        st.subheader("Interpretation notes")
        for warning in result["warnings"]:
            st.warning(warning)
    else:
        st.caption("No basic duration, level, or confidence warning was triggered.")

    st.download_button(
        "Download prediction JSON",
        data=prediction_json(result),
        file_name="robustser_prediction.json",
        mime="application/json",
        use_container_width=True,
    )

    # Show request ID for tracing
    st.caption(f"Request ID: {result.get('request_id', 'N/A')}")