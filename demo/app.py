"""College demonstration UI for the selected RobustSER-Tiny E5 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.inference import predict_wav_bytes, prediction_json


RUN_DIR = PROJECT_ROOT / "results" / "E5_seed20261108"

st.set_page_config(page_title="RobustSER-Tiny Demo", page_icon="🎙️", layout="wide")
st.title("RobustSER-Tiny: Speech Emotion Recognition")
st.caption("Software-only college demonstration • E5 validation-selected checkpoint • local inference")
st.warning(
    "This prototype classifies patterns in acted speech into six dataset labels. "
    "It is not a mental-health, truthfulness, or emotion assessment tool."
)

with st.sidebar:
    st.header("Research snapshot")
    st.metric("Clean test macro-F1", "47.41%")
    st.metric("Model parameters", "930")
    st.metric("Input context", "2 seconds")
    st.caption("Speaker-disjoint CREMA-D test protocol; results are three-seed means.")
    st.markdown("**Compared with the paper**")
    st.write("Stricter speaker-disjoint evaluation, external-corpus testing, and held-out MUSAN noise testing.")

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
            result = predict_wav_bytes(audio_bytes, run_dir=RUN_DIR, device=device)
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
