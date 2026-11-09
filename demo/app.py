"""Accessible local college demonstration for the RobustSER-Tiny E5 model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.features.log_mel import SAMPLE_RATE, extract_mfcc, fit_audio_window
from robustser.inference import (
    MAX_DEMO_DURATION_SECONDS,
    decode_wav_bytes,
    predict_wav_bytes,
    prediction_json,
    resample_to_model_rate,
)


RUN_DIR = PROJECT_ROOT / "results" / "E5_seed20261108"
REFERENCE_PAPER_URL = "https://ioro.org/ijcsa/article/577979314071/tinyml-based-speech-emotion-recognition-on-microcontrollers-using-quantized-cnns-and-mel-filterbank-energy-features"
LOCAL_REFERENCE_CLIP = PROJECT_ROOT / "dataset" / "AudioWAV" / "1002_DFA_ANG_XX.wav"


def inject_styles() -> None:
    """Apply a contrast-checked, local research-portal design system."""
    st.markdown(
        """
        <style>
        :root {
          --rs-primary: #112250; /* Royal Blue */
          --rs-secondary: #3C5070; /* Sapphire */
          --rs-accent: #A87A2E; /* contrast-safe Quicksand accent */
          --rs-ink: #112250;
          --rs-muted: #3C5070;
          --rs-surface: #FFFDF9;
          --rs-border: #D9CBC2; /* Shellstone */
          --rs-wash: #F5F0E9; /* Swan Wing */
        }
        .stApp, [data-testid="stAppViewContainer"] { background: var(--rs-wash); color: var(--rs-ink); }
        .stApp p, .stApp label, .stApp span, .stApp li, .stApp [data-testid="stMarkdownContainer"] { color: var(--rs-ink); }
        [data-testid="stSidebar"] { background: #EDE6DC; border-right: 1px solid var(--rs-border); }
        [data-testid="stSidebar"] [data-testid="stMetric"] { background: #FFFDF9 !important; }
        .block-container { max-width: 1260px; padding-top: 2rem; padding-bottom: 4rem; }
        .rs-hero { background: linear-gradient(135deg, #112250 0%, #203966 55%, #3C5070 100%);
          border-radius: 20px; color: white; padding: clamp(1.75rem, 4vw, 3.25rem); margin-bottom: 1.5rem;
          box-shadow: 0 16px 34px rgba(17, 34, 80, .24); }
        .rs-hero h1, .rs-hero h1 * { color: #FFFFFF !important; font-family: Georgia, "Times New Roman", serif !important; font-size: clamp(2.25rem, 5vw, 4rem) !important;
          margin: 0 0 .65rem 0 !important; line-height: 1.08 !important; letter-spacing: -.025em; }
        .rs-hero p { color: #F5F0E9 !important; font-size: 1.05rem; max-width: 60ch; margin: 0; line-height: 1.55; }
        .rs-kicker { color: #E0C58F !important; font-size: .8rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin-bottom: .8rem; }
        .rs-hero-meta { border-top: 1px solid rgba(245,240,233,.35); margin-top: 1.4rem; padding-top: 1rem; color: #F5F0E9 !important; font-size: .9rem; }
        .rs-card { background: var(--rs-surface); border: 1px solid var(--rs-border); border-radius: 14px; padding: 1.15rem 1.2rem;
          box-shadow: 0 6px 18px rgba(17, 34, 80, .07); min-height: 136px; }
        .rs-card-label { color: var(--rs-muted); font-size: .82rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
        .rs-card-value { color: var(--rs-primary); font-size: 1.8rem; font-weight: 800; margin: .32rem 0; font-variant-numeric: tabular-nums; }
        .rs-card-detail { color: var(--rs-muted); font-size: .9rem; line-height: 1.4; }
        .rs-section-title { font-family: Georgia, "Times New Roman", serif; color: var(--rs-ink); margin-top: 1.3rem; }
        .rs-note { border-left: 4px solid var(--rs-accent); background: #F7E8C9; color: var(--rs-ink); border-radius: 0 12px 12px 0;
          padding: .85rem 1rem; line-height: 1.5; }
        .rs-section { margin-top: 2.25rem; }
        .rs-eyebrow { color: var(--rs-secondary) !important; font-size: .75rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .35rem; }
        .rs-section h2, .rs-section h3 { color: var(--rs-primary) !important; font-family: Georgia, "Times New Roman", serif !important; }
        .rs-lead { font-size: 1.08rem; line-height: 1.65; max-width: 78ch; color: var(--rs-ink) !important; }
        .rs-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin: 1.25rem 0; }
        .rs-compare-card { background: var(--rs-surface); border: 1px solid var(--rs-border); border-radius: 14px; padding: 1.25rem; }
        .rs-compare-card h3 { color: var(--rs-primary) !important; margin: 0 0 .75rem; font-size: 1.15rem; }
        .rs-compare-card ul { margin: 0; padding-left: 1.2rem; line-height: 1.65; }
        .rs-result-strip { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 1rem; margin: 1rem 0 1.5rem; }
        .rs-evidence { background: #FFFDF9; border-top: 4px solid var(--rs-secondary); border-radius: 12px; padding: 1rem; }
        .rs-evidence strong { display: block; color: var(--rs-primary); font-size: 1.4rem; font-variant-numeric: tabular-nums; }
        .rs-evidence span { color: var(--rs-muted) !important; font-size: .9rem; line-height: 1.35; }
        .rs-flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .65rem; margin: 1.25rem 0; }
        .rs-step { border: 1px solid var(--rs-border); background: var(--rs-surface); border-radius: 12px; padding: .85rem; min-height: 116px; }
        .rs-step-number { display: inline-block; color: #FFFFFF !important; background: var(--rs-secondary); border-radius: 999px; width: 1.55rem; height: 1.55rem; text-align: center; line-height: 1.55rem; font-size: .8rem; font-weight: 700; margin-bottom: .55rem; }
        .rs-step strong { color: var(--rs-primary); display: block; font-size: .95rem; margin-bottom: .25rem; }
        .rs-step p { color: var(--rs-muted) !important; font-size: .84rem; line-height: 1.4; margin: 0; }
        .stButton > button, [data-testid="stDownloadButton"] > button { border-radius: 10px; min-height: 44px; font-weight: 700; }
        .stButton > button { color: var(--rs-primary) !important; background: var(--rs-surface); border-color: var(--rs-secondary); }
        .stButton > button[kind="primary"] { background: var(--rs-primary) !important; border-color: var(--rs-primary) !important; color: #FFFFFF !important; }
        .stButton > button:hover { background: #EDE6DC; border-color: var(--rs-primary); color: var(--rs-primary) !important; }
        .stButton > button[kind="primary"]:hover { background: var(--rs-secondary) !important; color: #FFFFFF !important; }
        .stButton > button:focus-visible, a:focus-visible, button[data-baseweb="tab"]:focus-visible { outline: 3px solid var(--rs-accent); outline-offset: 3px; }
        button[data-baseweb="tab"] { color: var(--rs-secondary) !important; font-weight: 700; }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--rs-primary) !important; border-bottom-color: var(--rs-accent) !important; }
        [data-testid="stMetric"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { background: var(--rs-surface); color: var(--rs-ink) !important; border-color: var(--rs-border) !important; }
        [data-testid="stMetric"] { border: 1px solid var(--rs-border); border-radius: 14px; padding: .8rem; }
        [data-testid="stFileUploader"] { background: var(--rs-surface); border: 1px dashed var(--rs-secondary); border-radius: 14px; padding: .5rem; }
        [data-testid="stFileUploader"] * { color: var(--rs-ink) !important; }
        [data-testid="stAudioInput"] { background: var(--rs-surface); border: 1px solid var(--rs-border); border-radius: 14px; padding: .5rem; }
        [data-testid="stAlert"] { background: var(--rs-surface); color: var(--rs-ink); border-color: var(--rs-border); }
        [data-testid="stAlert"] * { color: var(--rs-ink) !important; }
        [data-testid="stDataFrame"], [data-testid="stDataFrame"] * { color: var(--rs-ink) !important; }
        [data-testid="stDataFrame"] { background: var(--rs-surface); border: 1px solid var(--rs-border); border-radius: 12px; }
        @media (max-width: 760px) {
          .block-container { padding: 1rem; } .rs-hero { border-radius: 16px; }
          .rs-compare, .rs-result-strip { grid-template-columns: 1fr; }
          .rs-flow { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def project_cards() -> None:
    columns = st.columns(4)
    cards = [
        ("Robust test macro F1", "47.41%", "Three-seed, unseen-speaker mean"),
        ("Measured improvement", "+5.81 pp", "E5 over E4 clean test macro F1"),
        ("Model size", "930", "Trainable parameters in E5"),
        ("Evaluation protocol", "64 / 14 / 13", "Speaker train / validation / test split"),
    ]
    for column, (label, value, detail) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="rs-card"><div class="rs-card-label">{label}</div>'
                f'<div class="rs-card-value">{value}</div><div class="rs-card-detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )


def render_research_view() -> None:
    st.markdown('<section class="rs-section"><div class="rs-eyebrow">Evidence dossier</div><h2>Why this is a stronger test</h2></section>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rs-lead">The published system and RobustSER-Tiny share the same core idea—compact 16 kHz audio features fed to a CNN—but test different questions. The paper asks whether a model can recognize samples from a random split. This project asks whether the model still works for speakers it has never seen.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '''<div class="rs-compare">
          <article class="rs-compare-card"><h3>Published paper</h3><ul><li>Modified TESS corpus</li><li>Random 80/20 split</li><li>Five emotions plus a custom Noise class</li><li>Reported metric: validation accuracy</li><li>Result: <strong>99.5% validation accuracy</strong></li></ul></article>
          <article class="rs-compare-card"><h3>RobustSER-Tiny E5</h3><ul><li>CREMA-D acted-speech corpus</li><li>Frozen, speaker-disjoint 64/14/13 split</li><li>Six speech-emotion labels</li><li>Primary metric: macro F1, equal weight per class</li><li>Result: <strong>47.41% held-out test macro F1</strong></li></ul></article>
        </div>''',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rs-result-strip"><div class="rs-evidence"><strong>+5.81 pp</strong><span>E5 clean-test macro F1 improvement over E4; the 95% confidence interval excludes zero.</span></div><div class="rs-evidence"><strong>0.35 pp</strong><span>Mean validation-to-test gap: validation 47.76%, test 47.41%. This is evidence against a large selection gap.</span></div><div class="rs-evidence"><strong>34.58%</strong><span>E5 RAVDESS external macro F1: cross-corpus transfer remains the main open challenge.</span></div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rs-note"><strong>Presentation-ready conclusion:</strong> E5 is not a direct replacement for the paper\'s reported score. It is a stricter robustness result. The next report will show both a paper-style random-split track and the speaker-independent RobustSER track side by side.</div>',
        unsafe_allow_html=True,
    )
    st.link_button("Read the reference paper", REFERENCE_PAPER_URL, use_container_width=False)
    st.caption("Source: Ahnaf (2026), TinyML-Based Speech Emotion Recognition on Microcontrollers Using Quantized CNNs and Mel-Filterbank Energy Features.")


def render_protocol_view() -> None:
    st.markdown('<section class="rs-section"><div class="rs-eyebrow">Local inference protocol</div><h2>What happens to your recording</h2><p class="rs-lead">The demo performs its full inference pipeline on this computer. It does not send your audio to a cloud service.</p></section>', unsafe_allow_html=True)
    st.markdown('''<div class="rs-flow"><article class="rs-step"><span class="rs-step-number">1</span><strong>Capture</strong><p>Accept a WAV file or browser microphone recording.</p></article><article class="rs-step"><span class="rs-step-number">2</span><strong>Standardize</strong><p>Downmix to mono and resample to 16 kHz.</p></article><article class="rs-step"><span class="rs-step-number">3</span><strong>Window</strong><p>Use a fixed centre two-second segment.</p></article><article class="rs-step"><span class="rs-step-number">4</span><strong>Represent</strong><p>Extract a 32 × 62 MFCC feature matrix.</p></article><article class="rs-step"><span class="rs-step-number">5</span><strong>Classify</strong><p>Run E5 and return six probabilities plus warnings.</p></article></div>''', unsafe_allow_html=True)
    st.info(
        "Training clips averaged 2.55 seconds (median 2.50). For a representative demo, record one clearly acted "
        "sentence for about 2.5 seconds. The demo also accepts recordings up to "
        f"{MAX_DEMO_DURATION_SECONDS / 60:g} minutes, but only the centre two seconds are classified."
    )
    st.warning(
        "This is a research demonstration of acted-speech labels. It is not a mental-health, truthfulness, "
        "or real-emotion assessment tool."
    )


def render_classifier() -> None:
    st.subheader("Classify a recording", anchor="classify-a-recording")
    st.caption("Use a WAV file or record one clearly acted sentence. No audio is sent to a cloud service.")
    use_reference_clip = bool(st.session_state.get("use_reference_clip", False))
    if LOCAL_REFERENCE_CLIP.is_file():
        st.caption("For a quick classroom smoke test, use the local 2.569-second reference clip below. It is not used for training or for reporting any metric.")
        if st.button("Try the local reference recording", use_container_width=True):
            st.session_state["use_reference_clip"] = True
            use_reference_clip = True
        if use_reference_clip and st.button("Clear the local reference recording", use_container_width=True):
            st.session_state.pop("use_reference_clip", None)
            use_reference_clip = False
    else:
        st.caption("The optional local reference clip is unavailable because the raw CREMA-D dataset is not present.")
    uploaded = st.file_uploader(
        "Upload a WAV recording",
        type=["wav"],
        help=("For the most representative result, aim for about 2.5 seconds. "
              f"The maximum accepted duration is {MAX_DEMO_DURATION_SECONDS / 60:g} minutes."),
    )
    recorded = st.audio_input("Or record from your microphone")
    audio_file = recorded if recorded is not None else uploaded
    audio_bytes = LOCAL_REFERENCE_CLIP.read_bytes() if use_reference_clip else (audio_file.getvalue() if audio_file is not None else None)
    if audio_bytes is None:
        st.info("Choose one input method above. The model uses the centre two seconds of your clip.")
        return
    if use_reference_clip:
        st.info("Showing a local validation clip whose dataset label is Angry. This is a demo input only, not a reported evaluation result.")
    st.audio(audio_bytes, format="audio/wav")
    if not st.button("Classify this recording", type="primary", use_container_width=True):
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with st.spinner("Preparing the audio, extracting MFCC features, and running E5 locally..."):
            result = predict_wav_bytes(audio_bytes, run_dir=RUN_DIR, device=device)
            original_audio, original_rate = decode_wav_bytes(audio_bytes)
            model_audio = resample_to_model_rate(original_audio, original_rate)
            model_window = fit_audio_window(model_audio, target_samples=2 * SAMPLE_RATE, policy="center_crop_or_zero_pad")
            mfcc_preview = extract_mfcc(model_window)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return

    left, right = st.columns(2)
    with left:
        st.success(f"Predicted dataset label: {result['predicted_label']}")
        st.metric("Model confidence", f"{100 * result['confidence']:.1f}%")
        st.metric("Margin over second choice", f"{100 * result['margin_to_second_choice']:.1f} points")
    with right:
        audio_info = result["audio"]
        st.metric("Recording duration", f"{audio_info['duration_seconds']:.2f} seconds")
        st.metric("Original sample rate", f"{audio_info['source_sample_rate_hz']} Hz")
        st.metric("Audio level", f"{audio_info['rms_dbfs']:.1f} dBFS")

    probabilities = pd.DataFrame(
        {"Emotion label": list(result["probabilities"]), "Probability": list(result["probabilities"].values())}
    ).sort_values("Probability", ascending=False)
    st.subheader("All six model probabilities")
    st.bar_chart(probabilities, x="Emotion label", y="Probability", height=320)
    st.dataframe(probabilities.style.format({"Probability": "{:.1%}"}), hide_index=True, use_container_width=True)

    with st.expander("Inspect the model input"):
        waveform_step = max(1, original_audio.size // 1_000)
        waveform = pd.DataFrame(
            {
                "Time (seconds)": np.arange(0, original_audio.size, waveform_step) / original_rate,
                "Amplitude": original_audio[::waveform_step],
            }
        )
        st.caption("Waveform of the uploaded recording. The classifier uses a centre-cropped or zero-padded two-second window.")
        st.line_chart(waveform, x="Time (seconds)", y="Amplitude", height=220)
        st.caption("32×62 MFCC feature matrix sent to the CNN: coefficients by time frame.")
        st.image(mfcc_preview, clamp=True, caption="E5 MFCC input")

    if result["warnings"]:
        st.subheader("Interpretation notes")
        for warning in result["warnings"]:
            st.warning(warning)
    else:
        st.success("No duration, level, confidence, or ambiguity warning was triggered.")
    st.download_button(
        "Download prediction JSON",
        data=prediction_json(result),
        file_name="robustser_prediction.json",
        mime="application/json",
        use_container_width=True,
    )


st.set_page_config(page_title="RobustSER-Tiny Demo", page_icon=None, layout="wide")
inject_styles()

st.markdown(
    """
    <section class="rs-hero">
      <div class="rs-kicker">College research demonstration · software-only</div>
      <h1>RobustSER-Tiny</h1>
      <p>A compact speech-emotion classifier evaluated on speakers it never saw during training—built as a transparent comparison to TinyML-style random-split evaluation.</p>
      <div class="rs-hero-meta">CREMA-D · speaker-disjoint evaluation · 3 seeded replications · 930-parameter depthwise-separable CNN</div>
    </section>
    """,
    unsafe_allow_html=True,
)
project_cards()

with st.sidebar:
    st.header("E5 at a glance")
    st.metric("Macro F1", "47.41%")
    st.metric("Parameters", "930")
    st.metric("Model window", "2 seconds")
    st.divider()
    st.markdown("**Evidence used**")
    st.write("CREMA-D speaker-disjoint test, MUSAN noise stress test, and RAVDESS external evaluation.")
    st.link_button("Reference paper", REFERENCE_PAPER_URL, use_container_width=True)
    st.caption("All results are software-only; no device performance claim is made.")

classifier_tab, research_tab, protocol_tab = st.tabs(["Live classifier", "Research evidence", "Method and limits"])
with classifier_tab:
    render_classifier()
with research_tab:
    render_research_view()
with protocol_tab:
    render_protocol_view()
