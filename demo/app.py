"""Local, presentation-ready demo for RobustSER-Tiny E5."""

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
from robustser.inference import MAX_DEMO_DURATION_SECONDS, decode_wav_bytes, predict_wav_bytes, prediction_json, resample_to_model_rate

RUN_DIR = PROJECT_ROOT / "results" / "E5_seed20261108"
REFERENCE_PAPER_URL = "https://ioro.org/ijcsa/article/577979314071/tinyml-based-speech-emotion-recognition-on-microcontrollers-using-quantized-cnns-and-mel-filterbank-energy-features"
LOCAL_REFERENCE_CLIP = PROJECT_ROOT / "dataset" / "AudioWAV" / "1002_DFA_ANG_XX.wav"


def styles() -> None:
    st.markdown("""
    <style>
      :root { --navy:#1E3A5F; --blue:#2563EB; --ink:#0F172A; --muted:#475569; --line:#CBD5E1; --wash:#F8FAFC; --paper:#FFFFFF; --gold:#A16207; }
      .stApp { background:var(--paper); color:var(--ink); }
      .block-container { max-width:1120px; padding:2.8rem 2rem 5rem; }
      .rs-masthead { border-top:8px solid var(--navy); border-bottom:1px solid var(--line); padding:1.35rem 0 1.5rem; margin-bottom:2rem; }
      .rs-kicker { color:var(--blue)!important; font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin:0 0 .55rem; }
      .rs-masthead h1 { color:var(--navy)!important; font-family:Georgia,"Times New Roman",serif!important; font-size:clamp(2.25rem,5vw,4rem)!important; letter-spacing:-.03em; margin:0!important; }
      .rs-subtitle { color:var(--muted)!important; font-size:1.12rem; line-height:1.6; max-width:70ch; margin:.65rem 0 0!important; }
      .rs-meta { display:flex; flex-wrap:wrap; gap:.5rem 1.25rem; color:var(--muted)!important; font-size:.86rem; margin-top:1.25rem; }
      .rs-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-top:1px solid var(--line); border-left:1px solid var(--line); margin:1.5rem 0 2.35rem; }
      .rs-stat { border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:1rem; min-height:120px; }
      .rs-stat small { color:var(--muted)!important; display:block; font-size:.73rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
      .rs-stat strong { color:var(--navy)!important; display:block; font-size:1.7rem; margin:.35rem 0; font-variant-numeric:tabular-nums; }
      .rs-stat span { color:var(--muted)!important; font-size:.88rem; line-height:1.4; }
      h2,h3 { color:var(--navy)!important; font-family:Georgia,"Times New Roman",serif!important; }
      .rs-lead { color:var(--muted)!important; font-size:1.05rem; line-height:1.65; max-width:75ch; }
      .rs-note { background:var(--wash); border-left:4px solid var(--navy); padding:1rem 1.15rem; margin:1.25rem 0; line-height:1.55; }
      .rs-table { width:100%; border-collapse:collapse; margin:1rem 0 1.5rem; font-size:.95rem; }
      .rs-table th { background:var(--navy); color:#fff; text-align:left; padding:.7rem .8rem; }
      .rs-table td { border-bottom:1px solid var(--line); padding:.7rem .8rem; vertical-align:top; }
      .rs-table tr:nth-child(even) td { background:var(--wash); }
      button[data-baseweb="tab"] { color:var(--muted)!important; font-weight:700; }
      button[data-baseweb="tab"][aria-selected="true"] { color:var(--navy)!important; border-bottom:3px solid var(--blue)!important; }
      .stButton>button,[data-testid="stDownloadButton"]>button { min-height:44px; border-radius:4px; font-weight:700; }
      .stButton>button[kind="primary"] { background:var(--navy)!important; border-color:var(--navy)!important; color:#fff!important; }
      .stButton>button:focus-visible,a:focus-visible,button[data-baseweb="tab"]:focus-visible { outline:3px solid var(--gold); outline-offset:3px; }
      [data-testid="stFileUploader"] { border:1px dashed var(--line); border-radius:4px; padding:.65rem; }
      [data-testid="stMetric"] { border-left:3px solid var(--navy); background:var(--wash); padding:.7rem; }
      @media(max-width:760px){ .block-container{padding:1.4rem 1rem 3rem}.rs-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.rs-masthead h1{font-size:2.35rem!important;} }
    </style>
    """, unsafe_allow_html=True)


def stat_grid() -> None:
    cards = [
        ("Robust E5 result", "47.41%", "Held-out macro F1 on unseen speakers"),
        ("Paper-style E8", "99.50%", "TESS random-split validation macro F1"),
        ("Model size", "930", "Trainable parameters in E5"),
        ("Measured gain", "+5.81 pp", "E5 improvement over E4"),
    ]
    html = "<div class='rs-grid'>" + "".join(f"<article class='rs-stat'><small>{a}</small><strong>{b}</strong><span>{c}</span></article>" for a,b,c in cards) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def evidence() -> None:
    st.markdown("<h2>Research evidence</h2><p class='rs-lead'>This project reports two different, clearly labelled results: a controlled paper-style replication and a tougher unseen-speaker evaluation. Neither result is used to hide the other.</p>", unsafe_allow_html=True)
    st.markdown("""
    <table class='rs-table'><thead><tr><th>Track</th><th>Question answered</th><th>Evaluation</th><th>Result</th></tr></thead><tbody>
    <tr><td><strong>E8 · TESS replication</strong></td><td>Can the compact paper-style CNN learn a clean controlled benchmark?</td><td>Five emotions, random 80/20 split</td><td><strong>99.50% validation macro F1</strong></td></tr>
    <tr><td><strong>E7 · CREMA-D comparison</strong></td><td>Does simply using a random split reproduce that result on a varied corpus?</td><td>Three random-split runs</td><td><strong>47.27% mean validation macro F1</strong></td></tr>
    <tr><td><strong>E5 · RobustSER benchmark</strong></td><td>Does the model generalise to speakers it never heard?</td><td>64/14/13 speaker-disjoint split, three seeds</td><td><strong>47.41% held-out test macro F1</strong></td></tr>
    </tbody></table>
    <div class='rs-note'><strong>How to read this:</strong> E8 demonstrates that the paper-style protocol can achieve a very high controlled score. E5 is deliberately harder: every final-test speaker is new to the model, and macro F1 gives every emotion equal weight. The two scores answer different questions.</div>
    """, unsafe_allow_html=True)
    st.link_button("Open the reference paper", REFERENCE_PAPER_URL)


def method() -> None:
    st.markdown("<h2>Method and limits</h2><p class='rs-lead'>The live demo uses E5, our six-emotion speaker-independent model. Audio stays on this computer.</p>", unsafe_allow_html=True)
    st.markdown("""
    <table class='rs-table'><thead><tr><th>Stage</th><th>What happens</th></tr></thead><tbody>
    <tr><td>1. Prepare audio</td><td>WAV audio is downmixed and resampled to mono 16 kHz.</td></tr>
    <tr><td>2. Select speech</td><td>The centre two seconds are used; shorter recordings are padded.</td></tr>
    <tr><td>3. Create features</td><td>A 32 × 62 MFCC representation is created.</td></tr>
    <tr><td>4. Predict</td><td>The 930-parameter E5 CNN returns six probabilities and uncertainty warnings.</td></tr>
    </tbody></table>
    """, unsafe_allow_html=True)
    st.info(f"Training clips average 2.55 seconds. For the clearest demo, record about 2.5 seconds. The app accepts files up to {MAX_DEMO_DURATION_SECONDS / 60:g} minutes.")
    st.warning("This is a research demonstration of acted-speech dataset labels, not a real-emotion, mental-health, or truthfulness assessment.")


def classifier() -> None:
    st.markdown("<h2>Classify a recording</h2><p class='rs-lead'>Upload a WAV recording or use your microphone. Inference is local.</p>", unsafe_allow_html=True)
    use_reference = bool(st.session_state.get("use_reference_clip", False))
    if LOCAL_REFERENCE_CLIP.is_file() and st.button("Use the built-in reference recording"):
        st.session_state["use_reference_clip"] = True; use_reference = True
    upload = st.file_uploader("Upload a WAV recording", type=["wav"], help="A short acted sentence gives the most representative result.")
    recorded = st.audio_input("Or record from your microphone")
    audio_source = recorded if recorded is not None else upload
    audio_bytes = LOCAL_REFERENCE_CLIP.read_bytes() if use_reference else (audio_source.getvalue() if audio_source else None)
    if audio_bytes is None:
        st.caption("Tip: use one clearly acted sentence. The model classifies only the centre two seconds.")
        return
    if use_reference: st.caption("Built-in validation recording (Angry): demonstration input only, not a reported metric.")
    st.audio(audio_bytes, format="audio/wav")
    if not st.button("Classify recording", type="primary"):
        return
    try:
        with st.spinner("Running local inference…"):
            result = predict_wav_bytes(audio_bytes, run_dir=RUN_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
            audio, rate = decode_wav_bytes(audio_bytes)
            window = fit_audio_window(resample_to_model_rate(audio, rate), target_samples=2*SAMPLE_RATE, policy="center_crop_or_zero_pad")
            mfcc = extract_mfcc(window)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error)); return
    a,b,c = st.columns(3); info = result["audio"]
    a.metric("Predicted label", result["predicted_label"]); b.metric("Confidence", f"{100*result['confidence']:.1f}%"); c.metric("Duration", f"{info['duration_seconds']:.2f} s")
    probabilities = pd.DataFrame({"Emotion": list(result["probabilities"]), "Probability": list(result["probabilities"].values())}).sort_values("Probability", ascending=False)
    st.subheader("Model probabilities"); st.bar_chart(probabilities, x="Emotion", y="Probability"); st.dataframe(probabilities.style.format({"Probability":"{:.1%}"}), hide_index=True, use_container_width=True)
    with st.expander("Inspect the model input"):
        st.image(mfcc, clamp=True, caption="32 × 62 MFCC input to E5")
    for warning in result["warnings"]: st.warning(warning)
    st.download_button("Download prediction JSON", prediction_json(result), "robustser_prediction.json", "application/json")


st.set_page_config(page_title="RobustSER-Tiny", layout="wide", initial_sidebar_state="collapsed")
styles()
st.markdown("""<header class='rs-masthead'><p class='rs-kicker'>College research project · speech emotion recognition</p><h1>RobustSER-Tiny</h1><p class='rs-subtitle'>A compact model, a controlled replication, and an honest test of whether speech-emotion recognition generalises to people the model has never heard.</p><div class='rs-meta'><span>CREMA-D + TESS</span><span>Speaker-independent evaluation</span><span>Local inference</span></div></header>""", unsafe_allow_html=True)
stat_grid()
live_tab, evidence_tab, method_tab = st.tabs(["Live classifier", "Research evidence", "Method & limits"])
with live_tab: classifier()
with evidence_tab: evidence()
with method_tab: method()
