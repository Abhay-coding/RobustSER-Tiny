# RobustSER-Tiny college demo

## What the demo shows

The demo runs the validation-selected E5 checkpoint locally on a WAV upload or
browser microphone recording. It converts the audio to mono 16 kHz, applies
the fixed two-second centre-crop/zero-pad policy, extracts MFCC features, and
shows all six class probabilities. It is a demonstration of a research model,
not a system for determining a person's actual emotion or mental-health state.

## Start it

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m streamlit run demo/app.py
```

The browser opens at the local address printed by Streamlit. No physical
hardware, cloud account, or external API is needed.

## Two-minute presentation flow

1. Introduce the original paper: a small quantized CNN, one-second 32×31
   Mel-filterbank input, a random 80/20 TESS-derived split, and a reported
   99.5% validation accuracy.
2. Explain the stronger evaluation protocol here: CREMA-D speaker-disjoint
   64/14/13 train/validation/test speakers, removal of label-conflicting exact
   duplicates, macro F1, three seeds, MUSAN held-out noise, and external
   RAVDESS testing.
3. Record or upload an acted two-second sentence. Point out the predicted
   class, probability chart, confidence, and audio-quality warnings.
4. State the honest headline: E5 reaches 0.4741 clean-test macro F1, compared
   with E4's 0.4160, a paired +0.0581 improvement whose 95% t interval excludes
   zero. At 20 dB MUSAN noise E5 also improves from 0.4006 to 0.4282 macro F1.
5. Close with the limitation: external RAVDESS and severe-noise improvements
   are not established, which is more credible than presenting only a very high
   random-split validation accuracy.

## Suggested live demo clips

Use short, clearly acted speech rather than a single word. Record each clip in
a quiet room and keep the microphone distance similar. The demo flags quiet,
short, and low-confidence recordings. Do not claim a successful prediction
proves it can infer a person's real emotional state.
