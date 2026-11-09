# RobustSER-Tiny

RobustSER-Tiny is a **software-only speech emotion recognition research
project**. It predicts six acted-speech labels—Angry, Disgust, Fear, Happy,
Neutral, and Sad—from short WAV recordings using a compact CNN.

This is a college research project, not a mental-health or real-emotion
assessment system.

## Why this project matters

The reference paper reported 99.5% validation accuracy from a random 80/20
split of a modified TESS dataset. That figure cannot be compared directly with
this project because the datasets, split, and metric differ.

RobustSER-Tiny strengthens the experiment by using:

- CREMA-D with a frozen, speaker-disjoint 64/14/13 train/validation/test split
- removal of exact audio files that have conflicting labels
- macro F1 and UAR instead of accuracy alone
- three-seed replication with confidence intervals
- MUSAN noise held out from augmentation for noisy evaluation
- RAVDESS as an external-only evaluation corpus

## Best validated result

E5 keeps the compact 930-parameter depthwise-separable CNN from E4 and changes
only the speech context from one to two seconds.

| Result | E4: 1 second | E5: 2 seconds |
| --- | ---: | ---: |
| Clean speaker-disjoint test macro F1 | 0.4160 | **0.4741** |
| Clean-test UAR | 0.4339 | **0.4853** |
| Parameters | 930 | 930 |

The paired E5−E4 clean macro-F1 improvement is **+0.0581** with a
three-seed 95% t confidence-interval half-width of **0.0085**. The interval
excludes zero, so this is the project's first supported improvement.

At 20 dB MUSAN noise, E5 also improves macro F1 from 0.4006 to 0.4282. Results
at 10 dB and below, and on external RAVDESS, remain inconclusive. See
[the final report](docs/final_report.md) for the complete, honest results.

## Run the demo

The demo accepts a WAV upload or a browser microphone recording. It runs
locally, converts audio to mono 16 kHz, shows all six probabilities, and flags
short, quiet, or low-confidence inputs.

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
python -m streamlit run demo/app.py
```

Open the local URL printed in the terminal. For the best demonstration, record
a clearly acted sentence of roughly two seconds in a quiet room.

More presentation guidance is in [the demo guide](docs/demo_guide.md).

## Install dependencies

Python 3.11 or later is recommended. The project was developed with CUDA-capable
PyTorch, but the demo also works on CPU.

```powershell
python -m pip install -e ".[demo]"
```

## Verify the project

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/preflight_baseline.py --config configs/experiments/e5_two_second_context.yaml
```

## Project layout

```text
configs/       Frozen experiment and data-protocol configurations
data/manifests/Reproducible audits and speaker/noise split manifests
src/           Features, datasets, models, metrics, and local inference
scripts/       Auditing, training, evaluation, calibration, and export tools
demo/          Streamlit college demonstration application
results/       Completed experiment artifacts and summaries (local)
docs/          Research protocol, reports, comparison, and demo guide
tests/         Automated regression tests
```

## Research boundaries

- The system classifies **acted speech labels**, not a person's true internal
  emotion.
- It is not suitable for healthcare, hiring, surveillance, or consequential
  decisions.
- No physical-device RAM, flash, energy, microphone, or latency result is
  claimed. The project is software-only.
- RAVDESS was never used for training or checkpoint selection.

## Reference-paper comparison

Use [paper comparison](docs/paper_comparison.md) for your viva or report. It
explains the methodological improvements and includes a concise presentation
statement that avoids invalidly claiming to beat the paper's 99.5% score.
