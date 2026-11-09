# How RobustSER-Tiny improves on the reference paper

Reference: *TinyML-Based Speech Emotion Recognition on Microcontrollers Using
Quantized CNNs and Mel-Filterbank Energy Features* (Ahnaf, 2026).

## Important: do not compare the headline accuracies directly

The paper reports 99.5% validation accuracy and weighted F1 of 0.99 on its
modified TESS dataset using an 80/20 split. RobustSER-Tiny reports 0.4741
three-seed mean **macro F1** on speaker-disjoint CREMA-D test speakers. These
are different corpora, labels, metrics, and split protocols, so they are not a
head-to-head accuracy contest. A lower but leakage-resistant result is more
useful evidence of expected performance on unfamiliar speakers.

## Improvements brought by this project

| Area | Reference paper | RobustSER-Tiny improvement | Evidence |
| --- | --- | --- | --- |
| Evaluation split | Random 80/20 train/test split | Frozen 64/14/13-speaker train/validation/test split | No speaker appears in more than one split |
| Data scale/diversity | Modified TESS: 2,000 speech clips plus 720 Noise examples | 7,436 eligible CREMA-D recordings from 91 speakers | Raw data audit and manifest |
| Metrics | Validation accuracy and weighted F1 | Macro F1 primary, UAR, per-class results, confidence intervals | Three independently seeded runs |
| Noise robustness | Adds a separate Noise class | Mixes actual MUSAN noise into speech at fixed SNRs, with held-out noise files for evaluation | 20, 10, 5, 0 dB protocol |
| External validation | No second corpus evaluation | RAVDESS evaluated only after model selection | 1,056 external records, never used for training |
| Model efficiency | Quantized deployed model, hardware measurements | 930-parameter depthwise-separable CNN, 94.1% fewer parameters than our 15,702-parameter baseline | Exact parameter count, ONNX export |
| Temporal information | One-second 32×31 Mel energy input | Two-second 32×62 MFCC context, same 930-parameter model | Controlled E4→E5 ablation |
| Honest limitations | Notes field degradation qualitatively | Quantifies where the model fails: low-SNR and external-corpus uncertainty | Held-out MUSAN and RAVDESS results |

## Strongest measured result

The controlled E5 change doubles only the audio context from one to two
seconds. It raises clean-test macro F1 from **0.4160** to **0.4741**:

| Measure | Result |
| --- | ---: |
| E5 minus E4, clean macro F1 | +0.0581 |
| Three-seed 95% t confidence-interval half-width | 0.0085 |
| E5 parameter count | 930 |
| 20 dB MUSAN macro F1, E4 → E5 | 0.4006 → 0.4282 |

The clean-test interval excludes zero, so this is an experimentally supported
improvement within this project's protocol. At 10 dB and below, and on
RAVDESS, the evidence is inconclusive; say this explicitly in the viva.

## A concise viva statement

> The paper shows that a compact CNN can fit a clean, small dataset and run on
> a microcontroller. Our project improves the scientific reliability of the
> claim: we prevent speaker leakage, test with real held-out noise, test on a
> separate corpus, report macro F1 and uncertainty, and still obtain a
> statistically supported 5.81-point gain by using two seconds of speech
> context with only 930 model parameters. We do not claim to beat its 99.5%
> number because the datasets and evaluation protocols are different.
