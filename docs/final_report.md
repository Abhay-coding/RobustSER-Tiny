# RobustSER-Tiny: software-only research report

## Scope

This project evaluates speaker-independent speech emotion recognition using
CREMA-D and performs external RAVDESS evaluation. It is software-only: no
claims are made about physical-device RAM, flash, energy, microphone behavior,
or end-to-end embedded latency.

## Protocol

CREMA-D was structurally audited, exact label-conflicting duplicates were
excluded from modeling, and a frozen speaker-disjoint 64/14/13-speaker
train/validation/test split was used throughout. Macro F1 is the primary
metric. MUSAN augmentation and evaluation noise were SHA-256 partitioned to
prevent reuse across augmentation and noisy evaluation.

## Main results

| Experiment | Primary change | Clean test macro F1 (3-seed mean) |
| --- | --- | ---: |
| E1 | Paper-inspired log-Mel CNN baseline | 0.3815 |
| E2 | E1 + controlled MUSAN augmentation | 0.3738 |
| E3 | MFCC features | 0.4080 |
| E4 | MFCC + depthwise-separable CNN | 0.4160 |
| E5 | E4 + two-second temporal context | 0.4741 |

E4 reduced the trainable parameter count from 15,702 to 930 (94.1%) while
retaining comparable measured accuracy. The E4-versus-E3 paired macro-F1
difference was +0.0079 with a three-seed 95% t CI half-width of 0.0593, so it
is an efficiency finding rather than a statistically confirmed accuracy gain.

E5 retains E4's 930 trainable parameters but doubles only the input time
context. Its paired E5-minus-E4 clean-test macro-F1 difference is **+0.0581**
with a three-seed 95% t CI half-width of **0.0085**, which excludes zero. This
is the project's first statistically supported performance improvement.

## Noise robustness

E2 was promising but unconfirmed at low SNR. Relative to E1, E2-minus-E1 mean
macro-F1 differences were -0.0089 at 20 dB, +0.0185 at 10 dB, +0.0217 at 5 dB,
and +0.0244 at 0 dB. All three-seed confidence intervals included zero.

The controlled E5-versus-E4 comparison shows E5 is better at moderate 20 dB
noise (0.4282 versus 0.4006 macro F1; paired difference +0.0275, 95% t
CI half-width 0.0243). The 10 dB, 5 dB, and 0 dB comparisons are inconclusive;
E5 must not be claimed to improve severe-noise robustness.

## External generalization

The three E4 checkpoints achieved RAVDESS external macro F1 of 0.3216 with a
95% t CI half-width of 0.0885. This substantial decline from in-corpus results
demonstrates that cross-corpus generalization remains challenging.

E5's RAVDESS mean is 0.3458, but its three-seed 95% t CI half-width is 0.1528
and the paired E5-minus-E4 difference (+0.0242) has a 0.2357 half-width.
Therefore, no external-generalization improvement is established.

## Conclusions

The project does not reproduce the published 99.5% validation claim because it
uses a stricter speaker-disjoint protocol and different corpus. It establishes
that a 930-parameter MFCC depthwise-separable CNN benefits materially from
two seconds of context under the current clean protocol, while generalization
across corpora and severe noise remains the limiting problem.

## Limitations and future work

- Only three random seeds were used, producing wide confidence intervals.
- RAVDESS was external-only and not used for transfer learning.
- No physical-device deployment measurements were performed.
- Calibration did not improve validation ECE in the canonical E4 run.
