# Experiment log

## E0 — paper-inspired small CNN, six-emotion CREMA-D

**Status:** completed train/validation run and one held-out test evaluation.

| Field | Value |
| --- | --- |
| Run directory | `results/E0_six_emotion_20261109_092621` |
| Task | Six-emotion CREMA-D (Angry, Disgust, Fear, Happy, Neutral, Sad) |
| Protocol | Frozen speaker-disjoint split, 64 / 14 / 13 train / validation / test speakers |
| Selected checkpoint | Epoch 47 by validation macro F1 |
| Best validation macro F1 | 0.4354 |
| Held-out test macro F1 | 0.3934 |
| Held-out balanced accuracy / UAR | 0.4049 |
| Held-out accuracy | 0.4053 |
| Test records | 1,066 |
| Test evaluation count | One |
| Device | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA) |

This is a single seed, paper-inspired baseline result. It must not be
represented as an improvement over the published work, whose dataset,
partitioning, task labels, and reported evaluation protocol differ.

### Held-out per-class F1

| Emotion | F1 |
| --- | ---: |
| Angry | 0.5744 |
| Disgust | 0.3212 |
| Fear | 0.3321 |
| Happy | 0.2361 |
| Neutral | 0.3720 |
| Sad | 0.5246 |

### Initial error observations

- Happy has the weakest recall (36 / 182) and is often predicted as Angry,
  Disgust, or Neutral.
- Fear is frequently predicted as Sad (59 / 182).
- Disgust is often confused with Neutral (36 / 182) and Sad (39 / 182).

These are hypotheses for later controlled feature, augmentation, and model
ablations—not conclusions about which intervention will help. The frozen test
set must not be reused to tune E0. Future variants should be chosen using the
validation partition and evaluated once on the test set after selection.

## E1 — multi-seed speaker-independent baseline replication

**Status:** completed for pre-registered seeds 20261108, 20261109, and
20261110. Each seed used its validation-selected checkpoint and exactly one
held-out test evaluation.

| Metric | Mean | Sample SD | 95% t CI half-width |
| --- | ---: | ---: | ---: |
| Validation macro F1 | 0.4306 | 0.0098 | 0.0244 |
| Held-out test macro F1 | 0.3815 | 0.0107 | 0.0265 |
| Held-out balanced accuracy / UAR | 0.3988 | 0.0053 | 0.0132 |
| Validation-to-test macro-F1 gap | 0.0492 | — | — |

Mean held-out per-class F1 is Angry 0.56, Sad 0.52, Neutral 0.38, Fear 0.29,
Disgust 0.28, and Happy 0.26. Mean recall is particularly low for Fear (0.22)
and Happy (0.23). These estimates are based on only three seeds, so their
confidence intervals are wide; no run or seed is selected as "best" for later
comparison.

### Decision for the next experiment

E2 should introduce only controlled noise augmentation while retaining the
frozen split, feature representation, model, seeds, and validation selection
rule. It requires MUSAN (or another documented, license-compatible noise
corpus) with training and evaluation noise files kept disjoint. That corpus is
not present in this workspace, so E2 must not be run or claimed until its data
provenance and split are audited.

## E2 — controlled MUSAN noise augmentation, seed 20261108

**Status:** completed train/validation, one clean test, and one pre-registered
held-out MUSAN noisy test suite. Training mixed MUSAN augmentation-partition
audio into 50% of train examples at uniform 5–20 dB SNR. Validation remained
clean. The 403 MUSAN evaluation-holdout files were used only for the noisy test
suite.

| Condition | Macro F1 | UAR |
| --- | ---: | ---: |
| Clean test | 0.3897 | 0.4129 |
| MUSAN, 20 dB | 0.3774 | 0.3997 |
| MUSAN, 10 dB | 0.3573 | 0.3683 |
| MUSAN, 5 dB | 0.3268 | 0.3370 |
| MUSAN, 0 dB | 0.2903 | 0.2967 |

The validation-selected checkpoint was epoch 38 with validation macro F1
0.4049. The clean test result is close to the E1 three-seed mean (0.3815), but
one E2 seed cannot establish a clean-performance or robustness improvement.
The expected degradation across decreasing SNR is observed. E2 must be
replicated across its two remaining pre-registered seeds before any comparison
against E1 is made.

## E2 — three-seed replication summary

| Metric | E1 mean | E2 mean | E2 95% t CI half-width |
| --- | ---: | ---: | ---: |
| Validation macro F1 | 0.4306 | 0.4074 | 0.0336 |
| Clean held-out macro F1 | 0.3815 | 0.3738 | 0.0529 |
| Clean held-out UAR | 0.3988 | 0.4044 | 0.0328 |

E2's clean macro F1 is 0.0077 below E1's mean, with wide uncertainty at three
seeds. This is **not evidence that the chosen MUSAN augmentation improves
clean performance**. It does improve Happy mean F1 relative to E1 (0.342 vs
0.262), but Disgust and Fear decrease materially; this per-class observation
is exploratory and not a basis for a claim.

| Held-out MUSAN SNR | Mean macro F1 | Mean UAR |
| --- | ---: | ---: |
| 20 dB | 0.3703 | 0.3961 |
| 10 dB | 0.3490 | 0.3686 |
| 5 dB | 0.3212 | 0.3392 |
| 0 dB | 0.2921 | 0.3026 |

No same-protocol noisy E1 baseline has yet been measured. Therefore these E2
noise results characterize E2 but cannot establish an augmentation-driven
robustness improvement over E1. The next controlled action is to evaluate the
already fixed E1 checkpoints with the same MUSAN evaluation-holdout manifest
and fixed SNR schedule, without retraining or selecting models.

## Matched E1 versus E2 held-out MUSAN comparison

The fixed E1 checkpoints were subsequently evaluated once at the same SNRs,
with the same CREMA-D test recordings, MUSAN evaluation-holdout partition, and
seed identities. Differences below are paired E2 minus E1 macro F1 across the
three corresponding seeds.

| SNR | E1 mean macro F1 | E2 mean macro F1 | Paired difference | 95% t CI half-width |
| --- | ---: | ---: | ---: | ---: |
| 20 dB | 0.3792 | 0.3703 | -0.0089 | 0.0362 |
| 10 dB | 0.3305 | 0.3490 | +0.0185 | 0.0276 |
| 5 dB | 0.2996 | 0.3212 | +0.0217 | 0.0670 |
| 0 dB | 0.2677 | 0.2921 | +0.0244 | 0.0669 |

The direction is favorable to E2 at 10, 5, and 0 dB, but every three-seed
confidence interval includes zero. The scientifically defensible statement is
that this augmentation setting is a **promising but unconfirmed low-SNR
robustness candidate**; it has not demonstrated a statistically reliable
improvement and does not improve clean macro F1.

## E3 — MFCC feature ablation, three-seed replication

| Metric | E1 log-Mel mean | E3 MFCC mean | E3 95% t CI half-width |
| --- | ---: | ---: | ---: |
| Validation macro F1 | 0.4306 | 0.4227 | 0.0453 |
| Clean held-out macro F1 | 0.3815 | 0.4080 | 0.0246 |
| Clean held-out UAR | 0.3988 | 0.4230 | 0.0400 |

The paired E3-minus-E1 test macro-F1 difference is +0.0266, with a three-seed
95% t CI half-width of 0.0497. The point estimate and two seeds favor MFCC,
but the interval includes zero. Thus MFCC is a **promising clean-performance
candidate**, not a statistically established improvement.

MFCC has higher mean F1 than E1 for Angry (0.617 vs 0.561) and Happy (0.372 vs
0.262), but lower mean F1 for Fear (0.262 vs 0.288). These per-class results
are descriptive only; the next experiment should use validation-only choices
to investigate a lightweight architecture or feature parameter change, then
evaluate once on the frozen test protocol.

## E4 — depthwise-separable CNN, three-seed replication

| Metric | E3 conventional CNN | E4 depthwise-separable CNN |
| --- | ---: | ---: |
| Trainable parameters | 15,702 | 930 |
| Mean clean-test macro F1 | 0.4080 | 0.4160 |
| Mean clean-test UAR | 0.4230 | 0.4339 |

E4 reduces parameter count by **94.1%** while producing the highest current
clean-test macro-F1 point estimate. Its paired E4-minus-E3 macro-F1 difference
is +0.0079 with a three-seed 95% t CI half-width of 0.0593, so there is no
statistically reliable accuracy-gain claim. The defensible finding is that E4
is a strong **efficiency candidate**: it retains comparable measured accuracy
while drastically reducing parameter count. Deployment measurements and INT8
conversion are required before making memory, flash, or latency claims.

## External RAVDESS evaluation — canonical E4 checkpoint

The canonical pre-registered E4 seed (20261108) was evaluated once on 1,056
RAVDESS audio-only speech recordings sharing the six CREMA-D labels. RAVDESS
remained external-only: it was resampled from 48 kHz to 16 kHz at evaluation
time and was never used for training or model selection.

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.3494 |
| Accuracy | 0.3835 |
| UAR | 0.3585 |

This is a single canonical-checkpoint cross-corpus result, not a three-seed
average. The decline from the in-corpus E4 point estimate is evidence that
cross-corpus generalization remains difficult. Neutral is especially weak
(F1 0.0982), while Angry (0.5749), Disgust (0.4136), Fear (0.4092), and Sad
(0.3900) transfer comparatively better in this single run.

Across all three E4 seeds, RAVDESS macro F1 is **0.3216** with sample standard
deviation 0.0356 and a three-seed 95% t CI half-width of 0.0885. This confirms
substantial cross-corpus uncertainty and a marked drop relative to the
in-corpus test estimate. RAVDESS remains external-only and was not used to
tune any model or select a checkpoint.

## E5 — two-second temporal context, three-seed replication

| Metric | E4 one-second context | E5 two-second context |
| --- | ---: | ---: |
| Validation macro F1 | 0.4270 | 0.4776 |
| Clean test macro F1 | 0.4160 | 0.4741 |
| Clean test UAR | 0.4339 | 0.4853 |
| Parameters | 930 | 930 |

The paired E5-minus-E4 clean-test macro-F1 improvement is **+0.0581** with a
three-seed 95% t CI half-width of **0.0085**. This interval excludes zero,
making E5 the first experimentally supported performance improvement in this
project. It supports the hypothesis that the one-second pipeline discarded
important emotional context. E5 retains E4's 930-parameter model; increased
DSP/input processing cost must be reported separately.

## External RAVDESS evaluation — E5 three-seed replication

The E5 checkpoints were evaluated on the same 1,056 external RAVDESS
recordings, using E5's two-second centre-crop/zero-pad policy. RAVDESS was not
used in E5 training, selection, or tuning.

| Metric | E4 one-second context | E5 two-second context |
| --- | ---: | ---: |
| Mean external macro F1 | 0.3216 | 0.3458 |
| E5 sample standard deviation | — | 0.0615 |
| E5 95% t CI half-width | — | 0.1528 |

The paired E5-minus-E4 external macro-F1 difference is +0.0242, but its
three-seed 95% t CI half-width is 0.2357. Consequently, the apparent external
increase is **not established**. The clean in-corpus E5 improvement is real
under the current replication protocol, while cross-corpus generalization
remains uncertain and must not be presented as improved.

## Held-out MUSAN noisy evaluation — controlled E4 versus E5 comparison

Both E4 and E5 were evaluated once per seed at each pre-registered SNR on the
same CREMA-D test speakers and the MUSAN evaluation holdout. E4 used its fixed
one-second input; E5 used its fixed two-second input. No checkpoint, seed, or
SNR was selected after observing these results.

| SNR | E4 mean macro F1 | E5 mean macro F1 | Paired E5−E4 | 95% t CI half-width | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| 20 dB | 0.4006 | 0.4282 | +0.0275 | 0.0243 | Supported improvement |
| 10 dB | 0.3679 | 0.3894 | +0.0215 | 0.1008 | Inconclusive |
| 5 dB | 0.3417 | 0.3405 | −0.0012 | 0.0495 | Inconclusive |
| 0 dB | 0.3123 | 0.3228 | +0.0106 | 0.0119 | Inconclusive |

E5 therefore improves clean performance and shows a supported gain under
moderate (20 dB) noise, but it has **not** established stronger robustness at
10 dB or below. The next improvement must target severe-noise robustness or
cross-corpus transfer without selecting on the frozen clean test set.
