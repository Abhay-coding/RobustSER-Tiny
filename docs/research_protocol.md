# RobustSER-Tiny research protocol

## Frozen source data

All baseline experiments use `data/manifests/crema_d_speaker_splits.csv` and
must preserve its assignments: 64 training speakers, 14 validation speakers,
and 13 held-out test speakers. The test split is not used for architecture,
augmentation, threshold, or hyperparameter selection.

## Separate label tasks

Two task definitions are required because their results answer different
questions.

| Task | Labels | Use |
| --- | --- | --- |
| `five_emotion_common` | Angry, Disgust, Fear, Happy, Neutral | Comparable later with corpora sharing these labels. Sad is excluded from this task, never relabeled. |
| `crema_d_six_emotion` | Angry, Disgust, Fear, Happy, Neutral, Sad | Full in-corpus CREMA-D benchmark. |

Metrics must never be compared across these different label spaces as though
they measure the same task. Macro F1 on the speaker-disjoint held-out test set
is the primary endpoint for each task; validation macro F1 governs model
selection.

## E0: paper-inspired baseline definition

`configs/baseline.yaml` fixes E0 before implementation. It retains the
paper-reported 16 kHz input, 32 Mel filters, 32 ms frame length/stride, Adam,
0.005 learning rate, batch size 32, and an 80-epoch maximum. A one-second
16-kHz window yields 31 complete 32-ms frames with no overlap, making the
specified feature shape 32 x 31.

The original publication's exact CNN and preprocessing details are not fully
recoverable from the reported description, including its 32x31 versus 4,018
input discrepancy. Therefore this experiment is explicitly **paper-inspired**,
not an exact reproduction. The compact two-block CNN and crop/padding policy
are documented assumptions, not claims about the published implementation.

No augmentation, quantization, external corpus, or confidence rejection is
permitted in E0. Those factors belong to later controlled experiments.

## Feature-extraction contract

`src/robustser/features/log_mel.py` implements the E0 feature contract without
altering source WAVs. Audio is converted by the future data loader to float
mono samples, fitted to one second (16,000 samples), divided into 31
non-overlapping 512-sample frames, Hann-windowed, transformed with a 512-point
FFT, mapped through 32 triangular HTK Mel filters from 20 Hz to 8 kHz, floored
at `1e-6`, log-transformed, and normalized per utterance. The output is always
a finite `float32` matrix shaped `[32, 31]`.

For evaluation, long utterances use the deterministic center one-second crop;
short utterances use right zero-padding. Training will use a seeded random crop
to avoid always learning from the same one-second region while remaining
reproducible.

## Pre-training readiness checks

`src/robustser/data/cremad.py` is the only E0 data access layer. It reads the
frozen split manifest, selects the requested task's labels, reads the audited
16-bit mono WAV files, and returns PyTorch tensors shaped `[1, 32, 31]` plus a
fixed label index. Training crop selection is a deterministic function of the
experiment seed, epoch, and manifest row; validation and test remain center
crops.

`src/robustser/models/paper_inspired_cnn.py` defines the explicit two-block
E0 CNN assumption and returns unnormalized class logits. It has 15,702
trainable parameters for the six-emotion task. `scripts/preflight_baseline.py`
loads a real training example, runs a no-gradient forward pass, and verifies
the data/model shapes without updating any model parameter. It is the required
readiness check immediately before training code is introduced.

## Training protocol

`scripts/train.py` uses only the frozen train and validation partitions. It
records train/validation metrics after every epoch, selects a checkpoint only
by validation macro F1, saves both the best and last checkpoints, and stops
after 12 non-improving validation epochs (or 80 epochs). It explicitly never
loads the test partition. The held-out test evaluation is a separate later
action after the completed run is reviewed.

`scripts/evaluate.py` loads only the validation-selected checkpoint and the
frozen test split. It refuses to run if `test_metrics.json` already exists in
the requested run directory, protecting this protocol from accidental repeated
test-set iteration. Its result records the checkpoint epoch, label mapping,
device, complete metrics, and confusion matrix.

## E1: multi-seed baseline replication

E1 is pre-registered in `configs/experiments/e1_speaker_independent_replicates.yaml`.
It retains E0's frozen data, feature, model, and train/validation protocol and
runs seeds 20261108, 20261109, and 20261110. Its purpose is to estimate
variation, not to select a lucky seed. Each completed run receives exactly one
held-out test evaluation; `scripts/summarize_experiment.py` reports mean,
sample standard deviation, and a 95% t confidence interval across all three.

## E2: controlled MUSAN noise augmentation

E2 is defined in `configs/experiments/e2_controlled_musan_noise.yaml`. For
each training example, it has a 50% chance of mixing one MUSAN file from the
`augmentation_train` partition at a uniformly sampled 5–20 dB RMS SNR. The
clean waveform is retained otherwise. Validation remains clean and determines
checkpoint selection. The held-out MUSAN partition is reserved for later noisy
test evaluation at 20, 10, 5, and 0 dB; it is never used as training noise.
`scripts/evaluate_noisy.py` evaluates all four pre-registered SNR conditions
in one safeguarded run and refuses to overwrite its result file.

## E3: MFCC feature ablation

E3 is defined in `configs/experiments/e3_mfcc_features.yaml`. It replaces the
normalized 32-channel log-Mel matrix with 32 MFCC coefficients computed from
the same 32 Mel filters and 31 frames. The matrix shape remains `[32, 31]`, so
the frozen speaker split, crop policy, CNN, optimizer, augmentation policy,
and training schedule are unchanged. E3 therefore isolates the feature
representation as its single experimental factor.

## E4: depthwise-separable CNN ablation

E4 is defined in `configs/experiments/e4_depthwise_separable_cnn.yaml`. It
retains E3's MFCC features and every data/training factor, while replacing the
paper-inspired CNN with a depthwise-separable CNN plus global average pooling.
The E4 model has fewer than 1,000 trainable parameters for six classes versus
15,702 for the paper-inspired model. It is a resource-efficiency hypothesis,
not an assumed accuracy improvement.

## E5: two-second temporal context

E5 uses `configs/experiments/e5_two_second_context.yaml`. Relative to E4, it
changes only the analysis window from one second (31 frames) to two seconds
(62 frames). The DS-CNN's global pooling makes it compatible with this longer
time axis without increasing its parameter count.
