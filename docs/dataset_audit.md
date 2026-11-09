# CREMA-D dataset audit

## Exact-duplicate investigation — 2026-11-08

The immutable source audit manifest is `data/manifests/crema_d_manifest.csv`.
The SHA-256 field is calculated from the complete WAV file bytes, so records
with the same digest are exact file duplicates, not merely acoustically
similar recordings.

Three duplicate groups were identified. Every group contains two recordings
from the same speaker but with different emotion labels:

| SHA-256 (prefix) | Recordings | Classification | Handling |
| --- | --- | --- | --- |
| `0de0a7bf…` | `1017_IWW_ANG_XX.wav`, `1017_IWW_FEA_XX.wav` | Exact-audio label conflict | Exclude both from the eligible modeling manifest pending provenance review. |
| `776a8c97…` | `1006_TIE_HAP_XX.wav`, `1006_TIE_NEU_XX.wav` | Exact-audio label conflict | Exclude both from the eligible modeling manifest pending provenance review. |
| `c9b6d56f…` | `1013_WSI_DIS_XX.wav`, `1013_WSI_SAD_XX.wav` | Exact-audio label conflict | Exclude both from the eligible modeling manifest pending provenance review. |

No raw WAV file, original manifest row, or label has been changed. The
conflicting records must not be assigned competing target labels in training,
validation, or testing. Their shared speaker IDs mean they do not create a
cross-partition speaker-leakage case on their own, but the label conflict still
makes them ineligible for supervised model development.

`scripts/create_modeling_manifest.py` is the reproducible policy mechanism.
It produces:

- `data/manifests/crema_d_duplicate_conflicts.csv`: every duplicate record and its disposition.
- `data/manifests/crema_d_eligible_modeling_manifest.csv`: structurally valid rows excluding any exact-audio label or speaker-metadata conflict.

For this source manifest, the expected eligible count is **7,436**. Future
speaker-independent splitting must use this eligible manifest and retain the
conflict report alongside the split manifest.

## MUSAN noise corpus audit — 2026-11-09

MUSAN was added under `dataset/MUSAN/` without modifying CREMA-D. The audit in
`data/manifests/musan_manifest.csv` found 2,016 valid 16 kHz, mono, 16-bit WAV
files: 660 music, 930 noise, and 426 speech recordings. It found no exact
duplicate file occurrences.

`data/manifests/musan_noise_splits.csv` freezes a SHA-256-disjoint 80/20
partition using seed `20261109`: 1,613 recordings are available for training
augmentation and 403 are reserved solely for held-out noisy evaluation. No
MUSAN file may be selected independently outside this manifest.

## Acoustic-quality audit protocol

`scripts/audit_audio_quality.py` reads the eligible manifest in chunks and
does not alter audio. It measures duration, sample rate, channels, bit depth,
RMS level (dBFS), peak level (dBFS), digital-silence proportion, and clipping
proportion for every eligible WAV file. Results are saved to
`data/manifests/crema_d_audio_quality.csv`.

Flags are review candidates rather than exclusions: duration under 0.5 s or
over 8 s, RMS below -45 dBFS, peak below -30 dBFS, at least 0.1% full-scale
samples, at least 95% digital zeros, audio read failures, and robust (median
absolute deviation) duration/RMS outliers. No sample should be removed until
the flagged records and their effect on class/speaker coverage are documented.

### Initial acoustic-quality results — 2026-11-08

The first run covered all **7,436** eligible recordings and had **zero** audio
read failures. It flagged **286** records for review (records can have more
than one flag): 246 for the conservative clipping threshold, 31 for low RMS,
14 for low peak level, one for mostly digital silence, one robust RMS outlier,
and one robust duration outlier. These flags are not exclusions and are not
evidence of corruption. The next audit decision is a targeted review of the
unique flagged records, including class/speaker distribution and waveform
inspection, before determining whether any exclusion is justified.

The targeted distribution review found that clipping flags cluster strongly in
high-energy emotional delivery (174 Angry, 51 Fear, 22 Happy, 30 Sad, and 9
Disgust records). The most clipped recordings have normal RMS levels around
-10 to -12 dBFS and reach 0 dBFS, consistent with saturated peaks rather than
unreadable or silent files. Because excluding these recordings would
selectively change the emotion distribution, no additional quality-based
exclusions are justified at this stage. The eligible modeling manifest remains
frozen at **7,436** records; all flags remain available for sensitivity
analysis and qualitative inspection.

## Speaker-independent partition — 2026-11-08

`scripts/create_speaker_splits.py` assigns every eligible record through its
speaker ID using seed `20261108`. A deterministic speaker-level swap optimizer
targets a 70/15/15 record ratio while maintaining whole-speaker membership.
The generated `crema_d_speaker_splits.csv` and its JSON summary are the frozen
benchmark protocol; models must not regenerate partitions independently.

| Split | Records | Speakers | Angry / Disgust / Fear / Happy / Neutral / Sad |
| --- | ---: | ---: | --- |
| Train | 5,247 | 64 | 896 / 896 / 896 / 896 / 767 / 896 |
| Validation | 1,123 | 14 | 192 / 192 / 192 / 192 / 163 / 192 |
| Test | 1,066 | 13 | 182 / 182 / 182 / 182 / 156 / 182 |

Automated validation confirms no speaker overlap, no exact-audio hash crosses a
partition boundary, all manifest paths resolve, and all six CREMA-D labels
occur in every split. No augmented recordings exist yet; when created, each
derivative must inherit its source recording's frozen split and must never be
assigned independently.
