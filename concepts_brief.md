# RobustSER-Tiny — Goldfish Synthesis Brief

* **Author:** Abhay Kumar
* **Date:** 2026-11-08

## Overview

RobustSER-Tiny aims to deliver speaker-independent, noise-robust, uncertainty-aware speech emotion recognition on microcontrollers. The project has completed CREMA-D data preparation (7,436 eligible samples after deduplication) with a SHA-256 manifest, but has not yet implemented any training, feature extraction, quantization, or deployment code. Staged experiments E0–E8 are planned but remain entirely on paper. This review synthesizes critiques from seven independent lenses to validate scope, expose risks, and set a pragmatic path forward.

## Analysis

**Strongest Objections / Risks**

1. **Nine staged experiments (E0–E8) is premature academic over-design** (Contrarian/YAGNI). Only E0 (baseline) and E1 (speaker-independent split) are validated by current evidence. If E1 fails to beat random on held-out speakers (macro F1 < 0.55), the entire hypothesis collapses and E2–E8 become wasted work. The plan should run E0→E1, then gate all subsequent experiments on E1's held-out-speaker macro F1 exceeding a meaningful threshold (e.g., >0.60).

2. **No target MCU selected** (Cost/Economics). Model sizing, memory budgeting, latency targets, and quantization strategy are impossible without committing to a specific device. STM32H747 (Cortex-M7, 1 MB RAM, CMSIS-NN) is the only viable candidate among common MCUs; ESP32-S3 is marginal without PSRAM; RP2040 is non-viable. Commit to STM32H747 now and size the model accordingly (≤200 KB INT8, ≤5–10M MACs, ≤300 KB RAM).

3. **Zero ergonomics/MLOps infrastructure** (UX/Ergonomics). No `pyproject.toml`, lockfile, config system (Hydra), experiment tracking (MLflow), data versioning (DVC), CI/CD, or onboarding docs. The project is cloneable but not runnable. These must be built as first-class deliverables alongside E0, not afterthoughts.

4. **No deployment lifecycle** (Migration/Compatibility). Model versioning, dataset versioning, OTA rollout, rollback strategy, inference contracts, quantization config versioning, and schema evolution are entirely absent. The data layer shows good hygiene (immutable manifests, SHA-256), but everything above it is a blank slate. Treat this as a requirements spec for the missing MLOps layer.

5. **Zero observability scaffolding** (Observability/SRE). No structured logging, metrics contracts, failure diagnosis tooling, confidence/calibration monitoring, edge telemetry, alerting, or runbooks. Observability must be designed before E0 training begins, not bolted on later.

6. **High technical debt surface area** (Tech Debt/Maintenance). Hardcoded paths, implicit assumptions (16 kHz, mono, CREMA-D filename pattern), single-dataset coupling, no dependency lockfile, no type-checking/linting gates, no integration tests, and bus factor = 1. Extract shared schemas/config now; add `pyproject.toml` + CI + ADRs + integration tests before model code arrives.

7. **Core contradiction: model capacity vs. edge constraints** (Inventive Catalyst). Accuracy, speaker independence, noise robustness, and uncertainty awareness fundamentally conflict with <400ms latency, INT8, and microcontroller memory. Optimization alone cannot resolve this. The IFR (Ideal Final Result) implies **all learning happens offline; deployment is a tiny frozen projector (~20 KB INT8 linear layer) running on pre-computed SSL embeddings**. This resolves the contradiction via TRIZ Principle 24 (Separation in Time): build-time intelligence, deploy-time pure inference.

**Inventive Contradictions & Resolutions**

| Contradiction | TRIZ Principle | Resolution Pattern |
|---------------|----------------|-------------------|
| Accuracy vs. Size | 15 (Dynamics) | Asymmetric features: heavy SSL offline → light projector online |
| Speaker Independence vs. 91 speakers | 28 (Mechanics Substitution) | Self-supervised pre-training on unlabeled speech (VoxCeleb, LibriSpeech) |
| Noise Robustness vs. Clean Data | 3 (Local Quality) | Multi-condition SSL pre-training; deploy only denoising projection |
| Uncertainty vs. Calibration Cost | 1 (Segmentation) | Cascaded inference: cheap deterministic head + uncertainty head on low-confidence only |
| Latency vs. Feature Richness | 10 (Preliminary Action) | Pre-compute speaker embeddings offline; cache on-device |

**Secondary Trade-offs Accepted**
- Offline GPU compute required for SSL pre-training and distillation (accepted; one-time build cost)
- Embedding distribution shift risk if test speakers differ from SSL corpus (mitigated by large pre-training corpus)
- Cascaded inference worst-case latency still hits full model (acceptable if threshold tuned for typical case)
- "Projector-only" deployment limits on-device adaptation (accepted; speaker independence achieved at build time)

**Ideal Final Result Direction**
The system classifies emotions perfectly with **zero neural network on the MCU** — only a frozen, quantized linear projector (≈20 KB INT8) operating on frame-pooled SSL embeddings. All speaker adaptation, noise robustness, and uncertainty calibration happen at build time via self-supervised pre-training + adversarial head training + distillation. On-device latency <50ms, energy ~2 μJ/inference, 100% speaker-independent by construction.

## Goals vs. Non-Goals

**In-Scope Goals (Validated by Review)**
- Speaker-independent SER on CREMA-D with macro F1 > 0.65 on held-out speakers
- INT8 TFLite Micro deployment on STM32H747 with <400ms total latency (target <50ms for projector-only)
- Noise robustness via offline multi-condition SSL pre-training (not on-device denoising)
- Uncertainty awareness via distillation temperature + Mahalanobis distance in embedding space (no Monte Carlo)
- Reproducible pipeline: Hydra config + MLflow tracking + DVC data versioning + CI gates
- Observability: structured JSON logging, metrics contract, diagnosis CLI, calibration export, edge telemetry schema
- Migration readiness: model/dataset versioning, inference contract schema, quantization config versioning, rollback design

**Out-of-Scope Non-Goals (Explicitly Excluded)**
- E2–E8 experiments (noise aug, alt features, DS-CNN, calibration, INT8 QAT, DSP opt, distillation) — deferred until E1 baseline validated
- Cross-corpus evaluation (RAVDESS, TESS, MUSAN) — single-corpus CREMA-D first
- On-device feature extraction (MFCC/log-mel) — precompute to disk, ship tensors
- Custom DS-CNN architecture — use MobileNetV3-small / frozen SSL projector
- NAS / architecture search — fixed tiny architecture only
- Real-time streaming / VAD — batch inference first; streaming as follow-up
- RP2040 / ESP32-S3 without PSRAM — commit to STM32H747
- Multi-language / multi-dataset abstraction — single dataset first, adapter pattern later

## Alternatives Weighed

| Option | Rejected Because |
|--------|------------------|
| **Full E0–E8 pipeline now** | Premature; E1 gate not passed; wastes compute if hypothesis invalid |
| **Custom mel-spectrogram + CMVN** | `torchaudio.transforms` + `torchaudio.functional` handles in C++/CUDA; precompute to disk |
| **Custom noise augmentation (MUSAN mixing)** | `audiomentations` / `torch-audiomentations` GPU-accelerated, battle-tested |
| **Custom DS-CNN architecture** | MobileNetV3-small (adapt first conv) or frozen SSL projector smaller, faster, pre-quantized |
| **Custom INT8 quantization + TFLite Micro delegate** | `torch.ao.quantization` PTQ/QAT → `ai_edge_torch` → TFLite Micro flatbuffer |
| **Custom calibration (E5)** | `torch.nn.functional.temperature_scale` or `netcal` library |
| **Custom training loop** | PyTorch Lightning / SpeechBrain handles DDP, mixed precision, checkpointing |
| **MLflow vs. W&B** | MLflow chosen: self-hosted, offline-capable, lightweight |
| **DVC vs. LakeFS** | DVC chosen: integrates with git, works with local/S3 remotes, simpler for single-user |
| **Hydra vs. plain YAML** | Hydra chosen: config composition, overrides, dataclass validation, experiment reproducibility |
| **Projector-only vs. full model on-device** | Projector-only resolves core contradiction; full model violates latency/memory budget |

## Key Decisions & Staging for Next Steps

1. **Commit to STM32H747** as target MCU. Define memory map: ≤200 KB INT8 model, ≤300 KB RAM (activations + buffers + audio), ≤5–10M MACs/inference.

2. **Adopt ergonomics stack now** (Week 0):
   - `pyproject.toml` + `uv.lock` + `README.md` + `justfile`
   - Hydra config skeleton (`conf/{config.yaml, model/, data/, trainer/, experiment/}`)
   - MLflow tracking (local) + DVC for data versioning
   - GitHub Actions: lint (ruff, mypy), test (pytest), train-smoke (1 epoch CPU), export-validate

3. **Run E0→E1 gate** (Week 1–2):
   - Speaker-disjoint split: 20% speakers held out, stratified by gender; freeze split manifest as artifact
   - Baseline: 3 conv blocks + GAP + FC (~50K params) on precomputed 40-dim log-mel (1s frames, 10ms hop)
   - Metric: macro F1 on held-out speakers. **Gate: if < 0.55, pivot hypothesis; if > 0.60, proceed.**

4. **If gate passed, pursue Projector-Only path** (Week 3+):
   - Offline: HuBERT-base pre-training on 10k hrs unlabeled speech (or use public checkpoint)
   - Offline: Speaker-adversarial emotion head on CREMA-D + noise-augmented copies
   - Offline: Distill to 2-layer MLP projector (768→256→6) with temperature-scaled logits
   - Offline: Quantize projector to INT8; compute calibration thresholds (ECE, Mahalanobis)
   - On-device: Load only INT8 projector (~20 KB), frame-pool → project → classify → uncertainty flag

5. **Parallel observability/migration deliverables** (Week 1–3):
   - Structured JSON logging + correlation IDs in all scripts
   - `metrics_contract.yaml` with SLO targets (macro F1 ≥ 0.65, p99 latency ≤ 50ms)
   - `diagnose.py` CLI for confusion matrix, per-class PR curves, audio replay
   - Inference contract v1 schema (Pydantic) + model artifact bundle format
   - 3 runbooks: accuracy regression, latency spike, OOD surge

6. **Tech debt prevention** (Continuous):
   - Shared `schemas.py` (Pydantic) for all manifest stages
   - `DataConfig` dataclass externalizing paths, sample rate, label map, filename parser
   - `DatasetAdapter` protocol for future corpora
   - ADR template + first 3 ADRs (data schema, duplicate policy, split strategy)
   - Integration test (full pipeline on fixture dataset) + property tests (Hypothesis)
   - Document all magic numbers with rationale in `docs/thresholds.md`

---

**Next Action:** Begin Week 0 ergonomics stack. Do not write model code until `pyproject.toml`, Hydra, MLflow, DVC, CI, and STM32H747 memory map are committed.