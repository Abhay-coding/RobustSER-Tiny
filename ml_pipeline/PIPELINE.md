# RobustSER-Tiny experiment record

This file records the completed research pipeline retrospectively so subsequent
experiments preserve the same leakage controls. Detailed artifacts are in
`docs/`, `data/manifests/`, `configs/experiments/`, and `results/`.

1. **Data inspection — done.** CREMA-D and MUSAN manifests/audits are recorded
   in `docs/dataset_audit.md`; RAVDESS is external-only.
2. **EDA — done.** Dataset composition, speaker counts, audio quality flags,
   and label mapping are documented in the audit.
3. **Prediction problem — done.** Six-class, speaker-independent speech-emotion
   recognition; macro F1 is primary. See `docs/research_protocol.md`.
- Gate A: approved 2026-11-08
4. **Data cleaning — done.** Six label-conflicting exact duplicate records are
   excluded from modeling; the original manifest is retained.
5. **Data engineering — done.** Reproducible manifests encode paths, speakers,
   labels, and data provenance.
6. **Train/validation/test split — done.** Frozen speaker-disjoint CREMA-D
   splits are stored in `data/manifests/crema_d_speaker_splits.csv`.
7. **Feature engineering — done.** Fixed 16 kHz MFCC/log-Mel extraction is
   implemented under `src/robustser/features/`.
8. **Preprocessing — done.** Per-utterance normalization and fixed crop/pad
   policies are stored in each experiment configuration.

Model rationale:
- traits: six-class audio classification, repeated speakers, 7,436 eligible CREMA-D recordings, fixed-length spectral features
- baseline: compact paper-inspired CNN on fixed log-Mel features
- candidates: MFCC ablation, depthwise-separable CNN, and two-second temporal context
- ruled out: random record split (speaker leakage); external RAVDESS training (invalid cross-corpus evaluation)
- metric: validation macro F1 primary; UAR and fixed held-out evaluations secondary

- Gate B: approved 2026-11-08
9. **Baseline — done.** E1 establishes the log-Mel CNN baseline.
10. **Model training — done.** E1–E5 configs and seed-specific outputs are in
    `configs/experiments/` and `results/`.
11. **Hyperparameter/ablation study — done.** Choices are validation-driven and
    recorded in `docs/experiment_log.md`.
12. **Model evaluation — done.** Clean, external RAVDESS, and held-out MUSAN
    results are recorded in `docs/experiment_log.md`.
- Gate C: approved 2026-11-09
13. **Error analysis — in progress.** Per-class metrics and confusion matrices
    exist; severe-noise and cross-corpus failure modes remain the next focus.
14. **Final test — done for E1–E5.** Each selected checkpoint has its own fixed
    clean-test result; no model was selected using these values.
15. **Deployment — software export done.** ONNX export is verified; no physical
    hardware measurements are claimed.
16. **Monitoring/retraining plan — todo.** Complete after the final robustness
    experiment is chosen.
