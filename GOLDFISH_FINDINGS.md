# RobustSER-Tiny — Goldfish 7-Lens Findings (College Demo Edition)

*Synthesized from 7 parallel Goldfish reviews | Project: Speaker-Independent SER on CREMA-D*

---

## 📋 Executive Summary

Your project **already exceeds the original paper's methodological rigor** (frozen speaker-disjoint splits, 3-seed CI, honest reporting). The 7 lenses confirm you have a solid research baseline (E1-E5, best: 0.4741 macro F1). Since this is a **college demo project** — not production — we filter out enterprise-grade infra and focus on **high-impact, low-effort improvements** that make the demo impressive and the code maintainable.

---

## 🎯 Lens 1: Contrarian / YAGNI — "Cut the Custom Code"

**Finding:** ~937 lines of custom code reimplement standard libraries.

| Custom Component | Lines | Replace With | Effort Saved |
|------------------|-------|--------------|--------------|
| Log-Mel/MFCC extraction | 135 | `torchaudio.transforms.MelSpectrogram` + `MFCC` | ~130 LOC |
| CREMA-D Dataset | 160 | `torchaudio.datasets.CREMA_D` + `audiomentations` | ~150 LOC |
| Training Loop | 65 | `lightning.pytorch.Trainer` | ~60 LOC |
| Metrics | 52 | `torchmetrics.classification` | ~50 LOC |
| Speaker Splits | 170 | `sklearn.model_selection.StratifiedGroupKFold` | ~160 LOC |
| Models | 85 | `timm.create_model('mobilenetv3_small_100')` | ~80 LOC |
| Inference/ONNX Export | 180 | `torchaudio.load` + `onnxruntime` | ~170 LOC |

**🎓 College Demo Action:** Replace **training loop + metrics + features** first (Week 1-2). Keeps your custom CNN architecture (that's your research contribution) but removes boilerplate.

---

## 💰 Lens 2: Cost/Economics — "Compute-Rich, Insight-Poor"

**Finding:** Training is trivial ($0.60 for all 15 runs). E5 fits STM32H747 with 200× latency headroom. No quota ceilings.

| Experiment | Test Macro F1 | Params | GPU-hours (3 seeds) | F1/$ |
|------------|---------------|--------|---------------------|------|
| **E5 (2-sec MFCC + DS-CNN)** | **0.474** | 930 | 0.04 | **7,900** |
| E4 (1-sec MFCC + DS-CNN) | 0.416 | 930 | 0.04 | 6,930 |
| E3 (1-sec Log-Mel + DS-CNN) | 0.408 | 930 | 0.04 | 6,800 |
| E2 (Noise aug) | 0.374 | ~15K | 0.05 | 4,980 |
| E1 (Baseline) | 0.381 | ~15K | 0.05 | 5,080 |

**🎓 College Demo Action:** 
- **Stop investing in E1-E3** — dominated by E4/E5
- **Double down on E5 variants**: E5 + SpecAugment (cheap, +0.01-0.02 F1), E5 + Temperature Scaling (free calibration)
- **Skip STM32H747 deployment** — demo runs on laptop/CPU. Your Streamlit demo IS the deployment.

---

## 🛠️ Lens 3: UX/Ergonomics — "Developer Friction at Every Step"

**Priority Fixes (College-Friendly):**

| # | Fix | Tool | Effort | Why It Matters for Demo |
|---|-----|------|--------|------------------------|
| 1 | **Config System** | Hydra + hydra-zen | 2 hrs | `python train.py +training.max_epochs=50` for quick experiments |
| 2 | **Experiment Tracking** | MLflow (local) | 1 hr | Compare seeds/runs in browser UI — looks great in presentation |
| 3 | **CLI** | Typer + Rich | 2 hrs | `robustser train --help` works; auto-completion |
| 4 | **Packaging** | uv + uv.lock | 30 min | `uv sync` — reproducible install for evaluators |
| 5 | **CI** | GitHub Actions (ruff+mypy+pytest) | 30 min | Green checkmark on repo — professional |
| 6 | **Error Messages** | Audit ValueErrors | 1 hr | Clear errors when evaluators run it |

**🎓 College Demo Action:** Do **#4, #5, #6 first** (1 hr total) — immediate polish. Then **#1, #2** for experiment velocity.

---

## 🔄 Lens 4: Migration/Compatibility — "Zero Versioning Strategy"

**Finding:** No versioning for checkpoints, configs, data protocol, inference contracts.

**Minimal Fix for College Demo:**

```python
# Add to checkpoint dict (train.py)
checkpoint = {
    "epoch": epoch,
    "model_version": "1.0.0",           # <-- ADD
    "config_schema_version": "1.0.0",   # <-- ADD
    "protocol_version": "1.0.0",        # <-- ADD
    "git_commit": get_git_hash(),       # <-- ADD
    "checksum_sha256": hash_weights(),  # <-- ADD
    # ... existing fields
}
```

```yaml
# Add to data_protocol.yaml
protocol_version: "1.0.0"  # <-- ADD
```

**🎓 College Demo Action:** Add 4 lines to checkpoint + 1 line to data_protocol.yaml. Takes 15 min. Shows reproducibility awareness in report.

---

## 📊 Lens 5: Observability — "No Structured Logging/Metrics"

**Finding:** 14-file observability stack created (logging, metrics contract, diagnose CLI, runbooks).

**College Demo Minimum:**

| Need | Minimal Implementation | Effort |
|------|------------------------|--------|
| Structured logs | `print(json.dumps({"epoch": e, "val_f1": f1}))` in train loop | 5 min |
| Diagnose CLI | Keep your `scripts/diagnose.py` — it's already good | 0 |
| Calibration | `torchmetrics.CalibrationError` + temperature scaling | 30 min |
| Demo metrics | Streamlit sidebar showing live F1/confidence — already works | 0 |

**🎓 College Demo Action:** Add JSON log lines to training + CalibrationError metric. Skip the full 14-file stack.

---

## 🧱 Lens 6: Tech Debt — "Systemic Coupling, Bus Factor = 1"

**9 Refactorings (Prioritized for College):**

| # | Refactoring | Priority | Effort | Demo Impact |
|---|-------------|----------|--------|-------------|
| 1 | `uv.lock` + GitHub Actions CI | **P0** | 30 min | Green checkmark, reproducible install |
| 2 | Ruff + MyPy + pre-commit | **P0** | 1 hr | Catches bugs before commit |
| 3 | Pydantic schemas (`schemas.py`) | **P0** | 1 hr | Documents data contracts in report |
| 4 | Config dataclasses (`config.py`) | **P0** | 2 hrs | Centralizes magic numbers for report |
| 5 | DatasetAdapter protocol | **P1** | 3 hrs | Enables "we also tested on RAVDESS" in report |
| 6 | Integration test (fixture pipeline) | **P1** | 2 hrs | Proves pipeline works end-to-end |
| 7 | Hypothesis property tests | **P1** | 1 hr | Shows rigor in testing section |
| 8 | ADR template + 3 ADRs | **P2** | 1 hr | Documents design decisions in report |
| 9 | `docs/thresholds.md` | **P2** | 1 hr | Explains every hyperparameter in report |

**🎓 College Demo Action:** Do **#1, #2, #3, #4** (2.5 hrs). These give you: reproducible install, quality gates, documented data contracts, centralized config — all citeable in your report.

---

## 🧠 Lens 7: Inventive Catalyst — "IFR = Zero NN on MCU"

**Finding:** Ideal Final Result = **SSL pre-training → distilled INT8 projector** (20 KB) + speaker cache (23 KB) + cascaded uncertainty.

**College Reality Check:** This requires ~72 GPU-hours ($181) + Weeks 3-6 work. **Not needed for college demo.**

**What to borrow instead:**
- **Cascaded uncertainty idea**: Add temperature scaling + "low confidence" warning in demo (already in your `inference.py`!)
- **Speaker cache idea**: Pre-compute 91 speaker centroids, show "nearest speaker" in demo
- **Calibration thresholds**: Your `calibrate_temperature.py` already does this

**🎓 College Demo Action:** Highlight in report: *"Future work: SSL pre-training path (per Inventive Catalyst) would move representation learning offline for true edge deployment."* Shows you know the research frontier without doing it.

---

## ✅ Consolidated Action Plan (College Demo Edition)

### Week 1: Foundation & Polish (2.5 hrs)
```bash
# 1. Reproducible install + CI (30 min)
uv init --no-readme
uv add --dev ruff mypy pytest hypothesis
uv lock

# 2. GitHub Actions CI (copy .github/workflows/ci.yml template)
# 3. Ruff + MyPy config + pre-commit (1 hr)
# 4. Fix .gitignore (see below)
# 5. Add checkpoint versioning (15 min)
```

### Week 2: Config & Tracking (3 hrs)
```bash
# 1. Hydra config system (2 hrs) - enables quick experiment sweeps
# 2. MLflow local (1 hr) - experiment comparison UI for presentation
```

### Week 3: Code Reduction (Optional, 2 hrs)
```bash
# Replace training loop with lightning.pytorch (30 min)
# Replace metrics with torchmetrics (15 min)
# Replace features with torchaudio.transforms (30 min)
# Replace speaker splits with sklearn (15 min)
```

### Week 4: Report & Demo Polish (2 hrs)
```bash
# 1. Add Pydantic schemas + config dataclasses (documents data contracts)
# 2. Update .gitignore
# 3. Final demo rehearsal + record video
# 4. Write report citing: CI, schemas, MLflow, thresholds.md, ADRs
```

---

## 📝 Updated .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Python package
*.egg-info/
dist/
build/
*.egg

# PyTorch checkpoints
*.pt
*.pth
*.ckpt

# ONNX exports
*.onnx

# MLflow
mlruns/

# DVC
.dvc/
data/raw/
*.dvc

# Logs
*.log
logs/
wandb/

# Coverage
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Demo
demo/__pycache__/
.streamlit/

# Local config
.env
.local.env

# Temporary files
*.tmp
*.temp
tmp/
```

---

## 📚 What to Highlight in Your College Report

| Section | What to Cite from Goldfish |
|---------|---------------------------|
| **Methodology** | Frozen speaker-disjoint splits (Lens 4), 3-seed CI (Lens 2), test-once protocol |
| **Metrics** | Macro F1 + UAR + CalibrationError (Lens 5), not just accuracy |
| **Experiments** | E5 best (0.4741), E1-E3 dominated (Lens 2), E5 + SpecAugment next |
| **Reproducibility** | Hydra config, MLflow tracking, uv.lock, CI, data_protocol.yaml versioning |
| **Data Contracts** | Pydantic schemas (Lens 6), data_protocol.yaml versioning (Lens 4) |
| **Error Analysis** | `scripts/diagnose.py` confusion matrix, PR curves, calibration (Lens 5) |
| **Future Work** | SSL pre-training path (Lens 7), cascaded uncertainty, speaker cache |
| **Design Decisions** | ADRs for data schema, duplicate policy, split strategy (Lens 6) |

---

## 🎓 Final Note for College Demo

**You don't need production-grade infra.** The evaluators care about:
1. **Methodological rigor** — you have it (speaker-disjoint, 3-seed, test-once)
2. **Honest reporting** — you have it (0.47 F1, not 99.5%)
3. **Code quality** — CI, linting, schemas, config centralization show maturity
4. **Demo works** — Streamlit with confidence warnings, live metrics
5. **Report depth** — cite the Goldfish insights as "multi-lens review methodology"

**Time investment:** ~8 hours total for all high-impact items.  
**Result:** A college demo that looks like a research project, not a homework assignment.

---

*Generated from 7 Goldfish lens reviews | Filtered for college demo scope | November 2026*