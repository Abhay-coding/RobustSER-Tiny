# Runbook: Accuracy Regression

**Severity:** WARNING → CRITICAL
**SLO Impact:** `validation_macro_f1` drops > 0.03 epoch-over-epoch or `test_macro_f1` < 0.50
**Detection:** Automated via training metrics logging and evaluation scripts

---

## Symptoms

- Validation macro F1 drops significantly (> 3 percentage points) between epochs
- Held-out test macro F1 falls below 0.50
- Per-class F1 shows specific class degradation
- Confusion matrix shows increased off-diagonal mass

---

## Immediate Triage (5 minutes)

### 1. Check Recent Training Logs
```bash
# View latest metrics
tail -20 results/<run_dir>/metrics.jsonl | jq '.validation.macro_f1'

# Check for NaN loss
tail -20 results/<run_dir>/metrics.jsonl | jq 'select(.train.loss | isnan or isinf)'
```

### 2. Verify Data Integrity
```bash
# Check manifest hasn't changed
git status data/manifests/

# Verify dataset splits are consistent
python -c "
from src.robustser.data.cremad import CREMADSERDataset
ds = CREMADSERDataset('data/manifests/crema_d_speaker_splits.csv', split='validation', task='crema_d_six_emotion', seed=20261108)
print(f'Validation samples: {len(ds)}')
"
```

### 3. Check for Configuration Drift
```bash
# Compare current config with run config
diff configs/baseline.yaml results/<run_dir>/config.yaml
```

---

## Investigation (15 minutes)

### 1. Run Full Diagnostics
```bash
python scripts/diagnose.py --run-dir results/<run_dir> --all
```

### 2. Analyze Confusion Matrix
- Look for specific class pairs being confused
- Check if confusion is symmetric or directional
- Identify if new confusions appeared vs. baseline

### 3. Check Per-Class Metrics
```bash
# From diagnosis output or metrics.jsonl
# Look for: per_class_f1, per_class_precision, per_class_recall
```

### 4. Calibration Check
```bash
python scripts/diagnose.py --run-dir results/<run_dir> --calibration
# Check if ECE increased significantly
```

---

## Common Root Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| **Data leakage** | Train F1 >> Val F1, sudden drop | Verify speaker-disjoint splits, check manifest |
| **Label noise** | Specific class F1 drops, confusion with similar classes | Audit annotations, check CREMA-D label mapping |
| **Learning rate too high** | Loss oscillates, F1 unstable | Reduce LR, add LR scheduler |
| **Overfitting** | Train F1 ↗, Val F1 ↘, gap widening | Increase dropout, add weight decay, early stopping |
| **Batch norm statistics** | Different behavior train vs eval | Check `model.train()`/`model.eval()` calls |
| **Seed/non-determinism** | Non-reproducible runs | Set `deterministic_algorithms: true`, fix seeds |
| **Hardware/environment change** | Different GPU, PyTorch version | Pin versions, check CUDA/cuDNN |

---

## Remediation Steps

### Quick Fixes (try in order)
1. **Restart from best checkpoint**
   ```bash
   # Resume training from best_validation_macro_f1.pt
   python scripts/train.py --run-dir results/<run_dir>_recovery --config configs/baseline.yaml
   ```

2. **Reduce learning rate**
   ```yaml
   # In config
   training:
     learning_rate: 0.001  # was 0.005
   ```

3. **Increase regularization**
   ```yaml
   model:
     dropout: 0.30  # was 0.20
   training:
     weight_decay: 1e-4  # add if missing
   ```

4. **Verify data splits**
   ```bash
   python scripts/create_speaker_splits.py --verify
   ```

### Deep Fixes (if quick fixes fail)
1. **Re-run with fresh seed**
   ```bash
   python scripts/train.py --config configs/baseline.yaml --seed 20261109
   ```

2. **Architecture audit**
   - Check model capacity vs. dataset size
   - Verify feature extraction matches model input

3. **Full retrain with logging**
   ```bash
   python scripts/train.py --config configs/baseline.yaml --run-dir results/full_retrain_$(date +%s)
   ```

---

## Escalation

**Page oncall if:**
- Multiple retries fail to recover F1
- Test F1 < 0.40 (critical threshold)
- Regression persists across seeds

**Create issue with:**
- Run directory paths
- Confusion matrix screenshots
- Metrics JSONL excerpts
- Config diff

---

## Prevention

- [ ] Add CI gate: `validation_macro_f1` must not drop > 0.03 in any epoch
- [ ] Add CI gate: `test_macro_f1` ≥ 0.50 for merge
- [ ] Automated confusion matrix comparison in CI
- [ ] Data integrity checks in pre-commit hooks
- [ ] Scheduled weekly full retrain with regression detection