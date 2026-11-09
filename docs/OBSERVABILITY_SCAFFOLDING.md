# RobustSER-Tiny Observability Scaffolding

**Generated:** 2026-11-09
**Purpose:** Complete observability infrastructure for training, evaluation, and inference

---

## 📁 File Inventory

### Core Observability Package
| File | Purpose |
|------|---------|
| `src/robustser/observability/__init__.py` | Package exports |
| `src/robustser/observability/logging.py` | Structured JSON logging with correlation IDs |
| `src/robustser/observability/metrics.py` | Metrics collector (JSONL, Prometheus, in-memory) |
| `src/robustser/observability/tracing.py` | Distributed tracing (W3C TraceContext) |

### Configuration
| File | Purpose |
|------|---------|
| `configs/metrics_contract.yaml` | **Complete SLO targets, metric definitions, alerting rules** |

### Enhanced Scripts
| File | Purpose |
|------|---------|
| `scripts/train_with_observability.py` | Training with structured logging, metrics, tracing |
| `scripts/evaluate_with_observability.py` | Evaluation with calibration metrics, OOD detection |
| `scripts/diagnose.py` | **Diagnosis CLI: confusion matrix, PR curves, calibration, misclassified analysis** |
| `src/robustser/inference_observability.py` | Inference with timing breakdown, edge telemetry |
| `demo/app_observability.py` | Streamlit demo with live metrics dashboard |

### Runbooks
| File | Purpose |
|------|---------|
| `docs/runbooks/accuracy_regression.md` | Accuracy drop > 0.03, test F1 < 0.50 |
| `docs/runbooks/latency_spike.md` | Inference p99 > 100ms |
| `docs/runbooks/ood_surge.md` | OOD rate > 20% |

---

## 🔑 Key Metric Names (from metrics_contract.yaml)

### Training Metrics (per epoch)
```
train_loss                    # cross_entropy, target < 2.0
train_macro_f1               # f1_score, target >= 0.60
validation_loss              # cross_entropy, target < 1.5
validation_macro_f1          # f1_score, target >= 0.60 (PRIMARY SELECTION METRIC)
validation_balanced_accuracy # accuracy, target >= 0.60
epoch_duration_seconds       # seconds, target < 30
learning_rate                # rate
```

### Calibration Metrics (per epoch on validation)
```
validation_ece                    # ece, target < 0.05
validation_ece_temperature_scaled # ece, target < 0.03
temperature_parameter             # scalar
reliability_diagram_bins          # histogram (15 bins)
```

### Test Metrics (once per run)
```
test_macro_f1                # f1_score, target >= 0.60
test_balanced_accuracy       # accuracy, target >= 0.60
test_accuracy                # accuracy
test_ece                     # ece, target < 0.05
test_per_class_f1            # f1_score per class, target >= 0.50
test_confusion_matrix        # histogram [true_class, predicted_class]
```

### Noisy Test Metrics (per SNR)
```
noisy_test_macro_f1  # f1_score by snr_db label
noisy_test_ece       # ece by snr_db label
```

### Serving/Inference Metrics
```
inference_latency_ms        # histogram, p99 <= 50ms
model_confidence            # histogram, mean > 0.70
prediction_margin           # histogram, mean > 0.15
ood_detection_rate          # rate, target < 0.10
audio_duration_seconds      # histogram
audio_rms_dbfs              # histogram
```

### Edge/MCU Telemetry
```
edge_inference_latency_ms     # histogram, p99 <= 50ms
edge_peak_ram_kb              # gauge, target < 200 KB
edge_energy_per_inference_mj  # gauge, target < 10 mJ
edge_ood_rate                 # rate, target < 0.10
edge_model_load_time_ms       # histogram, target < 1000ms
```

### System Metrics
```
gpu_memory_allocated_mb   # MB
gpu_memory_reserved_mb    # MB
cpu_utilization_percent   # percent
disk_usage_mb             # MB
```

---

## 🚨 Alerting Thresholds

| Alert | Metric | Condition | Severity |
|-------|--------|-----------|----------|
| validation_macro_f1_drop | validation_macro_f1 | drop > 0.03 epoch-over-epoch | WARNING |
| training_loss_nan | train_loss | is_nan OR is_inf | CRITICAL |
| gpu_oom | gpu_memory_allocated_mb | >= 95% GPU capacity | CRITICAL |
| early_stopping_triggered | epochs_without_improvement | >= patience | INFO |
| test_macro_f1_below_threshold | test_macro_f1 | < 0.50 | WARNING |
| calibration_degraded | validation_ece | > 0.10 | WARNING |
| inference_latency_spike | inference_latency_ms | p99 > 100ms | WARNING |
| edge_ram_exceeded | edge_peak_ram_kb | > 500 KB | CRITICAL |
| ood_surge | ood_detection_rate | > 0.35 | CRITICAL |

---

## 🔗 Correlation IDs (for tracing)

All log entries and metrics include these correlation IDs:
```
experiment_id    # e.g., "E5_paper_inspired_cnn_v0"
run_id           # e.g., "E5_seed20261108"
epoch            # current epoch (training)
seed             # random seed
split            # train, validation, test, noisy_test, external_test, inference
snr_db           # for noisy evaluations (5, 10, 15, 20)
device           # cpu, cuda, mcu
request_id       # for inference requests (UUID)
```

---

## 📊 Diagnosis CLI Usage

```bash
# Full diagnostic suite
python scripts/diagnose.py --run-dir results/E5_seed20261108 --all

# Individual diagnostics
python scripts/diagnose.py --run-dir results/E5_seed20261108 --confusion-matrix
python scripts/diagnose.py --run-dir results/E5_seed20261108 --pr-curves
python scripts/diagnose.py --run-dir results/E5_seed20261108 --calibration
python scripts/diagnose.py --run-dir results/E5_seed20261108 --misclassified --top-k 10
```

### Outputs Generated
| Command | Output Files |
|---------|--------------|
| `--confusion-matrix` | `confusion_matrix.png`, `confusion_matrix.json` |
| `--pr-curves` | `pr_curves.png` |
| `--calibration` | `reliability_diagram_T1.png`, `reliability_diagram_T*.png`, `ece_by_class.png`, `calibration_summary.json` |
| `--misclassified` | `misclassified.json`, `misclassified/manifest.jsonl` |
| `--all` | All of the above |

---

## 🔄 Enhanced Training Flow

```python
# scripts/train_with_observability.py
# Key additions:
# 1. Structured JSON logging to training.jsonl
# 2. Metrics recorded per epoch to metrics.jsonl
# 3. Calibration metrics (ECE, temperature) per epoch
# 4. Per-class metrics recorded
# 5. GPU memory tracking
# 6. Alert: validation F1 drop > 0.03
# 7. W3C trace context for each epoch
# 8. Correlation IDs: experiment_id, run_id, epoch, seed, split, device
```

### Log Output Example (training.jsonl)
```json
{
  "timestamp": "2026-11-09T10:30:45.123Z",
  "level": "INFO",
  "logger": "robustser.train",
  "message": "Epoch 005: train_macro_f1=0.6234 validation_macro_f1=0.5876 ece=0.0432 temp=1.234 duration=12.3s",
  "correlation_ids": {
    "experiment_id": "E5_paper_inspired_cnn_v0",
    "run_id": "E5_seed20261108",
    "epoch": 5,
    "seed": 20261108,
    "split": "validation",
    "device": "cuda:0"
  }
}
```

---

## 📈 Enhanced Evaluation Flow

```python
# scripts/evaluate_with_observability.py
# Key additions:
# 1. Full calibration analysis (ECE before/after temperature scaling)
# 2. Confidence & margin distributions recorded as histograms
# 3. OOD detection rates (confidence < 0.5, margin < 0.1)
# 4. Per-class metrics + confusion matrix
# 5. Alert checks: test F1 < 0.50, ECE > 0.10, OOD > 20%
```

---

## 🎯 Inference Observability

```python
# src/robustser/inference_observability.py
# Key additions:
# 1. Timing breakdown: decode, resample, load, window, features, forward
# 2. Per-stage latency metrics
# 3. Confidence, margin, OOD detection metrics
# 3. Audio metadata (duration, RMS dBFS)
# 4. Request ID for tracing
# 5. Edge telemetry schema (for MCU deployment)
# 6. Model warmup function
# 7. LRU cache for model serving
```

### Timing Breakdown Example
```
Total:       45.2ms
  Decode:     2.1ms
  Resample:   1.8ms
  Load Model: 0.3ms (cached)
  Window:     0.1ms
  Features:   3.2ms
  Forward:    37.7ms
```

---

## 📱 Demo App with Live Metrics

```python
# demo/app_observability.py
# Key additions:
# 1. Live metrics sidebar (latency, confidence, OOD rate)
# 2. Timing breakdown chart
# 3. Request ID displayed for tracing
# 4. Model warmup button
# 5. Structured logging to logs/demo/demo.jsonl
# 6. Metrics collection to logs/demo/demo_metrics.jsonl
```

---

## 🚀 Quick Start

### 1. Run Training with Observability
```bash
python scripts/train_with_observability.py --config configs/baseline.yaml --run-dir results/E5_obs_seed20261109
```

### 2. Run Evaluation with Observability
```bash
python scripts/evaluate_with_observability.py --run-dir results/E5_obs_seed20261109
```

### 3. Run Full Diagnostics
```bash
python scripts/diagnose.py --run-dir results/E5_obs_seed20261109 --all
```

### 4. Run Demo with Observability
```bash
streamlit run demo/app_observability.py
```

### 5. View Metrics
```bash
# View training metrics
cat results/E5_obs_seed20261109/metrics.jsonl | jq '.validation_macro_f1'

# View Prometheus format
python -c "
from src.robustser.observability import get_metrics_collector
collector = get_metrics_collector()
collector.export_jsonl(Path('results/E5_obs_seed20261109/metrics.jsonl'))
print(collector.export_prometheus())
"
```

---

## 🔧 Integration Checklist

### For Training
- [ ] Replace `scripts/train.py` with `scripts/train_with_observability.py`
- [ ] Add `import logging` at top of training script
- [ ] Verify `configs/metrics_contract.yaml` SLOs match project goals

### For Evaluation
- [ ] Replace `scripts/evaluate.py` with `scripts/evaluate_with_observability.py`
- [ ] Run `scripts/diagnose.py --all` after each evaluation

### For Inference/Demo
- [ ] Replace `demo/app.py` with `demo/app_observability.py`
- [ ] Use `inference_observability.py` for any new inference code
- [ ] Call `warmup_model()` at service startup

### For Edge Deployment
- [ ] Use `predict_wav_bytes_edge()` for MCU inference
- [ ] Implement `edge_peak_ram_kb` and `edge_energy_per_inference_mj` collection on device
- [ ] Export model to ONNX: `python scripts/export_onnx.py --run-dir results/E5_seed20261108`

### For Monitoring/Alerting
- [ ] Ship `metrics.jsonl` to log aggregation (Loki, Elasticsearch, Datadog)
- [ ] Configure alerts per `metrics_contract.yaml` alerting rules
- [ ] Build dashboards for: training curves, calibration, inference latency, OOD rate

---

## 📋 SLO Dashboard Panels (Grafana/Datadog)

### Training Panel
- `validation_macro_f1` over epochs (target line at 0.60)
- `validation_ece` over epochs (target line at 0.05)
- `epoch_duration_seconds` histogram

### Evaluation Panel
- `test_macro_f1` gauge (target >= 0.60)
- `test_ece` gauge (target < 0.05)
- `test_confidence` distribution
- `test_per_class_f1` bar chart

### Serving Panel
- `inference_latency_ms` p50/p95/p99 over time
- `model_confidence` distribution
- `ood_detection_rate` over time
- Request rate (inferences/minute)

### Edge Panel
- `edge_inference_latency_ms` p99
- `edge_peak_ram_kb` gauge
- `edge_energy_per_inference_mj` gauge
- `edge_ood_rate` over time

---

## 🎯 Next Steps

1. **Integrate into CI/CD**: Add metrics gates to PR checks
2. **Log Aggregation**: Ship JSONL to centralized logging
3. **Dashboard Creation**: Build SLO dashboards
4. **Alert Routing**: Connect alerts to PagerDuty/Slack
5. **Edge Validation**: Test ONNX export on target MCU
6. **Continuous Profiling**: Add PyTorch profiler to nightly runs