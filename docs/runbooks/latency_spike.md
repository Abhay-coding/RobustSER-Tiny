# Runbook: Inference Latency Spike

**Severity:** WARNING → CRITICAL
**SLO Impact:** `inference_latency_ms` p99 > 100ms (target: p99 ≤ 50ms)
**Detection:** Automated via demo app metrics, edge telemetry, or load testing

---

## Symptoms

- Demo app classification takes > 2 seconds
- Streamlit UI shows spinner for extended time
- p99 latency exceeds 100ms in metrics
- Edge device inference timeout
- GPU/CPU utilization spikes during inference

---

## Immediate Triage (5 minutes)

### 1. Check Current Latency
```bash
# Quick latency test
python -c "
import time, torch
from src.robustser.inference import predict_wav_bytes
with open('test.wav', 'rb') as f: audio = f.read()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start = time.time()
result = predict_wav_bytes(audio, run_dir='results/E5_seed20261108', device=device)
print(f'Latency: {(time.time()-start)*1000:.1f}ms')
"
```

### 2. Check Resource Utilization
```bash
# GPU memory
nvidia-smi

# CPU/memory
htop  # or Task Manager on Windows

# Disk I/O
iostat -x 1 5
```

### 3. Check for Model Loading Issues
```bash
# Is model being reloaded every request?
# Check inference.py load_selected_model LRU cache
grep -n "lru_cache" src/robustser/inference.py
```

---

## Investigation (15 minutes)

### 1. Profile Inference Pipeline
```bash
python -c "
import cProfile, pstats, io
from src.robustser.inference import predict_wav_bytes
import torch

with open('test.wav', 'rb') as f: audio = f.read()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pr = cProfile.Profile()
pr.enable()
for _ in range(10):
    predict_wav_bytes(audio, run_dir='results/E5_seed20261108', device=device)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())
"
```

### 2. Break Down Latency Components
```bash
python -c "
import time, torch
from src.robustser.inference import decode_wav_bytes, resample_to_model_rate, fit_audio_window
from src.robustser.features.log_mel import SAMPLE_RATE
from src.robustser.data.cremad import FEATURE_EXTRACTORS
from src.robustser.inference import load_selected_model

with open('test.wav', 'rb') as f: audio = f.read()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Decode
t0 = time.time()
waveform, sr = decode_wav_bytes(audio)
t1 = time.time()

# 2. Resample
waveform = resample_to_model_rate(waveform, sr)
t2 = time.time()

# 3. Window
window = fit_audio_window(waveform, target_samples=16000, policy='center_crop_or_zero_pad')
t3 = time.time()

# 4. Features
features = FEATURE_EXTRACTORS['log_mel_filterbank_energy'](window)
t4 = time.time()

# 5. Model load (cached)
model, config, labels = load_selected_model('results/E5_seed20261108', device)
t5 = time.time()

# 6. Forward pass
with torch.no_grad():
    logits = model(torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device))
t6 = time.time()

print(f'Decode:      {(t1-t0)*1000:.1f}ms')
print(f'Resample:    {(t2-t1)*1000:.1f}ms')
print(f'Window:      {(t3-t2)*1000:.1f}ms')
print(f'Features:    {(t4-t3)*1000:.1f}ms')
print(f'Model load:  {(t5-t4)*1000:.1f}ms')
print(f'Forward:     {(t6-t5)*1000:.1f}ms')
print(f'Total:       {(t6-t0)*1000:.1f}ms')
"
```

### 3. Check Batch Size Impact
```bash
# Test with batch inference
python -c "
import time, torch
import numpy as np
from src.robustser.inference import load_selected_model, decode_wav_bytes, resample_to_model_rate, fit_audio_window
from src.robustser.features.log_mel import SAMPLE_RATE
from src.robustser.data.cremad import FEATURE_EXTRACTORS

with open('test.wav', 'rb') as f: audio = f.read()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, config, labels = load_selected_model('results/E5_seed20261108', device)

waveform, sr = decode_wav_bytes(audio)
waveform = resample_to_model_rate(waveform, sr)
window = fit_audio_window(waveform, target_samples=16000, policy='center_crop_or_zero_pad')
features = FEATURE_EXTRACTORS['log_mel_filterbank_energy'](window)

for batch_size in [1, 4, 8, 16, 32]:
    batch = torch.from_numpy(features).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    torch.cuda.synchronize() if device=='cuda' else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(batch)
    torch.cuda.synchronize() if device=='cuda' else None
    t1 = time.time()
    print(f'Batch {batch_size:2d}: {(t1-t0)*1000/batch_size:.2f}ms/sample')
"
```

---

## Common Root Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| **Model reloading per request** | High "Model load" time in profile | Ensure `load_selected_model` LRU cache works |
| **No GPU / CPU fallback** | Device shows 'cpu' when GPU expected | Check `torch.cuda.is_available()` |
| **Large input audio** | Decode/resample time high | Validate audio duration ≤ 60s, add preprocessing |
| **Feature extraction bottleneck** | High "Features" time | Optimize log-mel extraction, consider caching |
| **Model too large** | Forward pass > 30ms | Quantize to INT8, prune, or use smaller model |
| **Python GIL / threading** | High CPU but low GPU utilization | Use `torch.compile()`, batch requests |
| **Memory pressure / swapping** | High disk I/O, OOM killer | Reduce batch size, clear cache |
| **Cold start** | First request slow, subsequent fast | Warm up model on startup |

---

## Remediation Steps

### Quick Fixes (try in order)

1. **Verify GPU is being used**
   ```python
   # In inference code
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")  # Should be 'cuda'
   ```

2. **Enable model compilation (PyTorch 2.0+)**
   ```python
   model = torch.compile(model)  # Add after model.load_state_dict()
   ```

3. **Warm up model on startup**
   ```python
   # In demo app initialization
   dummy_input = torch.randn(1, 1, 32, 31).to(device)
   with torch.no_grad():
       _ = model(dummy_input)  # Warm up
   ```

4. **Quantize model for inference**
   ```bash
   python scripts/export_onnx.py --run-dir results/E5_seed20261108 --quantize int8
   ```

5. **Batch requests in demo** (if multiple users)
   - Accumulate requests for ~10ms window
   - Process as batch
   - Return results

### Deep Fixes (if quick fixes fail)

1. **Profile with PyTorch Profiler**
   ```bash
   python -m torch.utils.profiler.profile --help
   ```

2. **Optimize feature extraction**
   - Move to C++/CUDA if critical
   - Use `torchaudio.compliance.kaldi.fbank` (optimized)

3. **Model architecture optimization**
   - Replace with depthwise separable CNN
   - Reduce channels/layers
   - Knowledge distillation

4. **Edge deployment preparation**
   ```bash
   # Export to ONNX for ONNX Runtime / TensorRT
   python scripts/export_onnx.py --run-dir results/E5_seed20261108
   ```

---

## Escalation

**Page oncall if:**
- p99 latency > 200ms (critical threshold)
- Demo app unusable (> 5s per classification)
- Edge device inference fails consistently

**Create issue with:**
- Latency breakdown (decode, resample, window, features, forward)
- Hardware specs (GPU/CPU, RAM)
- PyTorch version, CUDA version
- Profiler output

---

## Prevention

- [ ] Add CI latency benchmark: `inference_latency_ms` p99 ≤ 50ms
- [ ] Automated nightly latency regression test
- [ ] Model compilation in CI pipeline
- [ ] Warm-up endpoint in demo app
- [ ] Latency alerting: p99 > 100ms for 5min
- [ ] Edge deployment validation in CI
- [ ] Regular quantization accuracy checks