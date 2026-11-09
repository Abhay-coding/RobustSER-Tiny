# Runbook: OOD (Out-of-Distribution) Surge

**Severity:** WARNING → CRITICAL
**SLO Impact:** `ood_detection_rate` > 0.20 (target: < 0.10), `edge_ood_rate` > 0.20
**Detection:** Automated via demo app warnings, inference metrics, edge telemetry

---

## Symptoms

- High rate of "Low model confidence" warnings in demo
- High rate of "Top two classes are close" warnings
- `model_confidence` mean drops below 0.50
- `prediction_margin` mean drops below 0.10
- Edge device reports high OOD rate
- User complaints about unreliable predictions

---

## Immediate Triage (5 minutes)

### 1. Check Current OOD Rate
```bash
# From demo app logs or inference metrics
# Look for warnings in recent predictions

# Quick check via inference script
python -c "
import torch
from src.robustser.inference import predict_wav_bytes

# Test with various audio types
test_files = ['normal.wav', 'quiet.wav', 'short.wav', 'noise.wav', 'music.wav']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for f in test_files:
    try:
        with open(f, 'rb') as audio: data = audio.read()
        result = predict_wav_bytes(data, run_dir='results/E5_seed20261108', device=device)
        print(f'{f}: conf={result[\"confidence\"]:.3f}, margin={result[\"margin_to_second_choice\"]:.3f}, warnings={len(result[\"warnings\"])}')
    except: pass
"
```

### 2. Check Confidence Distribution
```bash
# From metrics or run a batch evaluation
python scripts/diagnose.py --run-dir results/E5_seed20261108 --calibration
# Look at confidence histogram in reliability diagram
```

### 3. Check Recent Data Changes
```bash
# Any new audio sources?
# Microphone changes?
# Environment noise changes?
```

---

## Investigation (15 minutes)

### 1. Analyze Confidence & Margin Distributions
```bash
# Run on test set to see baseline
python -c "
import torch
from torch.utils.data import DataLoader
import sys; sys.path.insert(0, 'src')
from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.models import build_model
import yaml

config = yaml.safe_load(open('results/E5_seed20261108/config.yaml'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('results/E5_seed20261108/best_validation_macro_f1.pt', map_location=device)
model = build_model(config['model']['name'], num_classes=len(TASK_LABELS[config['experiment']['task']])).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataset = CREMADSERDataset(config['data']['manifest'], split='test', task=config['experiment']['task'],
                          seed=config['experiment']['seed'], feature_type=config['features']['type'],
                          window_seconds=config['data']['train_window_seconds'])
loader = DataLoader(dataset, batch_size=32, shuffle=False)

all_conf, all_margin = [], []
with torch.no_grad():
    for feat, labels in loader:
        feat = feat.to(device)
        logits = model(feat)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        all_conf.extend(conf.cpu().tolist())
        all_margin.extend(margin.cpu().tolist())

import numpy as np
print(f'Confidence: mean={np.mean(all_conf):.3f}, p10={np.percentile(all_conf,10):.3f}, p50={np.percentile(all_conf,50):.3f}')
print(f'Margin:     mean={np.mean(all_margin):.3f}, p10={np.percentile(all_margin,10):.3f}, p50={np.percentile(all_margin,50):.3f}')
print(f'OOD rate (conf<0.5): {sum(c<0.5 for c in all_conf)/len(all_conf):.1%}')
print(f'OOD rate (margin<0.1): {sum(m<0.1 for m in all_margin)/len(all_margin):.1%}')
"
```

### 2. Check Noisy Test Performance
```bash
# Run noisy evaluation at different SNRs
python scripts/evaluate_noisy.py --run-dir results/E5_seed20261108 --snr 5
python scripts/evaluate_noisy.py --run-dir results/E5_seed20261108 --snr 10
python scripts/evaluate_noisy.py --run-dir results/E5_seed20261108 --snr 20

# Check if OOD rate correlates with noise
```

### 3. Check External Corpus (RAVDESS)
```bash
python scripts/evaluate_ravdess.py --run-dir results/E5_seed20261108 --chunk-size 1000
# Check confidence distribution on external data
```

### 4. Audio Quality Analysis
```bash
# Run audio quality audit
python scripts/audit_audio_quality.py --manifest data/manifests/crema_d_speaker_splits.csv
```

---

## Common Root Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| **Distribution shift (new speakers)** | High OOD on new users, low on test set | Collect new speaker data, retrain |
| **Background noise increase** | OOD correlates with low SNR, high noise audit | Add noise augmentation, noise suppression |
| **Microphone/recording changes** | Different sample rate, gain, format | Standardize input pipeline, add gain normalization |
| **Non-speech audio (music, silence)** | High OOD on non-speech, warnings trigger | Add VAD (voice activity detection), reject non-speech |
| **Adversarial / out-of-domain** | Very low confidence, uniform predictions | Add OOD detector, reject unknown domains |
| **Model degradation** | OOD on previously good data | Check for bit rot, reload checkpoint |
| **Feature extraction mismatch** | Different sample rate, window size | Verify `train_window_seconds`, sample rate |

---

## Remediation Steps

### Quick Fixes (try in order)

1. **Add input validation & rejection**
   ```python
   # In inference.py predict_wav_bytes - already has warnings
   # Make warnings actionable:
   if confidence < 0.30 or margin < 0.05:
       return {"error": "Input out of distribution", "retry": True}
   ```

2. **Add Voice Activity Detection (VAD)**
   ```python
   # Before feature extraction
   import torchaudio
   vad = torchaudio.transforms.Vad(sample_rate=16000)
   speech_prob = vad(waveform).mean()
   if speech_prob < 0.3:
       return {"error": "No speech detected"}
   ```

3. **Normalize input gain**
   ```python
   # In decode_wav_bytes or predict_wav_bytes
   target_rms = 0.1  # -20 dBFS
   current_rms = np.sqrt(np.mean(audio**2))
   if current_rms > 0:
       audio = audio * (target_rms / current_rms)
   ```

4. **Increase noise augmentation during training**
   ```yaml
   # In config
   data:
     noise_augmentation:
       manifest: "data/manifests/musan_noise.csv"
       probability: 0.5  # was 0.0 or lower
       snr_db_range: [0, 20]  # was [5, 20]
   ```

5. **Retrain with temperature scaling**
   ```bash
   python scripts/calibrate_temperature.py --run-dir results/E5_seed20261108
   ```

### Deep Fixes (if quick fixes fail)

1. **Train OOD detector**
   ```python
   # Use validation set in-distribution + noise/non-speech as OOD
   # Train binary classifier on model features/confidence
   ```

2. **Domain adaptation**
   - Collect target domain data
   - Fine-tune on mixed source + target
   - Use DANN or similar

3. **Ensemble for uncertainty**
   - Train multiple seeds
   - Use prediction variance as OOD score

4. **Feature-space density estimation**
   ```python
   # Fit Gaussian on training features
   # Mahalanobis distance as OOD score
   ```

5. **Input preprocessing pipeline**
   - Bandpass filter (300-3400 Hz for speech)
   - Spectral gating for noise reduction
   - VAD + segmentation

---

## Escalation

**Page oncall if:**
- OOD rate > 35% sustained (critical threshold)
- Demo app unusable (majority predictions flagged)
- Production user complaints spike

**Create issue with:**
- OOD rate time series
- Confidence/margin distributions
- Audio samples triggering OOD (anonymized)
- Environmental context (new deployment, hardware change)

---

## Prevention

- [ ] Add CI gate: OOD rate on test set < 10%
- [ ] Automated noise augmentation in training config
- [ ] VAD preprocessing in inference pipeline
- [ ] Input gain normalization
- [ ] OOD detector as separate model/head
- [ ] Monitoring: confidence distribution drift detection
- [ ] Scheduled evaluation on noisy/external corpora
- [ ] Data collection pipeline for continuous improvement
- [ ] Canary deployment for model updates