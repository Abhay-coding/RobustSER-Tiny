# RobustSER-Tiny

**A college research project exploring speech emotion recognition, compact models, unseen speakers, and background noise.**

The project takes a short speech recording and predicts one of six acted-speech labels: **Angry, Disgust, Fear, Happy, Neutral, or Sad**. It includes a local Streamlit demonstration and scripts for training and evaluating experiments.

Our starting point is the paper [TinyML-Based Speech Emotion Recognition on Microcontrollers Using Quantized CNNs and Mel-Filterbank Energy Features](https://ioro.org/ijcsa/article/577979314071/tinyml-based-speech-emotion-recognition-on-microcontrollers-using-quantized-cnns-and-mel-filterbank-energy-features). Our work is **software-only**: we do not claim measurements on physical hardware.

## The project story

1. **Recreate the paper's controlled setting.** E8 uses a small CNN and a TESS random split. It reached **99.50% validation macro F1**. The paper reports 99.5% validation *accuracy*; those are different metrics. E8 uses five shared emotion labels, so it is paper-style rather than an exact reproduction.
2. **Test unfamiliar speakers.** CREMA-D includes many actors. We separate speakers into training, validation, and test groups to examine whether the model works on voices it did not train on. The tiny E5 CNN reached **47.41% test macro F1**, averaged across three seeds.
3. **Explore a larger model.** E18 fine-tunes pretrained Wav2Vec2 and reached **72.62% validation macro F1** and **70.83% in an interim test diagnostic**, each for one checkpoint. It is much larger than E5 and is not a TinyML model.
4. **Study background noise.** In an exploratory three-pair pilot, adding MUSAN noise during training improved mean F1 on fixed 10 dB noisy validation recordings from **46.04% to 67.95%**.

These numbers answer different questions. They use different datasets, splits, models, or evaluation stages; they are not a single head-to-head leaderboard.

## Important results

| Experiment | Result | What was measured |
| --- | ---: | --- |
| E8: paper-style TESS | **99.50%** | Five-emotion random-split validation macro F1; one run |
| E5: tiny CNN | **47.41%** | Six-emotion CREMA-D speaker-disjoint test macro F1; three-seed mean |
| E18: pretrained model | **72.62%** | CREMA-D validation macro F1; one run |
| E18: interim diagnostic | **70.83%** | CREMA-D test macro F1; one checkpoint, 1,066 recordings |
| E19: different class weights | **71.25%** | CREMA-D validation macro F1; one run |

The CREMA-D test speakers had already been evaluated by earlier project models before the E18 diagnostic. Our validation speakers have guided many experiments, so further independent evaluation is needed. The Streamlit classifier offers **E18 when its local checkpoint is available** and E5 as a smaller alternative.

### Noise-training pilot

The pilot splits the original 64 CREMA-D training speakers into **51 pilot-training and 13 pilot-validation speakers**. The original CREMA-D validation and test speakers are not loaded. Each seed has a matched pair of models: one without added MUSAN training noise and one with it.

| Seed | Clean recordings: no noise → noise training | Fixed 10 dB recordings: no noise → noise training |
| --- | ---: | ---: |
| 20261108 | 71.04% → 73.65% | 44.75% → 68.93% |
| 20261109 | 72.28% → 73.71% | 45.41% → 67.21% |
| 20261110 | 72.98% → 74.94% | 47.96% → 67.70% |
| **Mean** | **72.10% → 74.10%** | **46.04% → 67.95%** |

The mean paired gains are **2.00 percentage points on clean audio** and **21.91 points on the fixed 10 dB set**. These are exploratory results on one speaker split, not final test scores or guarantees for other noise levels. The 10 dB set uses a MUSAN evaluation partition that was previously used. Seed 20261108 used a 30-epoch maximum and earlier evaluation RNG handling; the later pairs used a 60-epoch maximum and corrected evaluation RNG handling. Comparisons are matched within each pair, but the recipes are not identical across all pairs.

## What is macro F1?

Each emotion gets its own F1 score, which balances precision and recall. **Macro F1** averages those six scores, giving every emotion equal weight. It is different from accuracy, which counts the fraction of all predictions that are correct. The score does not establish a person's actual emotional state.

## How the audio becomes a prediction

```text
WAV / microphone recording
        ↓
Mono audio at 16 kHz
        ↓
Choose a fixed-length window
        ↓
E5: MFCC features → tiny CNN → six scores
E18/pilot: waveform → pretrained Wav2Vec2 → classifier → six scores
```

The demo uses the centre **two seconds** for E5, represented as a **32 × 62 MFCC** matrix, or the centre **three seconds** for E18. Shorter clips are padded. The app accepts recordings of up to five minutes, but only that centre window is classified. For a representative demonstration, use one clearly acted sentence of about 2.5 seconds.

## Project folders

| Folder | What it contains |
| --- | --- |
| `dataset/` | Locally downloaded CREMA-D, MUSAN, RAVDESS, and TESS audio; not committed to Git |
| `data/manifests/` | Dataset inventories, quality checks, and speaker/noise split CSV files |
| `configs/` | Experiment settings and the saved pilot speaker split |
| `src/robustser/` | Audio processing, models, training metrics, evaluation, and inference code |
| `scripts/` | Commands for dataset preparation, training, and evaluation |
| `results/` | Local checkpoints, logs, and summaries; not committed to Git |
| `demo/` | Streamlit website |
| `docs/` | Research protocol, experiment notes, and demo guide |
| `tests/` | Automated tests |

## Setup on Windows

The commands below use **PowerShell** and Python 3.11, which was used for the project on the original machine. Change the path if your project is elsewhere.

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
py -3.11 --version
py -3.11 -m pip install -e ".[demo]" transformers
$env:PYTHONPATH = 'src'
```

The project dependencies are listed in `pyproject.toml`; the additional `transformers` package is used by the pretrained experiments. Installing PyTorch through the standard dependency command does not by itself guarantee a CUDA-enabled build. Check your GPU setup before training:

```powershell
py -3.11 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
```

If CUDA reports `False`, install a CUDA-enabled PyTorch build using the instructions appropriate for your system before using the `--device cuda` commands. CPU inference is possible for E5, but training the pretrained experiments on CPU can be slow.

### Data and model files

Raw recordings and model checkpoints are excluded from Git. A fresh clone needs the datasets downloaded under `dataset/`, the required manifests prepared, and the relevant checkpoints restored or recreated by training. Respect each dataset's licence and terms.

| Local path | Used for |
| --- | --- |
| `dataset/AudioWAV/` | CREMA-D speech |
| `dataset/MUSAN/` | Background noise for the pilot and stress tests |
| `dataset/RAVDESS/` | Previously examined external evaluation |
| `dataset/TESS/` | E8 paper-style experiment |
| `results/E5_seed20261108/` | The E5 checkpoint and supporting files used by the live demo |
| `results/E18_overnight_seed20261108/` | E18 checkpoint and local interim diagnostic, when available |

The pilot requires `data/manifests/crema_d_speaker_splits.csv`, `data/manifests/musan_noise_splits.csv`, and `configs/experiments/musan_pilot_split.json`. Manifest paths must point to files that exist on your computer. Pretrained runs use `facebook/wav2vec2-base`; the pilot loads it from the local Hugging Face cache. It must be downloaded to that cache first; for example, the E11 training script loads it when run with network access.

## Run the website

Open **PowerShell** and paste these three lines. Change the folder path if your project is stored elsewhere:

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
py -3.11 -m streamlit run demo\app.py --server.port 8504
```

Open [http://localhost:8504](http://localhost:8504) in your browser and keep PowerShell open while using the website. Press **Ctrl+C** in PowerShell to stop it. If port 8504 is occupied, change the port in the command and browser address to a free one, such as `8505`. If the app cannot find the E5 checkpoint, restore the local `results/E5_seed20261108/` run files; they are not included in Git.

In **Live classifier**, choose E18 or E5, then select a WAV file, microphone recording, or the local classroom sample. E18 is offered when its checkpoint exists and requires `transformers` and the cached pretrained encoder. Select **Classify recording** to see the chosen model's label, scores, and warnings. The classroom sample requires its local CREMA-D audio file. The **Research evidence** tab explains E5, E18, and the noise pilot separately.

## Run training and evaluation

Training writes files into `results/`. Use a **new run directory** for each experiment; the scripts refuse to overwrite an existing run. These commands require the relevant datasets and manifests already on disk.

### Train the E5 tiny model

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
py -3.11 -u scripts\train.py --config configs\experiments\e5_two_second_context.yaml --run-dir results\E5_example --seed 20261108 --device cuda
```

This trains and selects a checkpoint on validation speakers; it does not load the test split.

### Run a matched noise-pilot pair

The following runs use the same seed and saved pilot split. Run the second command only after the first finishes successfully. They may take several hours, and early stopping can end a run before 60 epochs.

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'

py -3.11 -u scripts\pilot_noise.py --branch clean --seed 20261109 --epochs 60 --patience 10 --run-dir results\pilot_clean_example --device cuda
if ($LASTEXITCODE -eq 0) {
  py -3.11 -u scripts\pilot_noise.py --branch noise --seed 20261109 --epochs 60 --patience 10 --run-dir results\pilot_noise_example --device cuda
}
```

Each folder stores `best.pt`, `metrics.jsonl`, and `summary.json`. The checkpoint is chosen using **clean validation F1** for both branches; the summary also reports performance on the same fixed 10 dB noisy validation set. Keep the computer plugged in and awake while training.

### Evaluate a fine-tuned checkpoint on validation speakers

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
py -3.11 -u scripts\evaluate_finetuned_validation.py results\E18_overnight_seed20261108\best_validation_macro_f1.pt --device cuda
```

This prints per-class F1, recall, and the confusion matrix for the original validation split. The E18 checkpoint and cached pretrained encoder must exist locally.

### E18 interim test diagnostic

`scripts/evaluate_finetuned.py` supports the Wav2Vec2 checkpoint format. E18's diagnostic has **already been run** locally, so the script refuses to overwrite its `test_metrics.json`. The command below is for a run directory whose diagnostic has not yet been saved:

```powershell
py -3.11 -u scripts\evaluate_finetuned.py --run-dir results\YOUR_FINETUNED_RUN --device cuda
```

Replace `YOUR_FINETUNED_RUN` with the selected run. Use test data only after deciding the purpose of the evaluation; repeated test-driven model selection makes its result less independent. The existing CREMA-D test speakers have already been examined in this project.

## Run automated tests

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
py -3.11 -m unittest discover -s tests -v
```

## What to say during the presentation

> We started with a paper-inspired compact speech-emotion model and reproduced a high score in a controlled TESS setting. We then tested unfamiliar CREMA-D speakers, where the tiny model averaged 47.41% test macro F1. A larger pretrained model achieved 70.83% in an interim test diagnostic. In a separate exploratory pilot, adding background noise during training improved mean F1 from 46.04% to 67.95% on fixed noisy validation recordings. The settings and splits differ, so we report the results separately.

## Limitations

- These datasets contain acted speech; predictions are dataset labels, not a diagnosis of real emotions, mental health, honesty, or intent.
- The pilot uses one speaker split, three seed pairs, and a previously used fixed noise set. More speakers, noise conditions, and independent evaluation are needed.
- E18 is a larger pretrained research model; the website also offers the smaller E5. We do not claim hardware deployment or device-level measurements.
- Earlier validation and test observations influenced project development. Results should be reported with their split, metric, seed count, and prior-use context.
