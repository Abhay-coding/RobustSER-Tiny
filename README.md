# RobustSER-Tiny

> **Can a tiny model recognise emotion from speech—and will it still work when the speaker changes?**

RobustSER-Tiny is a college research project about **Speech Emotion
Recognition**. In simple words, it listens to a short acted-speech recording
and predicts a dataset label such as **Angry, Happy, Fear, Disgust, Neutral,**
or **Sad**.

It is inspired by this research paper:

> [TinyML-Based Speech Emotion Recognition on Microcontrollers Using Quantized CNNs and Mel-Filterbank Energy Features](https://ioro.org/ijcsa/article/577979314071/tinyml-based-speech-emotion-recognition-on-microcontrollers-using-quantized-cnns-and-mel-filterbank-energy-features)

---

## The project story, without technical jargon

### Step 1 — We first followed the paper's idea

The reference paper uses short audio clips, turns them into compact sound
features, and trains a small CNN model. It reports **99.5% validation
accuracy** on a controlled version of the TESS dataset.

We recreated that controlled idea with TESS, five common emotion labels,
16 kHz audio, 32-band Mel-energy features, a small CNN, and a random 80/20
train/validation split.

### Step 2 — The controlled replication worked

Our TESS paper-style replication (E8) achieved:

| Controlled replication result | Score |
| --- | ---: |
| Best validation macro F1 | **99.50%** |
| Best epoch | 19 |
| Dataset | TESS, five emotions |
| Split | Random 80% training / 20% validation |

This is a valid answer to: *“Can the compact paper-style model learn this
clean, controlled benchmark?”* The answer is **yes**.

> Important honesty note: E8 is a close paper-style replication, not an exact
> reproduction. The paper also adds a custom Noise class; E8 uses the five
> shared speech-emotion labels only.

### Step 3 — Then we asked a harder, more realistic question

The paper itself says that performance drops on natural recordings and
unfamiliar voices. A high score on a simple controlled split does not always
mean the model will work equally well for a new person in the real world.

So we moved to **CREMA-D**, a much more varied acted-speech dataset with many
different speakers. Instead of allowing the same person to appear in both
training and validation, we made sure speakers were kept separate:

```text
64 speakers for training → 14 speakers for validation → 13 completely unseen speakers for testing
```

This is called a **speaker-disjoint evaluation**. It is harder, but it tests
whether the model has learned emotion-related voice patterns rather than simply
memorising a familiar speaker's voice.

---

## Our results: tiny model and larger pretrained model

We explored two different approaches. **E5** is the tiny model used by the
website. **E18** uses a much larger pretrained Wav2Vec2 model and is our best
exploratory validation result so far.

| Approach | Result | What it measures |
| --- | ---: | --- |
| E5 tiny CNN | **47.41%** | Held-out CREMA-D test macro F1, mean of three seeds |
| E18 pretrained Wav2Vec2 | **72.62%** | CREMA-D validation macro F1, one run; test not evaluated |
| E19 pretrained Wav2Vec2 | **71.25%** | Same validation speakers, one run; test not evaluated |

These numbers should not be compared as a direct test-set improvement: they
use different models and evaluation stages. E18's validation speakers have
guided repeated experiments, so its score may be optimistic. A fresh evaluation
and independent runs are needed before claiming unseen-speaker performance.
E18's larger model is not a 930-parameter TinyML model.

### E5: the tiny model

| Result | Value | Plain-English meaning |
| --- | ---: | --- |
| E5 final clean test macro F1 | **47.41%** | Score on speakers the model never heard during training. |
| E5 validation macro F1 | **47.76%** | Score used to choose the checkpoint. |
| Validation-to-test difference | **0.35 points** | The final unseen-speaker result stayed close to validation. |
| Model size | **930 parameters** | The model is intentionally very small. |
| Improvement over E4 | **+5.81 points** | Two seconds of context improved the robust model. |

## Why the scores answer different questions

At first glance, 47.41% looks much lower than 99.50%. But these results come
from **different tests**. Comparing them directly would be like comparing a
practice quiz with an exam containing unfamiliar questions.

| Controlled E8 replication | Robust E5 evaluation |
| --- | --- |
| TESS: two actresses, clean scripted recordings | CREMA-D: many speakers and acting styles |
| Random 80/20 split | Speakers are completely separated |
| Validation result | Final held-out test result |
| Five emotion labels | Six emotion labels |
| Question: “Can it learn this controlled set?” | Question: “Does it work for new speakers?” |

E5 uses **macro F1**, a stricter score than ordinary accuracy. Every emotion
gets equal importance, so the model cannot look good simply by predicting the
easiest emotion often.

The most important evidence is that E5 improved from **41.60% to 47.41%** when
we increased speech context from one second to two seconds—without increasing
the 930-parameter model size. This +5.81-point improvement was tested across
three random seeds and is statistically supported.

---

## Datasets we used

| Dataset | Why we used it |
| --- | --- |
| **TESS** | Controlled paper-style replication. It shows how well the compact model can perform on a simple benchmark. |
| **CREMA-D** | Main robust experiment: many speakers, six acted-speech labels, and unseen-speaker testing. |
| **MUSAN** | Noise clips used for controlled noise augmentation and stress testing. |
| **RAVDESS** | External dataset used only after model selection to check cross-dataset generalisation. |

---

## What happens to audio before training or prediction?

```text
WAV recording
   ↓
Convert to mono 16 kHz audio
   ↓
Choose a fixed speech window
   ↓
Create a small sound-feature image (MFCC or Mel energy)
   ↓
Small CNN model
   ↓
Emotion probabilities and confidence warnings
```

For the robust E5 model, the input is a **two-second** recording window and a
**32 × 62 MFCC** feature matrix.

---

## Challenges we found and how we handled them

| Challenge | What we did |
| --- | --- |
| A very high controlled score may not work for unfamiliar people | Added speaker-disjoint CREMA-D evaluation. |
| A random split can place the same speaker in both groups | Froze a split where every speaker is in only one group. |
| Duplicate audio with conflicting labels can distort results | Detected duplicates with SHA-256 hashes and excluded only conflicts. |
| Noise could accidentally leak from training into evaluation | Used separate, hash-disjoint MUSAN partitions. |
| Two damaged TESS files were present in the download | The manifest generator checks WAV files and excludes only invalid containers. |
| One lucky training run can be misleading | Used three seeded runs for the main robust experiments. |

---

## Experiment journey

| Experiment | Main change | Result |
| --- | --- | ---: |
| E1 | Paper-inspired compact CNN baseline | 38.15% robust test macro F1 |
| E2 | Added controlled MUSAN noise augmentation | 37.38% robust test macro F1 |
| E3 | Switched to MFCC features | 40.80% robust test macro F1 |
| E4 | Used a 930-parameter depthwise-separable CNN | 41.60% robust test macro F1 |
| E5 | Used two seconds of speech context | **47.41% robust test macro F1** |
| E7 | Paper-style random-split comparison on CREMA-D | 47.27% validation macro-F1 mean |
| E8 | TESS paper-style controlled replication | **99.50% validation macro F1** |
| E9 | Wider CNN, two seconds of speech | 52.93% validation macro F1, one run |
| E10 | Wider CNN, three seconds of speech | 54.64% validation macro F1, one run |
| E11 | Frozen pretrained Wav2Vec2 encoder | 58.65% validation macro F1, one run |
| E12 | Fine-tuned the last two encoder layers | 60.86% validation macro F1, one run |
| E13 | Continued fine-tuning | 69.63% validation macro F1, one run |
| E14 | Fine-tuned six layers | 69.94% validation macro F1, one run |
| E15 | Fine-tuned eight layers with class weights | 70.06% validation macro F1, one run |
| E16 | Corrected weighted-loss accumulation | 70.86% validation macro F1, one run |
| E17 | Adjusted class weights | 71.10% validation macro F1, one run |
| E18 | Continued E17 training | **72.62% validation macro F1, one run** |
| E19 | Increased Sad class weight | 71.25% validation macro F1, one run |

E13 was interrupted; its saved checkpoint was subsequently evaluated on the
validation split. E15 used per-microbatch weighted loss, which can dilute the
intended weights; E16 and later use corrected accumulation. All E9–E19 numbers
above are validation results, not final test scores.

---

## Try the demo

The Streamlit website runs locally on your computer. It accepts a WAV file or
microphone recording and shows the predicted label, probabilities, confidence,
and the MFCC feature image used by the model. **The live classifier uses E5,**
not the E18 pretrained checkpoint; the research tab reports both separately.

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
py -3.11 -m streamlit run demo/app.py --server.port 8504
```

For the most representative demo prediction, speak one clearly acted sentence
for about 2.5 seconds. The app accepts up to five minutes, but E5 classifies
the centre two-second window.

> The demonstration predicts acted-speech **dataset labels**. It is not a
> mental-health, honesty, or real-emotion assessment tool.

---

## A short explanation for your teacher

> “We first recreated the compact CNN approach from the paper. On the same type
> of controlled TESS random-split evaluation, our replication reached 99.50%
> validation macro F1. The paper also notes that real-life recordings are more
> difficult, so we tested the same core idea on CREMA-D with completely unseen
> speakers. That stricter test gave 47.41% macro F1, and we improved it by 5.81
> points using two seconds of speech context while keeping the model tiny. We
> also explored a larger pretrained speech model, reaching 72.62% on validation
> speakers. That newer result has not yet been tested on a fresh held-out set.”

---

## Project folders

```text
configs/          Experiment settings
data/manifests/   Audits and fixed dataset splits
src/robustser/    Audio processing, model, training, metrics, inference
scripts/          Dataset preparation, training, and evaluation tools
demo/             Local Streamlit demonstration website
results/          Local checkpoints and outputs (not committed to Git)
docs/             Detailed report and presentation material
tests/            Automated checks
```

## Verify the code

```powershell
Set-Location 'D:\Vs code\RobustSER-Tiny'
$env:PYTHONPATH = 'src'
python scripts/verify_release.py
```

Raw datasets are intentionally not committed. Please respect the licence and
terms for TESS, CREMA-D, MUSAN, and RAVDESS.
