"""Train a validation-selected classifier on frozen pretrained speech embeddings.

Only train and validation speakers are loaded. The encoder is downloaded by
Transformers on the first run; the pretrained model is not a TinyML model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robustser.data.cremad import TASK_LABELS, load_split_rows, read_pcm16_mono
from robustser.features.log_mel import fit_audio_window
from robustser.training import set_global_seed
from robustser.training.metrics import classification_metrics

TASK = "crema_d_six_emotion"
LABELS = TASK_LABELS[TASK]
MANIFEST = "data/manifests/crema_d_speaker_splits.csv"


def embeddings(encoder, device, split):
    rows = load_split_rows(MANIFEST, split=split, task=TASK)
    vectors, labels = [], []
    encoder.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            audio = read_pcm16_mono(row["path"])
            audio = fit_audio_window(audio, target_samples=48000, policy="center_crop_or_zero_pad")
            audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
            waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
            hidden = encoder(waveform).last_hidden_state
            vectors.append(torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1).cpu().squeeze(0))
            labels.append(LABELS.index(row["emotion"]))
            if (index + 1) % 250 == 0 or index + 1 == len(rows):
                print(f"{split}: encoded {index + 1}/{len(rows)}", flush=True)
    return TensorDataset(torch.stack(vectors), torch.tensor(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise SystemExit(f"Run directory already exists: {args.run_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; check your PyTorch installation.")
    from transformers import Wav2Vec2Model

    set_global_seed(args.seed, deterministic_algorithms=True)
    device = torch.device(args.device)
    encoder = Wav2Vec2Model.from_pretrained(args.model).to(device)
    train = embeddings(encoder, device, "train")
    validation = embeddings(encoder, device, "validation")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    args.run_dir.mkdir(parents=True)
    model = nn.Sequential(nn.Linear(train.tensors[0].shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LABELS))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(validation.tensors[0].to(device))
            targets = validation.tensors[1].to(device)
            metrics = classification_metrics(total_loss=float(criterion(logits, targets)) * len(targets), targets=targets, predictions=logits.argmax(1), num_classes=len(LABELS))
        score = metrics.macro_f1
        print(f"Epoch {epoch:03d}: validation_macro_f1={score:.4f}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "encoder": args.model, "labels": LABELS, "seed": args.seed, "epoch": epoch}, args.run_dir / "best_validation_macro_f1.pt")
        elif epoch - best_epoch >= 15:
            break
    summary = {"status": "completed_train_validation_only", "encoder": args.model, "encoder_frozen": True, "seed": args.seed, "best_epoch": best_epoch, "best_validation_macro_f1": best, "test_split_loaded": False}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
