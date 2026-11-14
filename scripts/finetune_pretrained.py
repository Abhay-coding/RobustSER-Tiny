"""Fine-tune upper Wav2Vec2 layers using only CREMA-D train/validation speakers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robustser.data.cremad import TASK_LABELS, load_split_rows, read_pcm16_mono
from robustser.features.log_mel import fit_audio_window
from robustser.training import set_global_seed
from robustser.training.metrics import classification_metrics

TASK = "crema_d_six_emotion"
LABELS = TASK_LABELS[TASK]


class AudioDataset(Dataset):
    def __init__(self, split):
        self.rows = load_split_rows("data/manifests/crema_d_speaker_splits.csv", split=split, task=TASK)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = fit_audio_window(read_pcm16_mono(row["path"]), target_samples=48000, policy="center_crop_or_zero_pad")
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        return torch.from_numpy(audio), LABELS.index(row["emotion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-head", type=Path, default=Path("results/E11_wav2vec2_seed20261108/best_validation_macro_f1.pt"))
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-hours", type=float, help="Stop after the first completed epoch following this time limit.")
    parser.add_argument("--patience", type=int, default=6, help="Stop after this many epochs without validation improvement.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--unfreeze-layers", type=int, default=2, help="Number of final transformer layers to train.")
    parser.add_argument("--class-weights", type=float, nargs=6, metavar=("ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD"), help="Optional positive loss weights in label order.")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or (args.max_hours is not None and args.max_hours <= 0):
        parser.error("epochs, patience, and max-hours must be positive")
    if args.class_weights and any(weight <= 0 for weight in args.class_weights):
        parser.error("all class weights must be positive")
    if args.run_dir.exists():
        raise SystemExit(f"Run directory already exists: {args.run_dir}")
    if not args.initial_head.is_file():
        raise SystemExit(f"E11 checkpoint missing: {args.initial_head}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    from transformers import Wav2Vec2Model

    set_global_seed(args.seed, deterministic_algorithms=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.initial_head, map_location="cpu", weights_only=False)
    encoder = Wav2Vec2Model.from_pretrained(checkpoint["encoder"]).to(device)
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if not 1 <= args.unfreeze_layers <= len(encoder.encoder.layers):
        raise SystemExit("--unfreeze-layers must be between 1 and the number of transformer layers")
    for param in encoder.parameters():
        param.requires_grad = False
    for layer in encoder.encoder.layers[-args.unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    head = nn.Sequential(nn.Linear(encoder.config.hidden_size * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LABELS))).to(device)
    head.load_state_dict(checkpoint["head_state_dict"] if "head_state_dict" in checkpoint else checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-4},
    ], weight_decay=0.01)
    weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device) if args.class_weights else None
    # Accumulate the weighted loss numerator and normalize once per effective
    # batch. Per-microbatch weighted means cancel class weights at batch size 1.
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    train = DataLoader(AudioDataset("train"), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    validation = DataLoader(AudioDataset("validation"), batch_size=args.batch_size, shuffle=False)
    args.run_dir.mkdir(parents=True)
    best, best_epoch = -1.0, 0
    start_time = time.monotonic()
    stop_reason = "max_epochs"
    for epoch in range(1, args.epochs + 1):
        encoder.train(); head.train(); optimizer.zero_grad()
        group_weight = 0.0
        print(f"Epoch {epoch:03d}: training {len(train)} batches...", flush=True)
        for i, (audio, labels) in enumerate(train):
            hidden = encoder(audio.to(device)).last_hidden_state
            features = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)
            labels = labels.to(device)
            criterion(head(features), labels).backward()
            group_weight += float(weights[labels].sum()) if weights is not None else labels.numel()
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train):
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            param.grad.div_(group_weight)
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1.0)
                optimizer.step(); optimizer.zero_grad()
                group_weight = 0.0
            if (i + 1) % 100 == 0 or i + 1 == len(train):
                print(f"Epoch {epoch:03d}: batch {i + 1}/{len(train)}", flush=True)
        print(f"Epoch {epoch:03d}: validating...", flush=True)
        encoder.eval(); head.eval(); predictions, targets = [], []
        with torch.no_grad():
            for audio, labels in validation:
                hidden = encoder(audio.to(device)).last_hidden_state
                features = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)
                predictions.append(head(features).argmax(dim=1).cpu()); targets.append(labels)
        metrics = classification_metrics(total_loss=0.0, targets=torch.cat(targets), predictions=torch.cat(predictions), num_classes=len(LABELS))
        score = metrics.macro_f1
        print(f"Epoch {epoch:03d}: validation_macro_f1={score:.4f}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"encoder": checkpoint["encoder"], "encoder_state_dict": encoder.state_dict(), "head_state_dict": head.state_dict(), "labels": LABELS, "seed": args.seed, "epoch": epoch}, args.run_dir / "best_validation_macro_f1.pt")
        elif epoch - best_epoch >= args.patience:
            stop_reason = "early_stopping"
            break
        if args.max_hours is not None and time.monotonic() - start_time >= args.max_hours * 3600:
            stop_reason = "time_limit_after_epoch"
            print("Time limit reached; finishing after this completed epoch.", flush=True)
            break
    summary = {"status": "completed_train_validation_only", "encoder": checkpoint["encoder"], "initial_checkpoint": str(args.initial_head), "unfrozen_transformer_layers": args.unfreeze_layers, "class_weights": args.class_weights, "seed": args.seed, "best_epoch": best_epoch, "best_validation_macro_f1": best, "stop_reason": stop_reason, "elapsed_hours": (time.monotonic() - start_time) / 3600, "test_split_loaded": False}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
