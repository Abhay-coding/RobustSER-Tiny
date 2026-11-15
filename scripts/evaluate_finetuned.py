"""Run an interim CREMA-D test diagnostic for one fine-tuned checkpoint."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from finetune_pretrained import AudioDataset, LABELS, classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    result_path = args.run_dir / "test_metrics.json"
    checkpoint_path = args.run_dir / "best_validation_macro_f1.pt"
    if result_path.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {result_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint missing: {checkpoint_path}")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    from transformers import Wav2Vec2Model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("labels", ())) != LABELS:
        raise SystemExit("Checkpoint labels do not match the CREMA-D label order")
    encoder = Wav2Vec2Model.from_pretrained(checkpoint["encoder"], local_files_only=True).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    head = nn.Sequential(nn.Linear(encoder.config.hidden_size * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LABELS))).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    encoder.eval(); head.eval()
    loader = DataLoader(AudioDataset("test"), batch_size=args.batch_size, shuffle=False)
    predictions, targets = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for i, (audio, labels) in enumerate(loader):
            hidden = encoder(audio.to(device)).last_hidden_state
            features = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)
            logits = head(features)
            total_loss += float(nn.functional.cross_entropy(logits, labels.to(device), reduction="sum"))
            predictions.append(logits.argmax(dim=1).cpu()); targets.append(labels)
            if (i + 1) % 100 == 0:
                print(f"Test diagnostic: {i + 1}/{len(loader)} batches", flush=True)
    metrics = classification_metrics(total_loss=total_loss, targets=torch.cat(targets), predictions=torch.cat(predictions), num_classes=len(LABELS))
    result = {"status": "interim_test_diagnostic", "disclosure": "CREMA-D test speakers were previously evaluated by earlier project models; this is not a fresh final test.", "checkpoint": checkpoint_path.name, "checkpoint_epoch": checkpoint["epoch"], "encoder": checkpoint["encoder"], "labels": list(LABELS), "test_records": len(loader.dataset), "device": str(device), "metrics": metrics.to_dict(), "evaluated_at_utc": datetime.now(timezone.utc).isoformat()}
    with result_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved interim diagnostic: {result_path}", flush=True)


if __name__ == "__main__":
    main()
