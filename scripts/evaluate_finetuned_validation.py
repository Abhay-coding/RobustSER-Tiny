"""Evaluate a saved fine-tuned checkpoint on validation speakers only."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from finetune_pretrained import AudioDataset, LABELS, classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = Wav2Vec2Model.from_pretrained(checkpoint["encoder"], local_files_only=True).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    head = nn.Sequential(nn.Linear(encoder.config.hidden_size * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LABELS))).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    encoder.eval(); head.eval()
    loader = DataLoader(AudioDataset("validation"), batch_size=args.batch_size)
    predictions, targets = [], []
    with torch.inference_mode():
        for i, (audio, labels) in enumerate(loader):
            hidden = encoder(audio.to(device)).last_hidden_state
            features = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)
            predictions.append(head(features).argmax(dim=1).cpu()); targets.append(labels)
            if (i + 1) % 100 == 0:
                print(f"Validation: {i + 1}/{len(loader)} batches", flush=True)
    metrics = classification_metrics(total_loss=0.0, targets=torch.cat(targets), predictions=torch.cat(predictions), num_classes=len(LABELS))
    print(json.dumps({"epoch": checkpoint["epoch"], "labels": LABELS, "validation_macro_f1": metrics.macro_f1, "per_class_f1": metrics.per_class_f1, "per_class_recall": metrics.per_class_recall, "confusion_matrix": metrics.confusion_matrix, "test_split_loaded": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
