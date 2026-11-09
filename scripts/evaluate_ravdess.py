"""Evaluate a fixed selected checkpoint on external RAVDESS exactly once."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch, yaml
import numpy as np
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data import RAVDESSExternalDataset, TASK_LABELS
from robustser.models import build_model
from robustser.training.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/ravdess_speech_manifest.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=200, help="New external examples processed per invocation.")
    args = parser.parse_args()
    output = args.run_dir / "ravdess_external_metrics.json"
    if output.exists():
        raise SystemExit(f"Refusing to repeat external evaluation: {output}")
    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    checkpoint = torch.load(args.run_dir / "best_validation_macro_f1.pt", map_location=args.device, weights_only=True)
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = RAVDESSExternalDataset(args.manifest, task=task, feature_type=config["features"]["type"], window_seconds=config["data"]["train_window_seconds"])
    cache_path = args.run_dir / "ravdess_external_logits.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        logits, labels = cache["logits"], cache["labels"]
    else:
        logits = np.full((len(dataset), len(TASK_LABELS[task])), np.nan, dtype=np.float32)
        labels = np.full(len(dataset), -1, dtype=np.int64)
    pending = np.flatnonzero(labels < 0)[:args.chunk_size]
    with torch.no_grad():
        for index in pending:
            features, label = dataset[int(index)]
            logits[index] = model(features.unsqueeze(0).to(args.device)).cpu().numpy()[0]
            labels[index] = label
    np.savez_compressed(cache_path, logits=logits, labels=labels)
    remaining = int((labels < 0).sum())
    if remaining:
        print(json.dumps({"status": "partial", "processed": int((labels >= 0).sum()), "remaining": remaining, "cache": str(cache_path)}, indent=2))
        return
    tensor_logits, tensor_labels = torch.from_numpy(logits), torch.from_numpy(labels)
    losses = nn.CrossEntropyLoss(reduction="none")(tensor_logits, tensor_labels)
    metrics = classification_metrics(total_loss=float(losses.sum()), targets=tensor_labels, predictions=tensor_logits.argmax(dim=1), num_classes=len(TASK_LABELS[task]))
    result = {"checkpoint_epoch": checkpoint["epoch"], "task": task, "records": len(dataset), "metrics": metrics.to_dict()}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
