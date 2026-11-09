"""Perform one safeguarded held-out test evaluation for a completed run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.models import build_model
from robustser.training import evaluate


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result_path = args.run_dir / "test_metrics.json"
    if result_path.exists():
        raise SystemExit(f"Refusing to repeat held-out test evaluation: {result_path} already exists.")
    config_path = args.run_dir / "config.yaml"
    checkpoint_path = args.run_dir / "best_validation_macro_f1.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("Run directory must contain config.yaml and best_validation_macro_f1.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    if task not in TASK_LABELS:
        raise SystemExit(f"Unsupported task: {task}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("task") != task or tuple(checkpoint.get("labels", ())) != TASK_LABELS[task]:
        raise SystemExit("Checkpoint task/label mapping does not match the frozen run configuration")
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_data = CREMADSERDataset(config["data"]["manifest"], split="test", task=task, seed=config["experiment"]["seed"], feature_type=config["features"]["type"], window_seconds=config["data"]["train_window_seconds"])
    test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    metrics = evaluate(model=model, batches=test_loader, loss_function=nn.CrossEntropyLoss(), device=device, num_classes=len(TASK_LABELS[task]))
    result = {
        "status": "completed_held_out_test_once",
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "task": task,
        "labels": list(TASK_LABELS[task]),
        "test_records": len(test_data),
        "device": str(device),
        "metrics": metrics.to_dict(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Held-out test results saved: {result_path}")


if __name__ == "__main__":
    main()
