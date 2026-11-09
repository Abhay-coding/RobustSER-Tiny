"""Verify the frozen E0 data/model contract without training parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow direct execution from a source checkout before editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    manifest = config["data"]["manifest"]
    if task not in TASK_LABELS:
        raise SystemExit(f"Unsupported task in configuration: {task}")

    feature_type = config["features"]["type"]
    datasets = {
        split: CREMADSERDataset(manifest, split=split, task=task, feature_type=feature_type, window_seconds=config["data"]["train_window_seconds"])
        for split in ("train", "validation", "test")
    }
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task]))
    model.eval()
    with torch.no_grad():
        features, label = datasets["train"][0]
        logits = model(features.unsqueeze(0))
    expected_feature_shape = (1, *config["features"]["expected_shape"])
    if tuple(features.shape) != expected_feature_shape or tuple(logits.shape) != (1, len(TASK_LABELS[task])):
        raise SystemExit("Baseline contract shape check failed")

    print("Baseline preflight passed; no training was performed.")
    print(f"Task: {task}")
    print(f"Feature type: {feature_type}")
    print("Dataset records: " + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items()))
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Example label index: {label}")
    print(f"Model output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in model.parameters())}")


if __name__ == "__main__":
    main()
