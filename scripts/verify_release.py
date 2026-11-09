"""Run the automated tests and verify the selected E5 inference contract."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data import CREMADSERDataset, TASK_LABELS
from robustser.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/e5_two_second_context.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/E5_seed20261108"))
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit("Automated tests failed")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    dataset = CREMADSERDataset(
        config["data"]["manifest"], split="validation", task=task,
        feature_type=config["features"]["type"],
        window_seconds=config["data"]["train_window_seconds"],
    )
    features, _ = dataset[0]
    checkpoint = torch.load(args.run_dir / "best_validation_macro_f1.pt", map_location="cpu", weights_only=True)
    if tuple(checkpoint["labels"]) != TASK_LABELS[task]:
        raise SystemExit("Checkpoint labels do not match the configured task")
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        logits = model(features.unsqueeze(0))
    expected_features = (1, *config["features"]["expected_shape"])
    if tuple(features.shape) != expected_features or tuple(logits.shape) != (1, len(TASK_LABELS[task])):
        raise SystemExit("Selected-model inference contract failed")
    print("Release verification passed: tests, E5 feature contract, and selected checkpoint are valid.")


if __name__ == "__main__":
    main()
