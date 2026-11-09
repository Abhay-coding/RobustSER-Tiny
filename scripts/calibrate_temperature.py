"""Fit validation-only temperature scaling for a completed run."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch, yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.evaluation import expected_calibration_error, fit_temperature
from robustser.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = args.run_dir / "calibration.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing calibration: {output}")
    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    checkpoint = torch.load(args.run_dir / "best_validation_macro_f1.pt", map_location=args.device, weights_only=True)
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = CREMADSERDataset(
        config["data"]["manifest"], split="validation", task=task,
        feature_type=config["features"]["type"],
        window_seconds=config["data"]["train_window_seconds"],
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    logits, labels = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits.append(model(features.to(args.device)).cpu())
            labels.append(targets)
    logits, labels = torch.cat(logits), torch.cat(labels)
    temperature = fit_temperature(logits, labels)
    result = {"checkpoint_epoch": checkpoint["epoch"], "temperature": temperature,
              "validation_ece_before": expected_calibration_error(logits, labels),
              "validation_ece_after": expected_calibration_error(logits, labels, temperature=temperature),
              "validation_examples": len(dataset)}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
