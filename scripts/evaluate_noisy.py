"""Run the pre-registered MUSAN-held-out noisy test protocol exactly once."""

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

from robustser.data import NoisyCREMADSERTestDataset, TASK_LABELS
from robustser.models import build_model
from robustser.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--snr", type=float, help="Evaluate exactly one pre-registered SNR; safe for resumable runs.")
    parser.add_argument("--noise-manifest", type=Path,
                        help="Required only for a baseline run whose saved config has no noise section.")
    parser.add_argument("--snrs", nargs="+", type=float,
                        help="Required only for a baseline run whose saved config has no noisy-test schedule.")
    args = parser.parse_args()
    result_path = args.run_dir / "noisy_test_metrics.json"
    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    noise_config = config["data"].get("noise_augmentation") or {}
    noise_manifest = str(args.noise_manifest) if args.noise_manifest else noise_config.get("manifest")
    configured_snrs = [float(value) for value in config.get("evaluation", {}).get("noisy_test_snr_db", [])]
    if not configured_snrs and args.snrs:
        configured_snrs = args.snrs
    if not noise_manifest or not configured_snrs:
        raise SystemExit("Provide --noise-manifest and --snrs for a baseline run without a noisy-test protocol")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint = torch.load(args.run_dir / "best_validation_macro_f1.pt", map_location=device, weights_only=True)
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    requested_snrs = configured_snrs if args.snr is None else [args.snr]
    if any(snr not in configured_snrs for snr in requested_snrs):
        raise SystemExit(f"Requested SNR must be one of {configured_snrs}")
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        snr_results = result["snr_metrics"]
    else:
        snr_results: dict[str, object] = {}
        result = {
            "checkpoint": "best_validation_macro_f1.pt",
            "checkpoint_epoch": checkpoint["epoch"],
            "task": task,
            "labels": list(TASK_LABELS[task]),
            "noise_manifest": noise_manifest,
            "noise_split": "evaluation_holdout",
            "test_records_per_snr": 0,
            "device": str(device),
            "snr_metrics": snr_results,
        }
    for snr_db in requested_snrs:
        snr_key = str(int(snr_db)) if snr_db.is_integer() else str(snr_db)
        if snr_key in snr_results:
            raise SystemExit(f"Refusing to repeat held-out noisy evaluation at {snr_key} dB")
        dataset = NoisyCREMADSERTestDataset(
            config["data"]["manifest"], task=task, noise_manifest_path=noise_manifest,
            snr_db=float(snr_db), seed=config["experiment"]["seed"],
            feature_type=config["features"]["type"],
            window_seconds=config["data"]["train_window_seconds"],
        )
        loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        metrics = evaluate(model=model, batches=loader, loss_function=nn.CrossEntropyLoss(), device=device, num_classes=len(TASK_LABELS[task]))
        snr_results[snr_key] = metrics.to_dict()
        result["test_records_per_snr"] = len(dataset)
        result["snr_metrics"] = snr_results
        result["status"] = "completed_held_out_musan_noisy_test_once" if len(snr_results) == len(configured_snrs) else "partial_held_out_musan_noisy_test"
        result["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"SNR {snr_db} dB: macro_f1={metrics.macro_f1:.4f}, uar={metrics.uar:.4f}")
    print(f"Noisy held-out test results saved: {result_path}")


if __name__ == "__main__":
    main()
