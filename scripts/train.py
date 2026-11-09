"""Train E0 on train/validation only; the frozen test split is never loaded."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.models import build_model
from robustser.training import evaluate, set_global_seed, train_one_epoch


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def write_json_line(path: Path, data: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--run-dir", type=Path, help="New directory for checkpoints and metrics.")
    parser.add_argument("--seed", type=int, help="Override the configured seed for a pre-registered replication run.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a valid PyTorch device string.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    experiment = config["experiment"]
    data_config = config["data"]
    training = config["training"]
    task = experiment["task"]
    if task not in TASK_LABELS:
        raise SystemExit(f"Unsupported task: {task}")
    run_dir = args.run_dir or Path("results") / f"{experiment['id']}_seed{experiment['seed']}"
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {run_dir}. Choose --run-dir with a new name.")
    run_dir.mkdir(parents=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    device = resolve_device(args.device)
    set_global_seed(experiment["seed"], deterministic_algorithms=experiment["deterministic_algorithms"])
    noise_augmentation = data_config.get("noise_augmentation", {})
    train_data = CREMADSERDataset(
        data_config["manifest"], split="train", task=task, seed=experiment["seed"],
        noise_manifest_path=noise_augmentation.get("manifest"),
        noise_probability=noise_augmentation.get("probability", 0.0),
        snr_db_range=tuple(noise_augmentation.get("snr_db_range", (5.0, 20.0))),
        feature_type=config["features"]["type"],
        window_seconds=data_config["train_window_seconds"],
    )
    validation_data = CREMADSERDataset(data_config["manifest"], split="validation", task=task, seed=experiment["seed"], feature_type=config["features"]["type"], window_seconds=data_config["train_window_seconds"])
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_data_loader_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(experiment["seed"]), **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    loss_function = nn.CrossEntropyLoss()

    best_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, training["max_epochs"] + 1):
        train_data.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model, batches=train_loader, loss_function=loss_function, device=device,
            optimizer=optimizer, num_classes=len(TASK_LABELS[task]),
        )
        validation_metrics = evaluate(
            model=model, batches=validation_loader, loss_function=loss_function, device=device,
            num_classes=len(TASK_LABELS[task]),
        )
        record = {"epoch": epoch, "train": train_metrics.to_dict(), "validation": validation_metrics.to_dict()}
        write_json_line(metrics_path, record)
        print(f"Epoch {epoch:03d}: train_macro_f1={train_metrics.macro_f1:.4f} validation_macro_f1={validation_metrics.macro_f1:.4f}")
        checkpoint = {
            "epoch": epoch,
            "task": task,
            "labels": TASK_LABELS[task],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics.to_dict(),
            "config": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation_metrics.macro_f1 > best_f1:
            best_f1 = validation_metrics.macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dir / "best_validation_macro_f1.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= training["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}; best validation macro F1 was {best_f1:.4f} at epoch {best_epoch}.")
            break

    summary = {
        "status": "completed_train_validation_only",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_split_loaded": False,
        "checkpoint": "best_validation_macro_f1.pt",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
