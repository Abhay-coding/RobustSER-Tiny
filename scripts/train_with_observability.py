"""
Train E0 on train/validation only; the frozen test split is never loaded.
Enhanced with structured logging, metrics collection, and tracing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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
from robustser.evaluation.calibration import expected_calibration_error, fit_temperature
from robustser.observability import (
    setup_structured_logging,
    get_logger,
    LogContext,
    CorrelationContext,
    get_metrics_collector,
    record_metric,
    trace_span,
    get_trace_context,
)


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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

    # Setup structured logging
    log_file = run_dir / "training.jsonl"
    logger = setup_structured_logging(
        level=getattr(logging, args.log_level),
        log_file=log_file,
        json_output=True,
    )
    logger = get_logger("robustser.train")

    # Setup metrics collector
    metrics_file = run_dir / "metrics.jsonl"
    collector = get_metrics_collector()
    collector._jsonl_path = metrics_file  # Reuse existing metrics.jsonl

    device = resolve_device(args.device)
    set_global_seed(experiment["seed"], deterministic_algorithms=experiment["deterministic_algorithms"])

    # Correlation context for this run
    run_id = run_dir.name
    corr_context = LogContext(
        experiment_id=experiment["id"],
        run_id=run_id,
        seed=experiment["seed"],
        device=str(device),
    )

    logger.info("Starting training", extra={"config": config})

    with CorrelationContext(corr_context):
        noise_augmentation = data_config.get("noise_augmentation", {})
        train_data = CREMADSERDataset(
            data_config["manifest"], split="train", task=task, seed=experiment["seed"],
            noise_manifest_path=noise_augmentation.get("manifest"),
            noise_probability=noise_augmentation.get("probability", 0.0),
            snr_db_range=tuple(noise_augmentation.get("snr_db_range", (5.0, 20.0))),
            feature_type=config["features"]["type"],
            window_seconds=data_config["train_window_seconds"],
        )
        validation_data = CREMADSERDataset(
            data_config["manifest"], split="validation", task=task, seed=experiment["seed"],
            feature_type=config["features"]["type"], window_seconds=data_config["train_window_seconds"]
        )
        loader_options = {
            "batch_size": training["batch_size"],
            "num_workers": training["num_data_loader_workers"],
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(
            train_data, shuffle=True,
            generator=torch.Generator().manual_seed(experiment["seed"]),
            **loader_options
        )
        validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)

        model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
        loss_function = nn.CrossEntropyLoss()

        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Model initialized", extra={
            "model_name": config["model"]["name"],
            "num_parameters": num_params,
            "device": str(device),
        })
        record_metric("model_parameters", num_params, unit="count", description="Total model parameters")
        record_metric("model_trainable_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad), unit="count")

        best_f1 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0

        # System metrics
        if device.type == "cuda":
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e6
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e6
            record_metric("gpu_memory_allocated_mb", gpu_mem_allocated, unit="MB")
            record_metric("gpu_memory_reserved_mb", gpu_mem_reserved, unit="MB")

        for epoch in range(1, training["max_epochs"] + 1):
            epoch_start = time.time()
            train_data.set_epoch(epoch)

            # Update correlation context for this epoch
            epoch_context = LogContext(
                experiment_id=experiment["id"],
                run_id=run_id,
                epoch=epoch,
                seed=experiment["seed"],
                split="train",
                device=str(device),
            )

            with CorrelationContext(epoch_context):
                with trace_span("train_epoch", baggage={"epoch": str(epoch)}) as span:
                    train_metrics = train_one_epoch(
                        model=model, batches=train_loader, loss_function=loss_function,
                        device=device, optimizer=optimizer, num_classes=len(TASK_LABELS[task]),
                    )
                    span.set_attribute("train_loss", train_metrics.loss)
                    span.set_attribute("train_macro_f1", train_metrics.macro_f1)

            # Validation
            val_context = LogContext(
                experiment_id=experiment["id"],
                run_id=run_id,
                epoch=epoch,
                seed=experiment["seed"],
                split="validation",
                device=str(device),
            )

            with CorrelationContext(val_context):
                with trace_span("validation_epoch", baggage={"epoch": str(epoch)}) as span:
                    validation_metrics = evaluate(
                        model=model, batches=validation_loader, loss_function=loss_function,
                        device=device, num_classes=len(TASK_LABELS[task]),
                    )
                    span.set_attribute("validation_loss", validation_metrics.loss)
                    span.set_attribute("validation_macro_f1", validation_metrics.macro_f1)

            epoch_duration = time.time() - epoch_start

            # Record metrics
            record_metric("epoch_duration_seconds", epoch_duration, unit="seconds")
            record_metric("train_loss", train_metrics.loss, unit="cross_entropy")
            record_metric("train_macro_f1", train_metrics.macro_f1, unit="f1_score")
            record_metric("train_accuracy", train_metrics.accuracy, unit="accuracy")
            record_metric("train_balanced_accuracy", train_metrics.balanced_accuracy, unit="accuracy")
            record_metric("validation_loss", validation_metrics.loss, unit="cross_entropy")
            record_metric("validation_macro_f1", validation_metrics.macro_f1, unit="f1_score")
            record_metric("validation_accuracy", validation_metrics.accuracy, unit="accuracy")
            record_metric("validation_balanced_accuracy", validation_metrics.balanced_accuracy, unit="accuracy")

            # Per-class metrics
            for i, (prec, rec, f1) in enumerate(zip(
                validation_metrics.per_class_precision,
                validation_metrics.per_class_recall,
                validation_metrics.per_class_f1,
            )):
                record_metric("validation_per_class_precision", prec, labels={"class": TASK_LABELS[task][i]}, unit="precision")
                record_metric("validation_per_class_recall", rec, labels={"class": TASK_LABELS[task][i]}, unit="recall")
                record_metric("validation_per_class_f1", f1, labels={"class": TASK_LABELS[task][i]}, unit="f1_score")

            # Calibration metrics on validation set
            # Get logits for ECE computation
            val_logits_list = []
            val_targets_list = []
            with torch.no_grad():
                for features, labels in validation_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits = model(features)
                    val_logits_list.append(logits.cpu())
                    val_targets_list.append(labels.cpu())
            val_logits = torch.cat(val_logits_list)
            val_targets = torch.cat(val_targets_list)

            ece = expected_calibration_error(val_logits, val_targets, temperature=1.0, bins=15)
            record_metric("validation_ece", ece, unit="ece")

            # Temperature scaling
            temperature = fit_temperature(val_logits, val_targets)
            ece_scaled = expected_calibration_error(val_logits, val_targets, temperature=temperature, bins=15)
            record_metric("validation_ece_temperature_scaled", ece_scaled, unit="ece")
            record_metric("temperature_parameter", temperature, unit="scalar")

            # GPU memory
            if device.type == "cuda":
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1e6
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1e6
                record_metric("gpu_memory_allocated_mb", gpu_mem_allocated, unit="MB")
                record_metric("gpu_memory_reserved_mb", gpu_mem_reserved, unit="MB")

            # Log epoch summary
            logger.info(
                f"Epoch {epoch:03d}: "
                f"train_macro_f1={train_metrics.macro_f1:.4f} "
                f"validation_macro_f1={validation_metrics.macro_f1:.4f} "
                f"ece={ece:.4f} "
                f"temp={temperature:.4f} "
                f"duration={epoch_duration:.1f}s"
            )

            record = {
                "epoch": epoch,
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
                "calibration": {
                    "ece": ece,
                    "ece_temperature_scaled": ece_scaled,
                    "temperature": temperature,
                },
                "epoch_duration_seconds": epoch_duration,
            }
            write_json_line(metrics_file, record)

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
                logger.info(f"New best validation macro F1: {best_f1:.4f} at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            # Alert: validation macro F1 drop > 0.03
            if epoch > 1:
                prev_val_f1 = record["validation"]["macro_f1"]
                if best_f1 - validation_metrics.macro_f1 > 0.03:
                    logger.warning(
                        f"Validation macro F1 dropped by {(best_f1 - validation_metrics.macro_f1):.4f} "
                        f"from best ({best_f1:.4f})"
                    )

            if epochs_without_improvement >= training["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}; best validation macro F1 was {best_f1:.4f} at epoch {best_epoch}.")
                record_metric("early_stopping_triggered", 1, unit="bool")
                break

        # Final summary
        summary = {
            "status": "completed_train_validation_only",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "test_split_loaded": False,
            "checkpoint": "best_validation_macro_f1.pt",
            "total_epochs": epoch,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        logger.info("Training completed", extra=summary)
        record_metric("training_completed", 1, unit="bool")
        record_metric("best_epoch", best_epoch, unit="epoch")
        record_metric("best_validation_macro_f1", best_f1, unit="f1_score")

        collector.flush()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import logging
    main()