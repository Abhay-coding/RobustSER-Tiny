"""
Perform one safeguarded held-out test evaluation for a completed run.
Enhanced with structured logging, metrics collection, and tracing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from robustser.evaluation.calibration import expected_calibration_error, fit_temperature, softmax_confidence
from robustser.observability import (
    setup_structured_logging,
    get_logger,
    LogContext,
    CorrelationContext,
    get_metrics_collector,
    record_metric,
    record_histogram,
    trace_span,
)


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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

    # Setup structured logging
    log_file = args.run_dir / "evaluation.jsonl"
    logger = setup_structured_logging(
        level=getattr(__import__("logging"), args.log_level),
        log_file=log_file,
        json_output=True,
    )
    logger = get_logger("robustser.evaluate")

    # Setup metrics collector
    metrics_file = args.run_dir / "metrics.jsonl"
    collector = get_metrics_collector()
    collector._jsonl_path = metrics_file

    run_id = args.run_dir.name
    corr_context = LogContext(
        experiment_id=config["experiment"]["id"],
        run_id=run_id,
        seed=config["experiment"]["seed"],
        split="test",
    )

    logger.info("Starting held-out test evaluation", extra={
        "run_dir": str(args.run_dir),
        "checkpoint": checkpoint_path.name,
    })

    with CorrelationContext(corr_context):
        device = resolve_device(args.device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("task") != task or tuple(checkpoint.get("labels", ())) != TASK_LABELS[task]:
            raise SystemExit("Checkpoint task/label mapping does not match the frozen run configuration")

        model = build_model(config["model"]["name"], num_classes=len(TASK_LABELS[task])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_data = CREMADSERDataset(
            config["data"]["manifest"], split="test", task=task,
            seed=config["experiment"]["seed"],
            feature_type=config["features"]["type"],
            window_seconds=config["data"]["train_window_seconds"],
        )
        test_loader = DataLoader(
            test_data, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
        )

        logger.info(f"Test dataset size: {len(test_data)}")

        with trace_span("test_evaluation", baggage={"split": "test"}) as span:
            metrics = evaluate(
                model=model, batches=test_loader,
                loss_function=nn.CrossEntropyLoss(), device=device,
                num_classes=len(TASK_LABELS[task]),
            )
            span.set_attribute("test_loss", metrics.loss)
            span.set_attribute("test_macro_f1", metrics.macro_f1)

        # Get all logits for calibration analysis
        test_logits_list = []
        test_targets_list = []
        test_predictions_list = []
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                test_logits_list.append(logits.cpu())
                test_targets_list.append(labels.cpu())
                test_predictions_list.append(logits.argmax(dim=1).cpu())

        test_logits = torch.cat(test_logits_list)
        test_targets = torch.cat(test_targets_list)
        test_predictions = torch.cat(test_predictions_list)

        # Calibration metrics
        ece = expected_calibration_error(test_logits, test_targets, temperature=1.0, bins=15)
        temperature = fit_temperature(test_logits, test_targets)
        ece_scaled = expected_calibration_error(test_logits, test_targets, temperature=temperature, bins=15)

        # Confidence and margin distributions
        probs = torch.softmax(test_logits / temperature, dim=1)
        confidence, predictions = probs.max(dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Record all metrics
        record_metric("test_loss", metrics.loss, unit="cross_entropy")
        record_metric("test_macro_f1", metrics.macro_f1, unit="f1_score")
        record_metric("test_accuracy", metrics.accuracy, unit="accuracy")
        record_metric("test_balanced_accuracy", metrics.balanced_accuracy, unit="accuracy")
        record_metric("test_uar", metrics.uar, unit="accuracy")
        record_metric("test_ece", ece, unit="ece")
        record_metric("test_ece_temperature_scaled", ece_scaled, unit="ece")
        record_metric("temperature_parameter", temperature, unit="scalar")

        # Confidence/margin histograms
        record_histogram("test_confidence", confidence.tolist(), unit="probability")
        record_histogram("test_margin", margin.tolist(), unit="probability")

        # OOD detection rate
        ood_conf = (confidence < 0.5).float().mean().item()
        ood_margin = (margin < 0.1).float().mean().item()
        record_metric("test_ood_rate_confidence", ood_conf, unit="rate")
        record_metric("test_ood_rate_margin", ood_margin, unit="rate")

        # Per-class metrics
        for i, (prec, rec, f1) in enumerate(zip(
            metrics.per_class_precision,
            metrics.per_class_recall,
            metrics.per_class_f1,
        )):
            record_metric("test_per_class_precision", prec, labels={"class": TASK_LABELS[task][i]}, unit="precision")
            record_metric("test_per_class_recall", rec, labels={"class": TASK_LABELS[task][i]}, unit="recall")
            record_metric("test_per_class_f1", f1, labels={"class": TASK_LABELS[task][i]}, unit="f1_score")

        # Confusion matrix
        for i, row in enumerate(metrics.confusion_matrix):
            for j, count in enumerate(row):
                record_metric("test_confusion_matrix", count,
                            labels={"true_class": TASK_LABELS[task][i], "predicted_class": TASK_LABELS[task][j]},
                            unit="count", metric_type="counter")

        # Alert checks
        if metrics.macro_f1 < 0.50:
            logger.warning(f"Test macro F1 ({metrics.macro_f1:.4f}) below threshold (0.50)")

        if ece > 0.10:
            logger.warning(f"Test ECE ({ece:.4f}) exceeds 0.10 - model poorly calibrated")

        if ood_conf > 0.20:
            logger.warning(f"OOD rate by confidence ({ood_conf:.1%}) exceeds 20%")

        result = {
            "status": "completed_held_out_test_once",
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": checkpoint["epoch"],
            "task": task,
            "labels": list(TASK_LABELS[task]),
            "test_records": len(test_data),
            "device": str(device),
            "metrics": metrics.to_dict(),
            "calibration": {
                "ece": ece,
                "ece_temperature_scaled": ece_scaled,
                "temperature": temperature,
            },
            "confidence_stats": {
                "mean": float(confidence.mean()),
                "median": float(confidence.median()),
                "p10": float(torch.quantile(confidence, 0.1)),
                "p90": float(torch.quantile(confidence, 0.9)),
            },
            "margin_stats": {
                "mean": float(margin.mean()),
                "median": float(margin.median()),
                "p10": float(torch.quantile(margin, 0.1)),
                "p90": float(torch.quantile(margin, 0.9)),
            },
            "ood_rates": {
                "confidence_lt_05": ood_conf,
                "margin_lt_01": ood_margin,
            },
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        logger.info("Held-out test evaluation completed", extra={
            "test_macro_f1": metrics.macro_f1,
            "test_ece": ece,
            "temperature": temperature,
        })

        record_metric("evaluation_completed", 1, unit="bool")

        collector.flush()

        print(json.dumps(result, indent=2))
        print(f"Held-out test results saved: {result_path}")


if __name__ == "__main__":
    main()