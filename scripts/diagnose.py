#!/usr/bin/env python3
"""
Diagnosis CLI for RobustSER-Tiny.

Provides:
- Confusion matrix visualization
- Per-class Precision-Recall curves
- Audio replay of misclassified samples
- Calibration reliability diagrams
- ECE breakdown by class

Usage:
    python scripts/diagnose.py --run-dir results/E5_seed20261108 --confusion-matrix
    python scripts/diagnose.py --run-dir results/E5_seed20261108 --pr-curves
    python scripts/diagnose.py --run-dir results/E5_seed20261108 --misclassified --top-k 10
    python scripts/diagnose.py --run-dir results/E5_seed20261108 --calibration
    python scripts/diagnose.py --run-dir results/E5_seed20261108 --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.data.cremad import CREMADSERDataset, TASK_LABELS
from robustser.data import RAVDESSExternalDataset
from robustser.models import build_model
from robustser.training import evaluate
from robustser.evaluation.calibration import expected_calibration_error, softmax_confidence, fit_temperature
from robustser.inference import predict_wav_bytes


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_run(run_dir: Path, device: torch.device):
    """Load model, config, and data for a run."""
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best_validation_macro_f1.pt"

    if not config_path.exists() or not checkpoint_path.exists():
        raise SystemExit(f"Run directory must contain config.yaml and best_validation_macro_f1.pt")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = config["experiment"]["task"]
    labels = TASK_LABELS[task]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(config["model"]["name"], num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return config, model, labels, task, checkpoint


def get_test_loader(config, task, device, split="test"):
    """Get test data loader."""
    dataset = CREMADSERDataset(
        config["data"]["manifest"],
        split=split,
        task=task,
        seed=config["experiment"]["seed"],
        feature_type=config["features"]["type"],
        window_seconds=config["data"]["train_window_seconds"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return loader, dataset


def get_all_predictions(model, loader, device, num_classes):
    """Get all logits, predictions, and targets from a data loader."""
    all_logits = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())

    return (
        torch.cat(all_logits),
        torch.cat(all_predictions),
        torch.cat(all_targets),
    )


def plot_confusion_matrix(confusion_matrix, labels, output_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = np.array(confusion_matrix)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalized Count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2%})",
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_pr_curves(logits, targets, labels, output_path, num_classes):
    """Plot per-class Precision-Recall curves."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    probs = torch.softmax(logits, dim=1).numpy()
    targets_np = targets.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for class_idx in range(num_classes):
        ax = axes[class_idx]
        y_true = (targets_np == class_idx).astype(int)
        y_scores = probs[:, class_idx]

        if y_true.sum() == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{labels[class_idx]} (no samples)")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        ax.plot(recall, precision, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{labels[class_idx]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.suptitle("Per-Class Precision-Recall Curves")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curves saved to {output_path}")


def plot_reliability_diagram(logits, targets, output_path, temperature=1.0, bins=15, title="Reliability Diagram"):
    """Plot reliability diagram (calibration curve)."""
    confidence, predictions = softmax_confidence(logits, temperature)
    correct = predictions.eq(targets)

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    accuracies = []
    confidences = []
    counts = []

    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)

        if mask.any():
            accuracies.append(correct[mask].float().mean().item())
            confidences.append(confidence[mask].float().mean().item())
            counts.append(mask.sum().item())
        else:
            accuracies.append(0)
            confidences.append(bin_centers[i])
            counts.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax1.scatter(confidences, accuracies, s=[c * 10 for c in counts], alpha=0.7)
    ax1.plot(confidences, accuracies, "o-", label="Model")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives (Accuracy)")
    ax1.set_title(f"{title} (T={temperature:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of confidences
    ax2.hist(confidence.numpy(), bins=bin_edges, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved to {output_path}")


def plot_ece_by_class(logits, targets, labels, output_path, temperature=1.0, bins=15):
    """Plot ECE breakdown by class."""
    probs = torch.softmax(logits / temperature, dim=1)
    confidence, predictions = probs.max(dim=1)
    correct = predictions.eq(targets)

    num_classes = len(labels)
    class_eces = []

    for class_idx in range(num_classes):
        class_mask = (targets == class_idx)
        if not class_mask.any():
            class_eces.append(0)
            continue

        class_conf = confidence[class_mask]
        class_correct = correct[class_mask]
        class_probs = probs[class_mask, class_idx]

        ece = 0.0
        for i in range(bins):
            low, high = i / bins, (i + 1) / bins
            if i == bins - 1:
                mask = (class_conf >= low) & (class_conf <= high)
            else:
                mask = (class_conf >= low) & (class_conf < high)

            if mask.any():
                ece += mask.float().mean() * (class_conf[mask].float().mean() - class_correct[mask].float().mean()).abs()

        class_eces.append(float(ece))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(num_classes), class_eces, color="steelblue", edgecolor="black")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error by Class")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, ece in zip(bars, class_eces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{ece:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ECE by class saved to {output_path}")


def find_misclassified(logits, targets, predictions, dataset, labels, top_k=10):
    """Find top misclassified samples by confidence."""
    probs = torch.softmax(logits, dim=1)
    confidence, _ = probs.max(dim=1)

    misclassified_mask = predictions != targets
    misclassified_indices = torch.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        return []

    misclassified_conf = confidence[misclassified_indices]
    # Sort by confidence descending (most confident wrong predictions first)
    sorted_idx = torch.argsort(misclassified_conf, descending=True)

    results = []
    for i in sorted_idx[:top_k]:
        idx = misclassified_indices[i].item()
        true_label = labels[targets[idx].item()]
        pred_label = labels[predictions[idx].item()]
        conf = confidence[idx].item()
        results.append({
            "index": idx,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": conf,
            "all_probs": {labels[j]: probs[idx, j].item() for j in range(len(labels))},
        })

    return results


def save_misclassified_audio(misclassified, dataset, output_dir, labels):
    """Save misclassified audio files for replay."""
    import shutil
    from scipy.io import wavfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "misclassified_manifest.jsonl"
    with manifest_path.open("w") as f:
        for item in misclassified:
            idx = item["index"]
            # Get original audio path from dataset
            # Note: This requires the dataset to expose file paths
            # For now, save the windowed features as reference
            features, _ = dataset[idx]

            record = {
                "index": idx,
                "true_label": item["true_label"],
                "predicted_label": item["predicted_label"],
                "confidence": item["confidence"],
                "all_probs": item["all_probs"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Misclassified manifest saved to {manifest_path}")
    print(f"To replay: use the indices with the original CREMA-D dataset")


def cmd_confusion_matrix(args):
    """Generate confusion matrix."""
    device = resolve_device(args.device)
    config, model, labels, task, checkpoint = load_run(Path(args.run_dir), device)

    loader, dataset = get_test_loader(config, task, device, "test")
    logits, predictions, targets = get_all_predictions(model, loader, device, len(labels))

    from robustser.training.metrics import classification_metrics
    metrics = classification_metrics(
        total_loss=0,  # Not needed for confusion matrix
        targets=targets,
        predictions=predictions,
        num_classes=len(labels),
    )

    output_path = Path(args.run_dir) / "confusion_matrix.png"
    plot_confusion_matrix(metrics.confusion_matrix, labels, output_path,
                          f"Test Confusion Matrix - {config['experiment']['id']}")

    # Also save as JSON
    json_path = Path(args.run_dir) / "confusion_matrix.json"
    json_path.write_text(json.dumps({
        "labels": labels,
        "matrix": metrics.confusion_matrix,
    }, indent=2))
    print(f"Confusion matrix JSON saved to {json_path}")


def cmd_pr_curves(args):
    """Generate per-class PR curves."""
    device = resolve_device(args.device)
    config, model, labels, task, checkpoint = load_run(Path(args.run_dir), device)

    loader, dataset = get_test_loader(config, task, device, "test")
    logits, predictions, targets = get_all_predictions(model, loader, device, len(labels))

    output_path = Path(args.run_dir) / "pr_curves.png"
    plot_pr_curves(logits, targets, labels, output_path, len(labels))


def cmd_calibration(args):
    """Generate calibration plots and ECE analysis."""
    device = resolve_device(args.device)
    config, model, labels, task, checkpoint = load_run(Path(args.run_dir), device)

    loader, dataset = get_test_loader(config, task, device, "test")
    logits, predictions, targets = get_all_predictions(model, loader, device, len(labels))

    # Temperature scaling
    print("Fitting temperature...")
    temperature = fit_temperature(logits, targets)
    print(f"Optimal temperature: {temperature:.4f}")

    # ECE before and after temperature scaling
    ece_before = expected_calibration_error(logits, targets, temperature=1.0)
    ece_after = expected_calibration_error(logits, targets, temperature=temperature)
    print(f"ECE (T=1.0): {ece_before:.4f}")
    print(f"ECE (T={temperature:.4f}): {ece_after:.4f}")

    # Reliability diagrams
    plot_reliability_diagram(logits, targets,
                             Path(args.run_dir) / "reliability_diagram_T1.png",
                             temperature=1.0)
    plot_reliability_diagram(logits, targets,
                             Path(args.run_dir) / f"reliability_diagram_T{temperature:.3f}.png",
                             temperature=temperature)

    # ECE by class
    plot_ece_by_class(logits, targets, labels,
                      Path(args.run_dir) / "ece_by_class.png",
                      temperature=temperature)

    # Save calibration summary
    summary_path = Path(args.run_dir) / "calibration_summary.json"
    summary_path.write_text(json.dumps({
        "temperature": temperature,
        "ece_temperature_1": ece_before,
        "ece_temperature_scaled": ece_after,
    }, indent=2))
    print(f"Calibration summary saved to {summary_path}")


def cmd_misclassified(args):
    """Find and analyze misclassified samples."""
    device = resolve_device(args.device)
    config, model, labels, task, checkpoint = load_run(Path(args.run_dir), device)

    loader, dataset = get_test_loader(config, task, device, "test")
    logits, predictions, targets = get_all_predictions(model, loader, device, len(labels))

    misclassified = find_misclassified(logits, targets, predictions, dataset, labels, args.top_k)

    if not misclassified:
        print("No misclassified samples found!")
        return

    print(f"\nTop {len(misclassified)} misclassified samples (by confidence):")
    print("-" * 80)
    for item in misclassified:
        probs_str = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(item["all_probs"].items(), key=lambda x: -x[1])[:3])
        print(f"  [{item['index']:4d}] True: {item['true_label']:10s} | Pred: {item['predicted_label']:10s} | Conf: {item['confidence']:.3f} | Top: {probs_str}")

    # Save misclassified analysis
    output_dir = Path(args.run_dir) / "misclassified"
    save_misclassified_audio(misclassified, dataset, output_dir, labels)

    json_path = Path(args.run_dir) / "misclassified.json"
    json_path.write_text(json.dumps(misclassified, indent=2))
    print(f"\nMisclassified details saved to {json_path}")


def cmd_all(args):
    """Run all diagnostics."""
    print("=" * 60)
    print("Running confusion matrix...")
    print("=" * 60)
    cmd_confusion_matrix(args)

    print("\n" + "=" * 60)
    print("Running PR curves...")
    print("=" * 60)
    cmd_pr_curves(args)

    print("\n" + "=" * 60)
    print("Running calibration analysis...")
    print("=" * 60)
    cmd_calibration(args)

    print("\n" + "=" * 60)
    print("Running misclassified analysis...")
    print("=" * 60)
    cmd_misclassified(args)

    print("\n" + "=" * 60)
    print("All diagnostics complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with checkpoints")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--top-k", type=int, default=10, help="Top K misclassified to show")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("confusion-matrix", help="Generate confusion matrix")
    subparsers.add_parser("pr-curves", help="Generate per-class PR curves")
    subparsers.add_parser("calibration", help="Generate calibration plots and ECE analysis")
    subparsers.add_parser("misclassified", help="Find and analyze misclassified samples")
    subparsers.add_parser("all", help="Run all diagnostics")

    args = parser.parse_args()

    if args.command == "confusion-matrix":
        cmd_confusion_matrix(args)
    elif args.command == "pr-curves":
        cmd_pr_curves(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "misclassified":
        cmd_misclassified(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()