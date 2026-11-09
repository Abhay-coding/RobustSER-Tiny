"""Aggregate completed, pre-registered runs without selecting between them."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


# Two-sided 95% Student-t critical values for 1..10 degrees of freedom.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def aggregate(values: list[float]) -> dict[str, float | int | None]:
    count = len(values)
    if not count:
        raise ValueError("No completed values to aggregate")
    average = mean(values)
    if count == 1:
        return {"n": 1, "mean": average, "sample_std": None, "ci95_half_width": None}
    deviation = stdev(values)
    critical = T_975.get(count - 1, 1.96)
    return {"n": count, "mean": average, "sample_std": deviation, "ci95_half_width": critical * deviation / math.sqrt(count)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/e1_replication_summary.json"))
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    labels: tuple[str, ...] | None = None
    for run_dir in args.run_dirs:
        train_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        test_path = run_dir / "test_metrics.json"
        if not test_path.is_file():
            raise SystemExit(f"Missing one-time held-out test result: {test_path}")
        test = json.loads(test_path.read_text(encoding="utf-8"))
        run_labels = tuple(test["labels"])
        if labels is None:
            labels = run_labels
        elif run_labels != labels:
            raise SystemExit("Cannot aggregate runs with different task label mappings")
        runs.append({
            "run_dir": run_dir.as_posix(),
            "best_epoch": train_summary["best_epoch"],
            "validation_macro_f1": train_summary["best_validation_macro_f1"],
            "test_macro_f1": test["metrics"]["macro_f1"],
            "test_balanced_accuracy": test["metrics"]["balanced_accuracy"],
            "test_per_class_f1": test["metrics"]["per_class_f1"],
            "test_per_class_recall": test["metrics"]["per_class_recall"],
        })
    assert labels is not None
    per_class_f1 = {
        label: aggregate([float(run["test_per_class_f1"][index]) for run in runs])
        for index, label in enumerate(labels)
    }
    per_class_recall = {
        label: aggregate([float(run["test_per_class_recall"][index]) for run in runs])
        for index, label in enumerate(labels)
    }
    result = {
        "runs": runs,
        "validation_macro_f1": aggregate([float(run["validation_macro_f1"]) for run in runs]),
        "test_macro_f1": aggregate([float(run["test_macro_f1"]) for run in runs]),
        "test_balanced_accuracy": aggregate([float(run["test_balanced_accuracy"]) for run in runs]),
        "validation_to_test_macro_f1_gap": aggregate([
            float(run["validation_macro_f1"]) - float(run["test_macro_f1"]) for run in runs
        ]),
        "test_per_class_f1": per_class_f1,
        "test_per_class_recall": per_class_recall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
