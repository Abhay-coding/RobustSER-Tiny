"""Create hash-disjoint MUSAN augmentation/evaluation noise partitions."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/musan_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/musan_noise_splits.csv"))
    parser.add_argument("--seed", type=int, default=20_261_109)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid_rows = [row for row in rows if row["valid"].lower() == "true" and row["category"] in {"music", "noise", "speech"}]
    if not valid_rows:
        raise SystemExit("No valid MUSAN music/noise/speech rows found")
    hashes_by_category: dict[str, set[str]] = defaultdict(set)
    for row in valid_rows:
        hashes_by_category[row["category"]].add(row["sha256"])
    split_by_hash: dict[str, str] = {}
    for category, hashes in sorted(hashes_by_category.items()):
        ordered = sorted(hashes)
        random.Random(args.seed).shuffle(ordered)
        train_count = round(len(ordered) * args.train_ratio)
        # Keep both partitions non-empty whenever the category has >= 2 files.
        train_count = min(max(train_count, 1), len(ordered) - 1) if len(ordered) >= 2 else len(ordered)
        split_by_hash.update({digest: "augmentation_train" for digest in ordered[:train_count]})
        split_by_hash.update({digest: "evaluation_holdout" for digest in ordered[train_count:]})
    output_rows = [{**row, "noise_split": split_by_hash.get(row["sha256"], "excluded_invalid")} for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0].keys(), "noise_split"])
        writer.writeheader()
        writer.writerows(output_rows)
    overlap = defaultdict(set)
    for row in output_rows:
        if row["noise_split"] != "excluded_invalid":
            overlap[row["sha256"]].add(row["noise_split"])
    if any(len(splits) > 1 for splits in overlap.values()):
        raise RuntimeError("Exact noise audio crossed partition boundaries")
    counts = defaultdict(int)
    for row in output_rows:
        counts[(row["category"], row["noise_split"])] += 1
    print(f"Seed: {args.seed}")
    print(f"Noise split counts: {dict(sorted(counts.items()))}")
    print("Exact hash overlap: 0")
    print(f"Split manifest saved: {args.output}")


if __name__ == "__main__":
    main()
