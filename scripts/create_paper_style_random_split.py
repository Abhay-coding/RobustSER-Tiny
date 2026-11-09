"""Create a clip-random 80/20 split for a clearly labelled paper-style comparison.

This is deliberately not a speaker-independent split and must never replace
the frozen RobustSER E5 benchmark. It is not an exact paper reproduction: the
paper uses modified TESS and adds a custom Noise class, while this comparison
uses the already audited CREMA-D five-emotion subset.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/crema_d_eligible_modeling_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/crema_d_paper_style_random_80_20.csv"))
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["emotion"] in LABELS and row["valid"] == "True"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["emotion"]].append(row)
    if set(grouped) != set(LABELS):
        raise SystemExit("Input manifest does not contain every paper-style label")

    rng = np.random.default_rng(args.seed)
    output_rows: list[dict[str, str]] = []
    for label in LABELS:
        class_rows = grouped[label]
        permutation = rng.permutation(len(class_rows))
        validation_count = max(1, round(0.20 * len(class_rows)))
        validation_indices = set(permutation[:validation_count])
        for index, row in enumerate(class_rows):
            output_rows.append({
                "path": row["path"], "speaker_id": row["speaker_id"], "emotion": row["emotion"],
                "split": "validation" if index in validation_indices else "train", "sha256": row["sha256"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "speaker_id", "emotion", "split", "sha256"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}; this is a clip-random paper-style comparison, not the robust benchmark.")


if __name__ == "__main__":
    main()
