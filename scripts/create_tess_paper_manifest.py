"""Create a deterministic paper-style 80/20 random TESS split for five emotions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral")
FILENAME_LABELS = {label.lower(): label for label in LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/TESS"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/tess_paper_style_random_80_20.csv"))
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"TESS directory not found: {args.root}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_files: list[Path] = []
    for path in sorted(args.root.rglob("*.wav")):
        parts = path.stem.lower().split("_")
        label = FILENAME_LABELS.get(parts[-1])
        if label is None:
            continue
        try:
            with wave.open(str(path), "rb") as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    raise wave.Error("expected 16-bit mono PCM")
        except (wave.Error, EOFError):
            invalid_files.append(path)
            continue
        grouped[label].append({
            "path": str(path).replace("\\", "/"), "speaker_id": parts[0].upper(), "emotion": label,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if set(grouped) != set(LABELS):
        raise SystemExit(f"Expected {LABELS}; found {sorted(grouped)}")
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, str]] = []
    for label in LABELS:
        samples = grouped[label]
        validation_indices = set(rng.permutation(len(samples))[: round(len(samples) * 0.20)])
        for index, row in enumerate(samples):
            rows.append({**row, "split": "validation" if index in validation_indices else "train"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "speaker_id", "emotion", "split", "sha256"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} valid TESS rows to {args.output}: five emotions, random 80/20 split.")
    if invalid_files:
        print("Excluded invalid WAV containers: " + ", ".join(str(path) for path in invalid_files))


if __name__ == "__main__":
    main()
