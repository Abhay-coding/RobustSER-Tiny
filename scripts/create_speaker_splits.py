"""Create a deterministic, speaker-disjoint CREMA-D train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = 20_261_108
DEFAULT_RESTARTS = 1


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        required = {"filename", "speaker_id", "emotion", "sha256", "path"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        rows = list(reader)
        return rows, reader.fieldnames


def speaker_target_counts(speaker_count: int) -> dict[str, int]:
    raw = {split: speaker_count * RATIOS[split] for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    for split in sorted(SPLITS, key=lambda name: raw[name] - counts[name], reverse=True)[: speaker_count - sum(counts.values())]:
        counts[split] += 1
    return counts


def score_assignment(assignments: dict[str, str], speaker_labels: dict[str, Counter[str]], labels: list[str]) -> float:
    total = sum(sum(counts.values()) for counts in speaker_labels.values())
    label_totals = Counter()
    split_totals = Counter()
    split_labels = {split: Counter() for split in SPLITS}
    for speaker, split in assignments.items():
        counts = speaker_labels[speaker]
        split_labels[split].update(counts)
        split_totals[split] += sum(counts.values())
        label_totals.update(counts)

    score = 0.0
    for split in SPLITS:
        score += ((split_totals[split] - total * RATIOS[split]) / total) ** 2
        for label in labels:
            expected = label_totals[label] * RATIOS[split]
            if expected:
                score += ((split_labels[split][label] - expected) / expected) ** 2
    return score


def create_assignments(rows: list[dict[str, str]], seed: int, restarts: int = DEFAULT_RESTARTS) -> dict[str, str]:
    speaker_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        speaker_labels[row["speaker_id"]][row["emotion"]] += 1
    speakers = sorted(speaker_labels)
    labels = sorted({row["emotion"] for row in rows})
    targets = speaker_target_counts(len(speakers))

    # A deterministic swap search balances record and label counts while
    # preserving exact speaker counts in every split. More starts can be
    # requested for a changed corpus, but one reaches the optimum here.
    best: dict[str, str] | None = None
    best_score = float("inf")
    for restart in range(restarts):
        order = speakers[:]
        random.Random(seed + restart).shuffle(order)
        assignment: dict[str, str] = {}
        start = 0
        for split in SPLITS:
            for speaker in order[start : start + targets[split]]:
                assignment[speaker] = split
            start += targets[split]

        improved = True
        while improved:
            improved = False
            current_score = score_assignment(assignment, speaker_labels, labels)
            best_swap: tuple[str, str] | None = None
            for left_index, left in enumerate(speakers):
                for right in speakers[left_index + 1 :]:
                    if assignment[left] == assignment[right]:
                        continue
                    assignment[left], assignment[right] = assignment[right], assignment[left]
                    candidate_score = score_assignment(assignment, speaker_labels, labels)
                    assignment[left], assignment[right] = assignment[right], assignment[left]
                    if candidate_score + 1e-15 < current_score:
                        current_score = candidate_score
                        best_swap = (left, right)
            if best_swap:
                left, right = best_swap
                assignment[left], assignment[right] = assignment[right], assignment[left]
                improved = True

        candidate_score = score_assignment(assignment, speaker_labels, labels)
        if candidate_score < best_score:
            best, best_score = assignment.copy(), candidate_score

    assert best is not None
    return best


def validate(rows: list[dict[str, str]], *, check_paths: bool = False) -> dict[str, object]:
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    if any(not by_split[split] for split in SPLITS):
        raise ValueError("At least one split is empty")
    speaker_sets = {split: {row["speaker_id"] for row in by_split[split]} for split in SPLITS}
    if any(speaker_sets[left] & speaker_sets[right] for left in SPLITS for right in SPLITS if left < right):
        raise ValueError("Speaker overlap detected")
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_splits[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Exact recording overlap detected")
    all_labels = {row["emotion"] for row in rows}
    missing = {split: sorted(all_labels - {row["emotion"] for row in by_split[split]}) for split in SPLITS}
    if any(missing.values()):
        raise ValueError(f"Missing labels in split(s): {missing}")
    if check_paths:
        missing_paths = [row["path"] for row in rows if not Path(row["path"]).is_file()]
        if missing_paths:
            raise ValueError(f"Manifest paths do not resolve ({len(missing_paths)} missing); first: {missing_paths[0]}")
    return {
        "records": {split: len(by_split[split]) for split in SPLITS},
        "speakers": {split: len(speaker_sets[split]) for split in SPLITS},
        "class_counts": {split: dict(sorted(Counter(row["emotion"] for row in by_split[split]).items())) for split in SPLITS},
        "labels": sorted(all_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/crema_d_eligible_modeling_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/crema_d_speaker_splits.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/manifests/crema_d_speaker_splits_summary.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS,
                        help="Deterministic random initializations before local swap optimization.")
    args = parser.parse_args()

    rows, fields = read_manifest(args.input)
    assignments = create_assignments(rows, args.seed, args.restarts)
    output_rows = [{**row, "split": assignments[row["speaker_id"]]} for row in rows]
    summary = validate(output_rows, check_paths=True)
    summary.update({"seed": args.seed, "restarts": args.restarts, "ratios": RATIOS, "source_manifest": args.input.as_posix()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "split"])
        writer.writeheader()
        writer.writerows(output_rows)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Split manifest saved: {args.output}")
    print(f"Split summary saved: {args.summary_output}")


if __name__ == "__main__":
    main()
