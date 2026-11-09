"""Create traceable modeling manifests from the immutable CREMA-D audit.

The raw audit manifest remains the source of truth.  This utility identifies
every exact-hash duplicate group, records its handling, and excludes only
label-conflicting groups from the eligible modeling manifest.  It never edits
audio or the source manifest.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"filename", "speaker_id", "emotion", "sha256", "valid"}


def is_valid(row: dict[str, str]) -> bool:
    return row["valid"].strip().lower() == "true"


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        return list(reader), reader.fieldnames


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = row["sha256"].strip()
        if digest:
            groups[digest].append(row)

    conflicts: list[dict[str, str]] = []
    conflicting_hashes: set[str] = set()
    for digest, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        labels = {record["emotion"].strip() for record in group}
        speakers = {record["speaker_id"].strip() for record in group}
        if len(labels) > 1:
            disposition = "exclude_from_modeling_pending_provenance_review"
            reason = "exact_audio_label_conflict"
            conflicting_hashes.add(digest)
        elif len(speakers) > 1:
            disposition = "exclude_from_modeling_pending_provenance_review"
            reason = "exact_audio_speaker_metadata_conflict"
            conflicting_hashes.add(digest)
        else:
            disposition = "retain_as_group_for_split_leakage_prevention"
            reason = "exact_audio_duplicate_same_metadata"

        for record in group:
            conflicts.append(
                {
                    "sha256": digest,
                    "duplicate_group_size": str(len(group)),
                    "filename": record["filename"],
                    "speaker_id": record["speaker_id"],
                    "emotion": record["emotion"],
                    "reason": reason,
                    "disposition": disposition,
                }
            )

    eligible = [
        row
        for row in rows
        if is_valid(row) and row["sha256"].strip() not in conflicting_hashes
    ]
    return conflicts, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/manifests/crema_d_manifest.csv"),
        help="Immutable structural-audit manifest.",
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=Path("data/manifests/crema_d_duplicate_conflicts.csv"),
    )
    parser.add_argument(
        "--eligible-output",
        type=Path,
        default=Path("data/manifests/crema_d_eligible_modeling_manifest.csv"),
    )
    args = parser.parse_args()

    rows, source_fields = load_rows(args.input)
    conflicts, eligible = build_manifests(rows)
    conflict_fields = [
        "sha256", "duplicate_group_size", "filename", "speaker_id", "emotion", "reason", "disposition"
    ]
    write_csv(args.conflicts_output, conflict_fields, conflicts)
    write_csv(args.eligible_output, source_fields, eligible)

    excluded = len(rows) - len(eligible)
    label_conflict_groups = len({row["sha256"] for row in conflicts if row["reason"] == "exact_audio_label_conflict"})
    print(f"Source rows: {len(rows)}")
    print(f"Duplicate-record rows reported: {len(conflicts)}")
    print(f"Exact-audio label-conflict groups: {label_conflict_groups}")
    print(f"Excluded from eligible modeling manifest: {excluded}")
    print(f"Eligible modeling rows: {len(eligible)}")
    print(f"Conflict report saved: {args.conflicts_output}")
    print(f"Eligible manifest saved: {args.eligible_output}")


if __name__ == "__main__":
    main()
