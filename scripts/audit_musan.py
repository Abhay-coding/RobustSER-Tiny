"""Create an immutable structural audit manifest for a local MUSAN corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import wave
from collections import Counter
from pathlib import Path


def category_for(path: Path, root: Path) -> str:
    relative_parts = path.relative_to(root).parts
    return relative_parts[1] if len(relative_parts) > 1 and relative_parts[0] == "musan" else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/MUSAN"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/musan_manifest.csv"))
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Persist progress after this many files so interrupted audits can resume.")
    args = parser.parse_args()
    files = sorted(args.root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found beneath {args.root}")

    fields = ["relative_path", "category", "duration_s", "sample_rate", "channels", "bit_depth", "file_bytes", "sha256", "valid", "issue"]
    cached: dict[str, dict[str, str]] = {}
    if args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as handle:
            cached = {row["relative_path"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle = args.output.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for path in files:
        relative_path = path.as_posix()
        if relative_path in cached:
            row = cached[relative_path]
            rows.append(row)
            writer.writerow(row)
            continue
        issue = ""
        sample_rate = channels = bit_depth = 0
        duration_s = 0.0
        try:
            with wave.open(str(path), "rb") as source:
                sample_rate = source.getframerate()
                channels = source.getnchannels()
                bit_depth = source.getsampwidth() * 8
                duration_s = source.getnframes() / sample_rate if sample_rate else 0.0
                if not source.getnframes() or not sample_rate:
                    issue = "empty_or_invalid_audio"
        except (OSError, wave.Error) as error:
            issue = f"wave_read_error: {error}"
        row = {
            "relative_path": relative_path,
            "category": category_for(path, args.root),
            "duration_s": f"{duration_s:.6f}",
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_depth": bit_depth,
            "file_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "valid": not issue,
            "issue": issue,
        }
        rows.append(row)
        writer.writerow(row)
        if len(rows) % args.checkpoint_every == 0:
            handle.flush()
    handle.flush()
    handle.close()
    valid = [row for row in rows if str(row["valid"]).lower() == "true"]
    duplicates = Counter(str(row["sha256"]) for row in valid)
    print(f"Total files: {len(rows)}")
    print(f"Valid files: {len(valid)}")
    print(f"Invalid files: {len(rows) - len(valid)}")
    print(f"Categories: {dict(Counter(str(row['category']) for row in rows))}")
    print(f"Sample rates: {dict(Counter(int(row['sample_rate']) for row in valid))}")
    print(f"Exact duplicate occurrences: {sum(count - 1 for count in duplicates.values() if count > 1)}")
    print(f"Manifest saved: {args.output}")


if __name__ == "__main__":
    main()
