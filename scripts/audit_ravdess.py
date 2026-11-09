"""Build an external RAVDESS speech-only manifest; never merge it with CREMA-D."""

from __future__ import annotations

import argparse, csv, hashlib, wave
from pathlib import Path

EMOTIONS = {"01": "Neutral", "03": "Happy", "04": "Sad", "05": "Angry", "06": "Fear", "07": "Disgust"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/RAVDESS"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/ravdess_speech_manifest.csv"))
    args = parser.parse_args()
    files = sorted(args.root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No RAVDESS WAV files found beneath {args.root}")
    rows = []
    for path in files:
        parts = path.stem.split("-")
        # Modality 03 is audio-only; retain only common six emotions.
        if len(parts) != 7 or parts[0] != "03" or parts[2] not in EMOTIONS:
            continue
        with wave.open(str(path), "rb") as source:
            rows.append({"path": path.as_posix(), "actor_id": parts[6], "emotion": EMOTIONS[parts[2]],
                         "sample_rate": source.getframerate(), "channels": source.getnchannels(),
                         "duration_s": round(source.getnframes()/source.getframerate(), 6),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"External RAVDESS records: {len(rows)}")
    print(f"Manifest saved: {args.output}")


if __name__ == "__main__":
    main()
