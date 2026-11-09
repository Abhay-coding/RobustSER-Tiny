from pathlib import Path
from collections import Counter
import csv
import wave
import hashlib

root = Path("dataset/AudioWAV")
output = Path("data/manifests/crema_d_manifest.csv")
output.parent.mkdir(parents=True, exist_ok=True)

labels = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}

rows = []
hashes = Counter()

files = sorted(root.glob("*.wav"))

if not files:
    raise SystemExit("No WAV files found. Check the dataset path.")

for p in files:
    parts = p.stem.split("_")
    valid_name = len(parts) == 4 and parts[2] in labels

    speaker = parts[0] if valid_name else ""
    emotion = labels.get(parts[2], "") if valid_name else ""
    sentence = parts[1] if valid_name else ""
    intensity = parts[3] if valid_name else ""

    row = {
        "filename": p.name,
        "path": p.as_posix(),
        "speaker_id": speaker,
        "emotion": emotion,
        "sentence": sentence,
        "intensity": intensity,
        "duration_s": 0,
        "sample_rate": 0,
        "channels": 0,
        "bit_depth": 0,
        "file_bytes": p.stat().st_size,
        "sha256": "",
        "valid": False,
        "issue": ""
    }

    issues = []
    if not valid_name:
        issues.append("Invalid filename")

    try:
        with wave.open(str(p), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            channels = w.getnchannels()
            width = w.getsampwidth()
            audio = w.readframes(frames)

            row["sample_rate"] = rate
            row["duration_s"] = round(frames / rate, 3) if rate else 0
            row["channels"] = channels
            row["bit_depth"] = width * 8

            if len(audio) != frames * channels * width:
                issues.append("Truncated audio")

            if frames == 0 or rate == 0:
                issues.append("Empty or invalid audio")

        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        row["sha256"] = digest
        hashes[digest] += 1

    except Exception as e:
        issues.append(str(e))

    row["valid"] = not issues
    row["issue"] = "; ".join(issues)
    rows.append(row)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Total files:", len(rows))
print("Valid files:", sum(r["valid"] for r in rows))
print("Invalid files:", sum(not r["valid"] for r in rows))
print("Unique speakers:", len({r["speaker_id"] for r in rows if r["speaker_id"]}))
print("Sample rates:", dict(Counter(r["sample_rate"] for r in rows)))
print("Channels:", dict(Counter(r["channels"] for r in rows)))
print("Durations:", round(sum(r["duration_s"] for r in rows), 2), "seconds")
print("Exact duplicate files:", sum(n - 1 for n in hashes.values() if n > 1))
print("Manifest saved:", output)

for r in rows:
    if not r["valid"]:
        print("INVALID:", r["filename"], r["issue"])