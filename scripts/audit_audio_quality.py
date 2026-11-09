"""Measure acoustic-quality indicators without modifying CREMA-D WAV files.

The audit is deliberately conservative: flagged records are candidates for
manual review, not automatic exclusions.  It reads the frozen eligible
manifest and writes one metrics row per record.
"""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np


METRIC_FIELDS = [
    "filename", "path", "speaker_id", "emotion", "duration_s", "sample_rate", "channels",
    "bit_depth", "rms_dbfs", "peak_dbfs", "zero_sample_fraction", "clipped_sample_fraction",
    "quality_flags", "read_error",
]


def decode_pcm(raw: bytes, sample_width: int) -> tuple[np.ndarray, float]:
    """Return signed PCM samples and the full-scale value for common WAV widths."""
    if sample_width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128, 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2"), 32768.0
    if sample_width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        signed[signed >= 1 << 23] -= 1 << 24
        return signed, float(1 << 23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4"), float(1 << 31)
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def dbfs(value: float) -> float:
    return -float("inf") if value == 0 else 20.0 * math.log10(value)


def measure_wav(path: Path) -> dict[str, float | int]:
    squared_sum = 0.0
    peak = 0.0
    zero_samples = 0
    clipped_samples = 0
    samples = 0

    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frames = source.getnframes()
        while raw := source.readframes(65_536):
            pcm, full_scale = decode_pcm(raw, sample_width)
            normalized = pcm.astype(np.float64) / full_scale
            squared_sum += float(np.dot(normalized, normalized))
            peak = max(peak, float(np.max(np.abs(normalized))))
            zero_samples += int(np.count_nonzero(pcm == 0))
            clipped_samples += int(np.count_nonzero(np.abs(pcm.astype(np.int64)) >= full_scale - 1))
            samples += pcm.size

    rms = math.sqrt(squared_sum / samples) if samples else 0.0
    return {
        "duration_s": frames / sample_rate if sample_rate else 0.0,
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "zero_sample_fraction": zero_samples / samples if samples else 1.0,
        "clipped_sample_fraction": clipped_samples / samples if samples else 0.0,
    }


def robust_outliers(values: list[float], z_limit: float = 5.0) -> tuple[float, float, set[int]]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale == 0:
        return median, mad, set()
    return median, mad, {index for index, value in enumerate(values) if abs(value - median) / scale > z_limit}


def apply_flags(records: list[dict[str, str]]) -> None:
    durations = [float(record["duration_s"]) for record in records if not record["read_error"]]
    rms_values = [float(record["rms_dbfs"]) for record in records if not record["read_error"]]
    _, _, duration_outlier_indexes = robust_outliers(durations)
    _, _, rms_outlier_indexes = robust_outliers(rms_values)
    valid_index = 0

    for record in records:
        flags: list[str] = []
        if record["read_error"]:
            flags.append("audio_read_error")
        else:
            duration = float(record["duration_s"])
            rms_dbfs = float(record["rms_dbfs"])
            peak_dbfs = float(record["peak_dbfs"])
            zero_fraction = float(record["zero_sample_fraction"])
            clipping_fraction = float(record["clipped_sample_fraction"])
            if duration < 0.5:
                flags.append("unexpectedly_short_under_0.5s")
            if duration > 8.0:
                flags.append("unexpectedly_long_over_8s")
            if rms_dbfs < -45.0:
                flags.append("near_silent_rms_below_minus_45_dbfs")
            if peak_dbfs < -30.0:
                flags.append("very_low_peak_below_minus_30_dbfs")
            if clipping_fraction >= 0.001:
                flags.append("potential_clipping_at_least_0.1_percent")
            if zero_fraction >= 0.95:
                flags.append("mostly_digital_silence")
            if valid_index in duration_outlier_indexes:
                flags.append("robust_duration_outlier")
            if valid_index in rms_outlier_indexes:
                flags.append("robust_rms_outlier")
            valid_index += 1
        record["quality_flags"] = ";".join(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/crema_d_eligible_modeling_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/crema_d_audio_quality.csv"))
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise SystemExit(f"No records found in {args.input}")

    records: list[dict[str, str]] = []
    for source in source_rows:
        record = {field: source.get(field, "") for field in METRIC_FIELDS}
        try:
            metrics = measure_wav(Path(source["path"]))
            record.update({key: f"{value:.8f}" if isinstance(value, float) else str(value) for key, value in metrics.items()})
            record["read_error"] = ""
        except (OSError, ValueError, wave.Error) as error:
            record["read_error"] = str(error)
        records.append(record)

    apply_flags(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    flagged = [record for record in records if record["quality_flags"]]
    failures = [record for record in records if record["read_error"]]
    print(f"Audited recordings: {len(records)}")
    print(f"Read failures: {len(failures)}")
    print(f"Flagged for review: {len(flagged)}")
    print(f"Quality report saved: {args.output}")


if __name__ == "__main__":
    main()
