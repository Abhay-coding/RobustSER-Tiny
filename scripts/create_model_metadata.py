"""Create a portable metadata record for one selected model checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to model_metadata.json inside --run-dir.")
    args = parser.parse_args()
    run_dir = args.run_dir
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best_validation_macro_f1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    protocol_path = Path(config["data"]["protocol"])
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output = args.output or run_dir / "model_metadata.json"
    metadata = {
        "metadata_schema_version": "1.0.0",
        "experiment_id": config["experiment"]["id"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "labels": list(checkpoint["labels"]),
        "model_name": config["model"]["name"],
        "feature_type": config["features"]["type"],
        "expected_feature_shape": config["features"]["expected_shape"],
        "sample_rate_hz": config["data"]["sample_rate_hz"],
        "window_seconds": config["data"]["train_window_seconds"],
        "data_protocol": protocol["protocol_name"],
        "data_protocol_version": protocol["protocol_version"],
    }
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
