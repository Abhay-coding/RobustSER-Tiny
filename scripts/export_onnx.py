"""Export the canonical E4 PyTorch deployment candidate to a fixed-shape ONNX graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from robustser.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-config", type=Path, default=Path("configs/deployment/esp32s3_espdl.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/deployment/e4_esp32s3.onnx"))
    args = parser.parse_args()
    config = yaml.safe_load(args.deployment_config.read_text(encoding="utf-8"))
    checkpoint_path = Path(config["artifact"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model("depthwise_separable_cnn", num_classes=len(checkpoint["labels"])).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    input_shape = tuple(config["artifact"]["input_shape"])
    example = torch.zeros(input_shape, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed-shape legacy export avoids dynamic graph operators that are less
    # suitable for the ESP-DL interchange path.
    torch.onnx.export(
        model, example, args.output, input_names=["features"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    print(f"ONNX export saved: {args.output}")


if __name__ == "__main__":
    main()
