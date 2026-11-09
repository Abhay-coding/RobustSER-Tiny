# RobustSER-Tiny Migration & Backward Compatibility Critique

## Executive Summary

RobustSER-Tiny has **no versioning strategy** for model artifacts, configs, data protocols, or inference contracts. This creates critical risks for:
- **Deployment rollback** when a new model regresses
- **OTA updates** to edge devices (ESP32-S3)
- **Reproducibility** across experiment runs
- **Forward compatibility** when loading older checkpoints

---

## 1. Model Artifact Versioning — REQUIRED

### Current State
Checkpoint saved in `train.py:101-109` contains NO version field:
```python
checkpoint = {
    "epoch": epoch,
    "task": task,
    "labels": TASK_LABELS[task],
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "validation_metrics": validation_metrics.to_dict(),
    "config": config,  # Full config dumped - no schema version
}
```

### Required: Add Semantic Version to Every Checkpoint

**Pydantic Schema** (`src/robustser/schemas/checkpoint.py`):
```python
"""Versioned checkpoint schema with migration support."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class CheckpointVersion(str, Enum):
    """Semantic version of the checkpoint format."""
    V1_0_0 = "1.0.0"  # Initial versioned format
    # Future: V1_1_0 = "1.1.0" (added calibration, ONNX metadata, etc.)


class ModelConfigSchema(BaseModel):
    """Versioned model configuration schema."""
    config_schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the config schema this checkpoint was trained with"
    )
    experiment: Dict[str, Any]
    data: Dict[str, Any]
    features: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    replication: Optional[Dict[str, Any]] = None
    deployment: Optional[Dict[str, Any]] = None


class CheckpointMetadata(BaseModel):
    """Metadata embedded in every checkpoint for migration/rollback."""
    model_version: str = Field(
        default="1.0.0",
        description="Semantic version of the model artifact (major.minor.patch)"
    )
    config_schema_version: str = Field(
        default="1.0.0",
        description="Config schema version used during training"
    )
    protocol_version: str = Field(
        default="1.0.0",
        description="Data protocol version used for splits"
    )
    created_at: str = Field(description="ISO 8601 timestamp of checkpoint creation")
    git_commit: Optional[str] = Field(default=None, description="Git SHA for reproducibility")
    git_dirty: bool = Field(default=False, description="Whether working tree had uncommitted changes")
    training_seed: int = Field(description="Seed used for this training run")
    epoch: int = Field(description="Epoch number when checkpoint was saved")
    is_best: bool = Field(default=False, description="Whether this is the best validation checkpoint")
    selection_metric: str = Field(description="Metric used for model selection (e.g., validation_macro_f1)")
    selection_value: float = Field(description="Value of selection metric at this checkpoint")
    labels: List[str] = Field(description="Class labels in model output order")
    task: str = Field(description="Task identifier (e.g., crema_d_six_emotion)")
    checksum_sha256: Optional[str] = Field(default=None, description="SHA-256 of model_state_dict for integrity")


class VersionedCheckpoint(BaseModel):
    """Complete versioned checkpoint with migration metadata."""
    metadata: CheckpointMetadata
    model_state_dict: Dict[str, Any]  # PyTorch state dict (serialized)
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    config: ModelConfigSchema

    @model_validator(mode="after")
    def validate_compatibility(self) -> "VersionedCheckpoint":
        """Validate checkpoint can be loaded by current code."""
        from packaging import version
        current_config_version = "1.0.0"
        if version.parse(self.metadata.config_schema_version) > version.parse(current_config_version):
            raise ValueError(
                f"Checkpoint config schema {self.metadata.config_schema_version} "
                f"is newer than supported {current_config_version}. "
                f"Update code or use compatible checkpoint."
            )
        return self


# Migration utilities
def migrate_checkpoint_v1_to_current(checkpoint: Dict[str, Any]) -> VersionedCheckpoint:
    """Migrate legacy checkpoint (no version) to current versioned format."""
    # Legacy checkpoints have: epoch, task, labels, model_state_dict, optimizer_state_dict, validation_metrics, config
    from datetime import datetime
    import hashlib
    import json

    # Compute checksum of model weights
    model_bytes = b""
    for k, v in checkpoint["model_state_dict"].items():
        if hasattr(v, 'cpu'):
            model_bytes += v.cpu().numpy().tobytes()
        else:
            model_bytes += str(v).encode()
    checksum = hashlib.sha256(model_bytes).hexdigest()[:16]

    metadata = CheckpointMetadata(
        model_version="1.0.0",  # Legacy gets base version
        config_schema_version="1.0.0",
        protocol_version="1.0.0",
        created_at=datetime.utcnow().isoformat() + "Z",
        git_commit=None,
        git_dirty=False,
        training_seed=checkpoint.get("config", {}).get("experiment", {}).get("seed", 0),
        epoch=checkpoint["epoch"],
        is_best=True,  # Assume legacy best checkpoint
        selection_metric="validation_macro_f1",
        selection_value=checkpoint.get("validation_metrics", {}).get("macro_f1", 0.0),
        labels=checkpoint["labels"],
        task=checkpoint["task"],
        checksum_sha256=checksum,
    )

    config = ModelConfigSchema(
        config_schema_version="1.0.0",
        experiment=checkpoint["config"]["experiment"],
        data=checkpoint["config"]["data"],
        features=checkpoint["config"]["features"],
        model=checkpoint["config"]["model"],
        training=checkpoint["config"]["training"],
        evaluation=checkpoint["config"]["evaluation"],
        replication=checkpoint["config"].get("replication"),
        deployment=checkpoint["config"].get("deployment"),
    )

    return VersionedCheckpoint(
        metadata=metadata,
        model_state_dict=checkpoint["model_state_dict"],
        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
        validation_metrics=checkpoint.get("validation_metrics"),
        config=config,
    )


def load_checkpoint_with_migration(path: str) -> VersionedCheckpoint:
    """Load checkpoint with automatic migration from legacy format."""
    import torch
    from pathlib import Path

    path = Path(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)

    # Detect legacy format (no metadata field)
    if "metadata" not in raw:
        return migrate_checkpoint_v1_to_current(raw)

    # Already versioned - validate
    return VersionedCheckpoint.model_validate(raw)
```

---

## 2. Config Schema Versioning — REQUIRED

### Current State
All YAML configs (`configs/baseline.yaml`, `configs/experiments/*.yaml`, `configs/data_protocol.yaml`) lack any version field.

### Required: Add `config_schema_version` to Every Config File

**Pydantic Schema** (`src/robustser/schemas/config.py`):
```python
"""Versioned configuration schemas with validation."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    protocol: str
    manifest: str
    sample_rate_hz: int = 16000
    train_window_seconds: float
    evaluation_window_policy: Literal["center_crop_or_zero_pad"]
    train_window_policy: Literal["seeded_random_crop_or_zero_pad"]
    normalization: Literal["per_utterance_mean_variance"]
    augmentation: Union[Literal["none"], Dict[str, Any]]
    noise_augmentation: Optional[Dict[str, Any]] = None


class FeaturesConfig(BaseModel):
    type: Literal["log_mel_filterbank_energy", "mfcc"]
    mel_filters: int
    frame_length_ms: int
    frame_stride_ms: int
    fft_size: int
    log_floor: float
    expected_shape: List[int]


class ModelConfig(BaseModel):
    name: Literal["paper_inspired_small_cnn", "depthwise_separable_cnn"]
    input_channels: int = 1
    convolution_blocks: List[Dict[str, Any]]
    classifier_hidden_units: int
    dropout: float
    output_classes_from_task: bool = True


class TrainingConfig(BaseModel):
    optimizer: Literal["adam"]
    learning_rate: float
    batch_size: int
    max_epochs: int
    selection_metric: Literal["validation_macro_f1"]
    early_stopping_patience: int
    checkpoint_policy: Literal["best_validation_macro_f1"]
    num_data_loader_workers: int = 0


class EvaluationConfig(BaseModel):
    primary_metric: Literal["macro_f1"]
    secondary_metrics: List[str]
    report_confusion_matrix: bool = True
    evaluate_test_once_after_model_selection: bool = True


class ReplicationConfig(BaseModel):
    seeds: List[int]
    selection_rule: str
    reporting_rule: str


class DeploymentConfig(BaseModel):
    target_quantization: Literal["int8", "float32"]
    quantization_applied_in_experiment: bool = False


class ExperimentConfig(BaseModel):
    id: str
    task: Literal["five_emotion_common", "crema_d_six_emotion"]
    seed: int
    deterministic_algorithms: bool = True
    notes: Optional[str] = None


class FullExperimentConfig(BaseModel):
    """Root config with mandatory version field."""
    config_schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of this config schema"
    )
    experiment: ExperimentConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    replication: Optional[ReplicationConfig] = None
    deployment: Optional[DeploymentConfig] = None

    @field_validator("config_schema_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        from packaging import version
        supported = ["1.0.0"]
        if v not in supported:
            raise ValueError(f"Unsupported config_schema_version: {v}. Supported: {supported}")
        return v


# Data Protocol Config
class DataProtocolConfig(BaseModel):
    config_schema_version: str = Field(default="1.0.0")
    protocol_name: str
    source_manifest: str
    quality_report: str
    duplicate_conflict_report: str
    split_seed: int
    splits: Dict[str, str]
    tasks: Dict[str, Dict[str, Any]]
    integrity_rules: Dict[str, bool]


# Migration utilities
def migrate_config_to_current(config: Dict[str, Any], config_type: str = "experiment") -> BaseModel:
    """Migrate legacy config to current versioned format."""
    if "config_schema_version" not in config:
        config["config_schema_version"] = "1.0.0"

    if config_type == "experiment":
        return FullExperimentConfig.model_validate(config)
    elif config_type == "data_protocol":
        return DataProtocolConfig.model_validate(config)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


def load_config_with_migration(path: str, config_type: str = "experiment") -> BaseModel:
    """Load YAML config with automatic migration."""
    import yaml
    from pathlib import Path

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return migrate_config_to_current(raw, config_type)
```

---

## 3. Inference Contract — REQUIRED

### Current State
`inference.py:predict_wav_bytes()` returns a raw `dict` with no schema validation. Demo app assumes exact keys.

### Required: Pydantic Schema for Input/Output with Versioning

**Pydantic Schema** (`src/robustser/schemas/inference.py`):
```python
"""Versioned inference input/output schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import io


class AudioInfo(BaseModel):
    source_sample_rate_hz: int
    model_sample_rate_hz: int
    duration_seconds: float
    rms_dbfs: float
    model_window_seconds: float


class ModelInfo(BaseModel):
    experiment_id: str
    checkpoint: str
    feature_type: str
    model_version: str = Field(default="1.0.0", description="Model artifact version")
    config_schema_version: str = Field(default="1.0.0")
    protocol_version: str = Field(default="1.0.0")


class InferenceWarnings(BaseModel):
    warnings: List[str] = Field(default_factory=list)

    @classmethod
    def from_checks(cls, duration_seconds: float, window_seconds: float, rms_dbfs: float,
                    confidence: float, margin: float) -> "InferenceWarnings":
        warnings = []
        if duration_seconds < window_seconds:
            warnings.append(f"Recording is shorter than the {window_seconds:g}-second model window and was zero-padded.")
        if rms_dbfs < -45.0:
            warnings.append("Audio is quiet; speak closer to the microphone or use a clearer recording.")
        if confidence < 0.50:
            warnings.append("Low model confidence: treat this label as uncertain.")
        if margin < 0.10:
            warnings.append("The top two classes are close, so this prediction is ambiguous.")
        return cls(warnings=warnings)


class PredictionOutput(BaseModel):
    """Versioned prediction output schema."""
    inference_schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the inference output schema"
    )
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    margin_to_second_choice: float = Field(ge=0.0)
    probabilities: Dict[str, float]
    audio: AudioInfo
    model: ModelInfo
    warnings: InferenceWarnings

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        return v


class InferenceRequest(BaseModel):
    """Input validation for inference endpoint."""
    audio_bytes: bytes
    run_dir: str
    device: Literal["cpu", "cuda"] = "cpu"


# Updated predict_wav_bytes with schema validation
def predict_wav_bytes_validated(
    audio_bytes: bytes,
    *,
    run_dir: str,
    device: str = "cpu"
) -> PredictionOutput:
    """Type-safe inference with schema validation."""
    from robustser.inference import predict_wav_bytes as _predict
    from pathlib import Path

    # Run existing inference
    result = _predict(audio_bytes, run_dir=run_dir, device=device)

    # Load model metadata for version info
    from robustser.schemas.checkpoint import load_checkpoint_with_migration
    checkpoint_path = Path(run_dir) / "best_validation_macro_f1.pt"
    checkpoint = load_checkpoint_with_migration(str(checkpoint_path))

    # Build validated output
    model_info = ModelInfo(
        experiment_id=result["model"]["experiment_id"],
        checkpoint=result["model"]["checkpoint"],
        feature_type=result["model"]["feature_type"],
        model_version=checkpoint.metadata.model_version,
        config_schema_version=checkpoint.metadata.config_schema_version,
        protocol_version=checkpoint.metadata.protocol_version,
    )

    return PredictionOutput(
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        margin_to_second_choice=result["margin_to_second_choice"],
        probabilities=result["probabilities"],
        audio=AudioInfo(**result["audio"]),
        model=model_info,
        warnings=InferenceWarnings(warnings=result["warnings"]),
    )


# Backward compatibility: accept legacy output dict
def ensure_validated_output(result: Dict[str, Any]) -> PredictionOutput:
    """Convert legacy dict output to validated schema (for migration)."""
    return PredictionOutput.model_validate(result)
```

---

## 4. Dataset Manifest Versioning — REQUIRED

### Current State
`configs/data_protocol.yaml` has frozen splits but NO version field.

### Required: Add `protocol_version` to Data Protocol

**Updated data_protocol.yaml:**
```yaml
protocol_version: "1.0.0"
protocol_name: crema_d_speaker_independent_v1
source_manifest: data/manifests/crema_d_speaker_splits.csv
quality_report: data/manifests/crema_d_audio_quality.csv
duplicate_conflict_report: data/manifests/crema_d_duplicate_conflicts.csv
split_seed: 20261108

# These assignments are frozen. Model runs must consume this manifest rather
# than construct new train/validation/test partitions.
splits:
  train: train
  validation: validation
  test: test

tasks:
  five_emotion_common:
    labels: [Angry, Disgust, Fear, Happy, Neutral]
    excluded_labels: [Sad]
    primary_metric: macro_f1
    description: Common-label benchmark for later cross-corpus comparison.
  crema_d_six_emotion:
    labels: [Angry, Disgust, Fear, Happy, Neutral, Sad]
    excluded_labels: []
    primary_metric: macro_f1
    description: Full CREMA-D benchmark; not directly comparable with five-label results.

integrity_rules:
  speaker_disjoint: true
  exact_hash_disjoint: true
  raw_audio_immutable: true
  augmented_audio_inherits_source_split: true
  test_split_used_for_model_selection: false
```

**Schema** (already defined in `src/robustser/schemas/config.py` as `DataProtocolConfig`)

---

## 5. Rollback Strategy — REQUIRED

### Current State
Only keeps `best_validation_macro_f1.pt` and `last.pt`. No history, no CLI to list/rollback.

### Required: Checkpoint Retention + CLI

**Implementation** (`src/robustser/checkpoint_manager.py`):
```python
"""Checkpoint retention, listing, and rollback utilities."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

from robustser.schemas.checkpoint import VersionedCheckpoint, load_checkpoint_with_migration


@dataclass
class CheckpointInfo:
    """Summary of a checkpoint for listing."""
    path: Path
    epoch: int
    is_best: bool
    selection_value: float
    model_version: str
    created_at: str
    checksum_sha256: str


class CheckpointManager:
    """Manages checkpoint retention and rollback for a run directory."""

    def __init__(self, run_dir: Path, keep_n_best: int = 3, keep_n_recent: int = 5):
        self.run_dir = Path(run_dir)
        self.keep_n_best = keep_n_best
        self.keep_n_recent = keep_n_recent
        self.index_path = self.run_dir / "checkpoint_index.json"

    def save_checkpoint(
        self,
        epoch: int,
        task: str,
        labels: List[str],
        model_state_dict: dict,
        optimizer_state_dict: dict,
        validation_metrics: dict,
        config: dict,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint with versioned metadata and update index."""
        from robustser.schemas.checkpoint import CheckpointMetadata, ModelConfigSchema, VersionedCheckpoint
        import hashlib
        from datetime import datetime

        # Compute checksum
        model_bytes = b""
        for k, v in model_state_dict.items():
            if hasattr(v, 'cpu'):
                model_bytes += v.cpu().numpy().tobytes()

        metadata = CheckpointMetadata(
            model_version="1.0.0",
            config_schema_version="1.0.0",
            protocol_version="1.0.0",
            created_at=datetime.utcnow().isoformat() + "Z",
            git_commit=self._get_git_commit(),
            git_dirty=self._is_git_dirty(),
            training_seed=config.get("experiment", {}).get("seed", 0),
            epoch=epoch,
            is_best=is_best,
            selection_metric=config.get("training", {}).get("selection_metric", "validation_macro_f1"),
            selection_value=validation_metrics.get("macro_f1", 0.0),
            labels=labels,
            task=task,
            checksum_sha256=hashlib.sha256(model_bytes).hexdigest()[:16],
        )

        config_schema = ModelConfigSchema(
            config_schema_version="1.0.0",
            experiment=config["experiment"],
            data=config["data"],
            features=config["features"],
            model=config["model"],
            training=config["training"],
            evaluation=config["evaluation"],
            replication=config.get("replication"),
            deployment=config.get("deployment"),
        )

        checkpoint = VersionedCheckpoint(
            metadata=metadata,
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            validation_metrics=validation_metrics,
            config=config_schema,
        )

        # Save with epoch number for history
        checkpoint_path = self.run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint.model_dump(), checkpoint_path)

        # Also save as best if applicable
        if is_best:
            best_path = self.run_dir / "best_validation_macro_f1.pt"
            torch.save(checkpoint.model_dump(), best_path)

        # Update index
        self._update_index(checkpoint_path, metadata)

        # Prune old checkpoints
        self._prune_checkpoints()

        return checkpoint_path

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints with metadata."""
        if not self.index_path.exists():
            return []

        with self.index_path.open("r") as f:
            index = json.load(f)

        checkpoints = []
        for entry in index["checkpoints"]:
            path = self.run_dir / entry["filename"]
            if path.exists():
                checkpoints.append(CheckpointInfo(
                    path=path,
                    epoch=entry["epoch"],
                    is_best=entry["is_best"],
                    selection_value=entry["selection_value"],
                    model_version=entry["model_version"],
                    created_at=entry["created_at"],
                    checksum_sha256=entry["checksum_sha256"],
                ))
        return sorted(checkpoints, key=lambda c: c.selection_value, reverse=True)

    def rollback_to(self, epoch: int) -> Path:
        """Rollback to a specific epoch checkpoint."""
        checkpoints = self.list_checkpoints()
        target = next((c for c in checkpoints if c.epoch == epoch), None)
        if not target:
            raise ValueError(f"No checkpoint found for epoch {epoch}")

        # Copy target to best location
        best_path = self.run_dir / "best_validation_macro_f1.pt"
        shutil.copy2(target.path, best_path)

        # Update index to mark new best
        self._update_best_in_index(epoch)

        return best_path

    def rollback_to_best(self) -> Path:
        """Rollback to the best checkpoint by selection metric."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ValueError("No checkpoints available")
        return self.rollback_to(checkpoints[0].epoch)

    def _update_index(self, checkpoint_path: Path, metadata) -> None:
        """Update the checkpoint index JSON."""
        index = {"checkpoints": []}
        if self.index_path.exists():
            with self.index_path.open("r") as f:
                index = json.load(f)

        entry = {
            "filename": checkpoint_path.name,
            "epoch": metadata.epoch,
            "is_best": metadata.is_best,
            "selection_value": metadata.selection_value,
            "model_version": metadata.model_version,
            "created_at": metadata.created_at,
            "checksum_sha256": metadata.checksum_sha256,
        }

        # Remove existing entry for same epoch
        index["checkpoints"] = [e for e in index["checkpoints"] if e["epoch"] != metadata.epoch]
        index["checkpoints"].append(entry)

        with self.index_path.open("w") as f:
            json.dump(index, f, indent=2)

    def _update_best_in_index(self, epoch: int) -> None:
        """Mark a checkpoint as the new best in index."""
        if not self.index_path.exists():
            return
        with self.index_path.open("r") as f:
            index = json.load(f)
        for entry in index["checkpoints"]:
            entry["is_best"] = (entry["epoch"] == epoch)
        with self.index_path.open("w") as f:
            json.dump(index, f, indent=2)

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond retention limits."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= max(self.keep_n_best, self.keep_n_recent):
            return

        # Keep N best by metric + N most recent by epoch
        to_keep = set()
        to_keep.update(c.path for c in checkpoints[:self.keep_n_best])
        to_keep.update(c.path for c in sorted(checkpoints, key=lambda c: c.epoch, reverse=True)[:self.keep_n_recent])

        for c in checkpoints:
            if c.path not in to_keep and c.path.name != "best_validation_macro_f1.pt":
                c.path.unlink(missing_ok=True)

    def _get_git_commit(self) -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.run_dir.parent.parent
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _is_git_dirty(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.run_dir.parent.parent
            )
            return bool(result.stdout.strip()) if result.returncode == 0 else False
        except Exception:
            return False
```

**CLI** (`scripts/checkpoint_cli.py`):
```python
"""CLI for checkpoint management: list, rollback, inspect."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustser.checkpoint_manager import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint management for RobustSER-Tiny")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("--format", choices=["table", "json"], default="table")

    # rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a specific checkpoint")
    rollback_group = rollback_parser.add_mutually_exclusive_group(required=True)
    rollback_group.add_argument("--epoch", type=int, help="Epoch number to rollback to")
    rollback_group.add_argument("--best", action="store_true", help="Rollback to best checkpoint")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect checkpoint metadata")
    inspect_parser.add_argument("--epoch", type=int, required=True, help="Epoch to inspect")

    args = parser.parse_args()

    manager = CheckpointManager(args.run_dir)

    if args.command == "list":
        checkpoints = manager.list_checkpoints()
        if args.format == "json":
            import json
            print(json.dumps([{
                "epoch": c.epoch,
                "is_best": c.is_best,
                "selection_value": c.selection_value,
                "model_version": c.model_version,
                "created_at": c.created_at,
                "checksum": c.checksum_sha256,
            } for c in checkpoints], indent=2))
        else:
            print(f"{'Epoch':>6} | {'Best':>4} | {'Metric':>8} | {'Version':>8} | {'Created':<20} | Checksum")
            print("-" * 90)
            for c in checkpoints:
                best_mark = "★" if c.is_best else ""
                print(f"{c.epoch:>6} | {best_mark:>4} | {c.selection_value:>8.4f} | "
                      f"{c.model_version:>8} | {c.created_at:<20} | {c.checksum_sha256}")

    elif args.command == "rollback":
        if args.best:
            path = manager.rollback_to_best()
            print(f"Rolled back to best checkpoint: {path}")
        else:
            path = manager.rollback_to(args.epoch)
            print(f"Rolled back to epoch {args.epoch}: {path}")

    elif args.command == "inspect":
        checkpoints = manager.list_checkpoints()
        target = next((c for c in checkpoints if c.epoch == args.epoch), None)
        if not target:
            print(f"No checkpoint found for epoch {args.epoch}", file=sys.stderr)
            sys.exit(1)
        from robustser.schemas.checkpoint import load_checkpoint_with_migration
        checkpoint = load_checkpoint_with_migration(str(target.path))
        print(f"Checkpoint: {target.path}")
        print(f"  Model Version: {checkpoint.metadata.model_version}")
        print(f"  Config Schema: {checkpoint.metadata.config_schema_version}")
        print(f"  Protocol: {checkpoint.metadata.protocol_version}")
        print(f"  Epoch: {checkpoint.metadata.epoch}")
        print(f"  Selection Metric: {checkpoint.metadata.selection_metric} = {checkpoint.metadata.selection_value}")
        print(f"  Labels: {checkpoint.metadata.labels}")
        print(f"  Git Commit: {checkpoint.metadata.git_commit}")
        print(f"  Git Dirty: {checkpoint.metadata.git_dirty}")


if __name__ == "__main__":
    main()
```

---

## 6. Forward Compatibility — REQUIRED

### Current State
`inference.py:load_selected_model()` assumes exact config keys and fails on older/newer checkpoints.

### Required: Graceful Handling of Schema Differences

**Updated inference.py with forward compatibility:**
```python
"""Local, reproducible inference helpers with forward compatibility."""
from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from scipy import signal
from scipy.io import wavfile

from robustser.data.cremad import FEATURE_EXTRACTORS, TASK_LABELS
from robustser.features.log_mel import SAMPLE_RATE, fit_audio_window
from robustser.models import build_model
from robustser.schemas.checkpoint import load_checkpoint_with_migration, VersionedCheckpoint
from robustser.schemas.inference import PredictionOutput, predict_wav_bytes_validated

MAX_DEMO_DURATION_SECONDS = 60.0


def decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    # ... existing implementation unchanged ...
    pass


def resample_to_model_rate(audio: np.ndarray, source_rate: int) -> np.ndarray:
    # ... existing implementation unchanged ...
    pass


@lru_cache(maxsize=4)
def load_selected_model(run_dir_text: str, device_text: str) -> tuple[torch.nn.Module, VersionedCheckpoint, tuple[str, ...]]:
    """Load one validation-selected checkpoint with migration support."""
    run_dir = Path(run_dir_text)
    checkpoint_path = run_dir / "best_validation_macro_f1.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected selected checkpoint in {run_dir}")

    # Load with automatic migration from legacy format
    checkpoint = load_checkpoint_with_migration(str(checkpoint_path))

    task = checkpoint.metadata.task
    labels = TASK_LABELS[task]
    device = torch.device(device_text)

    # Build model from config in checkpoint (handles schema differences)
    model = build_model(
        checkpoint.config.model["name"],
        num_classes=len(labels)
    ).to(device).eval()

    model.load_state_dict(checkpoint.model_state_dict)
    return model, checkpoint, tuple(labels)


def predict_wav_bytes(
    audio_bytes: bytes,
    *,
    run_dir: str | Path,
    device: str = "cpu"
) -> dict[str, object]:
    """Classify WAV recording - backward compatible dict output."""
    # Use validated version internally
    validated = predict_wav_bytes_validated(audio_bytes, run_dir=run_dir, device=device)
    # Convert to legacy dict for backward compatibility
    return validated.model_dump()


def prediction_json(result: dict[str, object]) -> str:
    return json.dumps(result, indent=2) + "\n"


# Forward compatibility utilities
def get_model_compatibility_report(run_dir: str | Path) -> Dict[str, Any]:
    """Report on checkpoint compatibility with current code."""
    from packaging import version
    from robustser.schemas.checkpoint import load_checkpoint_with_migration

    checkpoint = load_checkpoint_with_migration(str(Path(run_dir) / "best_validation_macro_f1.pt"))

    current_versions = {
        "model": "1.0.0",
        "config_schema": "1.0.0",
        "protocol": "1.0.0",
    }

    report = {
        "compatible": True,
        "warnings": [],
        "checkpoint_versions": {
            "model": checkpoint.metadata.model_version,
            "config_schema": checkpoint.metadata.config_schema_version,
            "protocol": checkpoint.metadata.protocol_version,
        },
        "current_versions": current_versions,
    }

    # Check config schema compatibility
    if version.parse(checkpoint.metadata.config_schema_version) > version.parse(current_versions["config_schema"]):
        report["compatible"] = False
        report["warnings"].append(
            f"Checkpoint config schema {checkpoint.metadata.config_schema_version} "
            f"is newer than supported {current_versions['config_schema']}. "
            f"Some config fields may be unrecognized."
        )
    elif version.parse(checkpoint.metadata.config_schema_version) < version.parse(current_versions["config_schema"]):
        report["warnings"].append(
            f"Checkpoint config schema {checkpoint.metadata.config_schema_version} "
            f"is older than current {current_versions['config_schema']}. "
            f"Defaults will be used for new fields."
        )

    # Check protocol compatibility
    if checkpoint.metadata.protocol_version != current_versions["protocol"]:
        report["warnings"].append(
            f"Protocol version mismatch: checkpoint={checkpoint.metadata.protocol_version}, "
            f"current={current_versions['protocol']}. Label mapping may differ."
        )

    return report
```

---

## 7. OTA/Deployment Model Bundle Format — REQUIRED

### Current State
ONNX export (`export_onnx.py`) only saves model graph. No config, version, or checksum bundled.

### Required: Self-Contained Model Bundle

**Bundle Format Specification** (`src/robustser/schemas/bundle.py`):
```python
"""Self-contained model bundle for OTA/deployment."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ModelBundle:
    """
    Self-contained deployment bundle containing:
    - Model weights (PyTorch state_dict or ONNX)
    - Full config for reconstruction
    - Version metadata
    - Integrity checksums
    - License/attribution info
    """
    # Version info
    bundle_format_version: str = "1.0.0"
    model_version: str = "1.0.0"
    config_schema_version: str = "1.0.0"
    protocol_version: str = "1.0.0"

    # Model artifacts
    pytorch_state_dict: Optional[Dict[str, Any]] = None
    onnx_path: Optional[str] = None
    onnx_bytes: Optional[bytes] = None

    # Config for full reconstruction
    config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit: Optional[str] = None
    git_dirty: bool = False
    training_seed: int = 0
    selection_metric: str = "validation_macro_f1"
    selection_value: float = 0.0
    labels: List[str] = field(default_factory=list)
    task: str = ""

    # Integrity
    pytorch_checksum_sha256: Optional[str] = None
    onnx_checksum_sha256: Optional[str] = None
    config_checksum_sha256: Optional[str] = None

    # Deployment info
    target_platform: Optional[str] = None  # e.g., "esp32s3_espdl"
    quantization: Optional[str] = None     # e.g., "int8"
    input_shape: Optional[List[int]] = None

    def __post_init__(self):
        self._compute_checksums()

    def _compute_checksums(self):
        if self.pytorch_state_dict:
            model_bytes = b""
            for k, v in self.pytorch_state_dict.items():
                if hasattr(v, 'cpu'):
                    model_bytes += v.cpu().numpy().tobytes()
            self.pytorch_checksum_sha256 = hashlib.sha256(model_bytes).hexdigest()

        if self.onnx_bytes:
            self.onnx_checksum_sha256 = hashlib.sha256(self.onnx_bytes).hexdigest()
        elif self.onnx_path and Path(self.onnx_path).exists():
            self.onnx_checksum_sha256 = hashlib.sha256(Path(self.onnx_path).read_bytes()).hexdigest()

        if self.config:
            self.config_checksum_sha256 = hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()
            ).hexdigest()

    def save(self, path: Path) -> None:
        """Save bundle as compressed JSON + separate weight files."""
        import zipfile
        import tempfile

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save metadata JSON
            meta = {
                k: v for k, v in self.__dict__.items()
                if k not in ("pytorch_state_dict", "onnx_bytes")
            }
            meta_path = tmpdir / "metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            # Save PyTorch weights
            if self.pytorch_state_dict:
                torch.save(self.pytorch_state_dict, tmpdir / "model_state_dict.pt")

            # Save ONNX
            if self.onnx_bytes:
                (tmpdir / "model.onnx").write_bytes(self.onnx_bytes)
            elif self.onnx_path:
                shutil.copy2(self.onnx_path, tmpdir / "model.onnx")

            # Create zip bundle
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in tmpdir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(tmpdir))

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        """Load bundle from zip file."""
        import zipfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(tmpdir)

            # Load metadata
            meta = json.loads((tmpdir / "metadata.json").read_text())

            # Load PyTorch weights
            pytorch_path = tmpdir / "model_state_dict.pt"
            if pytorch_path.exists():
                meta["pytorch_state_dict"] = torch.load(pytorch_path, map_location="cpu", weights_only=True)

            # Load ONNX
            onnx_path = tmpdir / "model.onnx"
            if onnx_path.exists():
                meta["onnx_bytes"] = onnx_path.read_bytes()
                meta["onnx_path"] = None

            return cls(**meta)

    def verify_integrity(self) -> Dict[str, bool]:
        """Verify all checksums match."""
        results = {}
        if self.pytorch_state_dict and self.pytorch_checksum_sha256:
            model_bytes = b""
            for k, v in self.pytorch_state_dict.items():
                if hasattr(v, 'cpu'):
                    model_bytes += v.cpu().numpy().tobytes()
            results["pytorch"] = hashlib.sha256(model_bytes).hexdigest() == self.pytorch_checksum_sha256

        if self.onnx_bytes and self.onnx_checksum_sha256:
            results["onnx"] = hashlib.sha256(self.onnx_bytes).hexdigest() == self.onnx_checksum_sha256

        if self.config and self.config_checksum_sha256:
            results["config"] = hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()
            ).hexdigest() == self.config_checksum_sha256

        return results


def create_deployment_bundle(
    run_dir: Path,
    output_path: Path,
    target_platform: str = "esp32s3_espdl",
    include_onnx: bool = True,
) -> ModelBundle:
    """Create deployment bundle from a training run directory."""
    from robustser.schemas.checkpoint import load_checkpoint_with_migration

    # Load best checkpoint with migration
    checkpoint = load_checkpoint_with_migration(str(run_dir / "best_validation_macro_f1.pt"))

    bundle = ModelBundle(
        bundle_format_version="1.0.0",
        model_version=checkpoint.metadata.model_version,
        config_schema_version=checkpoint.metadata.config_schema_version,
        protocol_version=checkpoint.metadata.protocol_version,
        pytorch_state_dict=checkpoint.model_state_dict,
        config=checkpoint.config.model_dump(),
        git_commit=checkpoint.metadata.git_commit,
        git_dirty=checkpoint.metadata.git_dirty,
        training_seed=checkpoint.metadata.training_seed,
        selection_metric=checkpoint.metadata.selection_metric,
        selection_value=checkpoint.metadata.selection_value,
        labels=checkpoint.metadata.labels,
        task=checkpoint.metadata.task,
        target_platform=target_platform,
        quantization="int8" if target_platform == "esp32s3_espdl" else None,
        input_shape=[1, 1, 32, 31],  # From deployment config
    )

    # Export ONNX if requested
    if include_onnx:
        from scripts.export_onnx import main as export_onnx
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)
        # ... run export ...
        bundle.onnx_bytes = onnx_path.read_bytes()
        onnx_path.unlink()

    bundle.save(output_path)
    return bundle
```

---

## Migration Checklist

| Component | Current | Required | Priority |
|-----------|---------|----------|----------|
| Checkpoint format | No version | `model_version`, `config_schema_version`, `protocol_version`, checksums | 🔴 CRITICAL |
| Config files | No version | `config_schema_version` in every YAML | 🔴 CRITICAL |
| Data protocol | No version | `protocol_version` in data_protocol.yaml | 🔴 CRITICAL |
| Inference I/O | Raw dict | Pydantic `PredictionOutput` with `inference_schema_version` | 🔴 CRITICAL |
| Checkpoint retention | 2 files | Keep N best + N recent, CLI for list/rollback | 🟠 HIGH |
| Forward compat | None | `load_checkpoint_with_migration()`, compatibility report | 🟠 HIGH |
| OTA bundle | ONNX only | Self-contained zip with metadata, checksums, config | 🟠 HIGH |

---

## Implementation Order

1. **Add schemas** (`src/robustser/schemas/`) - checkpoint, config, inference, bundle
2. **Update train.py** - use `CheckpointManager` for versioned saves
3. **Update inference.py** - use `predict_wav_bytes_validated` + migration loader
4. **Update data_protocol.yaml** - add `protocol_version: "1.0.0"`
5. **Add `config_schema_version: "1.0.0"`** to all experiment YAMLs
6. **Create checkpoint CLI** (`scripts/checkpoint_cli.py`)
7. **Update export_onnx.py** - create `ModelBundle` instead of raw ONNX
8. **Update demo/app.py** - use validated inference output
9. **Add tests** for migration: legacy checkpoint → current, older config → current