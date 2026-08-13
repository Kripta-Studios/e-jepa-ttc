#!/usr/bin/env python
"""Run one real-data epoch on 32 rows for every Scientific Recovery V7 arm."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DINOv3RelationalTeacherDataset,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import (  # noqa: E402
    CausalScaleEAPTrainingConfig,
    train_real_causal_scale,
)

ARMS = ("soft", "c2f", "t20", "cap_s")
CONFIG_ROOT = ROOT / "configs/experiment/scientific_recovery_v7_fold_chain"


class _FixedIndexView(Dataset[dict[str, Any]]):
    """Small deterministic view that preserves the shard-sampler contract."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        indices: list[int],
        *,
        relabel_nine_smoke_sequences: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.relabel_nine_smoke_sequences = relabel_nine_smoke_sequences

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[self.indices[index]])
        if self.relabel_nine_smoke_sequences:
            record["sequence_id"] = f"smoke_sequence_{index % 9}"
        return record

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(range(len(self))),)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config root is not a mapping: {path}")
    return payload


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (ROOT / path).resolve(strict=True)


def _sample_tokens(dataset: Dataset[dict[str, Any]]) -> set[str]:
    return {str(dataset[index]["sample_token"]) for index in range(len(dataset))}


def smoke(*, device_name: str, output_dir: Path) -> dict[str, Any]:
    """Exercise data, teachers, losses, optimizer and evaluation for all four arms."""

    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        raw = _read_yaml(CONFIG_ROOT / f"v7_{arm}_fold0_seed7.yaml")
        model_raw = _read_yaml(_resolve(str(raw["model_config"])))
        model_raw.pop("model", None)
        thresholds = model_raw.get("risk_thresholds_s")
        if isinstance(thresholds, list):
            model_raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
        model_config = CausalScaleTTCConfig(**model_raw)
        model_config = replace(
            model_config,
            min_abs_log_ratio=1.0e-12,
            min_sensor_support=0.0,
        )
        training_config = CausalScaleEAPTrainingConfig(**raw["training"])
        if training_config.soft_geometry_teacher_checkpoint is not None:
            training_config = replace(
                training_config,
                soft_geometry_teacher_checkpoint=str(
                    _resolve(training_config.soft_geometry_teacher_checkpoint)
                ),
            )
        training_config = replace(
            training_config,
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=0,
            foreground_warmup_epochs=0,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_workers=0,
            precision="fp32" if device.type == "cpu" else training_config.precision,
            maximum_runtime_hours=0.5,
        )
        loss_config = CausalScaleTTCLossConfig(**raw["loss"])
        bins = (model_config.in_channels - 2) // 2
        base = GarlTTCObjectEventV4Dataset(
            str(_resolve(str(raw["data"]["cache_manifest"]))),
            splits=("train",),
            bins_per_polarity=bins,
        )
        train_base = _FixedIndexView(base, list(range(32)))
        validation = _FixedIndexView(
            base,
            list(range(32, 64)),
            relabel_nine_smoke_sequences=True,
        )
        teacher = raw["data"]["dinov3_relational_teacher"]
        train = DINOv3RelationalTeacherDataset(
            train_base,
            manifest_path=_resolve(str(teacher["manifest"])),
            expected_artifact_sha256=str(teacher["artifact_sha256"]),
            expected_manifest_sha256=str(teacher["manifest_sha256"]),
            allowed_sample_tokens=_sample_tokens(train_base),
        )
        state_dir = output_dir / arm / "state"
        result = train_real_causal_scale(
            model_config,
            training_config,
            loss_config,
            train,
            validation,
            device,
            checkpoint_dir=state_dir,
            resume=(state_dir / "last.pt").is_file(),
            stop_after_epoch=1,
        )
        point = np.asarray(
            result.best_validation["point_prediction_ttc_s"], dtype=np.float64
        )
        if point.shape != (32,) or not np.isfinite(point).all():
            raise RuntimeError(f"{arm} smoke did not produce 32 finite point predictions")
        if not all(
            np.isfinite(float(value))
            for row in result.history
            for value in row["train"].values()
        ):
            raise RuntimeError(f"{arm} smoke produced a non-finite training loss")
        arm_reports[arm] = {
            "rows_train": 32,
            "rows_validation": 32,
            "epochs": 1,
            "best_epoch": result.best_epoch,
            "elapsed_seconds": result.elapsed_seconds,
            "point_finite_fraction": float(np.isfinite(point).mean()),
            "parameter_count": sum(
                parameter.numel() for parameter in result.model.parameters()
            ),
            "selection_only_smoke_sequence_relabel": True,
            "selection_only_abstention_thresholds_disabled": True,
            "soft_teacher_excluded_from_optimizer": result.initialization.get(
                "soft_geometry_teacher_excluded_from_optimizer"
            ),
        }
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_four_arm_real_smoke_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "device": str(device),
        "arms": arm_reports,
        "closed_evaluation": {
            "public_validation_used": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    sign_artifact(report)
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v7/smoke_v2",
    )
    args = parser.parse_args()
    try:
        report = smoke(device_name=args.device, output_dir=args.output_dir.resolve())
    except Exception as error:
        parser.exit(2, f"V7 smoke failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "artifact": report["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
