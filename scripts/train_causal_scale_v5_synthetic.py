"""Train and gate the v5 event foreground-scale arm on synthetic dynamics only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.synthetic_causal_scale import (  # noqa: E402
    SyntheticCausalScaleConfig,
    SyntheticCausalScaleDataset,
    SyntheticCausalScaleSample,
    synthetic_scale_config_identity,
)
from e_jepa_ttc.evaluation.causal_scale_v5 import (  # noqa: E402
    SYNTHETIC_LEARNING_THRESHOLDS,
    evaluate_synthetic_learning_gates,
)
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import environment_snapshot, resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_v5 import (  # noqa: E402
    CausalScaleSyntheticTrainingConfig,
    calibrate_ratio_uncertainty,
    checkpoint_payload,
    evaluate_synthetic_causal_scale,
    train_synthetic_causal_scale,
)
from scripts.evaluate_causal_scale_v5_operator import (  # noqa: E402
    _classify_worktree_status,
    _git,
)

DEFAULT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_synthetic_v5.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _AtomicOutput(AbstractContextManager[Path]):
    """Promote only a newly generated sibling directory after successful execution."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")

    def __enter__(self) -> Path:
        if self.target.exists():
            raise FileExistsError(f"output exists: {self.target}")
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc_type is not None:
            if self.staging.exists():
                if self.staging.parent != self.target.parent or not self.staging.name.startswith(
                    f".{self.target.name}.staging-"
                ):
                    raise PermissionError("refusing to clean an unverified staging path")
                shutil.rmtree(self.staging)
            return None
        os.replace(self.staging, self.target)
        return None

def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model config must be a mapping")
    model_name = raw.pop("model", None)
    if model_name not in {
        "e_jepa_causal_scale_event_v5",
        "e_jepa_causal_scale_event_v6",
    }:
        raise ValueError("model config must declare a supported causal-scale event model")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def _dataset_config(raw: dict[str, Any], split: str) -> SyntheticCausalScaleConfig:
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError("data config must be a mapping")
    common = data.get("common")
    section = data.get(split)
    if not isinstance(common, dict) or not isinstance(section, dict):
        raise ValueError(f"synthetic split {split!r} is missing")
    return SyntheticCausalScaleConfig(**{**common, **section})


def _validate_protocol(raw: dict[str, Any]) -> dict[str, float]:
    data = raw.get("data")
    if not isinstance(data, dict) or any(
        data.get(key) is not False
        for key in ("real_data_opened", "ttc_labels_opened", "eap_opened", "evttc_opened")
    ):
        raise ValueError("synthetic learning protocol must keep every real source closed")
    gates = raw.get("test_gates")
    if not isinstance(gates, dict) or set(gates) != set(SYNTHETIC_LEARNING_THRESHOLDS):
        raise ValueError("test_gates differ from the implemented frozen contract")
    return {key: float(value) for key, value in gates.items()}


def run(
    config_path: Path,
    output_dir: Path,
    *,
    stage: str,
    device_name: str,
) -> dict[str, Any]:
    """Train on train/validation and open the synthetic test group only in full stage."""

    if stage not in {"diagnostic", "full"}:
        raise ValueError("stage must be diagnostic or full")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")
    thresholds = _validate_protocol(raw)
    experiment = raw.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("experiment config must be a mapping")
    artifact_type = experiment.get(
        "artifact_type",
        "causal_scale_v5_synthetic_learning_gate_v1",
    )
    if not isinstance(artifact_type, str) or not artifact_type.startswith("causal_scale_v"):
        raise ValueError("experiment artifact_type is invalid")
    status_lines = _git(
        "-c",
        "core.quotepath=false",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    worktree = _classify_worktree_status(status_lines)
    code_dirty = bool(worktree["tracked_dirty"] or worktree["untracked_code_paths"])
    if stage == "full" and code_dirty:
        raise RuntimeError("full synthetic gate requires clean tracked/code state")
    model_path = ROOT / str(raw["model_config"])
    model_config = _model_config(model_path)
    training_raw = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(training_raw, dict) or not isinstance(loss_raw, dict):
        raise ValueError("training and loss configs must be mappings")
    training_config = CausalScaleSyntheticTrainingConfig(**training_raw)
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    train_config = _dataset_config(raw, "train")
    validation_config = _dataset_config(raw, "validation")
    if train_config.seed == validation_config.seed:
        raise ValueError("train and validation synthetic seed groups must differ")
    device = resolve_device(device_name)
    started = time.perf_counter()
    result = train_synthetic_causal_scale(
        model_config,
        training_config,
        loss_config,
        SyntheticCausalScaleDataset(train_config),
        SyntheticCausalScaleDataset(validation_config),
        device,
    )
    validation_loader: DataLoader[SyntheticCausalScaleSample] = DataLoader(
        SyntheticCausalScaleDataset(validation_config),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    calibration_raw = raw.get("calibration")
    if not isinstance(calibration_raw, dict):
        raise ValueError("calibration config must be a mapping")
    calibration = calibrate_ratio_uncertainty(
        result.model,
        validation_loader,
        device,
        target_coverage=float(calibration_raw["target_ratio_coverage"]),
    )
    validation_metrics = evaluate_synthetic_causal_scale(
        result.model,
        validation_loader,
        device,
        loss_config=loss_config,
        controls=True,
    )
    test_metrics: dict[str, float | None] | None = None
    gates: dict[str, bool] | None = None
    opened_splits = ["train", "validation"]
    test_config: SyntheticCausalScaleConfig | None = None
    if stage == "full":
        test_config = _dataset_config(raw, "test")
        if test_config.seed in {train_config.seed, validation_config.seed}:
            raise ValueError("synthetic test seed group must be held out")
        test_loader: DataLoader[SyntheticCausalScaleSample] = DataLoader(
            SyntheticCausalScaleDataset(test_config),
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
        )
        test_metrics = evaluate_synthetic_causal_scale(
            result.model,
            test_loader,
            device,
            loss_config=loss_config,
            controls=True,
        )
        gates = evaluate_synthetic_learning_gates(test_metrics, thresholds)
        opened_splits.append("test")
    with _AtomicOutput(output_dir) as staging:
        checkpoint_path = staging / "best.pt"
        torch.save(
            checkpoint_payload(result, training_config, loss_config),
            checkpoint_path,
        )
        payload: dict[str, Any] = {
            "artifact_type": artifact_type,
            "protocol_version": experiment["protocol_version"],
            "created_at": datetime.now(UTC).isoformat(),
            "status": (
                "diagnostic_nonselectable"
                if stage == "diagnostic"
                else ("completed_passed" if gates and gates["passed"] else "completed_gate_failed")
            ),
            "stage": stage,
            "selectable": False,
            "evidence_scope": "synthetic_event_learning_only",
            "metrics_are_not_real_dataset_results": True,
            "garl_ttc_comparison_performed": False,
            "sota_claim_authorized": False,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": code_dirty,
            "worktree": worktree,
            "config": {
                "path": config_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(config_path),
            },
            "model_config": {
                "path": model_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(model_path),
            },
            "dataset_groups": {
                "train": synthetic_scale_config_identity(train_config),
                "validation": synthetic_scale_config_identity(validation_config),
                "test": synthetic_scale_config_identity(test_config) if test_config else None,
            },
            "opened_splits": opened_splits,
            "test_evaluation_count": 1 if stage == "full" else 0,
            "data_access": raw["data"],
            "training_config": training_raw,
            "loss_config": loss_raw,
            "selection": {
                "split": "validation",
                "best_epoch": result.best_epoch,
                "best_score": result.best_selection_score,
            },
            "validation_calibration": calibration,
            "history": result.history,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_gates": gates,
            "thresholds": thresholds,
            "decision": raw["decision_contract"],
            "checkpoint": {"path": "best.pt", "sha256": _sha256(checkpoint_path)},
            "environment": environment_snapshot(),
            "device": str(device),
            "elapsed_s": time.perf_counter() - started,
        }
        sign_artifact(payload)
        serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        (staging / "summary.json").write_bytes(serialized.encode("utf-8"))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("diagnostic", "full"), default="diagnostic")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        payload = run(
            args.config.resolve(),
            args.output_dir.resolve(),
            stage=args.stage,
            device_name=args.device,
        )
    except Exception as error:
        parser.exit(2, f"synthetic causal-scale run failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stage": payload["stage"],
                "output_dir": str(args.output_dir),
                "selection": payload["selection"],
                "validation_metrics": payload["validation_metrics"],
                "test_metrics": payload["test_metrics"],
                "test_gates": payload["test_gates"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
