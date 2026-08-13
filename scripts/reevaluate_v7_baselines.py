#!/usr/bin/env python
"""Re-evaluate frozen V6 causal checkpoints under the V7 point/selective contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView  # noqa: E402
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import (  # noqa: E402
    CausalScaleEAPTrainingConfig,
    _loader,
    evaluate_real_causal_scale,
)

RUN_NAMES = {
    "a5": "scientific_recovery_v6_a5_causal_grouped_fold{fold}_seed7",
    "a8": "scientific_recovery_v5_a8_0_fold_chain_fold{fold}_seed7",
    "v6_1": "scientific_recovery_v6_1_dual_transport_r2_fold{fold}_seed7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"invalid signed artifact: {path}")
    return payload


def _frame(evaluation: dict[str, Any], *, fold: int, seed: int) -> pd.DataFrame:
    row_count = len(evaluation["sample_tokens"])
    return pd.DataFrame(
        {
            "sample_token": evaluation["sample_tokens"],
            "sequence_id": evaluation["sequence_ids"],
            "track_id": evaluation["track_ids"],
            "target_ttc_s": evaluation["target_ttc_s"],
            "prediction_ttc_s": evaluation["prediction_ttc_s"],
            "point_prediction_ttc_s": evaluation["point_prediction_ttc_s"],
            "auxiliary_prediction_ttc_s": evaluation["auxiliary_prediction_ttc_s"],
            "known_mask": evaluation["known_mask"],
            "guard_margin": evaluation["guard_margin"],
            "ttc_log_variance": evaluation["ttc_log_variance"],
            "ttc_variance": evaluation["ttc_variance"],
            "event_count_log1p": evaluation["event_count_log1p"],
            "event_rate_log1p": evaluation["event_rate_log1p"],
            "transport_flow_magnitude": evaluation["transport_flow_magnitude"],
            "fold": [fold] * row_count,
            "seed": [seed] * row_count,
        }
    )


def reevaluate(*, device_name: str, output_dir: Path) -> dict[str, Any]:
    """Evaluate A5, A8 and V6.1 without opening any held-out split."""

    protocol = _read_signed(
        ROOT / "configs/protocol/scientific_recovery_v7_balanced_oof.json"
    )
    grouped = _read_signed(
        ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
    )
    cache_manifest = (
        ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
    )
    cache = GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("train",))
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    for arm, run_pattern in RUN_NAMES.items():
        frames: list[pd.DataFrame] = []
        for fold in range(3):
            checkpoint = ROOT / "artifacts/runs" / run_pattern.format(fold=fold) / "model_best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model_config = CausalScaleTTCConfig(**payload["model_config"])
            training_config = CausalScaleEAPTrainingConfig(**payload["training_config"])
            loss_config = CausalScaleTTCLossConfig(**payload["loss_config"])
            model = CausalScaleTTC(model_config)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(device)
            dev_sequences = set(grouped["folds"][fold]["dev_sequence_ids"])
            view = SequenceIndexedView(cache, sequence_ids=dev_sequences)
            loader = _loader(view, training_config, train=False)
            evaluation = evaluate_real_causal_scale(
                model,
                loader,
                device,
                training_config,
                loss_config,
                use_auxiliary_dev_metadata=True,
            )
            frame = _frame(evaluation, fold=fold, seed=7)
            expected_rows = int(grouped["folds"][fold]["dev_rows"])
            if len(frame) != expected_rows:
                raise ValueError(f"{arm} fold {fold} row count differs")
            frame.to_csv(output_dir / f"{arm}_fold{fold}_predictions.csv", index=False)
            frames.append(frame)
            sources[f"{arm}_fold{fold}"] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "best_epoch": payload["best_epoch"],
            }
        oof = pd.concat(frames, ignore_index=True)
        if len(oof) != 8192 or oof["sample_token"].duplicated().any():
            raise ValueError(f"{arm} re-evaluation is not an exact OOF partition")
        if not np.isfinite(oof["point_prediction_ttc_s"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{arm} produced a non-finite point prediction")
        oof.to_csv(output_dir / f"{arm}_oof_predictions.csv", index=False)

    garl_source = ROOT / "artifacts/scientific_recovery_v6/results/garl_outer_dev_predictions.csv"
    garl = pd.read_csv(garl_source)
    if len(garl) != 8192 or garl["sample_token"].duplicated().any():
        raise ValueError("frozen Garl comparator is not an exact OOF partition")
    garl_point = garl["prediction_ttc_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(garl_point).all():
        raise ValueError("frozen Garl comparator contains non-finite predictions")
    garl["point_prediction_ttc_s"] = garl_point
    garl["auxiliary_prediction_ttc_s"] = np.nan
    garl["known_mask"] = True
    garl["guard_margin"] = np.inf
    garl["ttc_log_variance"] = np.nan
    garl["ttc_variance"] = np.nan
    garl["event_count_log1p"] = np.nan
    garl["event_rate_log1p"] = np.nan
    garl["transport_flow_magnitude"] = np.nan
    fold_by_sequence = {
        sequence: int(fold["fold"])
        for fold in grouped["folds"]
        for sequence in fold["dev_sequence_ids"]
    }
    garl["fold"] = garl["sequence_id"].map(fold_by_sequence).astype(int)
    garl["seed"] = 7
    garl.to_csv(output_dir / "garl_oof_predictions.csv", index=False)
    sources["garl"] = {
        "predictions": str(garl_source.resolve()),
        "predictions_sha256": _sha256(garl_source),
        "interpretation": "frozen_seed7_comparator; point equals historical prediction",
    }
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_baseline_reevaluation_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "rows_per_model": 8192,
        "sources": sources,
        "contracts": {
            "point_prediction_finite_for_causal_models": True,
            "historical_selective_prediction_preserved": True,
            "garl_seed7_frozen": True,
            "public_validation_used_for_selection": False,
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v7/baselines",
    )
    args = parser.parse_args()
    try:
        report = reevaluate(device_name=args.device, output_dir=args.output_dir.resolve())
    except Exception as error:
        parser.exit(2, f"V7 baseline re-evaluation failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
