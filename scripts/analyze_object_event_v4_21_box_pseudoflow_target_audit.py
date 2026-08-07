#!/usr/bin/env python3
"""Audit whether v4.20's box-affine pseudoflow target is physically aligned with TTC."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _read_ensemble  # noqa: E402
from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores  # noqa: E402
from e_jepa_ttc.training.object_event_v4_20 import box_affine_pseudoflow  # noqa: E402
from e_jepa_ttc.training.object_event_v4_21 import (  # noqa: E402
    ObjectEventV421AuditConfig,
    box_scale_proxies,
    pearson,
    train_orientation,
)


def _load_config(path: Path) -> tuple[ObjectEventV421AuditConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.21 config must be a mapping")
    probe = ObjectEventV421AuditConfig(**dict(raw.get("probe", {})))
    decision = {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()}
    return probe, decision


def _oracle_divergence(split: Any, config: ObjectEventV421AuditConfig) -> tuple[np.ndarray, dict[str, float]]:
    boxes = split.boxes_xyxy.float()
    target_f, mask_f = box_affine_pseudoflow(
        boxes,
        source_height=split.source_height,
        source_width=split.source_width,
        target_height=config.map_size,
        target_width=config.map_size,
        first_index=1,
        second_index=2,
        epsilon=config.epsilon,
    )
    target_r, mask_r = box_affine_pseudoflow(
        boxes,
        source_height=split.source_height,
        source_width=split.source_width,
        target_height=config.map_size,
        target_width=config.map_size,
        first_index=2,
        second_index=1,
        epsilon=config.epsilon,
    )
    ones_f = torch.ones_like(mask_f)
    ones_r = torch.ones_like(mask_r)
    div_f, _, trans_f = dense_flow_scores(
        target_f[:, 0], target_f[:, 1], mask_f, ones_f,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    div_r, _, trans_r = dense_flow_scores(
        target_r[:, 0], target_r[:, 1], mask_r, ones_r,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    score = 0.5 * (div_f - div_r)
    reverse_score = 0.5 * (div_r - div_f)
    return score.cpu().numpy().astype(np.float64), {
        "endpoint_swap_oddness_max_abs": float((score + reverse_score).abs().max()),
        "mean_target_translation": float((0.5 * (trans_f + trans_r)).mean()),
        "mean_forward_mask_fraction": float(mask_f.mean()),
        "mean_reverse_mask_fraction": float(mask_r.mean()),
    }


def _per_sequence(frame: pd.DataFrame, score: np.ndarray, orientation: float) -> pd.DataFrame:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    sequences = frame["sequence_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for sequence in sorted(set(sequences)):
        mask = sequences == sequence
        local = orientation * score[mask]
        local_target = target[mask]
        rows.append({
            "sequence_id": sequence,
            "count": int(mask.sum()),
            "oriented_pearson": pearson(local, local_target),
            "positive_accuracy": float(np.mean(local[local_target >= 0.0] >= 0.0)) if np.any(local_target >= 0.0) else 0.0,
            "negative_accuracy": float(np.mean(local[local_target < 0.0] < 0.0)) if np.any(local_target < 0.0) else 0.0,
        })
    return pd.DataFrame(rows)


def _feature_rows(frame: pd.DataFrame, features: dict[str, np.ndarray], train_orientations: dict[str, float] | None = None) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    orientations: dict[str, float] = {}
    for name, score in features.items():
        orientation = train_orientation(score, target) if train_orientations is None else train_orientations[name]
        orientations[name] = orientation
        oriented = orientation * score
        rows.append({
            "feature": name,
            "orientation": orientation,
            "raw_pearson": pearson(score, target),
            "oriented_pearson": pearson(oriented, target),
            "mean_abs_value": float(np.mean(np.abs(score))),
        })
    return rows, orientations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--v420-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; use --force")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config, decision_config = _load_config(args.config)
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(args.ensemble_validation))

    train_div, train_diag = _oracle_divergence(train_split, config)
    val_div, val_diag = _oracle_divergence(validation_split, config)
    train_features = {"oracle_box_affine_divergence": train_div, **box_scale_proxies(train_split.boxes_xyxy, epsilon=config.epsilon)}
    val_features = {"oracle_box_affine_divergence": val_div, **box_scale_proxies(validation_split.boxes_xyxy, epsilon=config.epsilon)}
    train_feature_rows, orientations = _feature_rows(train_frame, train_features)
    val_feature_rows, _ = _feature_rows(validation_frame, val_features, orientations)

    oracle_orientation = orientations["oracle_box_affine_divergence"]
    train_per_sequence = _per_sequence(train_frame, train_div, oracle_orientation)
    val_per_sequence = _per_sequence(validation_frame, val_div, oracle_orientation)
    train_target = train_frame["target_expansion"].to_numpy(dtype=np.float64)
    val_target = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    train_p = pearson(oracle_orientation * train_div, train_target)
    val_p = pearson(oracle_orientation * val_div, val_target)
    min_seq = float(val_per_sequence["oriented_pearson"].min())

    v419 = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    v420 = json.loads(args.v420_summary.read_text(encoding="utf-8"))
    raw_v419 = float(v419["decision"]["comparisons"]["best_validation_dense_score_abs_pearson"])
    refined_v420 = float(v420["validation_score_metrics"]["pearson_to_target_expansion"])

    if (
        abs(train_p) >= decision_config["supported_train_abs_pearson"]
        and abs(val_p) >= decision_config["supported_validation_abs_pearson"]
        and min_seq >= decision_config["supported_min_sequence_oriented_pearson"]
    ):
        recommendation = "box_pseudoflow_supervision_supported_proceed_partial_unfreeze"
    elif abs(train_p) >= decision_config["supported_train_abs_pearson"] and abs(val_p) < decision_config["sequence_shift_validation_abs_pearson"]:
        recommendation = "box_target_sequence_shift_stop_box_supervision_use_event_native_flow"
    else:
        recommendation = "box_target_insufficient_stop_box_supervision_use_event_native_flow"

    pd.DataFrame(train_feature_rows).to_csv(args.output_dir / "train_feature_audit.csv", index=False)
    pd.DataFrame(val_feature_rows).to_csv(args.output_dir / "validation_feature_audit.csv", index=False)
    train_per_sequence.to_csv(args.output_dir / "train_oracle_per_sequence.csv", index=False)
    val_per_sequence.to_csv(args.output_dir / "validation_oracle_per_sequence.csv", index=False)
    sample = validation_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    sample["oracle_box_affine_divergence"] = oracle_orientation * val_div
    sample.to_csv(args.output_dir / "validation_oracle_scores.csv", index=False)

    summary = {
        "artifact_type": "object_event_v4_21_box_pseudoflow_target_audit",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "config": {"probe": asdict(config), "decision": decision_config},
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "train_oracle_divergence": {"orientation": oracle_orientation, "pearson": train_p, **train_diag},
        "validation_oracle_divergence": {"orientation_from_train": oracle_orientation, "pearson": val_p, "minimum_sequence_pearson": min_seq, **val_diag},
        "train_feature_audit": train_feature_rows,
        "validation_feature_audit": val_feature_rows,
        "decision": {
            "recommendation": recommendation,
            "comparisons": {
                "raw_v419_validation_divergence_pearson": raw_v419,
                "refined_v420_validation_divergence_pearson": refined_v420,
                "oracle_box_target_train_pearson": train_p,
                "oracle_box_target_validation_pearson": val_p,
                "oracle_box_target_minimum_validation_sequence_pearson": min_seq,
            },
            "note": "Oracle diagnostic only. Validation boxes are not optimisation inputs. Official eAP test and EvTTC remain unopened.",
        },
        "scientific_contract": {
            "no_model_trained": True,
            "box_pseudoflow_target_audited_before_encoder_unfreeze": True,
            "orientation_fit_on_train_only": True,
            "validation_boxes_used_only_for_oracle_diagnostic": True,
            "boxes_not_forward_features": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
