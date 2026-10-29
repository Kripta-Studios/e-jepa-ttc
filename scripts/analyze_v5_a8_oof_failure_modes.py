#!/usr/bin/env python
"""Freeze and run the V6-D0 fold-local A8 failure-mode analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
    ObjectEventV4Batch,
    collate_object_event_v4,
)
from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
    CausalScaleTTCOutput,
    target_log_ratio_from_ttc,
)
from e_jepa_ttc.models.local_transport import TRANSPORT_FEATURE_NAMES  # noqa: E402
from e_jepa_ttc.reproducibility import seed_everything  # noqa: E402

RUN_NAMES = {
    "a6": "scientific_recovery_v5_a6_fold_chain_fold{fold}_seed7",
    "a8_0": "scientific_recovery_v5_a8_0_fold_chain_fold{fold}_seed7",
}
OOF_NAMES = {
    "a6": "a6_outer_dev_predictions.csv",
    "a8_0": "a8_0_outer_dev_predictions.csv",
    "garl": "garl_outer_dev_predictions.csv",
}
FAMILY_FEATURES = {
    "motion_scale": (
        "a8_transport_flow_magnitude",
        "a8_transport_foreground_flow_magnitude",
        "a8_transport_divergence_isotropic_abs",
        "target_log_ratio_abs",
    ),
    "history_sparsity": (
        "event_density_current",
        "event_density_min",
        "event_density_range",
        "transport_divergence_isotropic_acceleration",
        "transport_flow_magnitude_acceleration",
    ),
    "transport_confidence": (
        "a8_transport_confidence_margin",
        "a8_transport_entropy",
        "a8_transport_cycle_error",
        "a8_ttc_std_seconds",
    ),
    "roi_geometry": (
        "bbox_area_fraction_current",
        "bbox_border_distance_current",
        "bbox_aspect_log_abs_current",
        "bbox_centroid_y_displacement",
    ),
}
BRANCH_BY_FAMILY = {
    "motion_scale": "V6.1_MULTI_SCALE_TRANSPORT",
    "history_sparsity": "V6.1_LONGER_HISTORY_CAUSAL_TRANSPORT",
    "transport_confidence": "V6.1_CONFIDENCE_AWARE_FUSION",
    "roi_geometry": "V6.1_ROI_PERTURBATION_TRAINING",
}
COMPLEXITY_PRIORITY = {
    "transport_confidence": 0,
    "roi_geometry": 1,
    "motion_scale": 2,
    "history_sparsity": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read_json(path: Path, *, signed: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    if signed and not verify_artifact_hash(value):
        raise ValueError(f"invalid artifact signature: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _source(path: Path, *, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": _relative(path), "sha256": _sha256(path)}
    if artifact is not None:
        result["artifact_sha256"] = artifact.get("artifact_sha256")
    return result


def _validate_v5_inputs(protocol: dict[str, Any], aggregate: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("V5 grouped protocol has the wrong status")
    if aggregate.get("status") != "completed_development_gate_evaluation":
        raise ValueError("V5 aggregate has the wrong status")
    contracts = aggregate.get("contracts", {})
    if contracts.get("public_validation_used_for_selection") is not False:
        raise ValueError("V5 aggregate used public validation for selection")
    if contracts.get("private_test_opened") is not False:
        raise ValueError("V5 aggregate opened private/test")
    gate = aggregate.get("a8_0_gate", {})
    if gate.get("decision") != "FAIL":
        raise ValueError("V6-D0 is defined only after the failed A8.0 gate")


def freeze_protocol(
    *,
    v5_protocol_path: Path,
    v5_aggregate_path: Path,
    run_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze V6-D0 inputs and the postmortem decision rule before analysis."""

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("freeze requires a tracked-clean worktree")
    v5_protocol = _read_json(v5_protocol_path, signed=True)
    v5_aggregate = _read_json(v5_aggregate_path, signed=True)
    _validate_v5_inputs(v5_protocol, v5_aggregate)

    sources: dict[str, Any] = {
        "v5_grouped_protocol": _source(v5_protocol_path, artifact=v5_protocol),
        "v5_aggregate": _source(v5_aggregate_path, artifact=v5_aggregate),
        "event_cache_manifest": _source(
            ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
        ),
        "analysis_code": _source(Path(__file__)),
        "dataset_code": _source(ROOT / "src/e_jepa_ttc/data/object_event_v4.py"),
        "metric_code": _source(ROOT / "src/e_jepa_ttc/evaluation/garl_ttc_protocol.py"),
        "model_code": _source(ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py"),
        "transport_code": _source(ROOT / "src/e_jepa_ttc/models/local_transport.py"),
    }
    results_dir = v5_aggregate_path.parent
    for arm, filename in OOF_NAMES.items():
        sources[f"{arm}_oof"] = _source(results_dir / filename)
    for fold in range(3):
        for arm, pattern in RUN_NAMES.items():
            run_dir = run_root / pattern.format(fold=fold)
            summary_path = run_dir / "summary.json"
            summary = _read_json(summary_path, signed=True)
            if summary.get("status") != "completed_train_only_grouped_dev":
                raise ValueError(f"{arm} fold {fold} did not complete")
            fold_identity = summary.get("development_protocol", {}).get("fold_identity", {})
            if int(fold_identity.get("fold", -1)) != fold:
                raise ValueError(f"{arm} fold identity mismatch")
            for name in ("summary.json", "model_best.pt", "effective_config.yaml"):
                sources[f"{arm}_fold{fold}_{name.replace('.', '_')}"] = _source(run_dir / name)

    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v6_d0_oof_postmortem_protocol_v1",
        "status": "frozen_before_v6_d0_analysis",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "v5_protocol_artifact_sha256": v5_protocol["artifact_sha256"],
        "v5_aggregate_artifact_sha256": v5_aggregate["artifact_sha256"],
        "sources": sources,
        "analysis_contract": {
            "population": "exact_8192_fold_local_outer_dev_rows",
            "replay_batch_size": 32,
            "outcome_a8_vs_a6": "per_sample_raw_MiD_delta_and_failure_transition",
            "outcome_a8_vs_garl": "per_sample_raw_MiD_delta_and_failure_transition",
            "cluster_unit": "sequence_id_plus_track_id",
            "feature_families": {key: list(value) for key, value in FAMILY_FEATURES.items()},
            "minimum_family_score": 0.10,
            "minimum_sequence_sign_agreement": 0.60,
            "near_tie_relative_tolerance": 0.10,
            "near_tie_complexity_priority": sorted(
                COMPLEXITY_PRIORITY,
                key=COMPLEXITY_PRIORITY.get,  # type: ignore[arg-type]
            ),
            "decision_branches": BRANCH_BY_FAMILY,
            "no_family_branch": "V6.1_RETHINK_TRANSPORT_OBJECTIVE",
            "analysis_is_exploratory": True,
            "selected_v6_1_requires_new_preregistration": True,
        },
        "split_contract": {
            "outer_dev_is_development_not_test": True,
            "public_validation_opened": False,
            "private_test_opened": False,
            "diagnostic_parent_exposed_excluded": True,
            "no_optimizer_steps": True,
        },
    }
    sign_artifact(result)
    _atomic_json(output_path, result)
    return result


def raw_mid_per_sample(target: Iterable[float], prediction: Iterable[float]) -> np.ndarray:
    """Return the official raw MiD term before bucket and sequence weighting."""

    truth = np.asarray(list(target), dtype=np.float64)
    estimate = np.asarray(list(prediction), dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction shapes differ")
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_eta = 1.0 - 0.1 / truth
        estimate_eta = 1.0 - 0.1 / estimate
        return np.abs(np.log(truth_eta) - np.log(estimate_eta)) * 1.0e4


def _read_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} predictions lack {missing}")
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate sample tokens")
    return frame[list(required)].copy()


def align_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align exact identities and targets without dropping failed predictions."""

    reference_name = next(iter(frames))
    reference = frames[reference_name].copy()
    result = reference.rename(columns={"prediction_ttc_s": f"{reference_name}_prediction"})
    for label, frame in frames.items():
        if label == reference_name:
            continue
        current = frame.rename(columns={"prediction_ttc_s": f"{label}_prediction"})
        result = result.merge(
            current,
            on=["sample_token", "sequence_id", "track_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{label}"),
        )
        target_column = f"target_ttc_s_{label}"
        if len(result) != len(reference):
            raise ValueError(f"{label} sample population differs")
        if not np.allclose(
            result["target_ttc_s"].to_numpy(dtype=np.float64),
            result[target_column].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(f"{label} targets differ")
        result = result.drop(columns=target_column)
    return result


def _load_model(path: Path, device: torch.device) -> CausalScaleTTC:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != "causal_scale_eap_grouped_dev_checkpoint_v1":
        raise ValueError(f"unexpected checkpoint type: {path}")
    model = CausalScaleTTC(CausalScaleTTCConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def _autocast(device: torch.device) -> AbstractContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _tensor_values(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float64)


def _model_features(prefix: str, output: CausalScaleTTCOutput) -> dict[str, np.ndarray]:
    values = {
        f"{prefix}_prediction_replayed": _tensor_values(output.ttc_mean_seconds),
        f"{prefix}_known": _tensor_values(output.known_mask).astype(bool),
        f"{prefix}_analytic_log_ratio": _tensor_values(output.analytic_log_height_ratio[:, -1]),
        f"{prefix}_residual_log_ratio": _tensor_values(output.residual_log_height_ratio[:, -1]),
        f"{prefix}_final_log_ratio": _tensor_values(output.pair_log_height_ratio[:, -1]),
        f"{prefix}_ttc_std_seconds": np.exp(0.5 * _tensor_values(output.ttc_log_variance)),
        f"{prefix}_sensor_support": _tensor_values(output.sensor_support[:, -1]),
    }
    for name in TRANSPORT_FEATURE_NAMES:
        key = f"transport_{name}"
        if key in output.diagnostics:
            values[f"{prefix}_transport_{name}"] = _tensor_values(output.diagnostics[key][:, -1])
    return values


def _batch_features(
    batch: ObjectEventV4Batch,
    a6: CausalScaleTTCOutput,
    a8: CausalScaleTTCOutput,
    fold: int,
) -> pd.DataFrame:
    events = batch.events.float()
    density = (events.abs() > 1.0e-8).float().mean(dim=(-3, -2, -1))
    boxes = batch.boxes_xyxy.float()
    height = float(events.shape[-2])
    width = float(events.shape[-1])
    box_width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
    box_height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
    border = torch.stack(
        (boxes[..., 0], boxes[..., 1], width - boxes[..., 2], height - boxes[..., 3]),
        dim=-1,
    ).amin(dim=-1)
    centroid_y = 0.5 * (boxes[..., 1] + boxes[..., 3]) / height
    delta = batch.delta_t_s[:, None].expand(-1, events.shape[1] - 1)
    target_ratio, _ = target_log_ratio_from_ttc(batch.target_ttc_s, delta[:, -1])
    data: dict[str, Any] = {
        "sample_token": batch.sample_tokens,
        "sequence_id": batch.sequence_ids,
        "track_id": batch.track_ids,
        "fold": np.full(len(batch.sample_tokens), fold, dtype=np.int64),
        "target_ttc_s": _tensor_values(batch.target_ttc_s),
        "target_log_ratio": _tensor_values(target_ratio),
        "target_log_ratio_abs": np.abs(_tensor_values(target_ratio)),
        "regime": np.where(_tensor_values(batch.target_ttc_s) > 0.0, "approaching", "receding"),
        "event_density_t0": _tensor_values(density[:, 0]),
        "event_density_t1": _tensor_values(density[:, 1]),
        "event_density_current": _tensor_values(density[:, -1]),
        "event_density_min": _tensor_values(density.amin(dim=1)),
        "event_density_range": _tensor_values(density.amax(dim=1) - density.amin(dim=1)),
        "bbox_area_fraction_current": _tensor_values(
            box_width[:, -1] * box_height[:, -1] / (height * width)
        ),
        "bbox_border_distance_current": _tensor_values(border[:, -1] / min(height, width)),
        "bbox_aspect_log_abs_current": _tensor_values(
            (box_width[:, -1].clamp_min(1.0e-6) / box_height[:, -1].clamp_min(1.0e-6)).log().abs()
        ),
        "bbox_centroid_y_displacement": _tensor_values(
            (centroid_y[:, -1] - centroid_y[:, -2]).abs()
        ),
    }
    data.update(_model_features("a6", a6))
    data.update(_model_features("a8", a8))
    for name in ("divergence_isotropic", "flow_magnitude"):
        key = f"a8_transport_{name}"
        if key in data:
            pair = a8.diagnostics[f"transport_{name}"]
            data[f"transport_{name}_acceleration"] = np.abs(
                _tensor_values(pair[:, -1] - pair[:, -2])
            )
    data["a8_transport_divergence_isotropic_abs"] = np.abs(
        np.asarray(data["a8_transport_divergence_isotropic"])
    )
    return pd.DataFrame(data)


def _assert_replay(frame: pd.DataFrame, oof_label: str, replay_label: str) -> None:
    expected = frame[f"{oof_label}_prediction"].to_numpy(dtype=np.float64)
    observed = frame[f"{replay_label}_prediction_replayed"].to_numpy(dtype=np.float64)
    expected_finite = np.isfinite(expected)
    observed_finite = frame[f"{replay_label}_known"].to_numpy(dtype=bool) & np.isfinite(observed)
    if not np.array_equal(expected_finite, observed_finite):
        raise ValueError(f"{oof_label} replay finite mask differs from V5 OOF")
    if expected_finite.any() and not np.allclose(
        expected[expected_finite], observed[expected_finite], rtol=2.0e-3, atol=2.0e-3
    ):
        maximum = float(np.max(np.abs(expected[expected_finite] - observed[expected_finite])))
        raise ValueError(f"{oof_label} replay differs from V5 OOF; max_abs={maximum}")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 8 or np.unique(x[valid]).size < 2:
        return float("nan")
    value = spearmanr(x[valid], y[valid]).statistic
    return float(value) if math.isfinite(float(value)) else float("nan")


def feature_association(frame: pd.DataFrame, feature: str, outcome: str) -> dict[str, Any]:
    """Measure global and per-sequence rank association with an error delta."""

    global_correlation = _safe_spearman(
        frame[feature].to_numpy(dtype=np.float64), frame[outcome].to_numpy(dtype=np.float64)
    )
    per_sequence = {
        str(sequence): _safe_spearman(
            group[feature].to_numpy(dtype=np.float64), group[outcome].to_numpy(dtype=np.float64)
        )
        for sequence, group in frame.groupby("sequence_id", sort=True)
    }
    finite = np.asarray([value for value in per_sequence.values() if math.isfinite(value)])
    if finite.size:
        median_abs = float(np.median(np.abs(finite)))
        global_sign = np.sign(global_correlation) if math.isfinite(global_correlation) else 0.0
        sign_agreement = float(np.mean(np.sign(finite) == global_sign)) if global_sign else 0.0
    else:
        median_abs = float("nan")
        sign_agreement = 0.0
    return {
        "global_spearman": global_correlation,
        "per_sequence_spearman": per_sequence,
        "finite_sequence_count": int(finite.size),
        "median_absolute_sequence_spearman": median_abs,
        "sequence_sign_agreement": sign_agreement,
    }


def select_family(associations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen family score, sign agreement and low-complexity tie break."""

    family_rows: dict[str, Any] = {}
    for family, features in FAMILY_FEATURES.items():
        candidates = []
        for feature in features:
            item = associations[feature]
            score = item["median_absolute_sequence_spearman"]
            agreement = item["sequence_sign_agreement"]
            if math.isfinite(score):
                candidates.append((float(score), float(agreement), feature))
        candidates.sort(reverse=True)
        best = candidates[0] if candidates else (float("nan"), 0.0, None)
        family_rows[family] = {
            "score": best[0],
            "sequence_sign_agreement": best[1],
            "driving_feature": best[2],
        }
    eligible = [
        family
        for family, item in family_rows.items()
        if math.isfinite(item["score"])
        and item["score"] >= 0.10
        and item["sequence_sign_agreement"] >= 0.60
    ]
    if not eligible:
        return {
            "selected_family": None,
            "selected_branch": "V6.1_RETHINK_TRANSPORT_OBJECTIVE",
            "family_scores": family_rows,
            "reason": "no family passed the frozen score and sequence-agreement thresholds",
        }
    best_score = max(float(family_rows[family]["score"]) for family in eligible)
    near = [
        family for family in eligible if float(family_rows[family]["score"]) >= best_score * 0.90
    ]
    selected = min(near, key=COMPLEXITY_PRIORITY.get)  # type: ignore[arg-type]
    return {
        "selected_family": selected,
        "selected_branch": BRANCH_BY_FAMILY[selected],
        "family_scores": family_rows,
        "reason": "highest robust association with frozen low-complexity near-tie rule",
    }


def _metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    sequence = frame["sequence_id"].astype(str).to_numpy()
    return {
        "signed": signed_garl_metrics(target, prediction),
        "sequence_macro": sequence_macro_signed_metrics(target, prediction, sequence),
    }


def _group_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in frame.groupby(column, sort=True, dropna=False):
        result[str(value)] = {
            "rows": len(group),
            "a6": _metrics(group, "a6_prediction"),
            "a8_0": _metrics(group, "a8_0_prediction"),
            "garl": _metrics(group, "garl_prediction"),
            "mean_raw_mid_delta_a8_minus_a6": float(
                np.nanmean(group["a8_minus_a6_raw_mid"].to_numpy(dtype=np.float64))
            ),
            "mean_raw_mid_delta_a8_minus_garl": float(
                np.nanmean(group["a8_minus_garl_raw_mid"].to_numpy(dtype=np.float64))
            ),
        }
    return result


def _verify_sources(protocol: dict[str, Any]) -> None:
    for name, source in protocol.get("sources", {}).items():
        path = ROOT / str(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"V6-D0 source is missing or stale: {name}")


def analyze(
    *, protocol_path: Path, output_dir: Path, device: torch.device, batch_size: int
) -> dict[str, Any]:
    """Replay A6/A8 checkpoints and produce the signed V6-D0 diagnostic artifact."""

    protocol = _read_json(protocol_path, signed=True)
    if protocol.get("status") != "frozen_before_v6_d0_analysis":
        raise ValueError("V6-D0 protocol is not frozen")
    _verify_sources(protocol)
    expected_batch_size = int(protocol["analysis_contract"]["replay_batch_size"])
    if batch_size != expected_batch_size:
        raise ValueError(
            f"replay batch size must match V5 evaluation: {batch_size} != {expected_batch_size}"
        )
    seed_everything(7, deterministic=True)
    sources = protocol["sources"]
    predictions = align_predictions(
        {arm: _read_predictions(ROOT / sources[f"{arm}_oof"]["path"], arm) for arm in OOF_NAMES}
    )
    v5_protocol = _read_json(ROOT / sources["v5_grouped_protocol"]["path"], signed=True)
    fold_by_sequence = {
        sequence: int(fold["fold"])
        for fold in v5_protocol["folds"]
        for sequence in fold["dev_sequence_ids"]
    }
    predictions["fold"] = predictions["sequence_id"].map(fold_by_sequence)
    if predictions["fold"].isna().any():
        raise ValueError("OOF sequence is absent from the grouped protocol")

    cache_manifest = ROOT / sources["event_cache_manifest"]["path"]
    dataset = GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("train",))
    replay_parts: list[pd.DataFrame] = []
    for fold in range(3):
        sequences = set(v5_protocol["folds"][fold]["dev_sequence_ids"])
        view = SequenceIndexedView(dataset, sequence_ids=sequences)
        loader = DataLoader(
            view,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_object_event_v4,
        )
        models = {
            arm: _load_model(ROOT / sources[f"{arm}_fold{fold}_model_best_pt"]["path"], device)
            for arm in RUN_NAMES
        }
        for host_batch in loader:
            batch = host_batch.to(device)
            delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            with torch.inference_mode(), _autocast(device):
                a6_output = models["a6"](batch.events, delta)
                a8_output = models["a8_0"](batch.events, delta)
            replay_parts.append(_batch_features(batch, a6_output, a8_output, fold))

    replay = pd.concat(replay_parts, ignore_index=True)
    frame = predictions.merge(
        replay,
        on=["sample_token", "sequence_id", "track_id", "fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_replayed"),
    )
    if len(frame) != len(predictions):
        raise ValueError("diagnostic replay did not preserve the exact OOF population")
    if not np.allclose(
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s_replayed"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError("diagnostic replay targets differ from V5 OOF")
    frame = frame.drop(columns="target_ttc_s_replayed")
    _assert_replay(frame, "a6", "a6")
    _assert_replay(frame, "a8_0", "a8")

    for label in ("a6", "a8_0", "garl"):
        frame[f"{label}_raw_mid"] = raw_mid_per_sample(
            frame["target_ttc_s"], frame[f"{label}_prediction"]
        )
        frame[f"{label}_failure"] = ~np.isfinite(frame[f"{label}_prediction"]) | (
            frame[f"{label}_prediction"].abs() < 0.1
        )
    frame["a8_minus_a6_raw_mid"] = frame["a8_0_raw_mid"] - frame["a6_raw_mid"]
    frame["a8_minus_garl_raw_mid"] = frame["a8_0_raw_mid"] - frame["garl_raw_mid"]
    frame["a8_vs_a6_class"] = np.select(
        (
            frame["a8_0_failure"] & ~frame["a6_failure"],
            ~frame["a8_0_failure"] & frame["a6_failure"],
            frame["a8_minus_a6_raw_mid"] < -1.0e-9,
            frame["a8_minus_a6_raw_mid"] > 1.0e-9,
        ),
        ("new_failure", "recovered_failure", "improved", "worsened"),
        default="neutral",
    )
    absolute = frame["target_ttc_s"].abs()
    frame["abs_ttc_bucket"] = pd.cut(
        absolute,
        bins=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")],
        labels=["0-0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"],
        include_lowest=True,
    ).astype(str)

    associations = {
        feature: feature_association(frame, feature, "a8_minus_a6_raw_mid")
        for features in FAMILY_FEATURES.values()
        for feature in features
    }
    decision = select_family(associations)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "a8_oof_failure_modes_rows.csv"
    frame.sort_values("sample_token", kind="stable").to_csv(
        csv_path, index=False, lineterminator="\n"
    )
    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v6_d0_a8_oof_failure_modes_v1",
        "status": "completed_exploratory_outer_dev_diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": _source(protocol_path, artifact=protocol),
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "optimizer_steps": 0,
        },
        "population": {
            "rows": len(frame),
            "sequences": int(frame["sequence_id"].nunique()),
            "tracks": int(frame["track_id"].nunique()),
            "exact_oof_population": True,
        },
        "models": {
            "a6": _metrics(frame, "a6_prediction"),
            "a8_0": _metrics(frame, "a8_0_prediction"),
            "garl": _metrics(frame, "garl_prediction"),
        },
        "a8_vs_a6_class_counts": {
            str(key): int(value) for key, value in frame["a8_vs_a6_class"].value_counts().items()
        },
        "by_fold": _group_summary(frame, "fold"),
        "by_sequence": _group_summary(frame, "sequence_id"),
        "by_regime": _group_summary(frame, "regime"),
        "by_abs_ttc_bucket": _group_summary(frame, "abs_ttc_bucket"),
        "feature_associations_a8_minus_a6": associations,
        "decision": decision,
        "rows": _source(csv_path),
        "contracts": {
            "analysis_is_exploratory": True,
            "public_validation_opened": False,
            "private_test_opened": False,
            "diagnostic_parent_exposed_used": False,
            "promotion_authorized": False,
            "selected_v6_1_requires_new_preregistration": True,
        },
    }
    sign_artifact(result)
    _atomic_json(output_dir / "a8_oof_failure_modes.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze", help="Freeze D0 inputs and decision rule.")
    freeze_parser.add_argument(
        "--v5-protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json",
    )
    freeze_parser.add_argument(
        "--v5-aggregate",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v5/results/aggregate.json",
    )
    freeze_parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts/runs")
    freeze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze", help="Run the frozen OOF diagnostic.")
    analyze_parser.add_argument("--protocol", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    analyze_parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze_protocol(
                v5_protocol_path=args.v5_protocol.resolve(strict=True),
                v5_aggregate_path=args.v5_aggregate.resolve(strict=True),
                run_root=args.run_root.resolve(strict=True),
                output_path=args.output.resolve(),
            )
        else:
            if args.batch_size <= 0:
                raise ValueError("batch size must be positive")
            result = analyze(
                protocol_path=args.protocol.resolve(strict=True),
                output_dir=args.output_dir.resolve(),
                device=torch.device(args.device),
                batch_size=args.batch_size,
            )
    except Exception as error:
        parser.exit(2, f"V6-D0 failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": result["status"], "artifact_sha256": result["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
