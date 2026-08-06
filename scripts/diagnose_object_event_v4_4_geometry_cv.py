#!/usr/bin/env python3
"""Train-only geometry residual calibration for Object Event TTC v4.4.

This diagnostic consumes the fixed v4.2 seed outputs and the corrected v4 event
cache.  It never fits on validation labels.  Geometry and residual calibrators
are selected by leave-one-training-sequence-out CV, refit on all training rows,
and evaluated once on the unchanged held-out validation sequences.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.object_event_v4_4 import (  # noqa: E402
    GEOMETRY_FEATURE_NAMES,
    StandardizedRidge,
    branch_metrics,
    event_geometry_features,
    fit_weighted_ridge,
    official_eap_metrics,
    sequence_sign_weights,
)
from scripts.train_e_jepa_object_event_v4_2 import MaterializedSplit, _materialize  # noqa: E402

IDENTITY_COLUMNS = ["sequence_id", "sample_token", "track_id"]
REFERENCE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
]


@dataclass(frozen=True)
class V44Config:
    seeds: tuple[int, ...] = (7, 13, 23)
    input_size: int = 64
    geometry_batch_size: int = 64
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    sequence_sign_weight_cap: float = 10.0
    max_abs_expansion: float = 0.25
    hybrid_pearson_max_drop: float = 0.01
    hybrid_mid_relative_improvement_gate: float = 0.03
    hybrid_balanced_sign_improvement_gate: float = 0.02
    hybrid_negative_accuracy_improvement_gate: float = 0.05
    per_sequence_negative_min_count: int = 20
    hybrid_min_sequence_negative_accuracy_gate: float = 0.20
    hybrid_expansion_mae_tolerance: float = 0.002
    hybrid_saturation_gate: float = 0.08
    geometry_only_pearson_gate: float = 0.10

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.4 requires at least three unique seeds")
        if self.input_size < 16 or self.geometry_batch_size < 1:
            raise ValueError("invalid geometry dimensions")
        if not self.ridge_alphas or min(self.ridge_alphas) < 0.0:
            raise ValueError("ridge_alphas must be non-empty and non-negative")
        if self.sequence_sign_weight_cap < 1.0:
            raise ValueError("sequence_sign_weight_cap must be at least 1")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")


def _load_config(path: Path) -> tuple[V44Config, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.4 config must be a mapping")
    references_raw = raw.pop("sota_references", {})
    if not isinstance(references_raw, dict):
        raise ValueError("sota_references must be a mapping")
    allowed = {field.name for field in fields(V44Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown v4.4 config fields: {unknown}")
    values = dict(raw)
    values["seeds"] = tuple(int(seed) for seed in values.get("seeds", (7, 13, 23)))
    values["ridge_alphas"] = tuple(
        float(alpha) for alpha in values.get("ridge_alphas", (0.01, 0.1, 1.0, 10.0, 100.0))
    )
    references = {str(key): float(value) for key, value in references_raw.items()}
    return V44Config(**values), references


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_ensemble(run_root: Path, seeds: tuple[int, ...], split: str) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    prediction_columns: list[str] = []
    for seed in seeds:
        path = run_root / f"seed-{seed}" / f"{split}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed {seed} {split} predictions: {path}")
        frame = pd.read_csv(path)
        required = {*REFERENCE_COLUMNS, "prediction_expansion"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        if frame.duplicated(IDENTITY_COLUMNS).any():
            raise ValueError(f"Duplicate identities in {path}")
        prediction_column = f"prediction_seed_{seed}"
        prediction_columns.append(prediction_column)
        current = frame[REFERENCE_COLUMNS + ["prediction_expansion"]].rename(
            columns={"prediction_expansion": prediction_column}
        )
        if aligned is None:
            aligned = current
            continue
        aligned = aligned.merge(
            current,
            on=IDENTITY_COLUMNS,
            how="inner",
            validate="one_to_one",
            suffixes=("", "_check"),
        )
        for column in ("delta_t_s", "target_ttc_s", "target_expansion"):
            check = f"{column}_check"
            if not np.allclose(
                aligned[column].to_numpy(dtype=np.float64),
                aligned[check].to_numpy(dtype=np.float64),
                rtol=1.0e-6,
                atol=1.0e-8,
            ):
                raise ValueError(f"Seed alignment changed {column}")
            aligned = aligned.drop(columns=check)
    if aligned is None:
        raise ValueError("no seeds configured")
    aligned["neural_ensemble_expansion"] = aligned[prediction_columns].mean(axis=1)
    return aligned.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


@torch.no_grad()
def _extract_geometry(split: MaterializedSplit, *, batch_size: int) -> pd.DataFrame:
    chunks: list[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        features = event_geometry_features(split.events[start : start + batch_size])
        chunks.append(features.cpu().numpy().astype(np.float64, copy=False))
    values = np.concatenate(chunks, axis=0)
    frame = pd.DataFrame(values, columns=list(GEOMETRY_FEATURE_NAMES))
    frame.insert(0, "track_id", split.track_ids)
    frame.insert(0, "sample_token", split.sample_tokens)
    frame.insert(0, "sequence_id", split.sequence_ids)
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("duplicate identities in materialized cache split")
    return frame


def _attach_geometry(predictions: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(geometry, on=IDENTITY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(predictions) or len(merged) != len(geometry):
        raise ValueError(
            f"prediction/cache alignment mismatch: prediction={len(predictions)}, "
            f"geometry={len(geometry)}, merged={len(merged)}"
        )
    return merged.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


def _geometry_design(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(GEOMETRY_FEATURE_NAMES)].to_numpy(dtype=np.float64)


def _hybrid_design(frame: pd.DataFrame) -> np.ndarray:
    geometry = _geometry_design(frame)
    neural = frame["neural_ensemble_expansion"].to_numpy(dtype=np.float64)
    proxy = frame["geometry_proxy"].to_numpy(dtype=np.float64)
    return np.column_stack((geometry, neural, np.abs(neural), neural * proxy))


def _prediction_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    return {
        "expansion": branch_metrics(
            frame["target_expansion"].to_numpy(dtype=np.float64),
            prediction,
            frame["delta_t_s"].to_numpy(dtype=np.float64),
        ),
        "official_eap": official_eap_metrics(
            frame["target_expansion"].to_numpy(dtype=np.float64),
            prediction,
            frame["delta_t_s"].to_numpy(dtype=np.float64),
            frame["target_ttc_s"].to_numpy(dtype=np.float64),
        ),
    }


def _cv_score(metrics: Mapping[str, object]) -> float:
    expansion = cast(Mapping[str, object], metrics["expansion"])
    official = cast(Mapping[str, object], metrics["official_eap"])
    weighted_mid = official.get("weighted_mid")
    mid = float(weighted_mid) if weighted_mid is not None else float(official["mid_mean_unweighted"])
    balanced = float(expansion["balanced_sign_accuracy"])
    negative = float(expansion["negative_accuracy"])
    # MiD is the principal objective.  Sign terms prevent a low global MiD from
    # selecting a calibrator that repeats the v4.3 negative-range failure.
    return mid + 100.0 * (1.0 - balanced) + 50.0 * (1.0 - negative)


def _fit_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    alpha: float,
    residual: bool,
    config: V44Config,
) -> tuple[np.ndarray, StandardizedRidge]:
    train_x = _hybrid_design(train_frame) if residual else _geometry_design(train_frame)
    test_x = _hybrid_design(test_frame) if residual else _geometry_design(test_frame)
    train_target = train_frame["target_expansion"].to_numpy(dtype=np.float64)
    train_base = train_frame["neural_ensemble_expansion"].to_numpy(dtype=np.float64)
    target = train_target - train_base if residual else train_target
    weights = sequence_sign_weights(
        train_frame["sequence_id"].astype(str).tolist(),
        train_target,
        cap=config.sequence_sign_weight_cap,
    )
    model = fit_weighted_ridge(train_x, target, sample_weight=weights, alpha=alpha)
    prediction = model.predict(test_x)
    if residual:
        prediction = test_frame["neural_ensemble_expansion"].to_numpy(dtype=np.float64) + prediction
    limit = config.max_abs_expansion * 0.999
    return np.clip(prediction, -limit, limit), model


def _select_alpha_grouped(
    train_frame: pd.DataFrame,
    *,
    residual: bool,
    config: V44Config,
) -> tuple[float, pd.DataFrame, list[dict[str, object]]]:
    sequences = sorted(train_frame["sequence_id"].astype(str).unique())
    if len(sequences) < 3:
        raise ValueError("grouped calibration requires at least three training sequences")
    candidates: list[dict[str, object]] = []
    best_prediction: np.ndarray | None = None
    best_alpha: float | None = None
    best_score = float("inf")
    for alpha in config.ridge_alphas:
        oof = np.full(len(train_frame), np.nan, dtype=np.float64)
        folds: list[dict[str, object]] = []
        for sequence in sequences:
            hold_mask = train_frame["sequence_id"].astype(str).to_numpy() == sequence
            fit_frame = train_frame.loc[~hold_mask]
            hold_frame = train_frame.loc[hold_mask]
            prediction, _ = _fit_predict(
                fit_frame,
                hold_frame,
                alpha=alpha,
                residual=residual,
                config=config,
            )
            oof[np.flatnonzero(hold_mask)] = prediction
            fold_metrics = _prediction_metrics(hold_frame, prediction)
            folds.append(
                {
                    "held_sequence": sequence,
                    "fit_sequence_count": len(sequences) - 1,
                    "held_count": int(hold_mask.sum()),
                    "metrics": fold_metrics,
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError("grouped OOF predictions contain missing values")
        metrics = _prediction_metrics(train_frame, oof)
        score = _cv_score(metrics)
        candidate = {"alpha": alpha, "score": score, "metrics": metrics, "folds": folds}
        candidates.append(candidate)
        if score < best_score - 1.0e-9 or (
            abs(score - best_score) <= 1.0e-9 and (best_alpha is None or alpha < best_alpha)
        ):
            best_score = score
            best_alpha = alpha
            best_prediction = oof.copy()
    if best_alpha is None or best_prediction is None:
        raise RuntimeError("failed to select grouped ridge alpha")
    oof_frame = train_frame[REFERENCE_COLUMNS].copy()
    oof_frame["prediction_expansion"] = best_prediction
    return best_alpha, oof_frame, candidates


def _serialize_model(model: StandardizedRidge, feature_names: list[str]) -> dict[str, object]:
    return {
        "alpha": model.alpha,
        "feature_names": feature_names,
        "mean": model.mean,
        "scale": model.scale,
        "intercept": float(model.coefficients[0]),
        "coefficients": model.coefficients[1:],
    }


def _per_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, group in frame.groupby("sequence_id", sort=True):
        row: dict[str, object] = {
            "sequence_id": sequence,
            "count": len(group),
            "negative_count": int((group["target_expansion"] < 0.0).sum()),
        }
        for name, column in (
            ("neural", "neural_ensemble_expansion"),
            ("geometry", "geometry_only_expansion"),
            ("hybrid", "hybrid_expansion"),
        ):
            metrics = branch_metrics(
                group["target_expansion"].to_numpy(dtype=np.float64),
                group[column].to_numpy(dtype=np.float64),
                group["delta_t_s"].to_numpy(dtype=np.float64),
            )
            for metric_name, value in metrics.items():
                if metric_name in {"count", "negative_count", "positive_count"}:
                    continue
                row[f"{name}_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    *,
    cache_manifest: Path,
    run_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    created_at = datetime.now(UTC)
    config, sota_references = _load_config(config_path)
    train_predictions = _load_ensemble(run_root, config.seeds, "train")
    validation_predictions = _load_ensemble(run_root, config.seeds, "validation")
    train_split, train_manifest = _materialize(cache_manifest, "train", input_size=config.input_size)
    train_geometry = _extract_geometry(train_split, batch_size=config.geometry_batch_size)
    del train_split
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=config.input_size
    )
    validation_geometry = _extract_geometry(
        validation_split, batch_size=config.geometry_batch_size
    )
    del validation_split
    train = _attach_geometry(train_predictions, train_geometry)
    validation = _attach_geometry(validation_predictions, validation_geometry)

    geometry_alpha, geometry_oof, geometry_candidates = _select_alpha_grouped(
        train, residual=False, config=config
    )
    hybrid_alpha, hybrid_oof, hybrid_candidates = _select_alpha_grouped(
        train, residual=True, config=config
    )
    geometry_validation, geometry_model = _fit_predict(
        train, validation, alpha=geometry_alpha, residual=False, config=config
    )
    hybrid_validation, hybrid_model = _fit_predict(
        train, validation, alpha=hybrid_alpha, residual=True, config=config
    )
    validation["geometry_only_expansion"] = geometry_validation
    validation["hybrid_expansion"] = hybrid_validation

    neural_metrics = _prediction_metrics(
        validation,
        validation["neural_ensemble_expansion"].to_numpy(dtype=np.float64),
    )
    geometry_metrics = _prediction_metrics(validation, geometry_validation)
    hybrid_metrics = _prediction_metrics(validation, hybrid_validation)
    per_sequence = _per_sequence(validation)
    eligible_sequences = per_sequence[
        per_sequence["negative_count"] >= config.per_sequence_negative_min_count
    ]
    minimum_sequence_negative = (
        float(eligible_sequences["hybrid_negative_accuracy"].min())
        if not eligible_sequences.empty
        else 0.0
    )

    neural_expansion = cast(Mapping[str, float], neural_metrics["expansion"])
    geometry_expansion = cast(Mapping[str, float], geometry_metrics["expansion"])
    hybrid_expansion = cast(Mapping[str, float], hybrid_metrics["expansion"])
    neural_official = cast(Mapping[str, object], neural_metrics["official_eap"])
    hybrid_official = cast(Mapping[str, object], hybrid_metrics["official_eap"])
    neural_mid_value = neural_official.get("weighted_mid")
    hybrid_mid_value = hybrid_official.get("weighted_mid")
    if neural_mid_value is None or hybrid_mid_value is None:
        mid_relative_improvement = float("-inf")
    else:
        neural_mid = float(neural_mid_value)
        hybrid_mid = float(hybrid_mid_value)
        mid_relative_improvement = (neural_mid - hybrid_mid) / max(neural_mid, 1.0e-12)

    gates = {
        "geometry_only_signal": float(geometry_expansion["pearson"])
        >= config.geometry_only_pearson_gate,
        "hybrid_preserves_pearson": float(hybrid_expansion["pearson"])
        >= float(neural_expansion["pearson"]) - config.hybrid_pearson_max_drop,
        "hybrid_improves_official_mid": mid_relative_improvement
        >= config.hybrid_mid_relative_improvement_gate,
        "hybrid_improves_balanced_sign": float(hybrid_expansion["balanced_sign_accuracy"])
        >= float(neural_expansion["balanced_sign_accuracy"])
        + config.hybrid_balanced_sign_improvement_gate,
        "hybrid_improves_negative_accuracy": float(hybrid_expansion["negative_accuracy"])
        >= float(neural_expansion["negative_accuracy"])
        + config.hybrid_negative_accuracy_improvement_gate,
        "hybrid_min_sequence_negative_accuracy": minimum_sequence_negative
        >= config.hybrid_min_sequence_negative_accuracy_gate,
        "hybrid_mae": float(hybrid_expansion["expansion_mae"])
        <= float(neural_expansion["expansion_mae"])
        + config.hybrid_expansion_mae_tolerance,
        "hybrid_saturation": float(hybrid_expansion["ttc_saturation_rate"])
        <= config.hybrid_saturation_gate,
    }
    passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    geometry_oof.to_csv(output_dir / "train_geometry_oof_predictions.csv", index=False)
    hybrid_oof.to_csv(output_dir / "train_hybrid_oof_predictions.csv", index=False)
    model_payload = {
        "geometry": _serialize_model(geometry_model, list(GEOMETRY_FEATURE_NAMES)),
        "hybrid_residual": _serialize_model(
            hybrid_model,
            [*GEOMETRY_FEATURE_NAMES, "neural_ensemble", "abs_neural_ensemble", "neural_x_geometry"],
        ),
    }
    (output_dir / "calibrators.json").write_text(
        json.dumps(_json_safe(model_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cv_payload = {
        "geometry_candidates": geometry_candidates,
        "hybrid_candidates": hybrid_candidates,
    }
    (output_dir / "grouped_train_cv.json").write_text(
        json.dumps(_json_safe(cv_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = cast(
        dict[str, object],
        _json_safe(
            {
                "artifact_type": "object_event_v4_4_train_only_geometry_cv",
                "status": "geometry_cv_passed" if passed else "geometry_cv_failed",
                "created_at": created_at.isoformat(),
                "cache_manifest": cache_manifest.resolve().as_posix(),
                "run_root": run_root.resolve().as_posix(),
                "config": asdict(config),
                "sota_references": sota_references,
                "selected_alpha": {
                    "geometry_only": geometry_alpha,
                    "hybrid_residual": hybrid_alpha,
                },
                "train_split": train_manifest,
                "validation_split": validation_manifest,
                "validation_metrics": {
                    "neural_ensemble": neural_metrics,
                    "geometry_only": geometry_metrics,
                    "hybrid": hybrid_metrics,
                    "hybrid_mid_relative_improvement": mid_relative_improvement,
                    "minimum_eligible_sequence_negative_accuracy": minimum_sequence_negative,
                },
                "gates": gates,
                "geometry_cv_passed": passed,
                "scientific_contract": {
                    "event_only": True,
                    "receives_boxes": False,
                    "receives_observable_motion": False,
                    "receives_rgb": False,
                    "calibrators_fit_on_train_only": True,
                    "alpha_selected_by_training_sequence_cv": True,
                    "validation_labels_used_for_fitting": False,
                    "reports_official_eap_mid_formula": True,
                    "validation_is_not_official_eap_test": True,
                    "evttc_not_evaluated_by_this_patch": True,
                    "advance_to_v4_5_differentiable_geometry": passed,
                    "advance_to_motion_or_rgb_fusion": False,
                    "claim_sota": False,
                },
            }
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "artifacts/runs/e_jepa_garl_object_event_screen_v4_2/scratch",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_geometry_cv_v4_4.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/debug/object_event_v4_4_geometry_cv",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output exists: {output_dir}; pass --force")
            shutil.rmtree(output_dir)
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            run_root=args.run_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["geometry_cv_passed"]) else 2
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_4_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
