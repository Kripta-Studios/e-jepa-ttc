from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e_jepa_ttc.object_event_v4_13 import (  # noqa: E402
    ObjectEventV413Config,
    conservative_dual_head_prediction,
    selective_fusion_gates,
)
from e_jepa_ttc.object_event_v4_4 import (  # noqa: E402
    branch_metrics,
    official_eap_metrics,
    pearson,
)

IDENTITY = ["sequence_id", "sample_token", "track_id"]


def _per_sequence(frame: pd.DataFrame, prediction: np.ndarray, minimum_negatives: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    for sequence_id, indices in frame.groupby("sequence_id", sort=True).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        y = target[idx]
        p = prediction[idx]
        neg = y < 0.0
        pos = ~neg
        rows.append({
            "sequence_id": str(sequence_id),
            "count": int(len(idx)),
            "negative_count": int(neg.sum()),
            "positive_count": int(pos.sum()),
            "pearson": pearson(y, p),
            "expansion_mae": float(np.mean(np.abs(y - p))),
            "positive_accuracy": float(np.mean(p[pos] >= 0.0)) if pos.any() else 0.0,
            "negative_accuracy": float(np.mean(p[neg] < 0.0)) if neg.any() else 0.0,
        })
    result = pd.DataFrame(rows)
    eligible = result[result["negative_count"] >= minimum_negatives]
    result.attrs["minimum_sequence_negative_accuracy"] = (
        float(eligible["negative_accuracy"].min()) if len(eligible) else 0.0
    )
    result.attrs["minimum_sequence_pearson"] = float(result["pearson"].min())
    return result


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, minimum_negatives: int) -> tuple[dict[str, Any], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    metrics = branch_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
    )
    per_sequence = _per_sequence(frame, prediction, minimum_negatives)
    metrics["minimum_sequence_negative_accuracy"] = per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    metrics["minimum_sequence_pearson"] = per_sequence.attrs["minimum_sequence_pearson"]
    metrics["official_eap"] = official_eap_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
    )
    return metrics, per_sequence


def run(*, config_path: Path, v412_summary_path: Path, predictions_path: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fusion = ObjectEventV413Config(**cast(dict[str, Any], raw["fusion"]))
    thresholds = {str(k): float(v) for k, v in cast(Mapping[str, Any], raw["gates"]).items()}
    summary412 = json.loads(v412_summary_path.read_text(encoding="utf-8"))
    if summary412.get("artifact_type") != "object_event_v4_12_reversal_balanced_directional_sign":
        raise RuntimeError("Unexpected v4.12 summary artifact")
    if not summary412.get("scientific_contract", {}).get("exact_descriptor_antisymmetry"):
        raise RuntimeError("v4.12 exact odd symmetry contract is missing")

    frame = pd.read_csv(predictions_path)
    required = set(IDENTITY + [
        "delta_t_s", "target_ttc_s", "target_expansion",
        "baseline_prediction_expansion", "negative_probability",
        "zero_events_prediction_expansion", "shuffled_prediction_expansion",
    ])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing v4.12 prediction columns: {missing}")

    baseline = frame["baseline_prediction_expansion"].to_numpy(dtype=np.float64)
    probability = frame["negative_probability"].to_numpy(dtype=np.float64)
    routed, directional, blend, override = conservative_dual_head_prediction(
        baseline, probability, config=fusion
    )
    baseline_metrics, _ = _metrics(frame, baseline, minimum_negatives=20)
    routed_metrics, per_sequence = _metrics(frame, routed, minimum_negatives=20)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    diagnostics = {
        "override_count": int(override.sum()),
        "override_rate": float(override.mean()),
        "mean_blend": float(blend.mean()),
        "directional_probability_mean": float(probability.mean()),
        "zero_event_pearson_drop": pearson(target, routed)
        - pearson(target, frame["zero_events_prediction_expansion"].to_numpy(dtype=np.float64)),
        "shuffled_event_pearson_drop": pearson(target, routed)
        - pearson(target, frame["shuffled_prediction_expansion"].to_numpy(dtype=np.float64)),
    }
    gates = selective_fusion_gates(
        routed=cast(Mapping[str, float], routed_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        diagnostics=diagnostics,
        thresholds=thresholds,
    )
    passed = all(gates.values())

    output = frame.loc[:, IDENTITY + ["delta_t_s", "target_ttc_s", "target_expansion"]].copy()
    output["baseline_prediction_expansion"] = baseline
    output["negative_probability"] = probability
    output["soft_directional_expansion"] = directional
    output["fusion_blend"] = blend
    output["high_confidence_override"] = override
    output["selective_prediction_expansion"] = routed
    output.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)

    result = {
        "artifact_type": "object_event_v4_13_conservative_dual_head_fusion",
        "status": "selective_fusion_passed" if passed else "selective_fusion_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(fusion),
        "thresholds": thresholds,
        "source_v412_summary": v412_summary_path.resolve().as_posix(),
        "source_v412_status": summary412.get("status"),
        "baseline_validation_metrics": baseline_metrics,
        "selective_validation_metrics": routed_metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "event_only_inference": True,
            "magnitude_source_is_frozen_v410_ensemble": True,
            "direction_source_is_frozen_v412_odd_probe": True,
            "only_high_confidence_positive_to_negative_overrides": True,
            "fusion_parameters_locked_before_v413_run": True,
            "parameters_informed_by_v412_seed7_development_results": True,
            "v413_is_not_independent_validation": True,
            "no_sequence_or_track_id_features": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_locked_multiseed_replication": passed,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v412-summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            config_path=args.config,
            v412_summary_path=args.v412_summary,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            force=args.force,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"artifact_type": "object_event_v4_13_failure", "error_type": type(exc).__name__, "error": str(exc)}
        (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
