#!/usr/bin/env python3
"""Evaluate the predeclared v4.2/v4.8 event-only fixed fusion screen."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.preflight_object_event_v4_9 import (  # noqa: E402
    assess_v42_fusion_baseline,
)
from e_jepa_ttc.object_event_v4_9 import (  # noqa: E402
    FixedFusionConfig,
    add_fusion_columns,
    align_prediction_frames,
    alpha_sweep,
    dependence_metrics,
    evaluate_prediction,
    fusion_gates,
)


def _construct_config(path: Path) -> FixedFusionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("fusion"), dict):
        raise ValueError("v4.9 config must contain a fusion mapping")
    values = raw["fusion"]
    allowed = {field.name for field in fields(FixedFusionConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown FixedFusionConfig fields: {unknown}")
    return FixedFusionConfig(**values)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_summary(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != artifact_type:
        raise RuntimeError(
            f"Unexpected artifact type in {path}: {payload.get('artifact_type')!r}"
        )
    return payload


def _evaluate_split(
    base_path: Path,
    dense_path: Path,
    *,
    split_name: str,
    config: FixedFusionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    aligned = align_prediction_frames(
        pd.read_csv(base_path), pd.read_csv(dense_path), split_name=split_name
    )
    rows = add_fusion_columns(aligned, alpha=config.alpha)
    base_metrics, _ = evaluate_prediction(rows, "base_prediction_expansion", config=config)
    dense_metrics, _ = evaluate_prediction(rows, "dense_prediction_expansion", config=config)
    fused_metrics, per_sequence = evaluate_prediction(
        rows, "fused_prediction_expansion", config=config
    )
    dependence = dependence_metrics(rows)
    metrics = {
        "base_v4_2": base_metrics,
        "dense_v4_8": dense_metrics,
        "fixed_fusion": fused_metrics,
        "event_dependence": dependence,
        "expert_prediction_pearson": float(
            rows[["base_prediction_expansion", "dense_prediction_expansion"]]
            .corr()
            .iloc[0, 1]
        ),
        "expert_disagreement_mean": float(rows["expert_disagreement"].mean()),
        "expert_disagreement_p95": float(rows["expert_disagreement"].quantile(0.95)),
    }
    return rows, per_sequence, metrics, alpha_sweep(rows, config=config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _construct_config(args.config)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}; pass --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    v42_summary = _load_summary(
        args.v42_summary, "object_event_v4_2_full_event_only_screen"
    )
    v48_summary = _load_summary(
        args.v48_summary, "object_event_v4_8_dense_foreground_motion"
    )
    v42_assessment = assess_v42_fusion_baseline(
        v42_summary,
        allow_marginal_negative_accuracy_only=bool(
            getattr(args, "allow_marginal_v42_negative_accuracy_only", False)
        ),
    )
    if not bool(v42_assessment["accepted_for_fusion"]):
        raise RuntimeError(
            "v4.2 source screen is not eligible: "
            f"{json.dumps(v42_assessment, sort_keys=True)}"
        )
    if not bool(v48_summary.get("passed")) or v48_summary.get("mode") != "screen":
        raise RuntimeError("v4.8 source screen is not eligible")

    train_rows, train_per_sequence, train_metrics, train_sweep = _evaluate_split(
        args.v42_train_predictions,
        args.v48_train_predictions,
        split_name="train",
        config=config,
    )
    validation_rows, validation_per_sequence, validation_metrics, validation_sweep = (
        _evaluate_split(
            args.v42_validation_predictions,
            args.v48_validation_predictions,
            split_name="validation",
            config=config,
        )
    )
    gates = fusion_gates(
        base=validation_metrics["base_v4_2"],
        dense=validation_metrics["dense_v4_8"],
        fused=validation_metrics["fixed_fusion"],
        dependence=validation_metrics["event_dependence"],
        config=config,
    )
    passed = all(gates.values())

    train_rows.to_csv(output / "train_predictions.csv", index=False)
    validation_rows.to_csv(output / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(output / "train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(output / "validation_per_sequence.csv", index=False)
    train_sweep.assign(split="train").to_csv(output / "train_alpha_sweep.csv", index=False)
    validation_sweep.assign(split="validation").to_csv(
        output / "validation_alpha_sweep_diagnostic.csv", index=False
    )

    summary = {
        "artifact_type": "object_event_v4_9_fixed_event_fusion",
        "status": "fusion_screen_passed" if passed else "fusion_screen_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "config": args.config.resolve().as_posix(),
        "fusion_config": asdict(config),
        "source_artifacts": {
            "v4_2_summary": args.v42_summary.resolve().as_posix(),
            "v4_2_assessment": v42_assessment,
            "v4_8_summary": args.v48_summary.resolve().as_posix(),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "event_only_inference": True,
            "fixed_alpha_before_v4_9_run": True,
            "alpha_informed_by_prior_development_results": True,
            "alpha_not_fit_inside_v4_9": True,
            "alpha_sweep_is_diagnostic_only": True,
            "no_boxes_or_heights_used_by_fusion": True,
            "standalone_v4_9_remains_strict_by_default": True,
            "marginal_v42_exception_requires_explicit_flag": True,
            "failed_v42_screen_is_not_relabelled": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_multiseed_fusion": passed,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v42-summary", type=Path, required=True)
    parser.add_argument("--v42-train-predictions", type=Path, required=True)
    parser.add_argument("--v42-validation-predictions", type=Path, required=True)
    parser.add_argument("--v48-summary", type=Path, required=True)
    parser.add_argument("--v48-train-predictions", type=Path, required=True)
    parser.add_argument("--v48-validation-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-marginal-v42-negative-accuracy-only",
        action="store_true",
        help=(
            "Allow only the audited v4.10 replication exception where the "
            "sole failed v4.2 gate is negative accuracy and all minimum "
            "non-degeneracy checks pass."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_9_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
