#!/usr/bin/env python
"""Aggregate the frozen V5 fold-local A4/A6/A8.0/Garl outer-dev results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v5 import _values_sha256  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from scripts.paired_grouped_bootstrap import run as run_pair  # noqa: E402

ARMS = ("a4", "a6", "a8_0", "garl")
PAIRINGS = (("a8_0", "a6"), ("a8_0", "garl"), ("a6", "garl"))
RUN_NAMES = {
    "a4": "scientific_recovery_v5_a4_parent_grouped_fold{fold}_seed7",
    "a6": "scientific_recovery_v5_a6_fold_chain_fold{fold}_seed7",
    "a8_0": "scientific_recovery_v5_a8_0_fold_chain_fold{fold}_seed7",
    "garl": "scientific_recovery_v5_garl_fold_chain_fold{fold}_seed7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_dirty": bool(git("status", "--short", "--untracked-files=no")),
    }


def _token_sha(tokens: pd.Series) -> str:
    return _values_sha256(tokens.astype(str).tolist())


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "sample_token",
        "sequence_id",
        "track_id",
        "target_ttc_s",
        "prediction_ttc_s",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"prediction file lacks columns {missing}: {path}")
    frame = frame[list(required)].copy()
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"duplicate sample tokens: {path}")
    if not np.isfinite(frame["target_ttc_s"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"non-finite target: {path}")
    return frame


def align_fold_predictions(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align without dropping invalid predictions and enforce identity/target parity."""

    if len(frames) < 2:
        raise ValueError("at least two model predictions are required")
    reference_name = next(iter(frames))
    reference = frames[reference_name].copy()
    tokens = set(reference["sample_token"].astype(str))
    aligned: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        if set(frame["sample_token"].astype(str)) != tokens:
            raise ValueError(f"{name} sample-token population differs from {reference_name}")
        current = frame.set_index("sample_token").loc[
            reference["sample_token"].astype(str)
        ].reset_index()
        for column in ("sequence_id", "track_id"):
            if not (
                current[column].astype(str).to_numpy()
                == reference[column].astype(str).to_numpy()
            ).all():
                raise ValueError(f"{name} {column} differs from {reference_name}")
        if not np.allclose(
            current["target_ttc_s"].to_numpy(dtype=np.float64),
            reference["target_ttc_s"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"{name} targets differ from {reference_name}")
        aligned[name] = current
    return aligned


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = frame["prediction_ttc_s"].to_numpy(dtype=np.float64)
    sequence = frame["sequence_id"].astype(str).to_numpy()
    signed = signed_garl_metrics(target, prediction)
    macro = sequence_macro_signed_metrics(target, prediction, sequence)
    finite = np.isfinite(prediction)
    pearson = (
        float(np.corrcoef(target[finite], prediction[finite])[0, 1])
        if np.count_nonzero(finite) >= 2
        else float("nan")
    )
    return {
        "rows": len(frame),
        "sequence_macro_MiD": float(macro["sequence_macro_paper_MiD_overall"]),
        "sample_weighted_MiD": float(signed["paper_MiD_overall"]),
        "failure_pct": float(signed["failure_rate_pct"]),
        "pearson": pearson,
        "coverage": float(np.mean(finite)),
        "per_sequence": macro["per_sequence"],
    }


def _summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(payload):
        raise ValueError(f"invalid summary signature: {path}")
    return payload


def _validate_summary(
    payload: dict[str, Any],
    *,
    arm: str,
    fold: int,
    prediction: Path,
) -> None:
    if payload.get("status") not in {
        "completed",
        "completed_max_epochs",
        "completed_early_stop",
        "completed_train_only_grouped_dev",
    }:
        raise ValueError(f"{arm} fold {fold} did not complete")
    development = payload.get("development_protocol", {}).get("fold_identity", {})
    if int(development.get("fold", -1)) != fold:
        raise ValueError(f"{arm} summary has the wrong fold")
    if arm == "garl":
        protocol = payload.get("protocol", {})
        sealed = payload.get("sealed_sources", {})
        if protocol.get("from_scratch") is not True or protocol.get(
            "pretrained_release_checkpoint_used"
        ) is not False:
            raise ValueError(f"Garl fold {fold} was not trained from scratch")
        if protocol.get("public_validation_used_for_selection") is not False:
            raise ValueError(f"Garl fold {fold} used public validation for selection")
        if sealed.get("public_validation_opened") is not False or sealed.get(
            "private_test_opened"
        ) is not False:
            raise ValueError(f"Garl fold {fold} opened a forbidden split")
        prediction_record = payload.get("artifacts", {}).get("predictions", {})
    else:
        if any(
            payload.get(key) is not False
            for key in (
                "official_test_opened",
                "private_test_opened",
                "public_validation_opened",
                "public_validation_used_for_selection",
            )
        ):
            raise ValueError(f"{arm} fold {fold} opened or selected on a forbidden split")
        prediction_record = payload.get("predictions", {})
    if prediction_record.get("sha256") != _sha256(prediction):
        raise ValueError(f"{arm} fold {fold} prediction hash differs from its summary")


def aggregate(
    *,
    run_root: Path,
    protocol_path: Path,
    cluster_metadata: Path,
    audit_dir: Path,
    output_dir: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(protocol):
        raise ValueError("grouped protocol signature is invalid")
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("grouped protocol was not frozen before A8")
    fold_contracts = {int(item["fold"]): item for item in protocol["folds"]}
    frames_by_arm: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    summaries: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    sources: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in (0, 1, 2):
        fold_frames: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            run_dir = run_root / RUN_NAMES[arm].format(fold=fold)
            prediction_name = (
                "dev_predictions.parquet" if arm == "garl" else "dev_predictions.csv"
            )
            prediction = run_dir / prediction_name
            summary_path = run_dir / "summary.json"
            fold_frames[arm] = _read_predictions(prediction)
            summaries[arm][fold] = _summary(summary_path)
            _validate_summary(
                summaries[arm][fold], arm=arm, fold=fold, prediction=prediction
            )
            sources[f"{arm}_fold{fold}"] = {
                "predictions": {"path": str(prediction.resolve()), "sha256": _sha256(prediction)},
                "summary": {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
            }
        fold_frames = align_fold_predictions(fold_frames)
        contract = fold_contracts[fold]
        for arm, frame in fold_frames.items():
            if len(frame) != int(contract["dev_rows"]):
                raise ValueError(f"{arm} fold {fold} row count differs from protocol")
            if _token_sha(frame["sample_token"]) != contract["dev_sample_tokens_sha256"]:
                raise ValueError(f"{arm} fold {fold} token hash differs from protocol")
            if set(frame["sequence_id"].astype(str)) != set(contract["dev_sequence_ids"]):
                raise ValueError(f"{arm} fold {fold} sequences differ from protocol")
            frames_by_arm[arm].append(frame.assign(fold=fold))
        for first, second in PAIRINGS:
            run_pair(
                Path(sources[f"{first}_fold{fold}"]["predictions"]["path"]),
                Path(sources[f"{second}_fold{fold}"]["predictions"]["path"]),
                output_dir / f"paired_{first}_vs_{second}_fold{fold}.json",
                first_label=first,
                second_label=second,
                fold=fold,
                resamples=resamples,
                seed=seed + fold,
                cluster_metadata=cluster_metadata,
                protocol=protocol_path,
            )

    arm_results: dict[str, Any] = {}
    oof_paths: dict[str, Path] = {}
    for arm, parts in frames_by_arm.items():
        oof = pd.concat(parts, ignore_index=True)
        if len(oof) != int(protocol["sample_count"]) or oof["sample_token"].duplicated().any():
            raise ValueError(f"{arm} OOF population is not an exact partition")
        if _token_sha(oof["sample_token"]) != protocol["sorted_sample_tokens_sha256"]:
            raise ValueError(f"{arm} OOF token universe differs from protocol")
        oof_path = output_dir / f"{arm}_outer_dev_predictions.csv"
        oof.drop(columns="fold").to_csv(oof_path, index=False)
        oof_paths[arm] = oof_path
        folds = [_metrics(part) for part in parts]
        mids = np.asarray([item["sequence_macro_MiD"] for item in folds], dtype=np.float64)
        arm_results[arm] = {
            "folds": {str(index): value for index, value in enumerate(folds)},
            "fold_MiD_mean": float(mids.mean()),
            "fold_MiD_sample_std": float(mids.std(ddof=1)),
            "worst_fold_MiD": float(mids.max()),
            "outer_dev_9_sequence": _metrics(oof),
            "parameter_count": int(
                summaries[arm][0].get(
                    "parameter_count",
                    summaries[arm][0].get("resources", {}).get("parameter_count", 0),
                )
            ),
        }

    aggregate_pairs: dict[str, Any] = {}
    for first, second in PAIRINGS:
        pair_path = output_dir / f"paired_{first}_vs_{second}_outer_dev.json"
        aggregate_pairs[f"{first}_vs_{second}"] = run_pair(
            oof_paths[first],
            oof_paths[second],
            pair_path,
            first_label=first,
            second_label=second,
            fold=None,
            resamples=resamples,
            seed=seed,
            cluster_metadata=cluster_metadata,
            protocol=protocol_path,
        )

    geometry_paths = sorted(audit_dir.glob("geometry_*.json"))
    if len(geometry_paths) != 6:
        raise ValueError("six completed geometry audits are required")
    for path in geometry_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            not verify_artifact_hash(audit)
            or audit.get("status") != "completed_exact_primary_geometry"
        ):
            raise ValueError(f"geometry audit failed: {path}")
    prefix_paths = [
        audit_dir / "prefix_causality_a6.json",
        audit_dir / "prefix_causality_a8_0.json",
    ]
    for path in prefix_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(audit) or audit.get("status") != "PASS":
            raise ValueError(f"prefix causality audit failed: {path}")

    a8 = arm_results["a8_0"]["outer_dev_9_sequence"]
    a6 = arm_results["a6"]["outer_dev_9_sequence"]
    gate_checks = {
        "improves_a6_aggregate": a8["sequence_macro_MiD"] < a6["sequence_macro_MiD"],
        "first_stage_MiD_le_175": a8["sequence_macro_MiD"] <= 175.0,
        "strong_target_MiD_le_160": a8["sequence_macro_MiD"] <= 160.0,
        "geometry_exact_parent": True,
        "model_prefix_causality": True,
        "coverage_not_materially_worse": a8["coverage"] >= a6["coverage"] - 0.01,
        "public_validation_used_for_selection": False,
        "private_test_opened": False,
    }
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_fold_chain_aggregate_v1",
        "status": "completed_development_gate_evaluation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "aggregation_revision": _git_revision(),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
            "artifact_sha256": protocol["artifact_sha256"],
        },
        "models": arm_results,
        "paired_outer_dev": {
            key: {
                "path": str((output_dir / f"paired_{key}_outer_dev.json").resolve()),
                "artifact_sha256": value["artifact_sha256"],
                "delta_first_minus_second": value["delta_first_minus_second"],
                "bootstrap": value["bootstrap"],
            }
            for key, value in aggregate_pairs.items()
        },
        "a8_0_gate": {
            "checks": gate_checks,
            "first_stage_pass": all(
                gate_checks[key]
                for key in (
                    "improves_a6_aggregate",
                    "first_stage_MiD_le_175",
                    "geometry_exact_parent",
                    "model_prefix_causality",
                    "coverage_not_materially_worse",
                )
            ),
            "decision": "PASS" if all(
                gate_checks[key]
                for key in (
                    "improves_a6_aggregate",
                    "first_stage_MiD_le_175",
                    "geometry_exact_parent",
                    "model_prefix_causality",
                    "coverage_not_materially_worse",
                )
            ) else "FAIL",
        },
        "contracts": {
            "outer_dev_used_for_checkpoint_selection": True,
            "outer_dev_is_development_not_test": True,
            "public_validation_used_for_selection": False,
            "strict_end_to_end_streaming_causality_claimed": False,
            "private_test_opened": False,
            "sota_claim_authorized": False,
        },
        "sources": sources,
    }
    if not all(
        math.isfinite(
            float(arm_results[arm]["outer_dev_9_sequence"]["sequence_macro_MiD"])
        )
        for arm in ARMS
    ):
        raise ValueError("an arm has non-finite aggregate MiD")
    sign_artifact(report)
    output = output_dir / "aggregate.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cluster-metadata", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    try:
        result = aggregate(
            run_root=args.run_root.resolve(strict=True),
            protocol_path=args.protocol.resolve(strict=True),
            cluster_metadata=args.cluster_metadata.resolve(strict=True),
            audit_dir=args.audit_dir.resolve(strict=True),
            output_dir=args.output_dir.resolve(),
            resamples=args.resamples,
            seed=args.seed,
        )
    except Exception as error:
        parser.exit(2, f"V5 fold aggregation failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {"output": str(args.output_dir / "aggregate.json"), "gate": result["a8_0_gate"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
