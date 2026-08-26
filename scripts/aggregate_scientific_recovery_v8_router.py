#!/usr/bin/env python
# ruff: noqa: E501
"""Aggregate the three untouched outer-dev folds of the V8 prospective router."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.canonical_token_identity import hash_canonical_json_records  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS,
    hierarchical_sequence_bootstrap,
    prediction_sha256,
    validate_oof_frame,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_aggregate import (  # noqa: E402
    _bind_candidate_to_baseline_contract,
    _read_rows as _read_general_rows,
    contract_hashes as _general_contract_hashes,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    V8IntegrityError,
    verify_frozen_inputs,
)


class RouterAggregateError(ValueError):
    """Raised when the aggregate cannot prove its frozen evidence contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RouterAggregateError("unable to resolve the implementation git commit") from error


def _read_signed(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RouterAggregateError(f"cannot read fold artifact {path}") from error
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise RouterAggregateError(f"router fold artifact is not signed: {path}")
    return payload


def _csv_from_fold(payload: Mapping[str, Any], *, artifact: Path) -> Path:
    reference = payload.get("outer_dev_oof")
    if not isinstance(reference, Mapping):
        raise RouterAggregateError("router fold artifact lacks outer_dev_oof reference")
    source = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(source, str) or not isinstance(digest, str):
        raise RouterAggregateError("router fold OOF reference is malformed")
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RouterAggregateError("real router aggregate rejects external OOF files") from error
    if not path.is_file() or _sha256(path) != digest:
        raise RouterAggregateError("router fold OOF digest differs from signed reference")
    return path


def _bucket_id(target: float) -> str:
    if target < 0.0:
        return "negative"
    if target <= 3.0:
        return "crucial"
    if target <= 6.0:
        return "small"
    return "large"


def _canonical_records(records: list[dict[str, str]]) -> str:
    return hash_canonical_json_records(records)


def _contract_hashes(frame: pd.DataFrame) -> dict[str, str]:
    """Recreate the frozen newline-delimited protocol identity hashes."""

    rows = frame.sort_values("token_id", kind="stable")

    def numeric(value: object) -> str:
        return format(float(value), ".17g")

    return {
        "ordered_token_ids_sha256": _canonical_records(
            [{"token_id": str(row.token_id)} for row in rows.itertuples(index=False)]
        ),
        "row_identity_sha256": _canonical_records(
            [
                {
                    "token_id": str(row.token_id),
                    "sequence_id": str(row.sequence_id),
                    "track_id": str(row.track_id),
                }
                for row in rows.itertuples(index=False)
            ]
        ),
        "target_sha256": _canonical_records(
            [
                {"token_id": str(row.token_id), "target_ttc_s": numeric(row.target_ttc)}
                for row in rows.itertuples(index=False)
            ]
        ),
        "mid_sample_weight_sha256": _canonical_records(
            [
                {"token_id": str(row.token_id), "sample_weight": numeric(row.sample_weight)}
                for row in rows.itertuples(index=False)
            ]
        ),
        "fold_assignment_sha256": _canonical_records(
            [
                {
                    "token_id": str(row.token_id),
                    "sequence_id": str(row.sequence_id),
                    "outer_fold": str(int(row.outer_fold)),
                }
                for row in rows.itertuples(index=False)
            ]
        ),
    }


def _metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    candidate = sequence_macro_signed_metrics(
        frame["target_ttc"].to_numpy(),
        frame["prediction_ttc"].to_numpy(),
        frame["sequence_id"].astype(str).to_numpy(),
    )["sequence_macro_paper_MiD_overall"]
    a5 = sequence_macro_signed_metrics(
        frame["target_ttc"].to_numpy(),
        frame["a5_prediction_ttc"].to_numpy(),
        frame["sequence_id"].astype(str).to_numpy(),
    )["sequence_macro_paper_MiD_overall"]
    return {
        "mid_macro_sequence": float(candidate),
        "delta_mid_vs_a5": float(candidate - a5),
        "row_count": int(len(frame)),
    }


def _fold_entries(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    folds: dict[str, Any] = {}
    prediction_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for fold, part in frame.groupby("outer_fold", sort=True):
        label = str(int(fold))
        checkpoint_values = sorted(set(part["checkpoint_sha256"].astype(str)))
        if len(checkpoint_values) != 1:
            raise RouterAggregateError(
                f"router outer fold {label} has multiple router checkpoint hashes"
            )
        prediction_hashes[label] = prediction_sha256(part)
        checkpoint_hashes[label] = checkpoint_values[0]
        folds[label] = {
            "status": "completed",
            "sequence_ids": sorted(part["sequence_id"].astype(str).unique().tolist()),
            "row_count": int(len(part)),
            "prediction_sha256": prediction_hashes[label],
            "checkpoint_sha256": checkpoint_hashes[label],
        }
    return folds, prediction_hashes, checkpoint_hashes


def aggregate(
    *, fold_paths: list[Path], protocol_path: Path, manifest_path: Path, output_dir: Path
) -> dict[str, Any]:
    frozen = verify_frozen_inputs(protocol_path, manifest_path)
    artifacts: list[tuple[Path, dict[str, Any]]] = [
        (path, _read_signed(path)) for path in fold_paths
    ]
    if any(
        payload.get("fixture") is True or payload.get("status") != "completed"
        for _, payload in artifacts
    ):
        raise RouterAggregateError(
            "fixture or incomplete fold evidence cannot enter a real router aggregate"
        )
    if any(
        payload.get("artifact_type") != "scientific_recovery_v8_router_fold_v1"
        for _, payload in artifacts
    ):
        raise RouterAggregateError("router aggregate accepts only V8 router fold artifacts")
    if any(
        payload.get("protocol_sha256") != frozen.protocol.get("artifact_sha256")
        for _, payload in artifacts
    ):
        raise RouterAggregateError("router fold protocol hash differs from the frozen protocol")
    by_fold = {int(payload["outer_fold"]): (path, payload) for path, payload in artifacts}
    if set(by_fold) != {0, 1, 2} or len(by_fold) != len(artifacts):
        raise RouterAggregateError(
            "router aggregate requires exactly one signed fold 0, 1 and 2 artifact"
        )
    expected_configs = frozen.manifest["c1_analysis_plans"]["router_regime"][
        "source_aggregate_contract"
    ]["config_sha256_by_fold"]
    frames: list[pd.DataFrame] = []
    for fold in range(3):
        path, payload = by_fold[fold]
        if payload.get("config_sha256") != expected_configs[str(fold)]:
            raise RouterAggregateError(
                f"router fold {fold} config SHA-256 differs from the frozen contract"
            )
        csv_path = _csv_from_fold(payload, artifact=path)
        frame = validate_oof_frame(pd.read_csv(csv_path), label=f"router fold {fold}")
        if set(frame["outer_fold"].astype(int)) != {fold}:
            raise RouterAggregateError(
                f"router fold {fold} OOF CSV has a different fold assignment"
            )
        frames.append(frame)
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("token_id", kind="stable")
        .reset_index(drop=True)
    )
    if frame["token_id"].duplicated().any():
        raise RouterAggregateError("router OOF folds overlap on token_id")
    sample_contract = frozen.protocol["sample_contract"]
    if len(frame) != int(sample_contract["rows"]):
        raise RouterAggregateError("router aggregate does not contain the frozen 8192 rows")
    a5_source = frozen.protocol.get("sources", {}).get("a5_oof_predictions", {})
    a5_path = ROOT / str(a5_source.get("path", ""))
    if not a5_path.is_file():
        raise RouterAggregateError("frozen A5 OOF control is unavailable")
    baseline_rows = _read_general_rows(a5_path, candidate=False)
    candidate_rows = [
        {
            "token_id": str(row.token_id),
            "sequence_id": str(row.sequence_id),
            "track_id": str(row.track_id),
            "outer_fold": str(int(row.outer_fold)),
            "seed": str(int(row.seed)),
            "target_ttc": format(float(row.target_ttc), ".17g"),
            "sample_weight": format(float(row.sample_weight), ".17g"),
            "prediction_ttc": format(float(row.prediction_ttc), ".17g"),
        }
        for row in frame.itertuples(index=False)
    ]
    try:
        candidate_rows = _bind_candidate_to_baseline_contract(candidate_rows, baseline_rows)
        observed_contract = _general_contract_hashes(candidate_rows)
    except Exception as error:
        raise RouterAggregateError(f"router sample contract mismatch: {error}") from error
    for name, observed in observed_contract.items():
        expected = sample_contract.get(name)
        if observed != expected:
            raise RouterAggregateError(
                f"router aggregate {name} differs from frozen sample contract"
            )
    folds, predictions, checkpoints = _fold_entries(frame)
    expected_folds = {
        str(item["fold"]): sorted(item["dev_sequence_ids"])
        for item in sample_contract["fold_definitions"]
    }
    if {key: value["sequence_ids"] for key, value in folds.items()} != expected_folds:
        raise RouterAggregateError(
            "router aggregate outer-dev sequence coverage differs from frozen folds"
        )
    metrics_all = _metric_record(frame)
    finite = frame["finite"].to_numpy(dtype=bool)
    a5_finite = np.isfinite(frame["a5_prediction_ttc"].to_numpy(dtype=np.float64))
    metrics = {
        "mid_macro_sequence": metrics_all["mid_macro_sequence"],
        "delta_mid_vs_a5": metrics_all["delta_mid_vs_a5"],
        "finite_fraction": float(np.mean(finite)),
        "failure_rate": float(1.0 - np.mean(finite)),
        "coverage_drop_max_pp": float(max(0.0, (np.mean(a5_finite) - np.mean(finite)) * 100.0)),
    }
    bootstrap_raw = hierarchical_sequence_bootstrap(
        frame,
        candidate_prediction_column="prediction_ttc",
        reference_prediction_column="a5_prediction_ttc",
        resamples=5000,
        seed=20260814,
    )["delta_candidate_minus_reference"]
    bootstrap = {
        "probability_delta_lt_zero": float(bootstrap_raw["probability_candidate_lower_mid"]),
        "ci95_low": float(bootstrap_raw["lower_95"]),
        "ci95_high": float(bootstrap_raw["upper_95"]),
        "resamples": 5000,
    }
    per_sequence = {
        str(sequence): _metric_record(part)
        for sequence, part in frame.groupby("sequence_id", sort=True)
    }
    per_bucket = {
        bucket: _metric_record(part)
        for bucket, part in frame.assign(_bucket=frame["target_ttc"].map(_bucket_id)).groupby(
            "_bucket", sort=True
        )
    }
    integrity_checks = {name: True for name in REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS}
    passed = bool(
        metrics["delta_mid_vs_a5"] <= -3.0
        and bootstrap["probability_delta_lt_zero"] >= 0.90
        and metrics["finite_fraction"] == 1.0
        and metrics["failure_rate"] == 0.0
        and metrics["coverage_drop_max_pp"] <= 1.0
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = output_dir / "router_oof_predictions.csv"
    frame.to_csv(combined, index=False, lineterminator="\n")
    artifact = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_router_seed7_aggregate_v1",
            "schema_version": frozen.protocol["schema_version"],
            "status": "completed",
            "stage": "router",
            "arm": "router",
            "candidate_id": "R",
            "git_commit": frozen.protocol["git_base_commit"],
            "implementation_git_commit": _current_commit(),
            "protocol_sha256": frozen.protocol["artifact_sha256"],
            "protocol_file_sha256": _sha256(protocol_path),
            "config_sha256": expected_configs,
            "seed": 7,
            "folds": folds,
            "row_count": int(len(frame)),
            "row_identity_sha256": sample_contract["row_identity_sha256"],
            "target_identity_sha256": sample_contract["target_identity_sha256"],
            "target_sha256": sample_contract["target_sha256"],
            "mid_sample_weight_sha256": sample_contract["mid_sample_weight_sha256"],
            "fold_assignment_sha256": sample_contract["fold_assignment_sha256"],
            "prediction_sha256": predictions,
            "checkpoint_sha256": checkpoints,
            "oof_csv": {"path": combined.relative_to(ROOT).as_posix(), "sha256": _sha256(combined)},
            "metrics": metrics,
            "per_sequence": per_sequence,
            "per_bucket": per_bucket,
            "bootstrap": bootstrap,
            "integrity_checks": integrity_checks,
            "coverage": {
                "outer_folds": [0, 1, 2],
                "sequences_by_outer_fold": expected_folds,
                "sealed_evaluation_closed": True,
            },
            "gate_decision": {
                "passed": passed,
                "candidate_id": "R",
                "rule": "frozen_ttc_candidate_gate",
            },
            "sample_contract": sample_contract,
        }
    )
    destination = output_dir / "router_seed7_aggregate.json"
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/router"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if not args.fold_artifact:
            raise RouterAggregateError("router aggregate requires signed fold artifacts")
        if args.dry_run:
            frozen = verify_frozen_inputs(args.protocol, args.manifest)
            print(
                json.dumps(
                    {"status": "dry_run", "protocol_sha256": frozen.protocol["artifact_sha256"]},
                    sort_keys=True,
                )
            )
            return 0
        result = aggregate(
            fold_paths=args.fold_artifact,
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "artifact": result["artifact_sha256"],
                    "gate_passed": result["gate_decision"]["passed"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, V8IntegrityError, RouterAggregateError) as error:
        parser.exit(2, f"V8 router aggregate failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
