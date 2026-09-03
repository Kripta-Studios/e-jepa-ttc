#!/usr/bin/env python
"""Verify immutable V8 evidence and build the signed E-Clock X0 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, sequence_macro_signed_metrics

EXPECTED_V8_ZIP_SHA256 = "8abab43e0fbef70252e7c3fd00111e11ca77daba908fc0507ae36876334602ac"
PARENT_COMMIT = "718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_manifest(repo_root: Path, manifest: dict[str, Any]) -> str:
    if manifest.get("source_git_commit") != PARENT_COMMIT:
        raise ValueError("V8 package parent commit mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1517:
        raise ValueError("V8 manifest file count mismatch")
    digest = hashlib.sha256()
    for entry in files:
        relative = str(entry["path"])
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V8 manifest member missing: {relative}")
        observed = compute_file_hash(str(path))
        if observed != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"V8 manifest member mismatch: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(observed.encode("ascii"))
    return digest.hexdigest()


def _recompute_available_baselines(repo_root: Path) -> dict[str, Any]:
    aggregate_path = (
        repo_root
        / "artifacts/scientific_recovery_v8/results/router/aggregate_seed7/"
        "router_seed7_aggregate.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(aggregate):
        raise ValueError("V8 router aggregate signature mismatch")
    oof_path = repo_root / aggregate["oof_csv"]["path"]
    if compute_file_hash(str(oof_path)) != aggregate["oof_csv"]["sha256"]:
        raise ValueError("V8 router OOF physical SHA-256 mismatch")
    frame = pd.read_csv(oof_path)
    required = {
        "token_id",
        "sequence_id",
        "outer_fold",
        "target_ttc",
        "prediction_ttc",
        "a5_prediction_ttc",
        "c2f_prediction_ttc",
        "sample_weight",
    }
    if len(frame) != 8192 or not required.issubset(frame.columns):
        raise ValueError("V8 router OOF row/schema contract mismatch")
    tokens = np.asarray(frame["token_id"], dtype=object)
    if bool(pd.isna(tokens).any()) or len(set(tokens.tolist())) != len(tokens):
        raise ValueError("V8 router OOF token identity is incomplete or duplicated")
    folds = np.asarray(pd.to_numeric(frame["outer_fold"], errors="coerce"), dtype=np.float64)
    if not np.isfinite(folds).all() or set(folds.astype(int).tolist()) != {0, 1, 2}:
        raise ValueError("V8 router OOF fold universe mismatch")
    numeric = (
        "target_ttc",
        "prediction_ttc",
        "a5_prediction_ttc",
        "c2f_prediction_ttc",
        "sample_weight",
    )
    numeric_arrays = {
        column: np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=np.float64)
        for column in numeric
    }
    if any(not np.isfinite(values).all() for values in numeric_arrays.values()):
        raise ValueError("V8 baseline OOF contains non-finite target/prediction/weight")
    target = numeric_arrays["target_ttc"]
    sequences = np.asarray(frame["sequence_id"], dtype=str)
    required_buckets = {name for name, _lower, _upper in BUCKETS}
    for sequence in sorted(set(sequences)):
        values = target[sequences == sequence]
        observed = {
            name
            for name, lower, upper in BUCKETS
            if bool(np.any((values > lower) & (values <= upper)))
        }
        if observed != required_buckets:
            raise ValueError(f"V8 baseline sequence lacks a TTC bucket: {sequence}")
    metrics: dict[str, float] = {}
    for name, column in (
        ("router_r", "prediction_ttc"),
        ("a5", "a5_prediction_ttc"),
        ("c2f", "c2f_prediction_ttc"),
    ):
        result = sequence_macro_signed_metrics(
            target,
            numeric_arrays[column],
            sequences,
        )
        value = float(result["sequence_macro_paper_MiD_overall"])
        if not np.isfinite(value):
            raise ValueError(f"recomputed V8 baseline MiD is non-finite: {name}")
        metrics[name] = value
    if not np.isclose(metrics["router_r"], aggregate["metrics"]["mid_macro_sequence"]):
        raise ValueError("recomputed V8 router MiD disagrees with signed aggregate")
    if not np.isclose(
        metrics["router_r"] - metrics["a5"],
        aggregate["metrics"]["delta_mid_vs_a5"],
    ):
        raise ValueError("recomputed V8 A5 delta disagrees with signed aggregate")
    return {
        "source_aggregate_file_sha256": compute_file_hash(str(aggregate_path)),
        "source_aggregate_artifact_sha256": aggregate["artifact_sha256"],
        "source_oof_file_sha256": aggregate["oof_csv"]["sha256"],
        "row_count": 8192,
        "folds": [0, 1, 2],
        "sequence_count": len(set(sequences)),
        "recomputed_sequence_macro_mid": metrics,
        "garl_physical_predictions_available": False,
    }


def build_reference(repo_root: Path, v8_zip: Path) -> dict[str, Any]:
    protocol_path = repo_root / "configs/protocol/scientific_recovery_v9_eclock_x0.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(protocol):
        raise ValueError("E-Clock protocol signature mismatch")
    if compute_file_hash(str(v8_zip)) != EXPECTED_V8_ZIP_SHA256:
        raise ValueError("V8 package physical SHA-256 mismatch")
    with zipfile.ZipFile(v8_zip) as archive:
        member = "artifacts/packages/e-jepa-ttc-v8-essential-results-20260903.manifest.json"
        manifest_bytes = archive.read(member)
        manifest = json.loads(manifest_bytes)
    manifest_path = repo_root / member
    if compute_file_hash(str(manifest_path)) != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("extracted V8 manifest differs physically from package manifest")
    member_set_sha256 = _verify_manifest(repo_root, manifest)
    baselines = _recompute_available_baselines(repo_root)
    return sign_artifact(
        {
            "artifact_type": "eclock_x0_reference_v1",
            "arm_id": "REFERENCE",
            "evidence_class": "reference",
            "scientific_result": False,
            "parent_git_commit": PARENT_COMMIT,
            "protocol_file_sha256": compute_file_hash(str(protocol_path)),
            "protocol_artifact_sha256": protocol["artifact_sha256"],
            "v8_zip_sha256": EXPECTED_V8_ZIP_SHA256,
            "v8_manifest_file_sha256": compute_file_hash(str(manifest_path)),
            "v8_member_set_sha256": member_set_sha256,
            "v8_files_verified": 1517,
            "v8_seed23_status": manifest["seed23_status"],
            "v8_final_multiseed_aggregate_present": False,
            "a5_physical_checkpoints_in_package": False,
            "available_v8_baselines": baselines,
            "sealed_paths_resolved": False,
            "upstream_roi_is_box_conditioned": True,
            "explicit_foreground_height_interface_bypassed": True,
            "loss_reduction": "not_applicable_reference",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--v8-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    reference = build_reference(args.repo_root.resolve(), args.v8_zip.resolve())
    if args.output is not None and not args.verify_only:
        _atomic_json(reference, args.output.resolve())
    print(json.dumps(reference, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
