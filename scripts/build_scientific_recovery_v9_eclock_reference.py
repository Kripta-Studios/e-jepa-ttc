#!/usr/bin/env python
"""Recompute and sign the canonical E-Clock X0 protocol and references.

Only frozen train/OOF evidence is read. No external evaluation split is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_protocol import canonical_records_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, sequence_macro_signed_metrics

EXPECTED_V8_ZIP_SHA256 = "8abab43e0fbef70252e7c3fd00111e11ca77daba908fc0507ae36876334602ac"
PARENT_COMMIT = "718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2"
X0_INITIAL_COMMIT = "1fd4e592887d4812c6abd9c4a48a8abd3eea0f0f"
REFERENCE_FAMILIES = (
    "official_a5_oof",
    "official_c2f_oof",
    "nested_router_retrained_a5_constituent",
    "nested_router_retrained_c2f_constituent",
    "prospective_router_r",
)


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    value = frame[name]
    if not isinstance(value, pd.Series):
        raise TypeError(f"expected exactly one column named {name}")
    return value


OFFICIAL_A5_RELATIVE = Path("artifacts/scientific_recovery_v7/baselines/a5_oof_predictions.csv")
OFFICIAL_A5_MANIFEST_RELATIVE = Path("artifacts/scientific_recovery_v7/baselines/manifest.json")
OFFICIAL_C2F_ARTIFACT_RELATIVE = Path("artifacts/scientific_recovery_v7/results/c2f_seed7_oof.json")
ROUTER_OOF_RELATIVE = Path(
    "artifacts/scientific_recovery_v8/results/router/aggregate_seed7/router_oof_predictions.csv"
)
ROUTER_AGGREGATE_RELATIVE = Path(
    "artifacts/scientific_recovery_v8/results/router/aggregate_seed7/router_seed7_aggregate.json"
)
CACHE_MANIFEST_RELATIVE = Path(
    "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
)


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


def _exact_frozen_schema(payload: dict[str, Any], *, schema_id: str) -> dict[str, Any]:
    """Return a closed schema for one immutable signed protocol/reference."""

    properties = {
        key: (
            {"type": "string", "pattern": "^[0-9a-f]{64}$"}
            if key == "artifact_sha256"
            else {"const": value}
        )
        for key, value in payload.items()
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(payload),
        "properties": properties,
    }


def _signed_json(path: Path, *, artifact_type: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"signed artifact is invalid: {path}")
    if artifact_type is not None and payload.get("artifact_type") != artifact_type:
        raise ValueError(f"artifact type mismatch: {path}")
    return payload


def _physical(root: Path, relative: Path, *, semantic_identity: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative.as_posix(),
        "file_sha256": compute_file_hash(str(path)),
        "bytes": path.stat().st_size,
        "semantic_identity": semantic_identity,
    }


def _verify_manifest(archive: zipfile.ZipFile, manifest: dict[str, Any]) -> str:
    if manifest.get("source_git_commit") != PARENT_COMMIT:
        raise ValueError("V8 package parent commit mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1517:
        raise ValueError("V8 manifest file count mismatch")
    digest = hashlib.sha256()
    for entry in files:
        relative = str(entry["path"])
        try:
            member = archive.getinfo(relative)
        except KeyError as error:
            raise FileNotFoundError(f"V8 package member missing: {relative}") from error
        member_digest = hashlib.sha256()
        with archive.open(member) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                member_digest.update(chunk)
        observed = member_digest.hexdigest()
        if observed != entry["sha256"] or member.file_size != int(entry["bytes"]):
            raise ValueError(f"V8 manifest member mismatch: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(observed.encode("ascii"))
    return digest.hexdigest()


def _normalized_official_a5(source_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_root / OFFICIAL_A5_RELATIVE)
    required = {
        "sample_token",
        "sequence_id",
        "track_id",
        "target_ttc_s",
        "point_prediction_ttc_s",
        "fold",
        "seed",
    }
    if not required.issubset(frame.columns):
        raise ValueError("official A5 OOF schema mismatch")
    return frame.loc[:, sorted(required)].rename(
        columns={"point_prediction_ttc_s": "prediction_ttc_s", "fold": "outer_fold"}
    )


def _normalized_official_c2f(
    source_root: Path, artifact: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[Path]]:
    sources = artifact.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("official C2F sources are missing")
    predictions = sources.get("fold_predictions")
    if not isinstance(predictions, list) or len(predictions) != 3:
        raise ValueError("official C2F requires exactly three fold prediction files")
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for expected_fold, record in enumerate(predictions):
        if not isinstance(record, Mapping):
            raise ValueError("official C2F prediction record is invalid")
        physical_path = Path(str(record["path"]))
        try:
            relative = physical_path.resolve().relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError("official C2F path escapes source root") from error
        path = source_root / relative
        if compute_file_hash(str(path)) != record.get("sha256"):
            raise ValueError("official C2F fold prediction SHA mismatch")
        frame = pd.read_csv(path)
        required = {
            "sample_token",
            "sequence_id",
            "track_id",
            "target_ttc_s",
            "point_prediction_ttc_s",
            "fold",
            "seed",
        }
        if not required.issubset(frame.columns):
            raise ValueError("official C2F fold OOF schema mismatch")
        observed_folds = set(
            np.asarray(pd.to_numeric(_series(frame, "fold"), errors="raise"), dtype=np.int64)
        )
        if observed_folds != {expected_fold}:
            raise ValueError("official C2F fold file identity mismatch")
        frames.append(
            frame.loc[:, sorted(required)].rename(
                columns={"point_prediction_ttc_s": "prediction_ttc_s", "fold": "outer_fold"}
            )
        )
        paths.append(relative)
    return pd.concat(frames, ignore_index=True), paths


def _normalized_router(source_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_root / ROUTER_OOF_RELATIVE)
    required = {
        "token_id",
        "sequence_id",
        "track_id",
        "outer_fold",
        "seed",
        "target_ttc",
        "sample_weight",
        "prediction_ttc",
        "a5_prediction_ttc",
        "c2f_prediction_ttc",
    }
    if not required.issubset(frame.columns):
        raise ValueError("nested Router OOF schema mismatch")
    return frame.rename(columns={"token_id": "sample_token", "target_ttc": "target_ttc_s"})


def _canonicalize_family(frame: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "sample_token",
        "sequence_id",
        "track_id",
        "outer_fold",
        "seed",
        "target_ttc_s",
        "prediction_ttc_s",
    )
    if not set(fields).issubset(frame.columns):
        raise ValueError("reference family normalized schema mismatch")
    observed = frame.loc[:, fields].copy()
    if len(observed) != 8192 or observed["sample_token"].nunique(dropna=False) != 8192:
        raise ValueError("reference family requires 8,192 unique tokens")
    expected = canonical.drop(columns="prediction_ttc_s")
    merged = observed.merge(
        expected,
        on="sample_token",
        how="outer",
        suffixes=("_source", "_canonical"),
        indicator=True,
        validate="one_to_one",
    )
    if set(merged["_merge"]) != {"both"}:
        raise ValueError("reference family token universe mismatch")
    for column in ("sequence_id", "track_id", "outer_fold", "seed"):
        if not bool((merged[f"{column}_source"] == merged[f"{column}_canonical"]).all()):
            raise ValueError(f"reference family {column} mismatch")
    source_target = np.asarray(
        pd.to_numeric(_series(merged, "target_ttc_s_source"), errors="coerce"),
        dtype=np.float64,
    )
    canonical_target = np.asarray(
        pd.to_numeric(_series(merged, "target_ttc_s_canonical"), errors="coerce"),
        dtype=np.float64,
    )
    if not np.isfinite(source_target).all() or not np.allclose(
        source_target, canonical_target, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("reference family target mismatch")
    result = pd.DataFrame(
        {
            "sample_token": merged["sample_token"].astype(str),
            "sequence_id": merged["sequence_id_canonical"].astype(str),
            "track_id": merged["track_id_canonical"].astype(str),
            "outer_fold": merged["outer_fold_canonical"].astype(np.int64),
            "seed": merged["seed_canonical"].astype(np.int64),
            "target_ttc_s": canonical_target,
            "source_target_ttc_s": source_target,
            "prediction_ttc_s": np.asarray(
                pd.to_numeric(_series(merged, "prediction_ttc_s"), errors="coerce"),
                dtype=np.float64,
            ),
            "sample_weight": np.asarray(
                pd.to_numeric(_series(merged, "sample_weight"), errors="coerce"),
                dtype=np.float64,
            ),
        }
    )
    return result.sort_values("sample_token", kind="stable").reset_index(drop=True)


def _family_facts(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.loc[
        :, ("outer_fold", "seed", "target_ttc_s", "prediction_ttc_s", "sample_weight")
    ].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(numeric).all())
    if not finite:
        raise ValueError("reference family contains non-finite values")
    result = sequence_macro_signed_metrics(
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        frame["prediction_ttc_s"].to_numpy(dtype=np.float64),
        frame["sequence_id"].astype(str).to_numpy(),
    )
    per_sequence = result["per_sequence"]
    mid = sum(
        float(per_sequence[sequence]["paper_MiD_overall"]) for sequence in sorted(per_sequence)
    ) / len(per_sequence)
    if not math.isfinite(mid):
        raise ValueError("reference family MiD is non-finite")
    return {
        "row_count": len(frame),
        "token_identity_sha256": canonical_records_hash(
            frame, ("sample_token", "sequence_id", "track_id")
        ),
        "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc_s")),
        "source_target_sha256": canonical_records_hash(
            frame, ("sample_token", "source_target_ttc_s")
        ),
        "fold_assignment_sha256": canonical_records_hash(
            frame, ("sample_token", "sequence_id", "outer_fold")
        ),
        "sample_weight_sha256": canonical_records_hash(frame, ("sample_token", "sample_weight")),
        "prediction_sha256": canonical_records_hash(frame, ("sample_token", "prediction_ttc_s")),
        "protocol_or_producer_identity": "garl_signed_v1_sequence_macro_point_full_coverage",
        "recomputed_mid": mid,
        "finite": finite,
        "coverage_fraction": 1.0,
        "folds": sorted(frame["outer_fold"].unique().astype(int).tolist()),
        "sequence_ids": sorted(frame["sequence_id"].astype(str).unique().tolist()),
    }


def _official_a5_checkpoints(
    source_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("official A5 checkpoint sources are missing")
    result: list[dict[str, Any]] = []
    for fold in range(3):
        record = sources.get(f"a5_fold{fold}")
        if not isinstance(record, Mapping):
            raise ValueError("official A5 fold checkpoint record is missing")
        path = Path(str(record["checkpoint"]))
        try:
            relative = path.resolve().relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError("official A5 checkpoint escapes source root") from error
        physical = _physical(
            source_root,
            relative,
            semantic_identity=f"official_a5_fold{fold}_checkpoint",
        )
        if physical["file_sha256"] != record.get("checkpoint_sha256"):
            raise ValueError("official A5 checkpoint physical SHA mismatch")
        result.append({"outer_fold": fold, **physical})
    return result


def _source_frames(
    source_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[Path],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    a5_manifest = _signed_json(source_root / OFFICIAL_A5_MANIFEST_RELATIVE)
    c2f_artifact = _signed_json(source_root / OFFICIAL_C2F_ARTIFACT_RELATIVE)
    router_aggregate = _signed_json(source_root / ROUTER_AGGREGATE_RELATIVE)
    a5 = _normalized_official_a5(source_root)
    c2f, c2f_paths = _normalized_official_c2f(source_root, c2f_artifact)
    router = _normalized_router(source_root)
    canonical = a5.copy()
    canonical["sample_weight"] = (
        canonical[["sample_token"]]
        .merge(
            router[["sample_token", "sample_weight"]],
            on="sample_token",
            how="left",
            validate="one_to_one",
        )["sample_weight"]
        .to_numpy()
    )
    return canonical, c2f, router, c2f_paths, a5_manifest, c2f_artifact, router_aggregate


def build_protocol(source_root: Path) -> dict[str, Any]:
    """Derive the immutable production universe from frozen signed evidence."""

    canonical, _c2f, _router, _paths, _a5_manifest, _c2f_artifact, _router_aggregate = (
        _source_frames(source_root)
    )
    checked = _canonicalize_family(canonical, canonical)
    facts = _family_facts(checked)
    sequence_to_fold: dict[str, int] = {}
    for sequence, values in (
        checked.groupby("sequence_id", sort=True)["outer_fold"].unique().items()
    ):
        if len(values) != 1:
            raise ValueError("canonical sequence appears in more than one fold")
        sequence_to_fold[str(sequence)] = int(values[0])
    if len(sequence_to_fold) != 9:
        raise ValueError("canonical sequence-to-fold assignment is not one-to-one")
    bucket_counts: dict[str, dict[str, int]] = {}
    for sequence, subset in checked.groupby("sequence_id", sort=True):
        target = subset["target_ttc_s"].to_numpy(dtype=np.float64)
        bucket_counts[str(sequence)] = {
            name: int(np.count_nonzero((target > lower) & (target <= upper)))
            for name, lower, upper in BUCKETS
        }
        if any(value <= 0 for value in bucket_counts[str(sequence)].values()):
            raise ValueError("canonical sequence lacks a required TTC bucket")

    cache_path = source_root / CACHE_MANIFEST_RELATIVE
    cache_manifest = _signed_json(cache_path, artifact_type="garlttc_official_lhr_object_cache_v4")
    train_shards = [entry for entry in cache_manifest["shards"] if entry.get("split") == "train"]
    if len(train_shards) != 32 or sum(int(entry["count"]) for entry in train_shards) != 8192:
        raise ValueError("canonical cache does not contain exactly 32 train shards / 8,192 rows")
    split_path = Path(str(cache_manifest["split_path"]))
    if (
        not split_path.is_file()
        or compute_file_hash(str(split_path)) != cache_manifest["split_sha256"]
    ):
        raise ValueError("canonical split manifest physical identity mismatch")

    fold_summaries = []
    for fold in range(3):
        dev = cast(pd.DataFrame, checked.loc[_series(checked, "outer_fold") == fold, :])
        train = cast(pd.DataFrame, checked.loc[_series(checked, "outer_fold") != fold, :])
        fold_summaries.append(
            {
                "outer_fold": fold,
                "dev_rows": len(dev),
                "train_rows": len(train),
                "dev_token_subset_sha256": canonical_records_hash(dev, ("sample_token",)),
                "train_token_subset_sha256": canonical_records_hash(train, ("sample_token",)),
                "dev_sequence_ids": sorted(_series(dev, "sequence_id").unique().tolist()),
                "train_sequence_ids": sorted(_series(train, "sequence_id").unique().tolist()),
            }
        )

    return sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_eclock_protocol_v2",
            "protocol_version": "eclock-x0-v2",
            "parent_git_commit": PARENT_COMMIT,
            "x0_initial_git_commit": X0_INITIAL_COMMIT,
            "authorized_seed": 7,
            "production_row_count": 8192,
            "canonical_sequence_ids": facts["sequence_ids"],
            "canonical_sequence_to_fold": sequence_to_fold,
            "canonical_outer_folds": [0, 1, 2],
            "canonical_bucket_counts_by_sequence": bucket_counts,
            "canonical_hashes": {
                "token_identity_sha256": facts["token_identity_sha256"],
                "target_sha256": facts["target_sha256"],
                "fold_assignment_sha256": facts["fold_assignment_sha256"],
                "sample_weight_sha256": facts["sample_weight_sha256"],
            },
            "fold_summaries": fold_summaries,
            "cache_binding": {
                "path": CACHE_MANIFEST_RELATIVE.as_posix(),
                "file_sha256": compute_file_hash(str(cache_path)),
                "artifact_sha256": cache_manifest["artifact_sha256"],
                "bytes": cache_path.stat().st_size,
                "artifact_type": cache_manifest["artifact_type"],
                "schema_version": cache_manifest["schema_version"],
                "preprocessing_version": cache_manifest["input_schema"]["version"],
                "train_shards": [
                    {
                        "path": str(entry["path"]),
                        "file_sha256": str(entry["sha256"]),
                        "bytes": int(entry["size_bytes"]),
                        "row_count": int(entry["count"]),
                    }
                    for entry in train_shards
                ],
            },
            "split_binding": {
                "path": "data/splits/eap_pilot12_v1.json",
                "file_sha256": cache_manifest["split_sha256"],
                "bytes": split_path.stat().st_size,
                "producer_identity": "frozen_eap_pilot12_v1",
            },
            "metric": {
                "protocol": "garl_signed_v1",
                "metric_delta_t_s": 0.1,
                "minimum_abs_prediction_ttc_s": 0.1,
                "deployment_ttc_clip_seconds": 60.0,
                "target_anchor": "t2_start_anchor_benchmark_phase",
                "scientific_coordinate": "predicted_benchmark_phase_float64",
                "zero_failure_is_partially_assisted_by_output_domain": True,
                "deployment_clipping_not_used_for_scientific_metric": True,
            },
            "bootstrap": {
                "method": "paired_hierarchical_sequence_then_track_cluster_bootstrap",
                "seed": 20260814,
                "draws": 5000,
            },
            "gates": {
                "reference_family": "official_a5_oof",
                "a5_replay_mid_tolerance": 1.0e-9,
                "delta_mid_vs_official_a5_oof_max": -3.0,
                "bootstrap_probability_delta_lt_zero_min": 0.90,
                "bootstrap_ci95_high_below_zero_required": True,
                "finite_fraction_required": 1.0,
                "failure_rate_required": 0.0,
            },
            "checkpoint_policy": "last_update_fixed_budget",
            "primary_comparison": "X0-DYN-U_vs_X0-BASE-U",
            "official_a5_reference_family": "official_a5_oof",
            "executable_arm_registry": [
                "X0-A5-REPLAY",
                "X0-PAIR-U",
                "X0-BASE-U",
                "X0-DYN-U",
            ],
            "global_transport_feature_names": [
                "translation_x",
                "translation_y",
                "divergence_x",
                "divergence_y",
                "divergence_isotropic",
                "flow_magnitude",
                "confidence_margin",
                "entropy",
                "cycle_error",
            ],
            "foreground_weight_used": False,
            "upstream_roi_is_box_conditioned": True,
            "explicit_foreground_height_interface_bypassed": True,
            "official_failure_region_excluded_by_parameterization": True,
            "sealed_evaluation": "closed",
            "x0_dyn_w_execution_authorized": False,
            "x0_dyn_w_status": "not_executed",
            "x0_dyn_w_loss_reduction": "normalized_weighted_absolute_phase_error",
            "future_coordinate_control": "documented_not_implemented",
            "maximum_future_claim": (
                "En el universo grouped-development y presupuesto preregistrado, conservar los "
                "nueve resúmenes de correspondencia global uniforme mejora frente al control que "
                "calcula y anula esos mismos slots, cuando ambos predicen benchmark phase y omiten "
                "la interfaz explícita de altura."
            ),
        }
    )


def build_reference(source_root: Path, *, protocol_path: Path, v8_zip: Path) -> dict[str, Any]:
    """Recompute exactly five disjoint reference families and their provenance."""

    protocol = _signed_json(
        protocol_path, artifact_type="scientific_recovery_v9_eclock_protocol_v2"
    )
    if compute_file_hash(str(v8_zip)) != EXPECTED_V8_ZIP_SHA256:
        raise ValueError("V8 package physical SHA-256 mismatch")
    with zipfile.ZipFile(v8_zip) as archive:
        member = "artifacts/packages/e-jepa-ttc-v8-essential-results-20260903.manifest.json"
        manifest_bytes = archive.read(member)
        manifest = json.loads(manifest_bytes)
        member_set_sha256 = _verify_manifest(archive, manifest)
    manifest_path = source_root / member
    if compute_file_hash(str(manifest_path)) != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("extracted V8 manifest differs physically from package manifest")

    canonical, c2f, router, c2f_paths, a5_manifest, c2f_artifact, router_aggregate = _source_frames(
        source_root
    )
    families: dict[str, dict[str, Any]] = {}
    family_inputs: tuple[tuple[str, pd.DataFrame, str, str], ...] = (
        (
            "official_a5_oof",
            canonical,
            "official_v7_point_full_coverage_oof",
            "Official A5 OOF comparator and the only X0-A5-REPLAY checkpoint family.",
        ),
        (
            "official_c2f_oof",
            c2f,
            "official_v7_c2f_seed7_fold_chain_oof",
            "Official C2F seed-7 OOF comparator; independent of nested Router retraining.",
        ),
        (
            "nested_router_retrained_a5_constituent",
            router.rename(columns={"a5_prediction_ttc": "prediction_ttc_s"}),
            "v8_nested_router_fold_local_retrained_a5_constituent",
            "A5 constituent retrained inside nested Router; never acceptable as official A5.",
        ),
        (
            "nested_router_retrained_c2f_constituent",
            router.rename(columns={"c2f_prediction_ttc": "prediction_ttc_s"}),
            "v8_nested_router_fold_local_retrained_c2f_constituent",
            "C2F constituent retrained inside nested Router; not the official C2F OOF family.",
        ),
        (
            "prospective_router_r",
            router.rename(columns={"prediction_ttc": "prediction_ttc_s"}),
            "v8_prospective_router_r_seed7_oof",
            "Prospective Router R output over nested constituents; "
            "distinct from both constituents.",
        ),
    )
    for name, frame, artifact_type, relation in family_inputs:
        checked = _canonicalize_family(frame, canonical)
        families[name] = {
            "reference_family": name,
            "artifact_type": artifact_type,
            **_family_facts(checked),
            "exact_relation": relation,
        }

    families["official_a5_oof"].update(
        {
            "path": OFFICIAL_A5_RELATIVE.as_posix(),
            "physical_references": [
                _physical(
                    source_root, OFFICIAL_A5_RELATIVE, semantic_identity="official_a5_oof_csv"
                )
            ],
            "artifact_reference": {
                **_physical(
                    source_root,
                    OFFICIAL_A5_MANIFEST_RELATIVE,
                    semantic_identity="official_a5_baseline_manifest",
                ),
                "artifact_sha256": a5_manifest["artifact_sha256"],
            },
            "official_fold_checkpoints": _official_a5_checkpoints(source_root, a5_manifest),
        }
    )
    families["official_c2f_oof"].update(
        {
            "path": OFFICIAL_C2F_ARTIFACT_RELATIVE.as_posix(),
            "physical_references": [
                _physical(
                    source_root, relative, semantic_identity=f"official_c2f_fold{fold}_oof_csv"
                )
                for fold, relative in enumerate(c2f_paths)
            ],
            "artifact_reference": {
                **_physical(
                    source_root,
                    OFFICIAL_C2F_ARTIFACT_RELATIVE,
                    semantic_identity="official_c2f_seed7_aggregate",
                ),
                "artifact_sha256": c2f_artifact["artifact_sha256"],
            },
        }
    )
    router_physical = _physical(
        source_root, ROUTER_OOF_RELATIVE, semantic_identity="v8_nested_router_seed7_oof_csv"
    )
    router_artifact_reference = {
        **_physical(
            source_root, ROUTER_AGGREGATE_RELATIVE, semantic_identity="v8_router_seed7_aggregate"
        ),
        "artifact_sha256": router_aggregate["artifact_sha256"],
    }
    for name in REFERENCE_FAMILIES[2:]:
        families[name].update(
            {
                "path": ROUTER_OOF_RELATIVE.as_posix(),
                "physical_references": [router_physical],
                "artifact_reference": router_artifact_reference,
            }
        )
    if tuple(families) != REFERENCE_FAMILIES:
        raise RuntimeError("reference family registry drifted")

    return sign_artifact(
        {
            "artifact_type": "eclock_x0_reference_v2",
            "evidence_class": "reference",
            "scientific_result": False,
            "parent_git_commit": PARENT_COMMIT,
            "x0_initial_git_commit": X0_INITIAL_COMMIT,
            "protocol": {
                "path": "configs/protocol/scientific_recovery_v9_eclock_x0.json",
                "file_sha256": compute_file_hash(str(protocol_path)),
                "artifact_sha256": protocol["artifact_sha256"],
                "bytes": protocol_path.stat().st_size,
                "producer_identity": "eclock-x0-v2-canonical-protocol",
            },
            "reference_family_registry": list(REFERENCE_FAMILIES),
            "families": families,
            "x0_a5_replay_reference_family": "official_a5_oof",
            "x0_pair_u_checkpoint_family": "official_a5_oof",
            "gate_reference_family": "official_a5_oof",
            "cache_binding": protocol["cache_binding"],
            "split_binding": protocol["split_binding"],
            "v8_package": {
                "path": "artifacts/packages/e-jepa-ttc-v8-essential-results-20260903.zip",
                "file_sha256": EXPECTED_V8_ZIP_SHA256,
                "bytes": v8_zip.stat().st_size,
                "producer_identity": "immutable_v8_essential_results_package",
            },
            "v8_manifest": {
                "path": member,
                "file_sha256": compute_file_hash(str(manifest_path)),
                "bytes": manifest_path.stat().st_size,
                "producer_identity": "immutable_v8_1517_member_manifest",
                "verified_member_count": 1517,
                "member_set_sha256": member_set_sha256,
            },
            "v8_seed23_status": manifest["seed23_status"],
            "sealed_evaluation": "closed",
            "zero_failure_is_partially_assisted_by_output_domain": True,
            "deployment_clipping_not_used_for_scientific_metric": True,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v8-zip", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument("--protocol-schema-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-schema-output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    protocol_path = (
        args.protocol_output.resolve()
        if args.protocol_output is not None
        else repo_root / "configs/protocol/scientific_recovery_v9_eclock_x0.json"
    )
    protocol = build_protocol(source_root)
    if not args.verify_only:
        _atomic_json(protocol, protocol_path)
    elif (
        not protocol_path.is_file()
        or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol
    ):
        raise ValueError("committed protocol differs from recomputed canonical protocol")
    if args.protocol_schema_output is not None:
        protocol_schema = _exact_frozen_schema(
            protocol,
            schema_id="scientific_recovery_v9_eclock_protocol_v2.schema.json",
        )
        protocol_schema_path = args.protocol_schema_output.resolve()
        if args.verify_only:
            if json.loads(protocol_schema_path.read_text(encoding="utf-8")) != protocol_schema:
                raise ValueError("committed protocol schema differs from generated closed schema")
        else:
            _atomic_json(protocol_schema, protocol_schema_path)
    reference = build_reference(
        source_root, protocol_path=protocol_path, v8_zip=args.v8_zip.resolve()
    )
    if args.output is not None:
        output = args.output.resolve()
        if args.verify_only:
            if not output.is_file() or json.loads(output.read_text(encoding="utf-8")) != reference:
                raise ValueError("committed reference differs from recomputed canonical reference")
        else:
            _atomic_json(reference, output)
    if args.reference_schema_output is not None:
        reference_schema = _exact_frozen_schema(
            reference,
            schema_id="scientific_recovery_v9_eclock_reference_v2.schema.json",
        )
        reference_schema_path = args.reference_schema_output.resolve()
        if args.verify_only:
            if json.loads(reference_schema_path.read_text(encoding="utf-8")) != reference_schema:
                raise ValueError("committed reference schema differs from generated closed schema")
        else:
            _atomic_json(reference_schema, reference_schema_path)
    print(json.dumps(reference, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
