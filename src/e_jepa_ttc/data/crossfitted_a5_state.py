"""Strict nested A5 state assembly for Stage 64 residual training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from e_jepa_ttc.artifacts.cache_components import component_record, verify_components
from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact
from e_jepa_ttc.data.nested_provenance import (
    validate_frozen_teacher_dependency,
    validate_nested_split,
    validate_uninitialized_producer_contract,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache


@dataclass(frozen=True)
class A5StateSplit:
    """Label-free A5 state and row identity for one train/eval role."""

    state: np.ndarray
    metadata: pd.DataFrame


def _read_signed_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(value):
        raise ValueError(f"artifact signature mismatch: {path}")
    return value


def _producer_state(
    *,
    feature_cache_base: Path,
    expert_root: Path,
    expected_role: str,
    coherent_state_root: Path | None = None,
    frozen_teacher_audit: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    arrays, metadata, cache_manifest = load_feature_cache(feature_cache_base.with_suffix(".npz"))
    artifact = _read_signed_artifact(expert_root / "expert_artifact.json")
    protocol = _read_signed_artifact(expert_root / "nested_protocol.json")
    oof_path = expert_root / "expert_oof.csv"
    if compute_file_hash(str(oof_path)) != artifact["oof_csv"]["sha256"]:
        raise ValueError(f"expert OOF SHA mismatch: {oof_path}")
    checkpoint = expert_root / "train" / "model_best.pt"
    expected_checkpoint = str(artifact["checkpoint"]["sha256"])
    if compute_file_hash(str(checkpoint)) != expected_checkpoint:
        raise ValueError(f"A5 checkpoint SHA mismatch: {checkpoint}")
    if (
        compute_file_hash(str(expert_root / "nested_protocol.json"))
        != artifact["nested_contract"]["sha256"]
    ):
        raise ValueError("A5 producer nested contract file hash mismatch")
    if cache_manifest["identity"]["a5_checkpoint_sha256"] != expected_checkpoint:
        raise ValueError("feature cache and A5 expert use different producer checkpoints")
    if artifact.get("role") != expected_role or artifact.get("expert") != "A5":
        raise ValueError("A5 expert role mismatch")
    oof = pd.read_csv(oof_path, dtype={"token_id": str, "sequence_id": str, "track_id": str})
    effective_path = expert_root / "train" / "effective_config.yaml"
    source_config_path = expert_root / "effective_config.yaml"
    if set(oof["config_sha256"].astype(str)) != {compute_file_hash(str(source_config_path))}:
        raise ValueError("A5 OOF rows do not bind the producer input configuration")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_uninitialized_producer_contract(
        checkpoint_payload,
        yaml.safe_load(source_config_path.read_text(encoding="utf-8")),
        protocol,
    )
    initialization_validation = validate_uninitialized_producer_contract(
        checkpoint_payload,
        yaml.safe_load(effective_path.read_text(encoding="utf-8")),
        protocol,
    )
    outer_fold = int(cache_manifest["identity"]["outer_fold"])
    inner_fold = cache_manifest["identity"]["inner_fold"]
    canonical = pd.read_csv(
        feature_cache_base.parent / f"outer{outer_fold}_final.metadata.csv",
        dtype={"sample_token": str, "sequence_id": str, "track_id": str},
    )
    nested_validation = validate_nested_split(
        protocol, artifact, canonical, oof, outer_fold=outer_fold, inner_fold=inner_fold
    )
    if frozen_teacher_audit is not None:
        validate_frozen_teacher_dependency(initialization_validation, frozen_teacher_audit)
        nested_validation["ancestry_validated"] = True
    if bool(oof["token_id"].duplicated().any()) or not bool(
        np.asarray(oof["finite"].astype(bool)).all()
    ):
        raise ValueError("A5 OOF table is duplicated or non-finite")
    indexed = metadata.reset_index(names="cache_index").merge(
        oof[
            [
                "token_id",
                "sequence_id",
                "track_id",
                "prediction_log_variance",
                "checkpoint_sha256",
                *(["inner_fold"] if "inner_fold" in oof else []),
            ]
        ],
        left_on=["sample_token", "sequence_id", "track_id"],
        right_on=["token_id", "sequence_id", "track_id"],
        validate="one_to_one",
    )
    if len(indexed) != len(oof):
        raise ValueError("feature cache does not cover the exact A5 OOF rows")
    if not bool(np.asarray(indexed["checkpoint_sha256"].astype(str) == expected_checkpoint).all()):
        raise ValueError("row-level A5 producer SHA mismatch")
    cache_indices = indexed["cache_index"].to_numpy(np.int64)
    phase = arrays["a5_phase"][cache_indices].astype(np.float32)
    support = arrays["pair_features"][cache_indices, -2].astype(np.float32)
    oof_log_variance = indexed["prediction_log_variance"].to_numpy(np.float32)
    log_variance = oof_log_variance
    dense_replay: dict[str, Any] | None = None
    # The Stage 61 outer-final cache is the exact replay consumed by Stage 62 and
    # carries the three A5 state fields used to construct its dense local field.
    # The older V8 expert_oof.csv is a separately materialized execution: its
    # uncertainty values are therefore provenance evidence, not an authority that
    # may silently replace the replay state. Inner caches do not contain a dense
    # field, so their signed producer OOF remains the only stored log-variance.
    if "patch_features" in arrays:
        repeated = arrays["patch_features"][cache_indices, :, -3:].astype(np.float32)
        broadcast_state = np.broadcast_to(repeated[:, :1, :], repeated.shape)
        if not np.array_equal(repeated, broadcast_state):
            raise ValueError("outer dense A5 state is not constant over patches")
        if not np.allclose(repeated[:, 0, 0], phase, rtol=0, atol=1e-7):
            raise ValueError("outer dense A5 phase disagrees with replay cache")
        if not np.allclose(repeated[:, 0, 2], support, rtol=0, atol=1e-7):
            raise ValueError("outer dense A5 support disagrees with replay cache")
        log_variance = repeated[:, 0, 1]
        difference = np.abs(log_variance.astype(np.float64) - oof_log_variance)
        dense_replay = {
            "state_source": "stage61_outer_final_dense_replay",
            "oof_log_variance_comparison_max_abs": float(difference.max()),
            "oof_log_variance_comparison_mean_abs": float(difference.mean()),
            "oof_log_variance_within_1e_6_fraction": float(np.mean(difference <= 1e-6)),
            "policy": "dense_replay_is_authoritative_for_outer_final",
        }
    state = np.column_stack((phase, log_variance, support)).astype(np.float32)
    coherent_manifest: dict[str, Any] | None = None
    if coherent_state_root is not None:
        coherent_root = coherent_state_root / feature_cache_base.name
        coherent_manifest = _read_signed_artifact(coherent_root / "manifest.json")
        if (
            coherent_manifest.get("artifact_type") != "coherent_frozen_a5_state_v2"
            or coherent_manifest.get("checkpoint_sha256") != expected_checkpoint
            or coherent_manifest.get("all_components_same_forward") is not True
            or coherent_manifest.get("exact_feature_phase_support_replay") is not True
            or coherent_manifest.get("producer") != feature_cache_base.name
        ):
            raise ValueError("coherent A5 replay producer identity mismatch")
        verify_components(
            coherent_root, coherent_manifest["components"], {"state.npy", "metadata.csv"}
        )
        coherent_metadata = pd.read_csv(coherent_root / "metadata.csv")
        for column in ("sample_token", "sequence_id", "track_id", "outer_fold"):
            if (
                coherent_metadata[column].astype(str).tolist()
                != metadata[column].astype(str).tolist()
            ):
                raise ValueError(f"coherent A5 replay metadata mismatch: {column}")
        coherent_values = np.load(coherent_root / "state.npy", allow_pickle=False)
        if coherent_values.shape != (len(metadata), 3) or coherent_values.dtype != np.float32:
            raise ValueError("coherent A5 replay layout mismatch")
        state = coherent_values[cache_indices]
        if not np.array_equal(state[:, 0], phase) or not np.array_equal(state[:, 2], support):
            raise ValueError("coherent A5 replay does not reproduce the stored phase/support")
        if dense_replay is not None and not np.array_equal(state[:, 1], log_variance):
            raise ValueError(
                "coherent A5 outer-final uncertainty does not reproduce its dense source"
            )
    if not np.isfinite(state).all():
        raise ValueError("A5 state is not finite")
    identity = pd.DataFrame(
        {
            "sample_token": indexed["sample_token"].astype(str),
            "sequence_id": indexed["sequence_id"].astype(str),
            "track_id": indexed["track_id"].astype(str),
            "outer_fold": indexed["outer_fold"].astype(int),
            "role": expected_role,
            "producer_sha": expected_checkpoint,
            "source_inner_fold": (
                indexed["inner_fold"].astype(int) if "inner_fold" in indexed else -1
            ),
        }
    )
    provenance = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "expert_artifact_path": str(expert_root / "expert_artifact.json"),
        "expert_artifact_sha256": compute_file_hash(str(expert_root / "expert_artifact.json")),
        "nested_protocol_path": str(expert_root / "nested_protocol.json"),
        "nested_protocol_sha256": compute_file_hash(str(expert_root / "nested_protocol.json")),
        "feature_cache_path": str(feature_cache_base.with_suffix(".npz")),
        "feature_cache_sha256": cache_manifest["cache"]["sha256"],
        "rows": len(identity),
        "sequence_ids": sorted(identity["sequence_id"].unique()),
        "nested_checks": protocol.get("checks", {}),
        "nested_relationship_validation": nested_validation,
        "initialization_relationship_validation": initialization_validation,
        "producer_input_config_sha256": compute_file_hash(str(source_config_path)),
        "trainer_effective_config_sha256": compute_file_hash(str(effective_path)),
        "state_component_sources": {
            "phase": "stage61_feature_cache.a5_phase",
            "log_variance": (
                "stage61_outer_final.patch_features[-2]"
                if dense_replay is not None
                else "signed_expert_oof.prediction_log_variance"
            ),
            "support": "stage61_feature_cache.pair_features[-2]",
        },
        "dense_replay_reconciliation": dense_replay,
        "scientific_state_accepted": coherent_manifest is not None,
        "coherent_replay": coherent_manifest,
        "frozen_teacher_dependency_audit": frozen_teacher_audit,
    }
    if coherent_state_root is not None:
        provenance["state_component_sources"] = {
            component: str(coherent_state_root / feature_cache_base.name / "state.npy")
            for component in ("phase", "log_variance", "support")
        }
    return identity, state, provenance


def build_crossfitted_a5_states(
    *,
    feature_cache_root: Path,
    router_root: Path,
    output_root: Path,
    coherent_state_root: Path,
    frozen_teacher_audit_path: Path,
) -> dict[str, Any]:
    """Create inner-OOF train and outer-final eval states for all outer folds."""

    frozen_teacher_audit = _read_signed_artifact(frozen_teacher_audit_path)
    teacher_manifest_path = Path(frozen_teacher_audit["teacher_manifest_path"])
    if (
        compute_file_hash(str(teacher_manifest_path))
        != frozen_teacher_audit["teacher_manifest_sha256"]
    ):
        raise ValueError("frozen teacher manifest changed since its dependency audit")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v9_crossfitted_a5_state_v1",
        "provenance_version": "coherent_nested_v2",
        "state_order": ["benchmark_phase", "log_variance", "sensor_support"],
        "contains_targets": False,
        "outer_folds": {},
    }
    ledger: list[dict[str, Any]] = []
    for outer in range(3):
        train_identity: list[pd.DataFrame] = []
        train_state: list[np.ndarray] = []
        train_provenance: list[dict[str, Any]] = []
        for inner in range(3):
            identity, state, provenance = _producer_state(
                feature_cache_base=feature_cache_root / f"outer{outer}_inner{inner}",
                expert_root=router_root / f"outer_fold{outer}_seed7" / "a5" / f"inner{inner}",
                expected_role="inner_oof",
                coherent_state_root=coherent_state_root,
                frozen_teacher_audit=frozen_teacher_audit,
            )
            if not bool(np.asarray(identity["source_inner_fold"] == inner).all()):
                raise ValueError("inner-fold label disagrees with producer path")
            train_identity.append(identity)
            train_state.append(state)
            train_provenance.append(provenance)
            ledger.append(
                {"outer_fold": outer, "inner_fold": inner, "role": "inner_oof", **provenance}
            )
        train_meta = pd.concat(train_identity, ignore_index=True)
        train_values = np.concatenate(train_state)
        if train_meta["sample_token"].duplicated().any():
            raise ValueError(f"outer{outer} inner-OOF rows overlap")
        expected_train = set(
            pd.read_csv(feature_cache_root / f"outer{outer}_final.metadata.csv")
            .query("outer_fold != @outer")["sample_token"]
            .astype(str)
        )
        if set(train_meta["sample_token"]) != expected_train:
            raise ValueError(f"outer{outer} inner-OOF rows do not equal outer-train")
        eval_meta, eval_values, eval_provenance = _producer_state(
            feature_cache_base=feature_cache_root / f"outer{outer}_final",
            expert_root=router_root / f"outer_fold{outer}_seed7" / "a5" / "outer_dev",
            expected_role="outer_dev",
            coherent_state_root=coherent_state_root,
            frozen_teacher_audit=frozen_teacher_audit,
        )
        if not bool(np.asarray(eval_meta["outer_fold"] == outer).all()):
            raise ValueError("outer-final evaluation rows contain a non-dev sequence")
        if set(train_meta["sample_token"]) & set(eval_meta["sample_token"]):
            raise ValueError("inner-OOF train and outer-final eval tokens overlap")
        ledger.append(
            {
                "outer_fold": outer,
                "inner_fold": None,
                "role": "outer_dev",
                **eval_provenance,
            }
        )
        fold_root = output_root / f"outer{outer}"
        fold_root.mkdir()
        order_train = np.argsort(train_meta["sample_token"].astype(str).to_numpy())
        order_eval = np.argsort(eval_meta["sample_token"].astype(str).to_numpy())
        train_meta = train_meta.iloc[order_train].reset_index(drop=True)
        eval_meta = eval_meta.iloc[order_eval].reset_index(drop=True)
        train_values = train_values[order_train]
        eval_values = eval_values[order_eval]
        np.save(fold_root / "train_state.npy", train_values)
        np.save(fold_root / "eval_state.npy", eval_values)
        train_meta.to_csv(fold_root / "train_metadata.csv", index=False, lineterminator="\n")
        eval_meta.to_csv(fold_root / "eval_metadata.csv", index=False, lineterminator="\n")
        manifest["outer_folds"][str(outer)] = {
            "components": {
                name: component_record(fold_root / name)
                for name in (
                    "train_state.npy",
                    "eval_state.npy",
                    "train_metadata.csv",
                    "eval_metadata.csv",
                )
            },
            "train_rows": len(train_meta),
            "eval_rows": len(eval_meta),
            "train_tokens_sha256": hashlib.sha256(
                "\n".join(train_meta["sample_token"]).encode("utf-8")
            ).hexdigest(),
            "eval_tokens_sha256": hashlib.sha256(
                "\n".join(eval_meta["sample_token"]).encode("utf-8")
            ).hexdigest(),
            "inner_producers": train_provenance,
            "outer_producer": eval_provenance,
        }
    pd.DataFrame(ledger).to_csv(output_root / "BASELINE_IDENTITY_LEDGER.csv", index=False)
    manifest = sign_stage63_65_artifact(manifest, evidence_type="nested_crossfitted_state")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_a5_state_split(root: Path, outer_fold: int, role: str) -> A5StateSplit:
    """Load one completed A5 state split and verify row alignment."""

    if role not in {"train", "eval"}:
        raise ValueError("A5 state role must be train or eval")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("provenance_version") != "coherent_nested_v2":
        raise ValueError("legacy/hybrid A5 states cannot authorize the corrected campaign")
    for fold_record in manifest["outer_folds"].values():
        for producer in [*fold_record["inner_producers"], fold_record["outer_producer"]]:
            if (
                producer.get("scientific_state_accepted") is not True
                or producer.get("nested_relationship_validation", {}).get("ancestry_validated")
                is not True
            ):
                raise ValueError("A5 producer lacks coherent state or verified nested ancestry")
    if not verify_artifact_hash(manifest):
        raise ValueError("cross-fitted A5 state manifest signature mismatch")
    fold = root / f"outer{outer_fold}"
    fold_manifest = manifest["outer_folds"][str(outer_fold)]
    verify_components(
        fold,
        fold_manifest.get("components", {}),
        {
            "train_state.npy",
            "eval_state.npy",
            "train_metadata.csv",
            "eval_metadata.csv",
        },
    )
    state = np.load(fold / f"{role}_state.npy")
    metadata = pd.read_csv(fold / f"{role}_metadata.csv", dtype={"sample_token": str})
    tokens = metadata["sample_token"].astype(str)
    if tokens.duplicated().any() or not tokens.is_monotonic_increasing:
        raise ValueError("A5 state token order is not unique and canonical")
    token_hash = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
    if token_hash != fold_manifest[f"{role}_tokens_sha256"]:
        raise ValueError("A5 state token identity mismatch")
    if state.shape != (len(metadata), 3) or not np.isfinite(state).all():
        raise ValueError("A5 state split is invalid")
    return A5StateSplit(state.astype(np.float32, copy=False), metadata)


__all__ = ["A5StateSplit", "build_crossfitted_a5_states", "load_a5_state_split"]
