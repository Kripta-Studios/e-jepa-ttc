"""Strict nested A5 state assembly for Stage 64 residual training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact
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
    *, feature_cache_base: Path, expert_root: Path, expected_role: str
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
    if cache_manifest["identity"]["a5_checkpoint_sha256"] != expected_checkpoint:
        raise ValueError("feature cache and A5 expert use different producer checkpoints")
    if artifact.get("role") != expected_role or artifact.get("expert") != "A5":
        raise ValueError("A5 expert role mismatch")
    oof = pd.read_csv(oof_path, dtype={"token_id": str, "sequence_id": str, "track_id": str})
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
    log_variance = indexed["prediction_log_variance"].to_numpy(np.float32)
    state = np.column_stack((phase, log_variance, support)).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("A5 state is not finite")
    # Outer-final caches carry an independent copy of all three state fields.
    if "patch_features" in arrays:
        repeated = arrays["patch_features"][cache_indices, :, -3:]
        if not np.allclose(repeated, state[:, None, :], rtol=0, atol=1e-6):
            raise ValueError("A5 state disagrees with outer dense-field cache")
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
    }
    return identity, state, provenance


def build_crossfitted_a5_states(
    *,
    feature_cache_root: Path,
    router_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create inner-OOF train and outer-final eval states for all outer folds."""

    output_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v9_crossfitted_a5_state_v1",
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
    if not verify_artifact_hash(manifest):
        raise ValueError("cross-fitted A5 state manifest signature mismatch")
    fold = root / f"outer{outer_fold}"
    state = np.load(fold / f"{role}_state.npy")
    metadata = pd.read_csv(fold / f"{role}_metadata.csv", dtype={"sample_token": str})
    if state.shape != (len(metadata), 3) or not np.isfinite(state).all():
        raise ValueError("A5 state split is invalid")
    return A5StateSplit(state.astype(np.float32, copy=False), metadata)


__all__ = ["A5StateSplit", "build_crossfitted_a5_states", "load_a5_state_split"]
