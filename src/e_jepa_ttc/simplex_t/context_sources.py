"""Role-separated D0 head sources from a complete amended temporal cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .cache import CachedQueries, fit_normalizer, training_mass
from .current_inputs import load_current_inputs
from .training import state_digest


def load_context_sources(
    compiled: Path,
    index_root: Path,
    dedup_root: Path,
    historical_root: Path,
    *,
    compiled_manifest_sha256: str,
    ancestry_sha256: str,
    allowed_sequences: set[str],
    feature_count: int,
) -> dict[str, CachedQueries]:
    """Load TRAIN and OLD_DEV separately, fitting one TRAIN-only normalizer.

    This does not authorize a fit. The caller must bind the returned identities
    to scientific freeze and phase gates. It never opens raw events or experts.
    """
    if feature_count not in {17, 145}:
        raise ValueError("unregistered feature count")
    manifest_path = compiled / "COMPILED.json"
    if compute_file_hash(str(manifest_path)) != compiled_manifest_sha256:
        raise ValueError("compiled manifest differs from requested freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["status"] != "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
        or manifest["queries"] != 8192
    ):
        raise ValueError("complete original D0 fold required")
    outer = manifest["outer"]
    if outer not in range(3):
        raise ValueError("invalid outer fold")
    arrays = {}
    for name, digest in manifest["arrays"].items():
        path = compiled / f"{name}.npy"
        if compute_file_hash(str(path)) != digest:
            raise ValueError("compiled array bytes changed")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    required = {"features145", "expert_ttc", "known", "anchor_us", "available_us"}
    if set(arrays) != required:
        raise ValueError("compiled array set mismatch")
    dedup_path = dedup_root / f"outer{outer}.npz"
    if compute_file_hash(str(dedup_path)) != manifest["dedup_sha256"]:
        raise ValueError("compiled history binding changed")
    with np.load(dedup_path, allow_pickle=False) as archive:
        history = archive["history"]
    index_path = index_root / "query_context_index.npz"
    if compute_file_hash(str(index_path)) != manifest["index_sha256"]:
        raise ValueError("compiled query index changed")
    with np.load(index_path, allow_pickle=False) as archive:
        tokens = archive["tokens"]
        assignments = archive["producer_family"][outer]
    if len(tokens) != 8192 or history.shape != (8192, 16):
        raise ValueError("query index schema mismatch")
    positions = {str(token): i for i, token in enumerate(tokens)}
    tables = {
        role: load_current_inputs(
            historical_root,
            outer,
            role,
            ancestry_sha256=ancestry_sha256,
            allowed_sequences=allowed_sequences,
        )
        for role in ("inner_oof", "outer_dev")
    }
    query_ids = {}
    for role, table in tables.items():
        metadata = table["metadata"]
        query_ids[role] = np.asarray([positions[str(t)] for t in metadata.sample_token], np.int64)
        expected_families = (
            np.full(len(metadata), outer * 4 + 3)
            if role == "outer_dev"
            else outer * 4 + metadata.inner_fold.to_numpy()
        )
        if not np.array_equal(assignments[query_ids[role]], expected_families):
            raise ValueError("temporal producer differs from role-specific query producer")
    train_ids, dev_ids = query_ids["inner_oof"], query_ids["outer_dev"]
    if set(train_ids) & set(dev_ids) or len(set(train_ids) | set(dev_ids)) != 8192:
        raise ValueError("incomplete or overlapping train/dev queries")
    features = arrays["features145"][:, :feature_count]
    train_history = history[train_ids]
    train_allowed = np.zeros(len(features), bool)
    unique_train = np.unique(train_history[train_history >= 0])
    train_allowed[unique_train] = True
    dev_history = history[dev_ids]
    if train_allowed[dev_history[dev_history >= 0]].any():
        raise ValueError("TRAIN and OLD_DEV share a consumed observation")
    normalizer = fit_normalizer(features, train_history, train_allowed)
    result = {}
    for role, table in tables.items():
        metadata = table["metadata"]
        mass = training_mass(metadata.target_ttc.to_numpy(), metadata.sequence_id.to_numpy())
        identity = state_digest(
            {
                "namespace": "SIMPLEX_T_QUERY_CONTEXT_AMENDMENT",
                "compiled": compiled_manifest_sha256,
                "index": compute_file_hash(str(index_path)),
                "table": table["reference"],
                "role": role,
                "features": feature_count,
                "normalizer_mean": torch.from_numpy(normalizer.mean),
                "normalizer_scale": torch.from_numpy(normalizer.scale),
                "normalizer_ids": normalizer.consumed_ids_sha256,
            }
        )
        result[role] = CachedQueries(
            features,
            arrays["anchor_us"],
            arrays["available_us"],
            history[query_ids[role]],
            table["arrays"]["target_phase"],
            mass,
            normalizer,
            identity,
        )
    return result
