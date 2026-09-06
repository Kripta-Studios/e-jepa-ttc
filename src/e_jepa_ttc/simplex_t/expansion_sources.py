"""Load the explicitly pinned D1 expansion cache and attach verified TRAIN targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .cache import CachedQueries, fit_normalizer
from .expansion_selection import selected_expansion_rows
from .expansion_targets import load_expansion_targets
from .training import state_digest


@dataclass(frozen=True)
class ExpansionBinding:
    """Content pins selected from acknowledged role/lineage interfaces before fitting."""

    compiled: Path
    compiled_sha256: str
    index_manifest: Path
    index_manifest_sha256: str
    dedup: Path
    pool: Path
    pool_sha256: str
    metadata: Path
    metadata_sha256: str
    labels: Path
    labels_sha256: str
    cache_identity_sha256: str


@dataclass(frozen=True)
class ExpansionInputs:
    """Aligned D1 expansion source and original supervision for registered controls."""

    source: CachedQueries
    tokens: np.ndarray
    sequences: np.ndarray
    target_ttc: np.ndarray


def _verify(path: Path, expected: str) -> None:
    if len(expected) != 64 or compute_file_hash(str(path)) != expected:
        raise ValueError(f"D1 content pin mismatch: {path.name}")


def load_expansion_inputs(
    binding: ExpansionBinding,
    *,
    outer: int,
    feature_count: int,
    allowed_sequences: set[str],
) -> ExpansionInputs:
    """Return expansion TRAIN in frozen pool order, plus aligned sequence identities.

    This API grants no fit, replay or data-role authority. Callers first resolve
    authoritative pins and later bind this source into the scientific freeze.
    No target file is opened until cache, input-only selection and metadata pass.
    """
    if outer not in range(3) or feature_count not in {17, 145}:
        raise ValueError("unregistered D1 fold or feature schema")
    compiled_path = binding.compiled / "COMPILED.json"
    for path, digest in (
        (compiled_path, binding.compiled_sha256),
        (binding.index_manifest, binding.index_manifest_sha256),
        (binding.pool, binding.pool_sha256),
        (binding.metadata, binding.metadata_sha256),
    ):
        _verify(path, digest)
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    if (
        compiled.get("pool") != "D1"
        or compiled["outer"] != outer
        or compiled["status"] != "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
        or compiled["cache_identity_sha256"] != binding.cache_identity_sha256
    ):
        raise ValueError("compiled D1 family/identity/status mismatch")
    manifest = json.loads(binding.index_manifest.read_text(encoding="utf-8"))
    pool = json.loads(binding.pool.read_text(encoding="utf-8"))
    if pool["expansion_metadata_sha256"] != binding.metadata_sha256:
        raise ValueError("pool metadata lineage changed")
    index_path = binding.index_manifest.parent / "query_context_index.npz"
    _verify(index_path, manifest["index_sha256"])
    if compiled["index_sha256"] != manifest["index_sha256"]:
        raise ValueError("compiled cache refers to another query index")
    _verify(binding.dedup, compiled["dedup_sha256"])
    with np.load(index_path, allow_pickle=False) as archive:
        index = {key: archive[key] for key in ("tokens", "sequences", "producer_family", "valid")}
    with np.load(binding.dedup, allow_pickle=False) as archive:
        history = archive["history"]
    rows = selected_expansion_rows(
        index,
        history,
        outer=outer,
        pool=pool,
        families=manifest["families"],
        allowed_expansion_sequences=allowed_sequences,
    )
    if (
        compiled["selected_query_ids"] != sorted(rows.tolist())
        or compiled["queries"] != len(rows)
        or compiled["indexed_queries"] != len(index["tokens"])
    ):
        raise ValueError("compiled selection differs from registered D1 pool")
    required = {"features145", "expert_ttc", "known", "anchor_us", "available_us"}
    if set(compiled["arrays"]) != required:
        raise ValueError("compiled D1 array schema mismatch")
    arrays = {}
    for name in sorted(required):
        path = binding.compiled / f"{name}.npy"
        _verify(path, compiled["arrays"][name])
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = compiled["observations"]
    if arrays["features145"].shape != (count, 145) or arrays["features145"].dtype != np.float32:
        raise ValueError("D1 observation feature schema mismatch")
    selected_history = history[rows]
    if (selected_history >= count).any():
        raise ValueError("D1 history exceeds compiled observation universe")
    features = arrays["features145"][:, :feature_count]
    allowed = np.zeros(count, bool)
    allowed[selected_history[selected_history >= 0]] = True
    if not allowed.all():
        raise ValueError("unconsumed observations in compiled D1 pool")
    normalizer = fit_normalizer(features, selected_history, allowed)
    tokens, sequences = index["tokens"][rows], index["sequences"][rows]
    metadata = pd.read_parquet(
        binding.metadata, columns=["sample_token", "sequence_id", "timestamp_us"]
    )
    if not metadata.sample_token.is_unique:
        raise ValueError("duplicate metadata token")
    metadata = metadata.set_index("sample_token").loc[tokens.tolist()]
    if not np.array_equal(metadata.sequence_id.to_numpy(), sequences):
        raise ValueError("input index and target-reference sequence mismatch")
    targets = load_expansion_targets(
        binding.labels,
        expected_sha256=binding.labels_sha256,
        tokens=tokens,
        sequences=sequences,
        timestamps_us=metadata.timestamp_us.to_numpy(),
    )
    identity = state_digest(
        {
            "namespace": "SIMPLEX_T_COMPILED_D1_TRAIN",
            "compiled": binding.compiled_sha256,
            "pool": binding.pool_sha256,
            "index": binding.index_manifest_sha256,
            "targets": targets.identity_sha256,
            "features": feature_count,
            "normalizer_mean": torch.from_numpy(normalizer.mean),
            "normalizer_scale": torch.from_numpy(normalizer.scale),
            "normalizer_ids": normalizer.consumed_ids_sha256,
        }
    )
    source = CachedQueries(
        features,
        arrays["anchor_us"],
        arrays["available_us"],
        selected_history,
        targets.target_phase,
        targets.mass,
        normalizer,
        identity,
    )
    return ExpansionInputs(source, tokens, sequences, targets.target_ttc)


def load_expansion_source(
    binding: ExpansionBinding, *, outer: int, feature_count: int, allowed_sequences: set[str]
) -> tuple[CachedQueries, np.ndarray]:
    """Preserve the original D1 source API; no extra target reads or numerical changes."""
    result = load_expansion_inputs(
        binding, outer=outer, feature_count=feature_count, allowed_sequences=allowed_sequences
    )
    return result.source, result.sequences
