"""Production file binding for the registered dense TRAIN observation pool."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .cache import CachedQueries, fit_normalizer
from .expansion_sources import ExpansionBinding
from .selected_targets import load_selected_train_targets
from .training import state_digest


@dataclass(frozen=True)
class DenseBinding(ExpansionBinding):
    """Same explicit file-pin layout as D1, but for DENSE_OLD only."""


@dataclass(frozen=True)
class DenseInputs:
    """Aligned TRAIN cache and original supervision for dense_source_pair."""

    source: CachedQueries
    tokens: np.ndarray
    sequences: np.ndarray
    target_ttc: np.ndarray


def load_dense_inputs(
    binding: DenseBinding, *, outer: int, allowed_sequences: set[str]
) -> DenseInputs:
    """Verify compiled files and registered selection before reading selected labels.

    Caller establishes role/time authority and transitive producer exclusions;
    valid hashes or a COMPLETE compiler status do not grant fit authorization.
    """
    if outer not in range(3) or not allowed_sequences:
        raise ValueError("authorized dense fold/group binding required")
    for path, digest in (
        (binding.compiled / "COMPILED.json", binding.compiled_sha256),
        (binding.index_manifest, binding.index_manifest_sha256),
        (binding.pool, binding.pool_sha256),
        (binding.metadata, binding.metadata_sha256),
    ):
        if len(digest) != 64 or compute_file_hash(str(path)) != digest:
            raise ValueError("dense source content pin mismatch")
    compiled = json.loads((binding.compiled / "COMPILED.json").read_text(encoding="utf-8"))
    if (
        compiled.get("pool") != "DENSE_OLD"
        or compiled["outer"] != outer
        or compiled["status"] != "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
        or compiled["cache_identity_sha256"] != binding.cache_identity_sha256
    ):
        raise ValueError("dense compiled pool/fold/extractor mismatch")
    manifest = json.loads(binding.index_manifest.read_text(encoding="utf-8"))
    index_path = binding.index_manifest.parent / "query_context_index.npz"
    if (
        compute_file_hash(str(index_path)) != manifest["index_sha256"]
        or compiled["index_sha256"] != manifest["index_sha256"]
        or compute_file_hash(str(binding.dedup)) != compiled["dedup_sha256"]
    ):
        raise ValueError("dense index/dedup content mismatch")
    with np.load(index_path, allow_pickle=False) as archive:
        index = {key: archive[key] for key in ("tokens", "sequences", "producer_family", "valid")}
    with np.load(binding.dedup, allow_pickle=False) as archive:
        history = archive["history"]
    plan = json.loads(binding.pool.read_text(encoding="utf-8"))
    folds = [fold for fold in plan["folds"] if fold["outer"] == outer]
    if len(folds) != 1:
        raise ValueError("ambiguous registered dense fold")
    selection = folds[0]["pools"]["DENSE_OLD"]
    tokens = np.asarray(selection["tokens"])
    if (
        tokens.ndim != 1
        or not len(tokens)
        or len(np.unique(tokens)) != len(tokens)
        or len(np.unique(index["tokens"])) != len(index["tokens"])
        or len(tokens) != folds[0]["nominal_common_count"]
        or not set(selection["sequences"]) <= allowed_sequences
    ):
        raise ValueError("invalid or unauthorized dense selection")
    positions = {str(token): row for row, token in enumerate(index["tokens"])}
    if not set(tokens) <= set(positions):
        raise ValueError("selected dense query absent from index")
    rows = np.asarray([positions[str(token)] for token in tokens], np.int64)
    sequences = index["sequences"][rows]
    if set(sequences) != set(selection["sequences"]):
        raise ValueError("dense sequence membership differs from registered pool")
    families = index["producer_family"][outer]
    if (
        history.shape != index["valid"].shape
        or history.shape != (len(families), 16)
        or history.dtype != np.int64
        or (history < -1).any()
        or not np.array_equal(history >= 0, index["valid"] & (families[:, None] >= 0))
        or not np.array_equal(np.sort(rows), np.flatnonzero(families >= 0))
    ):
        raise ValueError("dense active history coverage differs from selection")
    for row, sequence in zip(rows, sequences, strict=True):
        family_id = int(families[row])
        if family_id // 4 != outer or family_id % 4 == 3 or family_id >= len(manifest["families"]):
            raise ValueError("dense query has no valid inner producer")
        family = manifest["families"][family_id]
        if (
            family["outer_fold"] != outer
            or family["role"] == "outer_dev"
            or family["family_sha256"] != selection["sequence_family_sha256"].get(str(sequence))
        ):
            raise ValueError("dense query producer differs from registered family")
    if (
        compiled["selected_query_ids"] != sorted(rows.tolist())
        or compiled["queries"] != len(rows)
        or compiled["indexed_queries"] != len(families)
    ):
        raise ValueError("dense compiled query membership mismatch")
    count = compiled["observations"]
    schemas = {
        "features145": ((count, 145), np.float32),
        "expert_ttc": ((count, 3), np.float32),
        "known": ((count, 2), np.bool_),
        "anchor_us": ((count,), np.int64),
        "available_us": ((count,), np.int64),
    }
    if set(compiled["arrays"]) != set(schemas):
        raise ValueError("dense compiled field schema mismatch")
    arrays = {}
    for name, (shape, dtype) in schemas.items():
        path = binding.compiled / f"{name}.npy"
        if compute_file_hash(str(path)) != compiled["arrays"][name]:
            raise ValueError("dense compiled array changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("dense array shape/dtype mismatch")
        arrays[name] = array
    selected_history = history[rows]
    if not (selected_history[:, -1] >= 0).all() or not np.array_equal(
        np.unique(selected_history[selected_history >= 0]), np.arange(count)
    ):
        raise ValueError("dense history lacks current input or complete observation coverage")
    features = arrays["features145"][:, :17]
    normalizer = fit_normalizer(features, selected_history, np.ones(count, bool))
    metadata = pd.read_parquet(
        binding.metadata,
        columns=["sample_token", "sequence_id", "timestamp_us"],
        filters=[
            ("sample_token", "in", tokens.tolist()),
            ("sequence_id", "in", sorted(set(sequences))),
        ],
    )
    if len(metadata) != len(tokens) or not metadata.sample_token.is_unique:
        raise ValueError("dense metadata projection missing/duplicate queries")
    metadata = metadata.set_index("sample_token").loc[tokens.tolist()]
    if not np.array_equal(metadata.sequence_id.to_numpy(), sequences):
        raise ValueError("dense metadata identity mismatch")
    targets = load_selected_train_targets(
        binding.labels,
        expected_sha256=binding.labels_sha256,
        tokens=tokens,
        sequences=sequences,
        timestamps_us=metadata.timestamp_us.to_numpy(),
        allowed_sequences=set(selection["sequences"]),
        pool="DENSE_OLD",
    )
    identity = state_digest(
        {
            "namespace": "SIMPLEX_T_COMPILED_DENSE_TRAIN",
            "compiled": binding.compiled_sha256,
            "pool": binding.pool_sha256,
            "index": binding.index_manifest_sha256,
            "metadata": binding.metadata_sha256,
            "targets": targets.identity_sha256,
            "normalizer_mean": torch.from_numpy(normalizer.mean),
            "normalizer_scale": torch.from_numpy(normalizer.scale),
            "normalizer_ids": normalizer.consumed_ids_sha256,
        }
    )
    return DenseInputs(
        CachedQueries(
            features,
            arrays["anchor_us"],
            arrays["available_us"],
            selected_history,
            targets.target_phase,
            targets.mass,
            normalizer,
            identity,
        ),
        tokens,
        sequences,
        targets.target_ttc,
    )
