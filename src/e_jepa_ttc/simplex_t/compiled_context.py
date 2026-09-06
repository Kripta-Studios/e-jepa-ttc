"""Compile completed temporal blocks into bounded, immutable head input arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json

from .lifecycle import admitted


def validate_block(
    arrays: dict[str, np.ndarray],
    expected_ids: np.ndarray,
    anchors: np.ndarray,
    available_us: int,
) -> None:
    """Reject missing, reordered, nonfinite or timing-inconsistent consumed rows."""
    count = len(expected_ids)
    shapes = {
        "features145": (count, 145),
        "expert_ttc": (count, 3),
        "known": (count, 2),
        "pair_features": (count, 133),
        "observation_ids": (count,),
        "anchor_us": (count,),
        "available_us": (count,),
    }
    if set(arrays) != set(shapes) or any(arrays[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("temporal block schema mismatch")
    for name in ("features145", "expert_ttc", "pair_features"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError("nonfinite or non-FP32 cache values")
    for name in ("observation_ids", "anchor_us", "available_us"):
        if arrays[name].dtype != np.int64:
            raise ValueError("integer cache identity/time required")
    if arrays["known"].dtype != bool:
        raise ValueError("boolean known-support mask required")
    if not np.array_equal(arrays["observation_ids"], expected_ids):
        raise ValueError("consumed observation identities changed")
    if not np.array_equal(arrays["anchor_us"], anchors):
        raise ValueError("sensor times changed")
    if not (arrays["available_us"] == available_us).all():
        raise ValueError("ROI availability changed")


def compile_fold(cache: Path, index_root: Path, dedup_root: Path, output: Path, outer: int) -> None:
    """Require every D0 query block before allocating a fold's feature memmaps."""
    if outer not in range(3) or output.exists():
        raise ValueError("invalid fold or existing compiled output")
    identity = json.loads((cache / "IDENTITY.json").read_text(encoding="utf-8"))
    index_path = index_root / "query_context_index.npz"
    if compute_file_hash(str(index_path)) != identity["index_sha256"]:
        raise ValueError("cache/index mismatch")
    with np.load(index_path, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    dedup_manifest = json.loads((dedup_root / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    record = dedup_manifest["outputs"][outer]
    dedup_path = dedup_root / record["path"]
    if compute_file_hash(str(dedup_path)) != record["sha256"]:
        raise ValueError("deduplicated index changed")
    with np.load(dedup_path, allow_pickle=False) as archive:
        history, keys = archive["history"], archive["keys"]
    receipts = []
    for qi, family in enumerate(index["producer_family"][outer]):
        stem = cache / f"family{family:02d}_query{qi:05d}"
        if not stem.with_suffix(".json").exists():
            raise ValueError(f"TEMPORAL_CACHE_INCOMPLETE:outer{outer}:query{qi}")
        receipt = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        if receipt["query"] != qi or receipt["family"] != int(family):
            raise ValueError("receipt producer assignment changed")
        receipts.append((stem.with_suffix(".npz"), receipt))
    resources = admitted([output.parent])
    if not resources["has_headroom"]:
        raise RuntimeError("RESOURCE_PAUSE")
    output.mkdir()
    fields = {
        "features145": (np.float32, (len(keys), 145)),
        "expert_ttc": (np.float32, (len(keys), 3)),
        "known": (np.bool_, (len(keys), 2)),
        "anchor_us": (np.int64, (len(keys),)),
        "available_us": (np.int64, (len(keys),)),
    }
    destinations = {
        name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in fields.items()
    }
    consumed = np.zeros(len(keys), dtype=bool)
    for qi, (path, receipt) in enumerate(receipts):
        if compute_file_hash(str(path)) != receipt["sha256"]:
            raise ValueError("cache payload changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        mask = index["valid"][qi]
        ids = history[qi, mask]
        validate_block(
            arrays,
            ids,
            index["anchor_us"][qi] - index["lag_us"][mask],
            int(index["roi_available_us"][qi]),
        )
        if consumed[ids].any():
            raise ValueError("unexpected duplicate feature block")
        consumed[ids] = True
        for name, destination in destinations.items():
            destination[ids] = arrays[name]
    if not consumed.all():
        raise ValueError("incomplete consumed observation coverage")
    for destination in destinations.values():
        destination.flush()
    hashes = {name: compute_file_hash(str(output / f"{name}.npy")) for name in fields}
    write_new_json(
        output / "COMPILED.json",
        {
            "schema": "simplex_t_compiled_query_context_v1",
            "outer": outer,
            "cache_identity_sha256": compute_file_hash(str(cache / "IDENTITY.json")),
            "dedup_sha256": record["sha256"],
            "arrays": hashes,
            "queries": len(receipts),
            "observations": len(keys),
            "optimizer_updates": 0,
            "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
        },
    )
