"""Compile completed temporal blocks into bounded, immutable head input arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json

from .expert_phase import expert_phase_from_ttc
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
    for name in ("features145", "pair_features"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError("nonfinite or non-FP32 cache values")
    if arrays["expert_ttc"].dtype != np.float32:
        raise ValueError("non-FP32 expert points")
    if not np.isfinite(arrays["expert_ttc"][:, :2]).all():
        raise ValueError("nonfinite A5/C2F points")
    phase = expert_phase_from_ttc(arrays["expert_ttc"]).astype(np.float32)
    if not np.array_equal(arrays["features145"][:, 8:11], phase):
        raise ValueError("expert point/phase inconsistency")
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


def compile_fold(
    cache: Path,
    index_root: Path,
    dedup_root: Path,
    output: Path,
    outer: int,
    *,
    pool: str = "D0",
) -> None:
    """Require every selected query block; never fabricate inactive D1 observations.

    D0 remains the default and retains its existing compiled manifest schema.
    D1 requires an explicitly bound extraction identity and masks inactive rows.
    Compilation validates cache contents; it does not grant inference/fit authority.
    """
    if outer not in range(3) or output.exists() or pool not in {"D0", "D1"}:
        raise ValueError("invalid fold or existing compiled output")
    identity = json.loads((cache / "IDENTITY.json").read_text(encoding="utf-8"))
    if pool == "D1" and identity.get("pool") != "D1":
        raise ValueError("explicit D1 extraction identity required")
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
    families = index["producer_family"][outer]
    if history.shape != index["valid"].shape or history.shape != (len(families), 16):
        raise ValueError("query/history shape mismatch")
    if (history < -1).any() or (history >= len(keys)).any():
        raise ValueError("history observation index out of bounds")
    active = families >= 0
    if (families < -1).any() or (pool == "D0" and not active.all()):
        raise ValueError("invalid inactive producer assignment")
    upper = outer * 4 + (2 if pool == "D1" else 3)
    if not active.any() or ((families[active] < outer * 4) | (families[active] > upper)).any():
        raise ValueError("query assigned outside permitted producer family")
    if not np.array_equal(history >= 0, index["valid"] & active[:, None]):
        raise ValueError("history mask differs from active query support")
    receipts = []
    for qi in np.flatnonzero(active):
        family = families[qi]
        stem = cache / f"family{family:02d}_query{qi:05d}"
        if not stem.with_suffix(".json").exists():
            raise ValueError(f"TEMPORAL_CACHE_INCOMPLETE:outer{outer}:query{qi}")
        receipt = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        if receipt["query"] != qi or receipt["family"] != int(family):
            raise ValueError("receipt producer assignment changed")
        receipts.append((int(qi), stem.with_suffix(".npz"), receipt))
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
    for qi, path, receipt in receipts:
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
    expansion_fields = (
        {
            "pool": "D1",
            "indexed_queries": len(families),
            "selected_query_ids": np.flatnonzero(active).tolist(),
            "inactive_queries": int((~active).sum()),
        }
        if pool == "D1"
        else {}
    )
    write_new_json(
        output / "COMPILED.json",
        {
            "schema": "simplex_t_compiled_query_context_v1",
            "outer": outer,
            "cache_identity_sha256": compute_file_hash(str(cache / "IDENTITY.json")),
            "index_sha256": compute_file_hash(str(index_path)),
            "dedup_sha256": record["sha256"],
            "arrays": hashes,
            "queries": len(receipts),
            "observations": len(keys),
            "optimizer_updates": 0,
            "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
            **expansion_fields,
        },
    )
