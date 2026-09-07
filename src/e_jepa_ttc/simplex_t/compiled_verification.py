"""Read-only transport verification of compiled folds, not scientific admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def verify_compiled_fold(
    output: Path,
    cache: Path,
    index_root: Path,
    dedup_root: Path,
    outer: int,
    *,
    pool: str,
    resource_ok: Callable[[], bool],
) -> bool:
    """Return False only for an absent seal; reject changed or malformed sealed data.

    Checks byte bindings and array schemas without reading labels or loading models.
    Producer ancestry and scientific authority remain the source loader's responsibility.
    """
    if type(outer) is not int or outer not in range(3) or pool not in {"D0", "D1", "DENSE_OLD"}:
        raise ValueError("invalid compiled fold selection")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("RESOURCE_PAUSE: compiled verification")

    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                boundary()
                block = stream.read(1024 * 1024)
                if not block:
                    return result.hexdigest()
                result.update(block)

    boundary()
    seal = output / "COMPILED.json"
    if not seal.exists():
        return False
    if seal.stat().st_size > 8 * 1024**2:
        raise ValueError("oversized compiled seal")
    original = seal.read_bytes()
    manifest = json.loads(original)
    if (
        manifest.get("schema") != "simplex_t_compiled_query_context_v1"
        or manifest.get("status") != "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
        or type(manifest.get("outer")) is not int
        or manifest["outer"] != outer
        or manifest.get("pool", "D0") != pool
        or type(manifest.get("optimizer_updates")) is not int
        or manifest["optimizer_updates"] != 0
    ):
        raise ValueError("compiled seal identity mismatch")
    for field in ("queries", "observations"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError("invalid compiled coverage")
    if pool == "D0" and manifest["queries"] != 8192:
        raise ValueError("incomplete original query coverage")
    bindings = {
        cache / "IDENTITY.json": manifest["cache_identity_sha256"],
        index_root / "query_context_index.npz": manifest["index_sha256"],
        dedup_root / f"outer{outer}.npz": manifest["dedup_sha256"],
    }
    if "query_selection" in manifest:
        if pool != "D1":
            raise ValueError("density-selected compilation must belong to D1")
        mapping = output / "source_observation_ids.npy"
        bindings[mapping] = manifest["source_observation_ids_sha256"]
        ids = np.load(mapping, allow_pickle=False)
        if (
            ids.dtype != np.int64
            or ids.shape != (manifest["observations"],)
            or (ids < 0).any()
            or (np.diff(ids) <= 0).any()
        ):
            raise ValueError("invalid selected observation remapping")
    count = manifest["observations"]
    fields = {
        "features145": (np.dtype("float32"), (count, 145)),
        "expert_ttc": (np.dtype("float32"), (count, 3)),
        "known": (np.dtype("bool"), (count, 2)),
        "anchor_us": (np.dtype("int64"), (count,)),
        "available_us": (np.dtype("int64"), (count,)),
    }
    if not isinstance(manifest.get("arrays"), dict) or set(manifest["arrays"]) != set(fields):
        raise ValueError("compiled array set mismatch")
    bindings.update({output / f"{name}.npy": value for name, value in manifest["arrays"].items()})
    for path, expected in bindings.items():
        if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
            raise ValueError("compiled binding changed")
    for name, (dtype, shape) in fields.items():
        boundary()
        array = np.load(output / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        try:
            if array.dtype != dtype or array.shape != shape:
                raise ValueError("compiled array schema mismatch")
        finally:
            del array
    boundary()
    if seal.read_bytes() != original:
        raise ValueError("compiled seal changed during verification")
    return True
