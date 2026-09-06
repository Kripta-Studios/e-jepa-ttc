"""Reuse frozen observation blocks only across identical content-key bindings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .compiled_context import validate_block


def rebind_block(
    arrays: dict[str, np.ndarray],
    *,
    source_keys: np.ndarray,
    destination_keys: np.ndarray,
    destination_ids: np.ndarray,
    anchors_us: np.ndarray,
    available_us: int,
) -> dict[str, np.ndarray]:
    """Change observation row numbers, never expert outputs or input dependencies.

    Callers first verify block, index, deduplication and producer manifest hashes.
    Content keys bind raw data, preprocessing, producer, crop and availability.
    Equal query names alone are insufficient. This operation grants no new role,
    temporal or replay authority; it does not load targets or execute experts.
    """
    original_ids = arrays["observation_ids"]
    for ids, keys in ((original_ids, source_keys), (destination_ids, destination_keys)):
        if (
            ids.ndim != 1
            or ids.dtype != np.int64
            or not len(ids)
            or len(np.unique(ids)) != len(ids)
            or keys.ndim != 1
            or (ids < 0).any()
            or (ids >= len(keys)).any()
        ):
            raise ValueError("invalid reuse observation index")
    if original_ids.shape != destination_ids.shape:
        raise ValueError("reused history support changed")
    original = source_keys[original_ids]
    target = destination_keys[destination_ids]
    if not np.array_equal(original, target):
        raise ValueError("reused content key changes producer or input dependencies")
    if any(
        not isinstance(key, str) or len(key) != 64 or set(key) - set("0123456789abcdef")
        for key in original.tolist()
    ):
        raise ValueError("reuse requires canonical SHA256 observation keys")
    validate_block(arrays, original_ids, anchors_us, available_us)
    result = dict(arrays)
    result["observation_ids"] = destination_ids.copy()
    validate_block(result, destination_ids, anchors_us, available_us)
    return result


def load_reused_block(
    cache: Path,
    *,
    identity_sha256: str,
    receipt_sha256: str,
    query: int,
    family: int,
    source_ids: np.ndarray,
    source_keys: np.ndarray,
    destination_keys: np.ndarray,
    destination_ids: np.ndarray,
    anchors_us: np.ndarray,
    available_us: int,
) -> dict[str, np.ndarray]:
    """Load one pinned TRAIN block and rebind it without materializing a duplicate cache.

    The caller pins the source receipt and index/dedup manifests before invoking
    this loader. No path from the receipt is followed; paths derive from integer
    query/family IDs under the explicitly supplied cache root.
    """
    if type(query) is not int or query < 0 or family not in range(12) or family % 4 == 3:
        raise ValueError("reuse requires a TRAIN query and inner producer family")
    root = cache.resolve(strict=True)

    def verify(path: Path, expected: str) -> None:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("reuse file escapes declared cache root")
        if (
            len(expected) != 64
            or set(expected) - set("0123456789abcdef")
            or compute_file_hash(str(resolved)) != expected
        ):
            raise ValueError("reused cache content pin changed")

    verify(root / "IDENTITY.json", identity_sha256)
    path = root / f"family{family:02d}_query{query:05d}.npz"
    receipt_path = path.with_suffix(".json")
    verify(receipt_path, receipt_sha256)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["query"] != query or receipt["family"] != family:
        raise ValueError("reused receipt query/producer assignment changed")
    verify(path, receipt["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if not np.array_equal(arrays["observation_ids"], source_ids):
        raise ValueError("reused source history differs from pinned index")
    return rebind_block(
        arrays,
        source_keys=source_keys,
        destination_keys=destination_keys,
        destination_ids=destination_ids,
        anchors_us=anchors_us,
        available_us=available_us,
    )
