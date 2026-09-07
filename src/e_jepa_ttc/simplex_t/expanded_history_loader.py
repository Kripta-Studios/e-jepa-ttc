"""Read pinned expanded content indices without inventing inactive observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


class ExpandedHistoryLoader:
    """One-fold memory cache for metadata only; does not authorize raw inference."""

    def __init__(
        self,
        root: Path,
        *,
        manifest_sha256: str,
        index_manifest_sha256: str,
        pool: str,
        producer_family: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.path = self.root / "DEDUP_MANIFEST.json"
        self.digest = manifest_sha256
        if self.path.stat().st_size > 1_048_576 or sha256(self.path) != self.digest:
            raise ValueError("expanded dedup manifest changed")
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if pool == "D1":
            bound_index = manifest["identity"]["index_manifest_sha256"]
            expected_status = "D1_CONTENT_INDEX_READY_NOT_FEATURE_CACHE_OR_REPLAY_AUTHORIZATION"
        elif pool == "DENSE_OLD":
            bound_index = manifest["input_manifest_sha256"]
            expected_status = "DENSE_CONTENT_INDEX_READY_PENDING_TIME_ACK_AND_FEATURE_REPLAY"
        else:
            raise ValueError("registered expanded pool required")
        if bound_index != index_manifest_sha256 or manifest["status"] != expected_status:
            raise ValueError("dedup belongs to another index or pool")
        if (
            valid.ndim != 2
            or valid.shape[1] != 16
            or valid.dtype != bool
            or producer_family.shape != (3, len(valid))
            or len(manifest["outputs"]) != 3
        ):
            raise ValueError("expanded index dimensions differ")
        self.records = manifest["outputs"]
        self.families, self.valid = producer_family, valid
        self.outer: int | None = None
        self.history: np.ndarray | None = None

    def __call__(self, outer: int) -> np.ndarray:
        """Verify file bytes on reuse; only the current fold stays in memory."""
        if outer not in {0, 1, 2} or sha256(self.path) != self.digest:
            raise ValueError("dedup fold or manifest changed")
        record = self.records[outer]
        if record["path"] != f"outer{outer}.npz":
            raise ValueError("dedup fold path differs")
        path = (self.root / record["path"]).resolve(strict=True)
        if not path.is_relative_to(self.root) or sha256(path) != record["sha256"]:
            raise ValueError("dedup fold bytes changed")
        if self.outer == outer and self.history is not None:
            return self.history
        with np.load(path, allow_pickle=False) as archive:
            history = archive["history"]
            key_count = len(archive["keys"])
        active = self.families[outer] >= 0
        expected_valid = self.valid & active[:, None]
        if (
            history.dtype != np.int64
            or history.shape != self.valid.shape
            or (history < -1).any()
            or (history >= key_count).any()
            or not np.array_equal(history >= 0, expected_valid)
        ):
            raise ValueError("dedup history changes active observation identity or mask")
        history.setflags(write=False)
        self.history, self.outer = history, outer
        return history
