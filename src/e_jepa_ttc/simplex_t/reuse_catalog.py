"""Bind a complete compiled D0 fold as a read-only source of reusable TRAIN blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .cache_reuse import load_reused_block


class D0ReuseCatalog:
    """A hash-bound D0 source, not authority to assign new data roles or run fits."""

    def __init__(
        self,
        *,
        compiled: Path,
        compiled_sha256: str,
        cache: Path,
        index_root: Path,
        dedup: Path,
        outer: int,
    ) -> None:
        def verify(path: Path, digest: str) -> None:
            if len(digest) != 64 or compute_file_hash(str(path)) != digest:
                raise ValueError("D0 reuse catalog pin mismatch")

        verify(compiled / "COMPILED.json", compiled_sha256)
        manifest = json.loads((compiled / "COMPILED.json").read_text(encoding="utf-8"))
        if (
            outer not in range(3)
            or manifest["outer"] != outer
            or "pool" in manifest
            or manifest["status"] != "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
        ):
            raise ValueError("reuse requires the complete original D0 fold")
        verify(cache / "IDENTITY.json", manifest["cache_identity_sha256"])
        self.extraction_identity = json.loads((cache / "IDENTITY.json").read_text(encoding="utf-8"))
        if self.extraction_identity["index_sha256"] != manifest["index_sha256"]:
            raise ValueError("D0 cache/index identity mismatch")
        verify(index_root / "query_context_index.npz", manifest["index_sha256"])
        verify(dedup, manifest["dedup_sha256"])
        with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
            tokens, families = archive["tokens"], archive["producer_family"][outer]
        with np.load(dedup, allow_pickle=False) as archive:
            self.keys, self.history = archive["keys"], archive["history"]
        if (
            tokens.ndim != 1
            or len(np.unique(tokens)) != len(tokens)
            or families.shape != tokens.shape
            or self.history.shape != (len(tokens), 16)
        ):
            raise ValueError("D0 reuse index schema mismatch")
        self.positions = {str(token): row for row, token in enumerate(tokens)}
        self.families, self.outer, self.cache = families, outer, cache
        self.identity = {
            "compiled_sha256": compiled_sha256,
            "cache_identity_sha256": manifest["cache_identity_sha256"],
            "source_index_sha256": manifest["index_sha256"],
            "source_dedup_sha256": manifest["dedup_sha256"],
        }

    def verify_recipe(self, destination: dict) -> None:
        """Require identical numerical extraction while allowing another runner/index/pool."""
        fields = (
            "preprocessing_sha256",
            "extractor_sha256",
            "expert_phase_sha256",
            "voxel_sha256",
            "union_reader_sha256",
            "torch",
            "batch_size",
            "layout",
            "precision",
            "tf32",
        )
        if any(
            name not in destination
            or name not in self.extraction_identity
            or destination[name] != self.extraction_identity[name]
            for name in fields
        ):
            raise ValueError("dense and reused D0 numerical extraction recipes differ")

    def plan(self, tokens: np.ndarray, families: np.ndarray) -> dict[int, dict[str, int | str]]:
        """Pin source receipts before the compiler allocates any output arrays."""
        result = {}
        for query in np.flatnonzero(families >= 0):
            source = self.positions.get(str(tokens[query]))
            if source is None:
                continue
            family = int(families[query])
            if family != self.families[source] or family // 4 != self.outer or family % 4 == 3:
                raise ValueError("dense reuse would change producer or consume outer-dev")
            receipt = self.cache / f"family{family:02d}_query{source:05d}.json"
            digest = compute_file_hash(str(receipt))
            row = json.loads(receipt.read_text(encoding="utf-8"))
            if row["query"] != source or row["family"] != family:
                raise ValueError("D0 source receipt identity mismatch")
            result[int(query)] = {
                "query": source,
                "family": family,
                "receipt_sha256": digest,
            }
        return result

    def load(
        self,
        record: dict[str, int | str],
        *,
        destination_keys: np.ndarray,
        destination_ids: np.ndarray,
        anchors_us: np.ndarray,
        available_us: int,
    ) -> dict[str, np.ndarray]:
        """Load and rebind one source, rechecking immutable file pins at consumption."""
        source = int(record["query"])
        history = self.history[source]
        return load_reused_block(
            self.cache,
            identity_sha256=self.identity["cache_identity_sha256"],
            receipt_sha256=str(record["receipt_sha256"]),
            query=source,
            family=int(record["family"]),
            source_ids=history[history >= 0],
            source_keys=self.keys,
            destination_keys=destination_keys,
            destination_ids=destination_ids,
            anchors_us=anchors_us,
            available_us=available_us,
        )
