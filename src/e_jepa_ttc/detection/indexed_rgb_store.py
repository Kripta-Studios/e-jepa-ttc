"""Single-owner indexed RGB store with non-owning dataset views."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset, RawRGBSample
from e_jepa_ttc.detection.indexed_rgb_data import (
    IndexedEAPRGBDataset,
    IndexedTarRGBReader,
)


class IndexedRGBStore:
    """Own one union index and its bounded handles for several dataset views."""

    def __init__(
        self,
        root: Path | str,
        ownership: Mapping[str, tuple[str, str, str]],
        shard_sha256: Mapping[str, str],
        *,
        max_open_handles: int = 2,
    ) -> None:
        started = time.perf_counter()
        self.root = Path(root).resolve()
        self.ownership = dict(ownership)
        self.reader = IndexedTarRGBReader(
            self.root,
            self.ownership,
            shard_sha256,
            max_open_handles=max_open_handles,
        )
        self.index_build_seconds = time.perf_counter() - started
        self.read_count = 0
        self.read_bytes = 0
        self.read_seconds = 0.0
        self._closed = False

    @property
    def open_handle_count(self) -> int:
        """Expose the owned reader's bounded descriptor count."""
        return self.reader.open_handle_count

    @property
    def is_closed(self) -> bool:
        """Return whether the owner has ended the shared lifetime."""
        return self._closed

    def read(self, token: str) -> bytes:
        """Read through the shared index while accumulating store-level metrics."""
        if self._closed:
            raise RuntimeError("Indexed RGB store is closed")
        started = time.perf_counter()
        raw = self.reader.read(token)
        self.read_seconds += time.perf_counter() - started
        self.read_count += 1
        self.read_bytes += len(raw)
        return raw

    def dataset_view(
        self,
        sample_tokens: Sequence[str],
        *,
        media_path: str = "data/train.parquet",
        max_label_sequences: int = 2,
    ) -> BorrowedIndexedEAPRGBDataset:
        """Create a label-cache-owning view that borrows, but never closes, this store."""
        if self._closed:
            raise RuntimeError("Indexed RGB store is closed")
        return BorrowedIndexedEAPRGBDataset(
            self,
            sample_tokens,
            media_path=media_path,
            max_label_sequences=max_label_sequences,
        )

    def metrics(self) -> dict[str, int | float | bool]:
        """Return cumulative index/read metrics without resetting them."""
        return {
            "index_build_seconds": self.index_build_seconds,
            "read_count": self.read_count,
            "read_bytes": self.read_bytes,
            "read_seconds": self.read_seconds,
            "open_handle_count": self.open_handle_count,
            "closed": self._closed,
        }

    def close(self) -> None:
        """Close the single owned reader; borrowed views then fail closed."""
        if not self._closed:
            self.reader.close()
            self._closed = True

    def __enter__(self) -> IndexedRGBStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BorrowedIndexedEAPRGBDataset(IndexedEAPRGBDataset):
    """Dataset view with private label cache and a non-owning shared RGB store."""

    def __init__(
        self,
        store: IndexedRGBStore,
        sample_tokens: Sequence[str],
        *,
        media_path: str,
        max_label_sequences: int,
    ) -> None:
        EAPRGBDataset.__init__(
            self,
            store.root,
            sample_tokens,
            media_path=media_path,
            max_open_shards=1,
            max_label_sequences=max_label_sequences,
        )
        for token in self.sample_tokens:
            if token not in store.ownership:
                raise ValueError(f"Dataset token is absent from shared store: {token}")
            shard, member, _ = store.ownership[token]
            row = self.rows[token]
            if str(row["rgb_shard_path"]) != shard or str(row["rgb_member_path"]) != member:
                raise ValueError(f"Dataset media ownership differs from shared store: {token}")
        self.store = store
        self.indexed_reader = store

    def __getitem__(self, index: int) -> RawRGBSample:
        if self.store.is_closed:
            raise RuntimeError("Indexed RGB store is closed")
        return super().__getitem__(index)

    def close(self) -> None:
        """Clear this view's label cache without closing its borrowed RGB store."""
        EAPRGBDataset.close(self)
