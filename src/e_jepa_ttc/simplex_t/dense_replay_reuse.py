"""Bridge the verified D0 reuse catalog to expanded replay queue lookups."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .expanded_history_loader import ExpandedHistoryLoader
from .reuse_catalog import D0ReuseCatalog


class DenseReplayReuse:
    """Keep one source fold and its destination content keys in memory.

    Expected D0 identities come from the pinned compiled-fold configuration.
    A missing source fold raises; this never treats missing reuse as permission
    to rerun experts on all original queries. None means a genuinely new token.
    """

    def __init__(
        self,
        *,
        catalog_loader: Callable[[int], D0ReuseCatalog],
        expected_identities: dict[int, dict],
        histories: ExpandedHistoryLoader,
        index: dict[str, np.ndarray],
        extraction_identity: dict,
    ) -> None:
        if extraction_identity.get("pool") != "DENSE_OLD":
            raise ValueError("D0 replay reuse is restricted to DENSE_OLD")
        self.catalog_loader = catalog_loader
        self.expected_identities = expected_identities
        self.histories, self.index, self.identity = histories, index, extraction_identity
        self.catalog: D0ReuseCatalog | None = None
        self.outer: int | None = None
        self.records: dict[int, dict[str, int | str]] = {}
        self.keys: np.ndarray | None = None
        self.history: np.ndarray | None = None

    def __call__(self, family: int, query: int) -> dict[str, np.ndarray] | None:
        """Return a verified and remapped block, or None for a new query token."""
        if family not in {0, 1, 2, 4, 5, 6, 8, 9, 10}:
            raise ValueError("DENSE reuse requires an inner producer")
        outer = family // 4
        if (
            not 0 <= query < len(self.index["tokens"])
            or self.index["producer_family"][outer, query] != family
        ):
            raise ValueError("DENSE reuse query family differs")
        if self.outer != outer:
            if outer not in self.expected_identities:
                raise ValueError("complete D0 source identity missing for dense fold")
            catalog = self.catalog_loader(outer)
            if catalog.identity != self.expected_identities[outer] or catalog.outer != outer:
                raise ValueError("D0 reuse catalog differs from frozen source")
            catalog.verify_recipe(self.identity)
            history = self.histories(outer)
            record = self.histories.records[outer]
            path = self.histories.root / record["path"]
            if sha256(path) != record["sha256"]:
                raise ValueError("DENSE destination keys changed")
            with np.load(path, allow_pickle=False) as archive:
                keys = archive["keys"]
            records = catalog.plan(self.index["tokens"], self.index["producer_family"][outer])
            self.catalog, self.history, self.keys = catalog, history, keys
            self.records, self.outer = records, outer
        assert self.catalog is not None and self.keys is not None and self.history is not None
        self.catalog.verify_recipe(self.identity)
        record = self.records.get(query)
        if record is None:
            return None
        mask = self.index["valid"][query]
        return self.catalog.load(
            record,
            destination_keys=self.keys,
            destination_ids=self.history[query, mask],
            anchors_us=self.index["anchor_us"][query] - self.index["lag_us"][mask],
            available_us=int(self.index["roi_available_us"][query]),
        )
