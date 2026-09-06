"""Connect role-validated compiled caches to the registered scientific fit queue.

This adapter does not authorize fits or open evaluation scores. It retains only
one fold/feature family at a time and uses the same binding for preparation,
training, and fixed-endpoint inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .arms import resolve_arm
from .cache import CachedQueries
from .context_sources import load_context_sources
from .phase_manifest import fit_key
from .registry import FitSpec


@dataclass(frozen=True)
class CompiledFold:
    """Explicit immutable compiled manifest binding for one outer fold."""

    path: Path
    sha256: str


class CampaignSources:
    """Supply queue TRAIN callbacks and separate OLD_DEV endpoint inputs.

    Only D0 query-context caches have a production loader at present. A graph
    containing another pool is rejected before loading anything, never silently
    mapped onto D0. D1 requires its own acknowledged lineage and pool adapter.
    """

    def __init__(
        self,
        graph: list[FitSpec],
        folds: dict[int, CompiledFold],
        *,
        index_root: Path,
        dedup_root: Path,
        historical_root: Path,
        ancestry_sha256: str,
        allowed_sequences: set[str],
    ) -> None:
        if not graph or len({fit_key(spec) for spec in graph}) != len(graph):
            raise ValueError("nonempty unique registered graph required")
        for spec in graph:
            if resolve_arm(spec, graph).pool != "D0":
                raise ValueError("pool lacks an integrated authoritative source loader")
        if set(folds) != {0, 1, 2}:
            raise ValueError("all three compiled outer-fold bindings required")
        self.graph = list(graph)
        self.folds = dict(folds)
        self.index_root = index_root
        self.dedup_root = dedup_root
        self.historical_root = historical_root
        self.ancestry_sha256 = ancestry_sha256
        self.allowed_sequences = set(allowed_sequences)
        self._key: tuple[int, int] | None = None
        self._sources: dict[str, CachedQueries] = {}

    def release(self) -> None:
        """Drop retained arrays; callers must also release their returned sources."""
        self._sources.clear()
        self._key = None

    def source(self, spec: FitSpec, role: str) -> CachedQueries:
        """Bind one canonical arm without confusing TRAIN and OLD_DEV roles."""
        if role not in {"inner_oof", "outer_dev"}:
            raise ValueError("only TRAIN and OLD_DEV roles are authorized")
        binding = resolve_arm(spec, self.graph)
        pin = self.folds[spec.fold]
        manifest = pin.path / "COMPILED.json"
        # Recheck even for a retained source, so a changed pin fails before reuse.
        if compute_file_hash(str(manifest)) != pin.sha256:
            raise ValueError("compiled manifest changed since preparation")
        if json.loads(manifest.read_text(encoding="utf-8"))["outer"] != spec.fold:
            raise ValueError("compiled outer fold differs from registered fit")
        key = (spec.fold, binding.model.feature_count)
        if self._key != key:
            self.release()
            sources = load_context_sources(
                pin.path,
                self.index_root,
                self.dedup_root,
                self.historical_root,
                compiled_manifest_sha256=pin.sha256,
                ancestry_sha256=self.ancestry_sha256,
                allowed_sequences=self.allowed_sequences,
                feature_count=binding.model.feature_count,
            )
            self._sources, self._key = sources, key
        return binding.source(self._sources[role])

    def train(self, spec: FitSpec) -> CachedQueries:
        """Production source_loader callback for queue.run_phase."""
        return self.source(spec, "inner_oof")

    def identities(self) -> dict[str, dict[str, str]]:
        """Resolve every source hash before freeze, without fitting or scoring.

        Sorting reduces repeated fold loading but does not change queue order.
        Returned source identities bind TRAIN-fitted normalization and controls.
        """
        result = {}
        try:
            for spec in sorted(
                self.graph,
                key=lambda s: (s.fold, resolve_arm(s, self.graph).model.feature_count, fit_key(s)),
            ):
                result[fit_key(spec)] = {
                    role: self.source(spec, role).identity_sha256
                    for role in ("inner_oof", "outer_dev")
                }
        finally:
            self.release()
        return result
