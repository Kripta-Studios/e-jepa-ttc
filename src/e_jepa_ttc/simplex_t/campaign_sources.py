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
from .current_inputs import load_current_inputs
from .dense_file_sources import DenseBinding, load_dense_inputs
from .dense_sources import dense_source_pair
from .expanded_sources import merge_validated_sources
from .expansion_sources import ExpansionBinding, load_expansion_source
from .phase_manifest import fit_key
from .registry import FitSpec


@dataclass(frozen=True)
class CompiledFold:
    """Explicit immutable compiled manifest binding for one outer fold."""

    path: Path
    sha256: str


class CampaignSources:
    """Supply queue TRAIN callbacks and separate OLD_DEV endpoint inputs.

    D1 requires explicit compiled expansion bindings for every outer fold and
    an acknowledged TRAIN sequence set. No pool is silently mapped onto D0.
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
        expansion_folds: dict[int, ExpansionBinding] | None = None,
        expansion_sequences: set[str] | None = None,
        dense_folds: dict[int, DenseBinding] | None = None,
    ) -> None:
        if not graph or len({fit_key(spec) for spec in graph}) != len(graph):
            raise ValueError("nonempty unique registered graph required")
        for spec in graph:
            pool = resolve_arm(spec, graph).pool
            supported = (
                pool == "D0"
                or (pool == "D1" and expansion_folds is not None and bool(expansion_sequences))
                or (pool == "DENSE_OLD" and dense_folds is not None)
            )
            if not supported:
                raise ValueError("pool lacks an integrated authoritative source loader")
        if dense_folds is not None and set(dense_folds) != {0, 1, 2}:
            raise ValueError("all three dense fold bindings required")
        if expansion_folds is not None and set(expansion_folds) != {0, 1, 2}:
            raise ValueError("all three expansion fold bindings required")
        if set(expansion_sequences or ()) & allowed_sequences:
            raise ValueError("expansion and original authorized sequences overlap")
        if set(folds) != {0, 1, 2}:
            raise ValueError("all three compiled outer-fold bindings required")
        self.graph = list(graph)
        self.folds = dict(folds)
        self.index_root = index_root
        self.dedup_root = dedup_root
        self.historical_root = historical_root
        self.ancestry_sha256 = ancestry_sha256
        self.allowed_sequences = set(allowed_sequences)
        self.expansion_folds = dict(expansion_folds or {})
        self.expansion_sequences = set(expansion_sequences or ())
        self.dense_folds = dict(dense_folds or {})
        self._key: tuple[int, int, str] | None = None
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
        if binding.pool == "D1":
            expansion_pin = self.expansion_folds[spec.fold]
            if compute_file_hash(str(expansion_pin.compiled / "COMPILED.json")) != (
                expansion_pin.compiled_sha256
            ):
                raise ValueError("expansion compiled manifest changed since preparation")
        if binding.pool == "DENSE_OLD":
            dense_pin = self.dense_folds[spec.fold]
            if compute_file_hash(str(dense_pin.compiled / "COMPILED.json")) != (
                dense_pin.compiled_sha256
            ):
                raise ValueError("dense compiled manifest changed since preparation")
        key = (spec.fold, binding.model.feature_count, binding.pool)
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
            if binding.pool == "D1":
                expansion, sequences = load_expansion_source(
                    self.expansion_folds[spec.fold],
                    outer=spec.fold,
                    feature_count=binding.model.feature_count,
                    allowed_sequences=self.expansion_sequences,
                )
                original = load_current_inputs(
                    self.historical_root,
                    spec.fold,
                    "inner_oof",
                    ancestry_sha256=self.ancestry_sha256,
                    allowed_sequences=self.allowed_sequences,
                )["metadata"]
                sources = merge_validated_sources(
                    sources,
                    expansion,
                    original_train_sequences=original.sequence_id.to_numpy(),
                    expansion_train_sequences=sequences,
                )
            elif binding.pool == "DENSE_OLD":
                dense = load_dense_inputs(
                    self.dense_folds[spec.fold],
                    outer=spec.fold,
                    allowed_sequences=self.allowed_sequences,
                )
                metadata = {
                    role: load_current_inputs(
                        self.historical_root,
                        spec.fold,
                        role,
                        ancestry_sha256=self.ancestry_sha256,
                        allowed_sequences=self.allowed_sequences,
                    )["metadata"]
                    for role in ("inner_oof", "outer_dev")
                }
                sources = dense_source_pair(
                    sources,
                    dense.source,
                    original_tokens=metadata["inner_oof"].sample_token.to_numpy(),
                    original_sequences=metadata["inner_oof"].sequence_id.to_numpy(),
                    dev_sequences=metadata["outer_dev"].sequence_id.to_numpy(),
                    dense_tokens=dense.tokens,
                    dense_sequences=dense.sequences,
                    dense_target_ttc=dense.target_ttc,
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
                key=lambda s: (
                    s.fold,
                    resolve_arm(s, self.graph).model.feature_count,
                    resolve_arm(s, self.graph).pool,
                    fit_key(s),
                ),
            ):
                result[fit_key(spec)] = {
                    role: self.source(spec, role).identity_sha256
                    for role in ("inner_oof", "outer_dev")
                }
        finally:
            self.release()
        return result
