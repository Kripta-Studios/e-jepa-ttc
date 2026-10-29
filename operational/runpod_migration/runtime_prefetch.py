"""Byte-bounded Runpod runtime overlay built on frozen OOF overlay APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from operational.final_refit.producers import ParallelRGBDecoder
from operational.oof_prefetch import EffectiveBatchRecipe, PopulationSource
from operational.oof_runtime_overlay import FoldPrefetchRuntime, RuntimeStatsCallback

from .bounded_prefetch import ByteBoundedFoldPrefetch, SharedByteBudget


class ByteBoundedFoldPrefetchRuntime(FoldPrefetchRuntime):
    """Add explicit combined memory admission without editing frozen runtime bytes."""

    def __init__(
        self,
        module: object | None = None,
        *,
        parallel_rgb_decode: bool = True,
        checkpoint_binding_kwargs: Mapping[str, str | None] | None = None,
        stats_callback: RuntimeStatsCallback | None = None,
        prefetch_depth: int = 2,
        shared_limit_bytes: int = 512 * 1024**2,
        reservation_bytes_per_batch: int = 192 * 1024**2,
        decoder_max_pending_bytes: int = 128 * 1024**2,
    ) -> None:
        if prefetch_depth not in {1, 2}:
            raise ValueError("prefetch depth must be one or two")
        if decoder_max_pending_bytes <= 0:
            raise ValueError("decoder byte bound must be positive")
        self.prefetch_depth = prefetch_depth
        self.byte_budget = SharedByteBudget(shared_limit_bytes)
        self.reservation_bytes_per_batch = reservation_bytes_per_batch
        self.decoder_max_pending_bytes = decoder_max_pending_bytes
        combined = prefetch_depth * reservation_bytes_per_batch + (
            decoder_max_pending_bytes if parallel_rgb_decode else 0
        )
        if combined > shared_limit_bytes:
            raise ValueError("combined decoder/prefetch bounds exceed shared byte limit")
        super().__init__(
            module,
            parallel_rgb_decode=parallel_rgb_decode,
            checkpoint_binding_kwargs=checkpoint_binding_kwargs,
            stats_callback=stats_callback,
        )

    def _ensure_prefetch(self, source: object, recipe: object) -> None:
        self._require_rgb(source)
        if self._cursor is None:
            raise RuntimeError("checkpoint save/restore must expose the cursor before batching")
        if self._prefetch is not None:
            if source is not self._source or recipe is not self._recipe:
                raise RuntimeError("one runtime overlay cannot serve multiple sources or recipes")
            return
        if self._original_effective_batch is None:
            raise RuntimeError("runtime overlay is not active")
        if self.parallel_rgb_decode:
            base = getattr(source, "base", None)
            if base is None:
                raise ValueError("parallel RGB decoding requires a fold source with a base source")
            self._decoder = ParallelRGBDecoder(
                base,
                workers=4,
                max_pending_bytes=self.decoder_max_pending_bytes,
            )
        original = self._original_effective_batch

        def build_batch(
            batch_source: PopulationSource,
            indices: tuple[int, ...],
            batch_recipe: EffectiveBatchRecipe,
        ) -> object:
            return original(batch_source, indices, batch_recipe)

        self._source = source
        self._recipe = recipe
        try:
            self._prefetch = cast(
                Any,
                ByteBoundedFoldPrefetch(
                    cast(PopulationSource, source),
                    cast(EffectiveBatchRecipe, recipe),
                    self._cursor,
                    build_batch,
                    depth=self.prefetch_depth,
                    byte_budget=self.byte_budget,
                    reservation_bytes_per_batch=self.reservation_bytes_per_batch,
                ),
            )
        except BaseException:
            if self._decoder is not None:
                self._decoder.close()
                self._decoder = None
            self._source = None
            self._recipe = None
            raise

    def snapshot(self) -> dict[str, object]:
        return {
            **super().snapshot(),
            "schema": "runpod_byte_bounded_oof_prefetch_v1",
            "prefetch_depth": self.prefetch_depth,
            "reservation_bytes_per_batch": self.reservation_bytes_per_batch,
            "decoder_max_pending_bytes": self.decoder_max_pending_bytes,
            "shared_byte_budget": self.byte_budget.snapshot(),
        }


__all__ = ["ByteBoundedFoldPrefetchRuntime"]
