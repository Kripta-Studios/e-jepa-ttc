"""Temporary runtime overlay for deterministic RGB OOF producer prefetch."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, cast

from operational.final_refit.producers import ParallelRGBDecoder
from operational.oof_prefetch import (
    BoundedFoldPrefetch,
    EffectiveBatchRecipe,
    PopulationSource,
)

RuntimeStatsCallback = Callable[[dict[str, object]], None]


class FoldPrefetchRuntime:
    """Patch one frozen producer module only while its original train call runs.

    The overlay replaces only the module-local checkpoint class and effective-batch
    function. It leaves model, optimizer, scheduler, loss, cursor mutation, and RNG
    ownership with the frozen training implementation.
    """

    def __init__(
        self,
        module: ModuleType | object | None = None,
        *,
        parallel_rgb_decode: bool = True,
        checkpoint_binding_kwargs: Mapping[str, str | None] | None = None,
        stats_callback: RuntimeStatsCallback | None = None,
    ) -> None:
        self.module = (
            importlib.import_module("operational.final_refit.scientific_fold_producers")
            if module is None
            else module
        )
        self.parallel_rgb_decode = parallel_rgb_decode
        self.checkpoint_binding_kwargs = dict(checkpoint_binding_kwargs or {})
        self.stats_callback = stats_callback
        self._original_checkpoint: type[object] | None = None
        self._original_effective_batch: Callable[[object, Sequence[int], object], object] | None = (
            None
        )
        self._cursor: Mapping[str, object] | None = None
        self._prefetch: BoundedFoldPrefetch[Any, Any, object] | None = None
        self._decoder: ParallelRGBDecoder | None = None
        self._source: object | None = None
        self._recipe: object | None = None
        self._active = False

    def _capture_cursor(self, cursor: object) -> Mapping[str, object]:
        if not isinstance(cursor, Mapping):
            raise TypeError("producer checkpoint cursor must be a mapping")
        typed = cast(Mapping[str, object], cursor)
        if self._prefetch is not None and typed is not self._cursor:
            raise RuntimeError("producer cursor identity changed while prefetch was active")
        self._cursor = typed
        return typed

    @staticmethod
    def _require_rgb(source: object) -> object:
        identity = getattr(source, "identity", None)
        if not isinstance(identity, Mapping) or identity.get("modality") != "rgb":
            raise ValueError("FoldPrefetchRuntime supports RGB fold sources only")
        return source

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
                max_pending_bytes=128 * 1024**2,
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
            self._prefetch = BoundedFoldPrefetch(
                cast(PopulationSource, source),
                cast(EffectiveBatchRecipe, recipe),
                self._cursor,
                build_batch,
                depth=2,
            )
        except BaseException:
            if self._decoder is not None:
                self._decoder.close()
                self._decoder = None
            self._source = None
            self._recipe = None
            raise

    def _batch(self, source: object, indices: Sequence[int], recipe: object) -> object:
        self._ensure_prefetch(source, recipe)
        if self._prefetch is None:
            raise RuntimeError("OOF prefetch was not initialized")
        return self._prefetch.get(indices)

    def snapshot(self) -> dict[str, object]:
        """Return technical runtime counters suitable for a checkpoint sidecar."""

        return {
            "schema": "oof_fold_prefetch_runtime_v1",
            "active": self._active,
            "cursor_captured": self._cursor is not None,
            "parallel_rgb_decode": self.parallel_rgb_decode,
            "prefetch": self._prefetch.snapshot() if self._prefetch is not None else None,
            "decoder": self._decoder.snapshot() if self._decoder is not None else None,
        }

    def _after_save(self) -> None:
        if self.stats_callback is not None:
            self.stats_callback(self.snapshot())

    def __enter__(self) -> FoldPrefetchRuntime:
        if self._active:
            raise RuntimeError("FoldPrefetchRuntime cannot be entered twice")
        checkpoint = getattr(self.module, "ProducerCheckpoint", None)
        effective_batch = getattr(self.module, "_effective_batch", None)
        if not isinstance(checkpoint, type) or not callable(effective_batch):
            raise TypeError("target module lacks the frozen producer runtime symbols")
        self._original_checkpoint = checkpoint
        self._original_effective_batch = cast(
            Callable[[object, Sequence[int], object], object], effective_batch
        )
        runtime = self

        class RuntimeProducerCheckpoint(checkpoint):  # type: ignore[valid-type, misc]
            def __init__(self, *args: object, **kwargs: object) -> None:
                injected = dict(kwargs)
                injected.update(runtime.checkpoint_binding_kwargs)
                super().__init__(*args, **injected)

            def restore(self, *args: object, **kwargs: object) -> Mapping[str, object]:
                cursor = super().restore(*args, **kwargs)
                return runtime._capture_cursor(cursor)

            def save(
                self,
                model: object,
                optimizer: object,
                scheduler: object,
                generator: object,
                cursor: Mapping[str, object],
                *,
                status: str,
            ) -> None:
                runtime._capture_cursor(cursor)
                super().save(
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    cursor,
                    status=status,
                )
                runtime._after_save()

        def effective_batch_wrapper(
            source: object, indices: Sequence[int], recipe: object
        ) -> object:
            return runtime._batch(source, indices, recipe)

        try:
            vars(self.module)["ProducerCheckpoint"] = RuntimeProducerCheckpoint
            vars(self.module)["_effective_batch"] = effective_batch_wrapper
        except BaseException:
            vars(self.module)["ProducerCheckpoint"] = checkpoint
            vars(self.module)["_effective_batch"] = effective_batch
            self._original_checkpoint = None
            self._original_effective_batch = None
            raise
        self._active = True
        return self

    def close(self) -> None:
        """Drain helpers and restore both frozen module symbols, even on failure."""

        if not self._active:
            return
        failures: list[BaseException] = []
        try:
            if self._prefetch is not None:
                try:
                    self._prefetch.close()
                except BaseException as error:
                    failures.append(error)
            if self._decoder is not None:
                try:
                    self._decoder.close()
                except BaseException as error:
                    failures.append(error)
        finally:
            if self._original_checkpoint is not None:
                vars(self.module)["ProducerCheckpoint"] = self._original_checkpoint
            if self._original_effective_batch is not None:
                vars(self.module)["_effective_batch"] = self._original_effective_batch
            self._active = False
            self._prefetch = None
            self._decoder = None
            self._source = None
            self._recipe = None
        if failures:
            raise failures[0]

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["FoldPrefetchRuntime", "RuntimeStatsCallback"]
