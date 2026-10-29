from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import operational.oof_runtime_overlay as overlay_module
from operational.oof_runtime_overlay import FoldPrefetchRuntime


class _Source:
    def __init__(self, *, modality: str = "rgb") -> None:
        self.identity = {"modality": modality}
        self.base = object()
        self.population_size = 5


class _Recipe:
    effective_batch_size = 2


class _Checkpoint:
    restored_cursor: dict[str, object] | None = None
    instances: list[_Checkpoint] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.saved: list[tuple[object, str]] = []
        type(self).instances.append(self)

    def restore(self, *args: object, **kwargs: object) -> dict[str, object]:
        assert not kwargs
        assert args
        assert self.restored_cursor is not None
        return self.restored_cursor

    def save(
        self,
        model: object,
        optimizer: object,
        scheduler: object,
        generator: object,
        cursor: object,
        *,
        status: str,
    ) -> None:
        assert (model, optimizer, scheduler, generator) == (1, 2, 3, 4)
        self.saved.append((cursor, status))


def _module(calls: list[tuple[int, ...]]) -> SimpleNamespace:
    def effective_batch(source: object, ids: tuple[int, ...], recipe: object) -> tuple[int, ...]:
        assert isinstance(source, _Source)
        assert isinstance(recipe, _Recipe)
        calls.append(tuple(ids))
        return tuple(ids)

    return SimpleNamespace(ProducerCheckpoint=_Checkpoint, _effective_batch=effective_batch)


def _cursor(position: int = 0) -> dict[str, object]:
    return {"epoch": 1, "position": position, "order": [0, 1, 2, 3, 4], "losses": []}


def test_new_run_save_captures_cursor_prefetches_and_restores_symbols() -> None:
    calls: list[tuple[int, ...]] = []
    module = _module(calls)
    original_checkpoint = module.ProducerCheckpoint
    original_batch = module._effective_batch
    snapshots: list[dict[str, object]] = []
    cursor = _cursor()
    source = _Source()

    with FoldPrefetchRuntime(module, parallel_rgb_decode=False, stats_callback=snapshots.append):
        assert module.ProducerCheckpoint is not original_checkpoint
        assert module._effective_batch is not original_batch
        checkpoint = module.ProducerCheckpoint("run", identity={})
        checkpoint.save(1, 2, 3, 4, cursor, status="READY")
        assert snapshots[-1]["cursor_captured"] is True
        recipe = _Recipe()
        assert module._effective_batch(source, [0, 1], recipe) == (0, 1)
        cursor["position"] = 2
        assert module._effective_batch(source, [2, 3], recipe) == (2, 3)

    assert module.ProducerCheckpoint is original_checkpoint
    assert module._effective_batch is original_batch
    assert calls[:2] == [(0, 1), (2, 3)]


def test_restore_captures_exact_cursor_reference_and_binding_kwargs() -> None:
    calls: list[tuple[int, ...]] = []
    module = _module(calls)
    cursor = _cursor(position=2)
    _Checkpoint.restored_cursor = cursor
    binding = {
        "execution_binding_sha256": "a" * 64,
        "execution_binding_origin_checkpoint_sha256": "b" * 64,
        "execution_binding_origin_prior_sha256": None,
    }

    with FoldPrefetchRuntime(module, parallel_rgb_decode=False, checkpoint_binding_kwargs=binding):
        checkpoint = module.ProducerCheckpoint("run", identity={})
        restored = checkpoint.restore(1)
        assert restored is cursor
        assert checkpoint.kwargs == {"identity": {}, **binding}
        source = _Source()
        assert module._effective_batch(source, [2, 3], _Recipe()) == (2, 3)


def test_parallel_decoder_uses_exact_bounds_and_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, ...]] = []
    module = _module(calls)
    created: list[object] = []

    class Decoder:
        def __init__(self, source: object, *, workers: int, max_pending_bytes: int) -> None:
            self.source = source
            self.workers = workers
            self.max_pending_bytes = max_pending_bytes
            self.closed = False
            created.append(self)

        def snapshot(self) -> dict[str, int]:
            return {"workers": self.workers}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(overlay_module, "ParallelRGBDecoder", Decoder)
    source = _Source()
    with FoldPrefetchRuntime(module, parallel_rgb_decode=True):
        checkpoint = module.ProducerCheckpoint()
        checkpoint.save(1, 2, 3, 4, _cursor(), status="READY")
        assert module._effective_batch(source, [0, 1], _Recipe()) == (0, 1)
        assert len(created) == 1
        decoder = cast(Decoder, created[0])
        assert decoder.source is source.base
        assert decoder.workers == 4
        assert decoder.max_pending_bytes == 128 * 1024**2
    assert decoder.closed is True


def test_non_rgb_source_is_rejected() -> None:
    module = _module([])
    original_batch = module._effective_batch
    with FoldPrefetchRuntime(module, parallel_rgb_decode=False):
        checkpoint = module.ProducerCheckpoint()
        checkpoint.save(1, 2, 3, 4, _cursor(), status="READY")
        with pytest.raises(ValueError, match="RGB"):
            module._effective_batch(_Source(modality="event"), [0, 1], _Recipe())
    assert module._effective_batch is original_batch


def test_mismatched_new_cursor_after_prefetch_fails_closed() -> None:
    module = _module([])
    with FoldPrefetchRuntime(module, parallel_rgb_decode=False):
        checkpoint = module.ProducerCheckpoint()
        checkpoint.save(1, 2, 3, 4, _cursor(), status="READY")
        source = _Source()
        module._effective_batch(source, [0, 1], _Recipe())
        with pytest.raises(RuntimeError, match="cursor identity changed"):
            checkpoint.save(1, 2, 3, 4, _cursor(position=2), status="RUNNING")


def test_builder_failure_propagates_and_symbols_are_restored() -> None:
    def broken(source: object, ids: tuple[int, ...], recipe: object) -> object:
        raise LookupError(f"bad batch {ids}")

    module = SimpleNamespace(ProducerCheckpoint=_Checkpoint, _effective_batch=broken)
    original_checkpoint = module.ProducerCheckpoint
    original_batch = module._effective_batch
    with pytest.raises(LookupError, match="bad batch"):
        with FoldPrefetchRuntime(module, parallel_rgb_decode=False):
            checkpoint = module.ProducerCheckpoint()
            checkpoint.save(1, 2, 3, 4, _cursor(), status="READY")
            module._effective_batch(_Source(), [0, 1], _Recipe())
    assert module.ProducerCheckpoint is original_checkpoint
    assert module._effective_batch is original_batch


def test_cleanup_failure_propagates_after_restoring_both_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module([])
    original_checkpoint = module.ProducerCheckpoint
    original_batch = module._effective_batch

    class BrokenPrefetch:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get(self, indices: object) -> tuple[int, ...]:
            return (0, 1)

        def snapshot(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            raise OSError("drain failed")

    monkeypatch.setattr(overlay_module, "BoundedFoldPrefetch", BrokenPrefetch)
    with pytest.raises(OSError, match="drain failed"):
        with FoldPrefetchRuntime(module, parallel_rgb_decode=False):
            checkpoint = module.ProducerCheckpoint()
            checkpoint.save(1, 2, 3, 4, _cursor(), status="READY")
            module._effective_batch(_Source(), [0, 1], _Recipe())
    assert module.ProducerCheckpoint is original_checkpoint
    assert module._effective_batch is original_batch
