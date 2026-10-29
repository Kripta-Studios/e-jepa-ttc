"""Move unchanged CPU input checks into the bounded prefetch worker."""
# ruff: noqa: ANN401 -- adapts the frozen producer's heterogeneous batch API.

from __future__ import annotations

import threading
import time
from typing import Any

from operational.oof_runtime_overlay import FoldPrefetchRuntime


def _fingerprint(part: Any) -> tuple[Any, ...]:
    events = part.events
    if events.device.type != "cpu":
        raise ValueError("Validation receipt requires CPU inputs")
    return (
        id(events),
        events.data_ptr(),
        events._version,
        tuple(events.shape),
        tuple(events.stride()),
        events.dtype,
        id(part.dinov3_relation_targets),
        id(part.dinov3_relation_valid),
    )


class ValidatedFoldPrefetchRuntime(FoldPrefetchRuntime):
    """Retain original validation and consume single-use, mutation-checked receipts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._validation_lock = threading.Lock()
        self._tickets: dict[int, tuple[Any, Any, tuple[Any, ...]]] = {}
        self._validation_seconds = 0.0
        self._validated_parts = 0
        self._consumed_parts = 0
        self._saved_validator: Any = None
        self._saved_builder: Any = None

    def __enter__(self) -> ValidatedFoldPrefetchRuntime:
        if self._active:
            raise RuntimeError("Validated runtime cannot be entered twice")
        original = self.module._effective_batch
        validate = self.module._validate_batch
        self._saved_builder, self._saved_validator = original, validate

        def build(source: Any, indices: Any, recipe: Any) -> Any:
            batch = original(source, indices, recipe)
            started = time.perf_counter()
            tickets = []
            for part in batch.parts:
                validate(part, recipe)
                tickets.append((id(part), (part, recipe, _fingerprint(part))))
            with self._validation_lock:
                for key, ticket in tickets:
                    if key in self._tickets:
                        raise RuntimeError("Duplicate validation receipt")
                    self._tickets[key] = ticket
                self._validated_parts += len(tickets)
                self._validation_seconds += time.perf_counter() - started
            return batch

        def consume(part: Any, recipe: Any) -> None:
            with self._validation_lock:
                ticket = self._tickets.pop(id(part), None)
            if ticket is None or ticket[0] is not part or ticket[1] is not recipe:
                raise ValueError("Missing single-use CPU validation receipt")
            if ticket[2] != _fingerprint(part):
                raise ValueError("CPU input changed after validation")
            self._consumed_parts += 1

        vars(self.module)["_effective_batch"] = build
        try:
            super().__enter__()
            vars(self.module)["_validate_batch"] = consume
        except BaseException:
            vars(self.module)["_effective_batch"] = original
            vars(self.module)["_validate_batch"] = validate
            raise
        return self

    def snapshot(self) -> dict[str, object]:
        result = super().snapshot()
        with self._validation_lock:
            result["cpu_validation"] = dict(
                validated_parts=self._validated_parts,
                consumed_parts=self._consumed_parts,
                seconds=self._validation_seconds,
                pending_receipts=len(self._tickets),
            )
        return result

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._saved_builder is not None:
                vars(self.module)["_effective_batch"] = self._saved_builder
                vars(self.module)["_validate_batch"] = self._saved_validator
            with self._validation_lock:
                self._tickets.clear()
