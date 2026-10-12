"""Bounded speculative input loading that never advances the authoritative sampler early."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Generic, TypeVar

import torch

Batch = TypeVar("Batch")


class UniformPrefetch(Generic[Batch]):
    """Load one future batch using a private sampler copy; commit draws only on consumption."""

    def __init__(
        self,
        load: Callable[[list[int]], Batch],
        population: int,
        batch_size: int,
        sampler: torch.Generator,
    ) -> None:
        if population < 1 or batch_size < 1 or sampler.device.type != "cpu":
            raise ValueError("Prefetch requires a positive population/batch and CPU sampler")
        self.load = load
        self.population = population
        self.batch_size = batch_size
        self.sampler = sampler
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v18-input")
        self.future: Future[tuple[list[int], Batch]] | None = None
        self.closed = False

    def _forecast(self) -> list[int]:
        shadow = torch.Generator().set_state(self.sampler.get_state())
        return torch.randint(self.population, (self.batch_size,), generator=shadow).tolist()

    def _load(self, indices: list[int]) -> tuple[list[int], Batch]:
        return indices, self.load(indices)

    def __enter__(self) -> UniformPrefetch[Batch]:
        if self.closed or self.future is not None:
            raise RuntimeError("A prefetch instance can only be entered once")
        self.future = self.executor.submit(self._load, self._forecast())
        return self

    def next(self) -> tuple[list[int], Batch]:
        """Consume the next canonical draw and preload one successor without reserving it."""
        if self.closed or self.future is None:
            raise RuntimeError("Enter the prefetch context before consuming a batch")
        # Verify the forecast before changing the checkpointed generator state.
        predicted, batch = self.future.result()
        if predicted != self._forecast():
            raise ValueError("Authoritative sampler changed outside the input consumer")
        indices = torch.randint(
            self.population, (self.batch_size,), generator=self.sampler
        ).tolist()
        if indices != predicted:
            raise AssertionError("Sampler forecast differs from the committed draw")
        self.future = self.executor.submit(self._load, self._forecast())
        return indices, batch

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish or cancel the unconsumed read before its dataset owner closes handles."""
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
