"""Ordered, bounded CPU preparation for unchanged single-frame GPU evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from e_jepa_ttc.detection.bounded_prefetch import BoundedOrderedPrefetch


class EvaluationCache(Protocol):
    """CPU-only, label-free cache owned by one evaluation worker at a time."""

    dataset: Any
    label_free_model_input: bool

    def get(self, index: int) -> tuple[Any, Any]: ...


class _EvaluationLoader:
    def __init__(
        self,
        cache: EvaluationCache,
        truth_builder: Callable[[Any], Any],
        legacy_builder: Callable[[Any], Any],
    ) -> None:
        self.cache = cache
        self.truth_builder = truth_builder
        self.legacy_builder = legacy_builder

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        started = time.perf_counter()
        prepared, raw = self.cache.get(index)
        if prepared.image.device.type != "cpu":
            raise ValueError("Evaluation prefetch may prepare CPU inputs only")
        if prepared.native_ground_truth or prepared.target["labels"].numel():
            raise ValueError("Evaluation model inputs must remain annotation-empty")
        loaded = time.perf_counter()
        truth = self.truth_builder(raw)
        historical = self.legacy_builder(raw)
        return (
            prepared,
            raw,
            truth,
            historical,
            {
                "load_prepare_cpu_seconds": loaded - started,
                "ground_truth_cpu_seconds": time.perf_counter() - loaded,
            },
        )

    def close(self) -> None:
        # The caller retains the cache/dataset across scheduled evaluations.
        # BoundedOrderedPrefetch joins the worker before returning ownership.
        pass


def prefetch_evaluation_rows(
    cache: EvaluationCache,
    truth_builder: Callable[[Any], Any],
    legacy_builder: Callable[[Any], Any],
) -> BoundedOrderedPrefetch[int, tuple[Any, ...]]:
    """Overlap CPU preparation with B1 inference; preserve order and exact inputs.

    At most two speculative frames exist. The worker exclusively owns cache
    access until the context closes. Model inference, decoding, scoring and RNG
    ownership remain in the main thread. Ground truth never enters model inputs.
    """
    if not cache.label_free_model_input:
        raise ValueError("Evaluation requires a label-free cache")
    return BoundedOrderedPrefetch(
        range(len(cache.dataset)),
        lambda: _EvaluationLoader(cache, truth_builder, legacy_builder),
        capacity=2,
        thread_name="evaluation-cpu-prefetch",
    )
