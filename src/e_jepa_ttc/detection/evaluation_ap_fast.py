"""Exact linear-time helpers for the local detection AP101 definition.

These helpers intentionally preserve the frozen scorers' curve construction,
threshold comparisons, and left-to-right sum order.  They are kept separate
until end-to-end parity has been demonstrated on retained predictions.
"""

from __future__ import annotations

from collections.abc import Sequence


def _suffix_max(values: Sequence[float]) -> list[float]:
    """Return inclusive suffix maxima without changing any input float."""
    result = [0.0] * len(values)
    maximum = 0.0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value > maximum:
            maximum = value
        result[index] = maximum
    return result


def _first_at_least(values: Sequence[float], threshold: float) -> int:
    """Match ``bisect_left`` for the nondecreasing recall sequence."""
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < threshold:
            low = middle + 1
        else:
            high = middle
    return low


def ap101_with_curve(
    flags: Sequence[int], count: int
) -> tuple[float | None, list[dict[str, float | int]]]:
    """Compute the frozen v2 AP101 result in linear time plus 101 lookups."""
    correct = 0
    curve: list[dict[str, float | int]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        recall = correct / count if count else 0.0
        precision = correct / rank
        curve.append({"rank": rank, "recall": recall, "precision": precision})
        recalls.append(recall)
        precisions.append(precision)
    if count == 0:
        return None, curve

    envelope = _suffix_max(precisions)
    samples: list[float] = []
    for step in range(101):
        index = _first_at_least(recalls, step / 100)
        samples.append(envelope[index] if index < len(envelope) else 0.0)
    return sum(samples) / 101, curve


def ap101_value(flags: Sequence[int], count: int) -> float | None:
    """Compute the frozen legacy scalar AP101 without materializing PR dicts."""
    if count == 0:
        return None
    correct = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        recalls.append(correct / count)
        precisions.append(correct / rank)

    envelope = _suffix_max(precisions)
    samples: list[float] = []
    for step in range(101):
        index = _first_at_least(recalls, step / 100)
        samples.append(envelope[index] if index < len(envelope) else 0.0)
    return sum(samples) / 101
