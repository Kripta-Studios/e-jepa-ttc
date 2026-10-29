"""Exactness tests for the isolated AP101 optimization candidate."""

from __future__ import annotations

import random

from e_jepa_ttc.detection import development_metrics, evaluation_v2
from e_jepa_ttc.detection.evaluation_ap_fast import ap101_value, ap101_with_curve


def frozen_v2_ap101(
    flags: list[int], count: int
) -> tuple[float | None, list[dict[str, float | int]]]:
    """Standalone copy of the hash-pinned pre-optimization v2 formula."""
    correct = 0
    curve: list[dict[str, float | int]] = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        curve.append(
            {
                "rank": rank,
                "recall": correct / count if count else 0.0,
                "precision": correct / rank,
            }
        )
    if count == 0:
        return None, curve
    return (
        sum(
            max(
                (
                    float(point["precision"])
                    for point in curve
                    if point["recall"] >= step / 100
                ),
                default=0.0,
            )
            for step in range(101)
        )
        / 101,
        curve,
    )


def frozen_legacy_ap101(flags: list[int], count: int) -> float | None:
    """Standalone copy of the hash-pinned pre-optimization legacy formula."""
    if count == 0:
        return None
    correct = 0
    points = []
    for index, flag in enumerate(flags):
        correct += flag
        points.append((correct / count, correct / (index + 1)))
    return sum(
        max((precision for recall, precision in points if recall >= step / 100), default=0.0)
        for step in range(101)
    ) / 101


def assert_exact_against_frozen(flags: list[int], count: int) -> None:
    expected_v2 = frozen_v2_ap101(flags, count)
    expected_legacy = frozen_legacy_ap101(flags, count)
    assert ap101_with_curve(flags, count) == expected_v2
    assert evaluation_v2._ap101(flags, count) == expected_v2
    assert ap101_value(flags, count) == expected_legacy
    assert development_metrics._ap101(flags, count) == expected_legacy


def test_adversarial_cases_match_both_frozen_helpers_exactly() -> None:
    cases = (
        ([], 0),
        ([], 7),
        ([0], 1),
        ([1], 1),
        ([0, 0, 0, 0], 3),
        ([1, 1, 1, 1], 4),
        ([0, 1] * 20, 20),
        ([1, 0] * 20, 20),
        ([0] * 99 + [1], 1),
        ([1] * 100 + [0] * 100, 100),
        ([1, 0, 0, 1, 0, 1, 0, 0, 1], 13),
    )
    for flags, count in cases:
        assert_exact_against_frozen(flags, count)


def test_seeded_random_flags_match_both_frozen_helpers_exactly() -> None:
    generator = random.Random(7)
    for _ in range(500):
        length = generator.randrange(0, 513)
        flags = [generator.randrange(2) for _ in range(length)]
        count = generator.randrange(0, 513)
        assert_exact_against_frozen(flags, count)
