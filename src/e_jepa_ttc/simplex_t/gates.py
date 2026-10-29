"""Resource allocation is not statistical acceptance. Missing evidence is blocked."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    delta_risk17: float
    delta_current_control: float
    sequence_wins: int
    sequence_count: int
    finite_fraction: float
    sign_error_delta: float
    crucial_delta: float
    integrity: bool = False


def may_replicate(e: Evidence) -> bool:
    numeric = (
        e.delta_risk17,
        e.delta_current_control,
        e.finite_fraction,
        e.sign_error_delta,
        e.crucial_delta,
    )
    if not all(math.isfinite(v) for v in numeric) or e.integrity is not True:
        return False
    return (
        type(e.sequence_count) is int
        and type(e.sequence_wins) is int
        and e.sequence_count == 9
        and 6 <= e.sequence_wins <= e.sequence_count
        and e.delta_risk17 <= -3
        and e.delta_current_control <= -1
        and e.finite_fraction == 1.0
        and e.sign_error_delta <= 0.005
        and e.crucial_delta <= 3.0
    )


def development_accepted(e: Evidence, ci_hier_high: float, ci_sequence_high: float) -> bool:
    return (
        may_replicate(e)
        and math.isfinite(ci_hier_high)
        and math.isfinite(ci_sequence_high)
        and ci_hier_high < 0
        and ci_sequence_high < 0
    )
