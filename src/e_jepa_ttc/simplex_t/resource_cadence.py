"""Short-lived resource admission reuse for bounded non-allocating work."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class ResourceCadence:
    """Reuse a positive sample for at most one second, never relax its limits.

    Callers must force a fresh check before allocations, writes that consume a
    new reservation, producer loading and publication. This is suitable only
    between bounded work units, not for admitting an unbounded operation. It
    does not monitor asynchronously or promise continuous host headroom.
    Exceptions and denied samples are never treated as successful admission.
    """

    def __init__(
        self,
        probe: Callable[[], bool],
        *,
        maximum_age_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(maximum_age_seconds) not in {int, float}
            or not math.isfinite(maximum_age_seconds)
            or not 0 < maximum_age_seconds <= 1.0
        ):
            raise ValueError("resource sample maximum age must be in (0, 1] seconds")
        self._probe = probe
        self._clock = clock
        self._maximum_age = maximum_age_seconds
        self._sampled_at: float | None = None
        self._allowed = False
        self.checks = 0
        self.reuses = 0

    def __call__(self, *, force: bool = False) -> bool:
        """Return a recent positive decision or synchronously take a fresh sample."""
        if type(force) is not bool:
            raise ValueError("explicit boolean force policy required")
        now = self._clock()
        if not math.isfinite(now):
            self._allowed = False
            raise ValueError("resource cadence requires a finite monotonic clock")
        if (
            not force
            and self._allowed
            and self._sampled_at is not None
            and 0 <= now - self._sampled_at < self._maximum_age
        ):
            self.reuses += 1
            return True
        # Revoke the old positive decision before calling a probe that may raise.
        self._allowed = False
        self._sampled_at = None
        self.checks += 1
        result = self._probe()
        if type(result) is not bool:
            raise ValueError("resource probe must return a boolean decision")
        self._sampled_at = now  # Sampling start, not completion, bounds stale reuse.
        self._allowed = result
        return result
