"""Input-only retrospective windows; NOT reconstructed same-track histories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryContextInput:
    """Current supplied ROI and three frozen producer event intervals, in microseconds."""

    query_token: str
    producer_family_sha256: str
    anchor_us: int
    roi_available_us: int
    allowed_cutoff_us: int
    source_start_us: int
    windows_us: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    square_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class QueryContextObservation:
    """A fixed-crop past sensor observation available only with the current ROI."""

    lag_us: int
    anchor_us: int
    sensor_available_us: int
    available_us: int
    windows_us: tuple[tuple[int, int], ...]
    input: QueryContextInput


def query_context(observation: QueryContextInput, length: int = 8) -> list[QueryContextObservation]:
    """Shift all producer intervals by50ms, retaining only complete source windows.

    Current is always included. No TTC, track lifetime, category, depth, velocity,
    earlier labelled query membership or inferred object association is accepted.
    All crops remain query-conditioned, including for earlier sensor timestamps.
    """
    import math

    if length not in {1, 4, 8, 16}:
        raise ValueError("unregistered context length")
    if not observation.query_token or len(observation.producer_family_sha256) != 64:
        raise ValueError("missing query or producer identity")
    if any(type(t) is not int for window in observation.windows_us for t in window):
        raise ValueError("integer microsecond windows required")
    if any(a >= b for a, b in observation.windows_us):
        raise ValueError("empty or reversed sensor interval")
    ends = [b for _, b in observation.windows_us]
    if ends != sorted(set(ends)):
        raise ValueError("producer windows must be chronological")
    x0, y0, x1, y1 = observation.square_xyxy
    if not all(math.isfinite(v) for v in observation.square_xyxy) or x0 >= x1 or y0 >= y1:
        raise ValueError("invalid supplied crop")
    if min(a for a, _ in observation.windows_us) < observation.source_start_us:
        raise ValueError("current observation lacks complete source support")
    available = max(observation.roi_available_us, ends[-1])
    if available > observation.allowed_cutoff_us:
        raise ValueError("current ROI or sensor exceeds allowed cutoff")
    result = []
    for index in reversed(range(length)):
        lag = index * 50_000
        windows = tuple((start - lag, end - lag) for start, end in observation.windows_us)
        if min(start for start, _ in windows) < observation.source_start_us:
            continue
        result.append(
            QueryContextObservation(
                lag_us=lag,
                anchor_us=observation.anchor_us - lag,
                sensor_available_us=ends[-1] - lag,
                available_us=max(observation.roi_available_us, ends[-1] - lag),
                windows_us=windows,
                input=observation,
            )
        )
    return result
