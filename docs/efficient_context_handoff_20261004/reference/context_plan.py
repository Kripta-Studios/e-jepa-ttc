"""Pure reference contracts, not an integrated eAP loader or trained model."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Sequence, TypeVar

T = TypeVar('T')
WIDE8_SLOTS = (0, 2, 4, 6, 9, 11, 13, 15)

@dataclass(frozen=True)
class WindowKey:
    """All model-independent preprocessing inputs belong in this key."""
    source_digest: str
    sequence: str
    start_us: int
    end_us: int
    roi_xyxy: tuple[float, float, float, float]
    roi_size: int
    event_pixel_diff: float
    preprocessing_digest: str
    bins_per_polarity: int = 5

    def __post_init__(self):
        if not self.source_digest or not self.preprocessing_digest or not self.sequence:
            raise ValueError('Missing source/preprocessing identity')
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError('Expected a positive half-open interval')
        if not all(math.isfinite(x) for x in self.roi_xyxy):
            raise ValueError('ROI must be finite')
        a,b,c,d = self.roi_xyxy
        if not (a<c and b<d) or self.roi_size < 1 or self.bins_per_polarity < 1:
            raise ValueError('Invalid ROI/dimensions')
        if not math.isfinite(self.event_pixel_diff):
            raise ValueError('Invalid coordinate offset')


def deduplicate(keys: Sequence[WindowKey]) -> tuple[list[WindowKey], list[int]]:
    """Stable first-use deduplication; no boundary rounding or ROI approximation."""
    seen: dict[WindowKey,int] = {}
    unique: list[WindowKey] = []
    inverse: list[int] = []
    for key in keys:
        if key not in seen:
            seen[key] = len(unique)
            unique.append(key)
        inverse.append(seen[key])
    return unique,inverse


def evaluate_unique(keys: Sequence[WindowKey], encode: Callable[[WindowKey],T]) -> list[T]:
    """Reference only: caller copies returned arrays if later consumers mutate."""
    unique,inverse=deduplicate(keys)
    outputs=[encode(k) for k in unique]
    return [outputs[j] for j in inverse]


def check_suffix(valid: Sequence[bool]) -> None:
    if len(valid) == 0 or not valid[-1]:
        raise ValueError('Mandatory current observation missing')
    if any(a and not b for a,b in zip(valid,valid[1:])):
        raise ValueError('Valid observations must form a suffix')


def selected_context(anchors_us: Sequence[int], available_us: Sequence[int],
                     valid: Sequence[bool], current_anchor_us: int,
                     current_available_us: int, slots: Sequence[int]=WIDE8_SLOTS):
    """Select 8 of 16 fixed slots; rebuild gaps, retain absolute ages/availability.

    Missing early history stays padded. Never pick slots using labels/scores.
    Returns selected slot IDs, validity and four timing channels in seconds.
    The model itself must zero invalid feature rows and skip their state updates.
    """
    if len(anchors_us)!=16 or len(available_us)!=16 or len(valid)!=16:
        raise ValueError('Historical H16 source required')
    check_suffix(valid)
    if tuple(sorted(set(slots)))!=tuple(slots) or len(slots) == 0 or slots[-1]!=15 or min(slots)<0:
        raise ValueError('Slots must be unique, chronological, and contain present')
    mask=[bool(valid[i]) for i in slots]
    check_suffix(mask)
    times=[]
    previous=None
    for i,ok in zip(slots,mask):
        if not ok:
            times.append((0.,0.,0.,0.))
            continue
        anchor,available=int(anchors_us[i]),int(available_us[i])
        age=current_anchor_us-anchor
        availability_age=current_available_us-available
        delay=available-anchor
        if age<0 or availability_age<0 or delay<0:
            raise ValueError('Future/invalid sensor availability')
        gap=0 if previous is None else anchor-previous
        if previous is not None and gap<=0:
            raise ValueError('Expected strictly increasing valid anchors')
        times.append((age/1e6,availability_age/1e6,gap/1e6,delay/1e6))
        previous=anchor
    return tuple(slots),tuple(mask),tuple(times)


def benchmark_phase(ttc: float, delta: float=.1)->float:
    """Uncapped analytic benchmark phase; production must reuse canonical emitter."""
    if not math.isfinite(delta) or delta <= 0 or not math.isfinite(ttc) or not (ttc<0 or ttc>delta):
        raise ValueError('Outside analytic signed-TTC domain')
    return -math.log1p(-delta/ttc)


def current_ttc_from_ratio(ratio: float, delta: float)->float:
    if not math.isfinite(ratio) or not math.isfinite(delta) or ratio<=0 or delta<=0 or ratio==1:
        raise ValueError('Invalid or infinite analytic TTC')
    return delta/(ratio-1)


def transport_ttc_constant_velocity(ttc: float, age_s: float, domain_floor_s: float = .1) -> float | None:
    """Analytic diagnostic only; None denotes a rejected past term, not a query.

    A past positive TTC that has crossed contact or the benchmark domain is not
    reinterpreted as a receding object. Infinite TTC is unchanged. Production
    aggregation must retain the current query and use its canonical phase emitter.
    """
    if not math.isfinite(age_s) or age_s < 0 or not math.isfinite(domain_floor_s) or domain_floor_s <= 0:
        raise ValueError('Invalid elapsed time or TTC domain')
    if math.isnan(ttc) or ttc == 0:
        return None
    if math.isinf(ttc):
        return ttc
    current = ttc - age_s
    if ttc > 0 and current <= domain_floor_s:
        return None
    return current
