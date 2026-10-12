"""Descriptive metrics for the published CSVs; NOT the official benchmark scorer."""
from __future__ import annotations
import math
import statistics
from collections.abc import Iterable


def phase(ttc: float) -> float:
    if not math.isfinite(ttc) or ttc == 0 or 0 < ttc <= 0.1:
        return math.nan
    try:
        value = -math.log1p(-0.1 / ttc)
    except (ValueError, ZeroDivisionError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def number(text: str) -> float:
    return float(text) if text.strip() else math.nan


def summarize(pairs: Iterable[tuple[float, float]]) -> dict:
    labelled = [(t, p) for t, p in pairs if math.isfinite(t)]
    finite = [(t, p) for t, p in labelled if math.isfinite(p)]
    mids = [1e4 * abs(phase(t) - phase(p)) for t, p in labelled]
    valid_mid = [v for v in mids if math.isfinite(v)]
    errors = [p - t for t, p in finite]
    ae = [abs(e) for e in errors]
    relative = [100 * abs(p - t) / abs(t) for t, p in finite if t != 0]
    return {
        'labelled': len(labelled), 'finite_pairs': len(finite),
        'missing_predictions': len(labelled) - len(finite),
        'invalid_MiD': len(labelled) - len(valid_mid),
        'mean_MiD_finite': statistics.fmean(valid_mid) if valid_mid else None,
        'strict_mean_MiD': statistics.fmean(mids) if mids and len(valid_mid)==len(mids) else None,
        'MAE_finite_s': statistics.fmean(ae) if ae else None,
        'median_AE_finite_s': statistics.median(ae) if ae else None,
        'RTE_finite_pct': statistics.fmean(relative) if relative else None,
        'bias_finite_s': statistics.fmean(errors) if errors else None,
        'wrong_sign_count': sum(t*p<0 for t,p in finite),
        'capped_count': sum(abs(p)>=59.999 for _,p in finite),
    }


def joined_by_sensor_time(left: list[dict], right: list[dict]) -> list[tuple[dict,dict]]:
    """Valid only after asserting one object/query per sequence+timestamp in these CSVs."""
    def index(rows: list[dict]) -> dict:
        result = {}
        for row in rows:
            key=(row['sequence_id'],int(row['anchor_us']))
            if key in result:
                raise ValueError('Ambiguous timestamp: add object/ROI identity, do not join by position')
            result[key]=row
        return result
    a,b=index(left),index(right)
    return [(a[k],b[k]) for k in sorted(a.keys() & b.keys())]
