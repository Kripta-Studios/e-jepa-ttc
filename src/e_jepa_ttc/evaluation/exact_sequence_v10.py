"""Exact sequence-only bootstrap and frozen-score omission sensitivity.

Not a replacement for sequence->track bootstrap. Not leave-one-sequence-out
retraining. The nine sequences are reused development data, not fresh evidence.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np


def _compositions(
    total: int, parts: int, prefix: tuple[int, ...] = ()
) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield prefix + (total,)
    else:
        for i in range(total + 1):
            yield from _compositions(total - i, parts - 1, prefix + (i,))


def exact_sequence_diagnostic(delta: np.ndarray) -> dict:
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim != 1 or not 2 <= len(x) <= 10 or not np.isfinite(x).all():
        raise ValueError("need 2..10 finite paired sequence differences")
    n = len(x)
    c = np.array(list(_compositions(n, n)), dtype=np.int64)
    p = np.array(
        [math.factorial(n) / (math.prod(math.factorial(int(v)) for v in row) * n**n) for row in c]
    )
    if not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ArithmeticError("bootstrap probability does not sum to one")
    v = c @ x / n
    order = np.argsort(v, kind="stable")
    cumulative = p[order].cumsum()
    ci = [
        float(v[order[min(np.searchsorted(cumulative, q), len(order) - 1)]]) for q in (0.025, 0.975)
    ]
    omit = (x.sum() - x) / (n - 1)
    flips = np.array(list(itertools.product((-1, 1), repeat=n))) @ x / n
    return {
        "point_delta": float(x.mean()),
        "ci95": ci,
        "fraction_negative": float(p[v < 0].sum()),
        "sequence_wins": int((x < 0).sum()),
        "sequence_count_vectors": len(c),
        "omit_one_score_deltas": omit.tolist(),
        "signflip_p_two_sided_descriptive": float(np.mean(np.abs(flips) >= abs(x.mean()) - 1e-12)),
        "retraining_performed": False,
        "confirmatory": False,
    }
