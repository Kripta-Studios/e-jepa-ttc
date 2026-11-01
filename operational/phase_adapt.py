"""Explicit exposed-data final refit contracts, separate from transfer certification."""

from __future__ import annotations

import math

import numpy as np

ROLE = "all_compatible_local_refit"


def balanced_mass(domains: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Give domains equal total mass and recordings equal mass within each domain."""
    if domains.shape != groups.shape or domains.ndim != 1 or len(domains) == 0:
        raise ValueError("Nonempty aligned domain/group vectors required")
    mass = np.zeros(len(domains), dtype=np.float64)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        selected = domains == domain
        names, inverse, counts = np.unique(
            groups[selected], return_inverse=True, return_counts=True
        )
        mass[selected] = 1 / (len(unique_domains) * len(names) * counts[inverse])
    return mass.astype(np.float32)


def compatible_target(table: np.ndarray, query_us: int) -> float | None:
    """Interpolate local CSV TTC at the native query clock without extrapolation."""
    if table.ndim != 2 or table.shape[1] != 5 or len(table) == 0:
        raise ValueError("Expected nonempty five-column EvTTC labels")
    times = np.rint(table[:, 1] * 1e6).astype(np.int64)
    if not np.isfinite(table[:, 1]).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Label clock must be finite and strictly increasing")
    index = int(np.searchsorted(times, query_us))
    if index < len(times) and times[index] == query_us:
        value = float(table[index, 4])
    elif index == 0 or index == len(times) or times[index] - times[index - 1] > 20_000:
        return None
    else:
        weight = (query_us - times[index - 1]) / (times[index] - times[index - 1])
        value = float((1 - weight) * table[index - 1, 4] + weight * table[index, 4])
    # Head emission has [-60,-0.1] union [~0.1,60] support. Do not clip labels.
    if not math.isfinite(value) or not (-60 <= value <= -0.100001 or 0.10004 <= value <= 60):
        return None
    return value


def validate_population(tokens: list[str], roles: list[str]) -> None:
    """Reject duplicate examples and roles that could silently consume held-out data."""
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Population requires unique nonempty tokens")
    if len(roles) != len(tokens) or set(roles) != {ROLE}:
        raise ValueError("Explicit final refit role required")


def epoch_count(population: int, *, batch_size: int, max_updates: int, max_epochs: int) -> int:
    """Spend the declared update budget only on complete passes over every example."""
    if min(population, batch_size, max_updates, max_epochs) <= 0:
        raise ValueError("Positive training budget required")
    result = min(max_epochs, max_updates // math.ceil(population / batch_size))
    if result < 1:
        raise ValueError("Budget cannot complete epoch")
    return result
