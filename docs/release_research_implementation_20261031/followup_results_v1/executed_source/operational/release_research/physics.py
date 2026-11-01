"""Signed physical state, causal controls and q-specific affine observability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def raw_phase_to_q(phase: float) -> float:
    """Invert raw canonical phase without the benchmark's clipping or 60 s floor."""
    if not math.isfinite(phase):
        raise ValueError("Raw phase must be finite")
    try:
        value = -math.expm1(-phase) / 0.1
    except OverflowError as error:
        raise ValueError("Raw phase inverse overflow") from error
    if not math.isfinite(value):
        raise ValueError("Raw phase inverse overflow")
    return value


def q_to_raw_phase(q: float) -> float:
    """Convert signed inverse seconds to phase, rejecting the physical singularity."""
    if not math.isfinite(q) or 1 - 0.1 * q <= 0:
        raise ValueError("q outside canonical phase domain")
    return -math.log1p(-0.1 * q)


def transport(q: float, dt: float) -> tuple[float, float]:
    """Return constant-axial-velocity prediction and its derivative with respect to q."""
    if not math.isfinite(q) or not math.isfinite(dt) or dt < 0:
        raise ValueError("Transport requires finite q and nonnegative seconds")
    denominator = 1 - dt * q
    if denominator <= 1e-12:
        raise ValueError("Transition crosses contact; use an explicit measurement/fallback")
    result = q / denominator, 1 / denominator**2
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite transport")
    return result


@dataclass(frozen=True)
class Measurement:
    """Independent current measurement in one query's spatial frame, before emission."""

    time_s: float
    available_s: float
    q: float
    variance: float
    frame: str


@dataclass(frozen=True)
class ClockConfig:
    """Fixed controls, expressed in seconds and inverse-second variance."""

    history_s: float = 0.65
    ewma_tau_s: float = 0.15
    process_variance_per_s: float = 0.01
    max_gain: float = 0.9

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Clock parameters must be finite")
        if not 0 < self.history_s <= 0.65 or self.ewma_tau_s <= 0:
            raise ValueError("Principal clock comparison permits at most 650 ms")
        if self.process_variance_per_s < 0 or not 0 < self.max_gain <= 1:
            raise ValueError("Invalid clock process noise/gain")


@dataclass(frozen=True)
class ClockResult:
    """Result with explicit fallback and admitted-history denominator."""

    q: float
    variance: float | None
    observations: int
    reason: str
    innovations: tuple[float, ...] = ()


DEFAULT_CLOCK = ClockConfig()


def observe(
    measurements: list[Measurement],
    query_s: float,
    frame: str,
    fallback_q: float,
    mode: Literal["ewma", "transport", "observer"],
    config: ClockConfig = DEFAULT_CLOCK,
) -> ClockResult:
    """Rebuild D1/D2/D3 within 650 ms; future input and other ROI frames have no effect.

    Variance is approximate: overlapping event windows are not independent. This
    routine deliberately does not label its output calibrated uncertainty.
    """
    if mode not in ("ewma", "transport", "observer"):
        raise ValueError("Unknown clock control")
    if not all(math.isfinite(v) for v in (query_s, fallback_q)) or not frame:
        raise ValueError("Query, fallback and frame are required")
    retained = []
    for m in measurements:
        if not all(math.isfinite(v) for v in (m.time_s, m.available_s, m.q, m.variance)):
            raise ValueError("Nonfinite measurement")
        if m.available_s < m.time_s or m.variance <= 0:
            raise ValueError("Invalid measurement availability/variance")
        if query_s - config.history_s <= m.time_s <= m.available_s <= query_s:
            if m.frame != frame:
                raise ValueError("Reconstruct history in the query ROI before filtering")
            retained.append(m)
    if not retained:
        return ClockResult(fallback_q, None, 0, "no_causal_history")
    retained.sort(key=lambda m: m.time_s)
    if any(a.time_s == b.time_s for a, b in zip(retained, retained[1:], strict=False)):
        raise ValueError("Duplicate measurement time would double-count evidence")
    if mode == "transport":
        retained = retained[-1:]
    q, variance, previous = retained[0].q, retained[0].variance, retained[0].time_s
    innovations: list[float] = []
    reason = "ok"
    for m in retained[1:]:
        dt = m.time_s - previous
        if mode == "ewma":
            q += -math.expm1(-dt / config.ewma_tau_s) * (m.q - q)
        else:
            try:
                predicted, derivative = transport(q, dt)
            except ValueError:
                q, variance = m.q, m.variance
                reason = "contact_reset_to_measurement"
            else:
                prior = derivative**2 * variance + config.process_variance_per_s * dt
                if not math.isfinite(prior):
                    q, variance = m.q, m.variance
                    reason = "variance_reset_to_measurement"
                else:
                    gain = min(config.max_gain, prior / (prior + m.variance))
                    innovation = m.q - predicted
                    innovations.append(innovation)
                    q = predicted + gain * innovation
                    variance = (1 - gain) ** 2 * prior + gain**2 * m.variance
        previous = m.time_s
    if mode != "ewma":
        try:
            q, derivative = transport(q, query_s - previous)
            variance = derivative**2 * variance + config.process_variance_per_s * (
                query_s - previous
            )
            if not math.isfinite(variance):
                raise ValueError("Variance overflow")
        except ValueError:
            return ClockResult(
                fallback_q, None, len(retained), "contact_fallback", tuple(innovations)
            )
    return ClockResult(
        q, variance if mode != "ewma" else None, len(retained), reason, tuple(innovations)
    )


def affine_design(points: Array, normals: Array) -> Array:
    """H for beta=[Axx,Axy,Ayx,Ayy,bx,by]; points share one isotropic spatial unit.

    Normal velocities must use that same spatial unit per second. Normalizing
    points without normalizing velocities changes the physical meaning of q.
    """
    if points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape:
        raise ValueError("Expected paired [N,2] positions and normals")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Nonfinite affine observations")
    x, y = points.T
    nx, ny = normals.T
    return np.column_stack((nx * x, nx * y, ny * x, ny * y, nx, ny))


@dataclass(frozen=True)
class Observability:
    """Estimability is distinct from accuracy or object membership."""

    q: float | None
    rank: int
    null_leakage: float
    sensitivity: float | None
    median_absolute_residual: float
    estimable: bool


def estimate_expansion(design: Array, velocity: Array, rtol: float = 1e-8) -> Observability:
    """Test c in row(H); do not silently convert a ridge prior into evidence."""
    if design.ndim != 2 or design.shape[1] != 6 or velocity.shape != (len(design),):
        raise ValueError("Expected H[N,6] and normal velocity[N]")
    if len(design) == 0 or not np.isfinite(design).all() or not np.isfinite(velocity).all():
        raise ValueError("Nonempty finite measurements required")
    if not 0 < rtol < 1:
        raise ValueError("SVD relative tolerance must be in (0,1)")
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    admitted = singular > singular[0] * rtol
    basis = vh[admitted]
    c = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    projection = basis @ c
    leakage = float(np.linalg.norm(c - basis.T @ projection) / np.linalg.norm(c))
    beta = basis.T @ ((u[:, admitted].T @ velocity) / singular[admitted])
    estimable = leakage <= rtol
    sensitivity = float(np.linalg.norm(projection / singular[admitted]))
    return Observability(
        float(c @ beta) if estimable else None,
        int(admitted.sum()),
        leakage,
        sensitivity if estimable else None,
        float(np.median(np.abs(design @ beta - velocity))),
        estimable,
    )
