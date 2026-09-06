"""Local time-plane and robust affine-normal-flow reference, float64 NumPy.

Geometry is an observable/quality feature, not automatically a TTC label.
The integration must preserve the raw clock and one common spatial transform.
No pixels, ground-truth depths, TTC labels or teacher masks are synthesized here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneNormal:
    valid: bool
    normal: np.ndarray
    speed: float
    r_squared: float
    residual_fraction: float


def fit_time_plane(xy: np.ndarray, t_seconds: np.ndarray) -> PlaneNormal:
    """t = a*x + b*y + c; n=g/||g|| and n dot v = 1/||g||.

    xy uses the fixed common ROI coordinate system, time is relative seconds.
    Fit >=9 unique pixel locations from a local latest-event surface. This
    function does not perform surface construction or polarity mixing.
    """
    xy = np.asarray(xy, dtype=np.float64)
    t = np.asarray(t_seconds, dtype=np.float64)
    invalid = PlaneNormal(False, np.zeros(2), 0.0, 0.0, 0.0)
    if xy.ndim != 2 or xy.shape[1] != 2 or t.shape != (len(xy),):
        raise ValueError("expected xy [N,2], t [N]")
    if not np.isfinite(xy).all() or not np.isfinite(t).all():
        raise ValueError("nonfinite plane data")
    if len(xy) < 9 or len(np.unique(xy, axis=0)) < 9:
        return invalid
    xc = xy - xy.mean(0)
    tc = t - t.mean()
    if np.linalg.matrix_rank(xc) < 2 or np.ptp(t) < 1e-6:
        return invalid
    # Standardize space for numerical conditioning, then recover the gradient.
    scale = np.maximum(np.std(xc, axis=0), 1e-12)
    z = xc / scale
    coef, _, rank, _ = np.linalg.lstsq(z, tc, rcond=None)
    if rank < 2:
        return invalid
    grad = coef / scale
    norm = float(np.linalg.norm(grad))
    if not 1e-8 <= norm <= 1e8:
        return invalid
    residual = tc - xc @ grad
    r2 = float(1.0 - (residual @ residual) / max(tc @ tc, 1e-24))
    fraction = float(np.median(np.abs(residual)) / max(np.ptp(t), 1e-12))
    return PlaneNormal(r2 >= 0.5 and fraction <= 0.2, grad / norm, 1.0 / norm, r2, fraction)


@dataclass(frozen=True)
class AffineNormalFit:
    valid: bool
    beta: np.ndarray
    condition: float
    residual: float
    inlier_fraction: float
    orientation_balance: float
    mode: str = "invalid"

    @property
    def kappa(self) -> float:
        return float((self.beta[0] + self.beta[3]) / 2.0)


def fit_affine_normal_flow(
    xy: np.ndarray,
    normals: np.ndarray,
    normal_speed: np.ndarray,
    weights: np.ndarray | None = None,
    iterations: int = 5,
) -> AffineNormalFit:
    """Robust n dot (A*x+b)=s; beta=[Axx,Axy,Ayx,Ayy,bx,by].

    A rank/condition failure returns valid=False, never a plausible invented TTC.
    The support/quality bits must be retained by the caller.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    s = np.asarray(normal_speed, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or n.shape != xy.shape or s.shape != (len(xy),):
        raise ValueError("normal-flow shapes disagree")
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != s.shape or any(not np.isfinite(v).all() for v in (xy, n, s, w)) or (w < 0).any():
        raise ValueError("invalid normal-flow values/weights")
    if iterations != 5:
        raise ValueError("IRLS iterations frozen at five")
    bad = AffineNormalFit(False, np.zeros(6), 1e12, 0.0, 0.0, 0.0)
    if len(s) < 32 or w.sum() <= 0:
        return bad
    if not np.allclose(np.linalg.norm(n, axis=1), 1.0, rtol=0, atol=1e-7):
        raise ValueError("normals must have unit length in the fitting coordinates")
    x, y = xy.T
    nx, ny = n.T
    design = np.column_stack((nx * x, nx * y, ny * x, ny * y, nx, ny))
    colscale = np.maximum(np.sqrt(np.average(design**2, axis=0, weights=w)), 1e-12)
    z = design / colscale
    singular = np.linalg.svd(np.sqrt(w)[:, None] * z, compute_uv=False)
    if singular[-1] <= 0:
        return _fit_isotropic(xy, n, s, w)
    condition = float(singular[0] / singular[-1])
    if np.linalg.matrix_rank(z) < 6 or condition > 1e6:
        return _fit_isotropic(xy, n, s, w)
    cur = w.copy()
    beta = np.zeros(6)
    for _ in range(iterations):
        coef, _, rank, _ = np.linalg.lstsq(np.sqrt(cur)[:, None] * z, np.sqrt(cur) * s, rcond=None)
        if rank < 6:
            return _fit_isotropic(xy, n, s, w)
        beta = coef / colscale
        residual = s - design @ beta
        sigma = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-8)
        cur = w * np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
    residual = s - design @ beta
    sigma = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-8)
    normalized = float(
        np.sqrt(np.average(residual**2, weights=cur))
        / max(np.sqrt(np.average(s**2, weights=w)), 1e-8)
    )
    eig = np.linalg.eigvalsh((n.T @ (w[:, None] * n)) / w.sum())
    balance = float(eig[0] / max(eig[-1], 1e-12))
    return AffineNormalFit(
        True,
        beta,
        condition,
        normalized,
        float(np.average(np.abs(residual) <= 3.0 * sigma, weights=w)),
        balance,
        "affine",
    )


def _fit_isotropic(xy: np.ndarray, n: np.ndarray, s: np.ndarray, w: np.ndarray) -> AffineNormalFit:
    """Predetermined fallback: translation plus isotropic expansion, not full flow.

    A circular expanding contour cannot identify rotation, but can identify
    expansion. Requiring full affine rank would wrongly reject this valid case.
    This reduced-model output always exposes mode='isotropic' to the caller.
    """
    bad = AffineNormalFit(False, np.zeros(6), 1e12, 0.0, 0.0, 0.0)
    design = np.column_stack((n[:, 0], n[:, 1], np.sum(n * xy, axis=1)))
    scale = np.maximum(np.sqrt(np.average(design**2, axis=0, weights=w)), 1e-12)
    z = design / scale
    singular = np.linalg.svd(np.sqrt(w)[:, None] * z, compute_uv=False)
    if singular[-1] <= 0 or np.linalg.matrix_rank(z) < 3 or singular[0] / singular[-1] > 1e6:
        return bad
    condition = float(singular[0] / singular[-1])
    cur = w.copy()
    for _ in range(5):
        coef, _, rank, _ = np.linalg.lstsq(np.sqrt(cur)[:, None] * z, np.sqrt(cur) * s, rcond=None)
        if rank < 3:
            return bad
        beta = coef / scale
        r = s - design @ beta
        sigma = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-8)
        cur = w * np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(r), 1e-12))
    r = s - design @ beta
    sigma = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-8)
    residual = float(
        np.sqrt(np.average(r * r, weights=cur)) / max(np.sqrt(np.average(s * s, weights=w)), 1e-8)
    )
    eig = np.linalg.eigvalsh(n.T @ (w[:, None] * n) / w.sum())
    fullbeta = np.array([beta[2], 0.0, 0.0, beta[2], beta[0], beta[1]])
    return AffineNormalFit(
        True,
        fullbeta,
        condition,
        residual,
        float(np.average(np.abs(r) <= 3 * sigma, weights=w)),
        float(eig[0] / max(eig[-1], 1e-12)),
        "isotropic",
    )


def inverse_ttc_at_anchor(kappa_at_reference: float, reference_minus_anchor_s: float) -> float:
    """Constant-relative-velocity diagnostic: q_anchor=q_ref/(1+q_ref*dt).

    Do not use this identity as a universal loss on turning/accelerating scenes.
    """
    q, dt = float(kappa_at_reference), float(reference_minus_anchor_s)
    if not np.isfinite([q, dt]).all():
        raise ValueError("nonfinite time alignment")
    denominator = 1.0 + q * dt
    if denominator <= 1e-6:
        raise ValueError("time transport crosses/approaches a contact singularity")
    return q / denominator
