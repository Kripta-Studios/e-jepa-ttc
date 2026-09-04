"""Fixed float64 two-output ridge router for Stage 65."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FullRegretRidge:
    """Weighted relative-cost ridge with an unpenalized intercept."""

    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    ridge: float = 0.01

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        losses: np.ndarray,
        mass: np.ndarray,
        *,
        ridge: float = 0.01,
    ) -> FullRegretRidge:
        """Fit both C2F and PAIR regret outputs in one deterministic solve."""

        x = np.asarray(features, dtype=np.float64)
        costs = np.asarray(losses, dtype=np.float64)
        weight = np.asarray(mass, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) == 0 or costs.shape != (len(x), 3):
            raise ValueError("full-regret feature/loss shapes disagree")
        if weight.shape != (len(x),) or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("full-regret mass is invalid")
        if ridge != 0.01:
            raise ValueError("Stage 65 ridge is frozen to 0.01")
        if any(not np.isfinite(value).all() for value in (x, costs, weight)):
            raise ValueError("full-regret inputs must be finite")
        weight = weight / weight.sum()
        mean = np.sum(weight[:, None] * x, axis=0)
        scale = np.maximum(np.sqrt(np.sum(weight[:, None] * np.square(x - mean), axis=0)), 1e-8)
        z = (x - mean) / scale
        targets = (costs[:, 1:] - costs[:, :1]) / 100.0
        intercept = np.sum(weight[:, None] * targets, axis=0)
        lhs = z.T @ (weight[:, None] * z) + ridge * np.eye(x.shape[1])
        rhs = z.T @ (weight[:, None] * (targets - intercept))
        coefficient = np.linalg.solve(lhs, rhs)
        residual = lhs @ coefficient - rhs
        if not np.allclose(residual, 0.0, rtol=0, atol=1e-10):
            raise ArithmeticError("full-regret ridge solve residual is too large")
        return cls(mean, scale, coefficient, intercept, ridge)

    def predict_regret(self, features: np.ndarray) -> np.ndarray:
        """Return regret estimates in the fixed A5/C2F/PAIR order."""

        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("full-regret inference feature schema mismatch")
        other = ((x - self.mean) / self.scale) @ self.coefficient + self.intercept
        return np.column_stack((np.zeros(len(x), dtype=np.float64), other))

    def select(self, features: np.ndarray) -> np.ndarray:
        """Choose argmin with NumPy's deterministic A5→C2F→PAIR tie order."""

        return np.argmin(self.predict_regret(features), axis=1).astype(np.int64)


__all__ = ["FullRegretRidge"]
