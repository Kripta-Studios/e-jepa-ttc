"""Stateful callbacks that do not mutate source code or experiment inputs."""

from __future__ import annotations


class EarlyStopping:
    """Validation-loss early stopping with explicit minimum epoch policy."""

    def __init__(self, patience: int, *, min_epochs: int = 0, mode: str = "min") -> None:
        if patience < 0 or min_epochs < 0 or mode not in {"min", "max"}:
            raise ValueError("Invalid early-stopping configuration.")
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record a validation value and return whether training should stop."""

        improved = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs > self.patience


__all__ = ["EarlyStopping"]
