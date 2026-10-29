"""JEPA predictor variants used by the small and dense protocols."""

from e_jepa_ttc.training.jepa import (
    DenseTemporalJEPAPredictor,
    DenseTemporalTransformerJEPAPredictor,
    JEPAPredictor,
    TemporalJEPAPredictor,
)

__all__ = [
    "DenseTemporalJEPAPredictor",
    "DenseTemporalTransformerJEPAPredictor",
    "JEPAPredictor",
    "TemporalJEPAPredictor",
]
