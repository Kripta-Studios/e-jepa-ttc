"""Baseline predictors."""

from e_jepa_ttc.baselines.event_rate import run_event_rate_baseline
from e_jepa_ttc.baselines.geometric import run_geometric_baseline
from e_jepa_ttc.baselines.trivial import run_trivial_baseline

__all__ = ["run_event_rate_baseline", "run_geometric_baseline", "run_trivial_baseline"]
