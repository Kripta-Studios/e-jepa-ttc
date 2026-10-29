"""Runtime aliases for applications that use the generic package layout."""

from e_jepa_ttc.runtime.benchmark import benchmark_object_ttc_model
from e_jepa_ttc.runtime.streaming import StreamingPrediction, StreamingTTCEstimator

__all__ = ["StreamingPrediction", "StreamingTTCEstimator", "benchmark_object_ttc_model"]
