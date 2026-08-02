"""Small, explicit promotion predicates used by experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable result of comparing a candidate to its matched control."""

    passed: bool
    reasons: tuple[str, ...]


def highres_kda_gate(
    *,
    control_mid: float,
    candidate_mid: float,
    control_rte_pct: float,
    candidate_rte_pct: float,
    control_latency_ms: float,
    candidate_latency_ms: float,
    control_failure_rate_pct: float,
    candidate_failure_rate_pct: float,
    control_small_object_error: float,
    candidate_small_object_error: float,
    control_peak_vram_bytes: int,
    candidate_peak_vram_bytes: int,
    control_max_temporal_steps: int,
    candidate_max_temporal_steps: int,
    temporal_lengths_evaluated: tuple[int, ...],
    temporal_lengths_with_advantage: tuple[int, ...],
    seeds: tuple[int, ...],
    seeds_with_advantage: tuple[int, ...],
    config_equivalent: bool,
    relative_tolerance: float = 0.01,
    latency_limit: float = 1.25,
) -> PromotionDecision:
    """Apply the full preregistered KDA gate to a matched S4 control."""

    reasons: list[str] = []
    if candidate_mid > control_mid * (1.0 + relative_tolerance):
        reasons.append("MiD_regression")
    if candidate_rte_pct > control_rte_pct * (1.0 + relative_tolerance):
        reasons.append("RTE_regression")
    if candidate_failure_rate_pct > control_failure_rate_pct * (1.0 + relative_tolerance):
        reasons.append("FR_regression")
    if candidate_small_object_error > control_small_object_error * (1.0 + relative_tolerance):
        reasons.append("small_object_regression")
    if candidate_latency_ms > control_latency_ms * latency_limit:
        reasons.append("runtime_regression")
    history_doubled = candidate_max_temporal_steps >= 2 * control_max_temporal_steps
    vram_reduced = candidate_peak_vram_bytes <= 0.8 * control_peak_vram_bytes
    if not (history_doubled or vram_reduced):
        reasons.append("no_history_or_VRAM_gain")
    if len(set(temporal_lengths_evaluated)) < 2:
        reasons.append("insufficient_T_coverage")
    elif len(set(temporal_lengths_with_advantage)) < 2:
        reasons.append("advantage_not_persistent_across_T")
    if tuple(sorted(set(seeds))) != (7, 13, 23):
        reasons.append("three_seed_protocol_missing")
    elif len(set(seeds_with_advantage).intersection(seeds)) < 2:
        reasons.append("advantage_not_consistent_across_seeds")
    if not config_equivalent:
        reasons.append("config_not_equivalent")
    return PromotionDecision(not reasons, tuple(reasons))


__all__ = ["PromotionDecision", "highres_kda_gate"]
