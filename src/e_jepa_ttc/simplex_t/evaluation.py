"""Full-cohort signed-phase diagnostics and paired factorial contrasts."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass

from .phase import MAX_PHASE, MIN_PHASE, phase_to_ttc


def prediction_frame(
    metadata: pd.DataFrame,
    expert_ttc: np.ndarray,
    outputs: dict[str, np.ndarray],
    history: np.ndarray,
    *,
    arm: str,
    seed: int,
    fold: int,
    output_mode: str = "residual",
) -> pd.DataFrame:
    """Score actually emitted finite TTC and retain all cold-start query identities."""
    required = {"sample_token", "sequence_id", "track_id", "target_ttc"}
    if not required <= set(metadata) or metadata.sample_token.duplicated().any():
        raise ValueError("missing/duplicate evaluation identity")
    count = len(metadata)
    if expert_ttc.shape != (count, 3) or history.shape[0] != count:
        raise ValueError("query population mismatch")
    target = metadata.target_ttc.to_numpy(np.float64)
    truth = benchmark_phase(target)
    experts = benchmark_phase(expert_ttc)
    costs = np.asarray(outputs["relative_cost"], dtype=np.float64)
    if costs.shape != (count, 3) or not np.isfinite(costs).all():
        raise ValueError("nonfinite/misaligned costs")
    if output_mode not in {"residual", "free", "selector"}:
        raise ValueError("unregistered output mode")
    point = np.asarray(outputs["point_phase"], dtype=np.float64)
    if point.shape != (count,) or not np.isfinite(point).all():
        raise ValueError("nonfinite/misaligned point phase")
    selected = costs.argmin(1)
    if output_mode == "selector":
        # This comparator emits an unchanged original expert. A phase roundtrip
        # can alter its floating value and the broad residual projection can
        # clip it; neither operation belongs to unchanged expert selection.
        prediction = expert_ttc[np.arange(count), selected].copy()
    else:
        prediction = phase_to_ttc(torch.from_numpy(point)).numpy()
    actual_phase = benchmark_phase(prediction)
    if not np.isfinite(actual_phase).all() or actual_phase.shape != (count,):
        raise ValueError("nonfinite/misaligned finite TTC")
    raw = np.asarray(outputs["raw_location"], dtype=np.float64)
    q10 = np.asarray(outputs["q10"], dtype=np.float64)
    q90 = np.asarray(outputs["q90"], dtype=np.float64)
    if not np.isfinite(q10).all() or not np.isfinite(q90).all() or (q10 > q90).any():
        raise ValueError("invalid phase quantiles")
    frame = metadata.copy()
    frame["arm"], frame["seed"], frame["outer_fold"] = arm, seed, fold
    frame["prediction_ttc_s"] = prediction
    frame["prediction_phase"] = actual_phase
    frame["target_phase"] = truth
    frame["raw_location"] = raw
    frame["raw_residual"] = outputs["raw_residual"]
    frame["q10"], frame["q90"] = q10, q90
    frame["phase_interval_covers"] = (q10 <= truth) & (truth <= q90)
    frame["phase_interval_width"] = q90 - q10
    frame["loss"] = 10000 * np.abs(actual_phase - truth)
    frame["wrong_sign"] = np.sign(prediction) != np.sign(target)
    frame["signed_phase_error"] = actual_phase - truth
    frame["history_count"] = (history >= 0).sum(1)
    frame["cold_start"] = frame.history_count == 1
    frame["hull_position"] = np.where(
        actual_phase < experts.min(1),
        "below",
        np.where(actual_phase > experts.max(1), "above", "inside"),
    )
    frame["phase_support_saturation"] = (raw < MIN_PHASE) | (raw > MAX_PHASE)
    frame["target_outside_support"] = (truth < MIN_PHASE) | (truth > MAX_PHASE)
    frame["finite_ttc_cap"] = np.abs(prediction) >= 60
    frame["sign_change_from_median"] = np.sign(actual_phase) != np.sign(np.median(experts, 1))
    median_ttc = phase_to_ttc(torch.from_numpy(np.median(experts, 1))).numpy()
    median_loss = 10000 * np.abs(benchmark_phase(median_ttc) - truth)
    frame["gain_over_current_median"] = median_loss - frame.loss
    outside = frame.hull_position != "inside"
    frame["escape_gain"] = np.where(outside, np.maximum(frame.gain_over_current_median, 0), 0)
    frame["escape_harm"] = np.where(outside, np.maximum(-frame.gain_over_current_median, 0), 0)
    frame["diagnostic_selector"] = selected
    for expert in range(3):
        frame[f"expert{expert}_ttc"] = expert_ttc[:, expert]
        frame[f"expert{expert}_phase"] = experts[:, expert]
        frame[f"original_cost{expert}"] = 10000 * np.abs(experts[:, expert] - truth)
        frame[f"predicted_relative_cost{expert}"] = costs[:, expert]
    for slot in range(history.shape[1]):
        frame[f"history_index{slot}"] = history[:, slot]
    return frame


def factorial_contrasts(losses: dict[tuple[int, int, int], np.ndarray]) -> dict[str, np.ndarray]:
    """Per-query D/H/C effects and interactions; negative means lower loss.

    Pair/triple interactions use differences of differences, averaged over factors
    not in the interaction. Arrays must already be joined by identical query IDs.
    """
    cells = [(d, h, c) for d, h, c in product((0, 1), repeat=3)]
    if set(losses) != set(cells) or len({a.shape for a in losses.values()}) != 1:
        raise ValueError("complete paired 2x2x2 factorial required")
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in losses.values()):
        raise ValueError("finite per-query losses required")
    result = {}
    for order in (1, 2, 3):
        for factors in combinations(range(3), order):
            name = "x".join("DHC"[factor] for factor in factors)
            total = np.zeros_like(losses[cells[0]], dtype=np.float64)
            for cell in cells:
                sign = np.prod([1 if cell[factor] else -1 for factor in factors])
                total += sign * losses[cell] / 2 ** (3 - order)
            result[name] = total
    return result


def strict_development_score(frame: pd.DataFrame) -> float:
    """Use the historical full-cohort metric, never TRAIN mass renormalization."""
    if len(frame) != 8192 or frame.sample_token.duplicated().any():
        raise ValueError("canonical 8192-query coverage required")
    mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
    return float(mass @ frame.loss.to_numpy())


def may_explore_context(
    *,
    delta_h1: float,
    finite: bool,
    sign_delta: float,
    crucial_delta: float,
    fraction_train_h8: tuple[float, float, float],
    integrity: bool,
) -> bool:
    """T3 practical gate independent of CI significance; no retrospective candidate swap."""
    return (
        integrity
        and finite
        and np.isfinite(delta_h1)
        and delta_h1 <= -1
        and np.isfinite(sign_delta)
        and sign_delta <= 0.005
        and np.isfinite(crucial_delta)
        and crucial_delta <= 3
        and all(np.isfinite(value) and 0.5 <= value <= 1 for value in fraction_train_h8)
    )
