"""Derive practical guardrail measurements from paired original TTC predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass

from .expert_phase import expert_benchmark_phase


def paired_practical_comparison(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Recompute losses/sign errors; never accept hand-entered aggregate deltas.

    Both frames must have already passed frozen publication/replay lineage
    validation. This computes measurements, not canonical candidate selection,
    gate authorization, confidence intervals or a confirmation claim. Infinite
    original-expert TTC remains phase zero, while finite coverage is reported
    separately and cannot pass the new candidate's finite-output guardrail.
    """
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    aligned = []
    for frame in (candidate, reference):
        if (
            not {*identity, "prediction_ttc_s"} <= set(frame)
            or len(frame) != 8192
            or frame.sample_token.duplicated().any()
            or frame.loc[:, identity].isna().to_numpy().any()
            or len(frame.sequence_id.unique()) != 9
            or set(frame.outer_fold) != {0, 1, 2}
        ):
            raise ValueError("complete nine-sequence OLD8192 practical comparison required")
        aligned.append(frame.sort_values("sample_token").reset_index(drop=True))
    current, base = aligned
    if not current.loc[:, identity].equals(base.loc[:, identity]):
        raise ValueError("practical comparison query/target identities differ")
    target = current.target_ttc.to_numpy(np.float64)
    sequences = current.sequence_id.to_numpy()
    truth = benchmark_phase(target)
    mass = strict_macro_mass(target, sequences)
    predictions = [frame.prediction_ttc_s.to_numpy(np.float64) for frame in aligned]
    losses = [10_000 * np.abs(expert_benchmark_phase(values) - truth) for values in predictions]
    difference = losses[0] - losses[1]
    crucial = (target > 0) & (target <= 3)
    if not np.isclose(mass[crucial].sum(), 0.5, atol=1e-12):
        raise ValueError("frozen crucial-bucket support changed")
    sequence_delta = {}
    for sequence in sorted(np.unique(sequences)):
        mask = sequences == sequence
        sequence_delta[str(sequence)] = float((mass[mask] / mass[mask].sum()) @ difference[mask])
    signs = [(np.sign(values) != np.sign(target)).astype(np.float64) for values in predictions]
    return {
        "candidate_score": float(mass @ losses[0]),
        "reference_score": float(mass @ losses[1]),
        "point_delta": float(mass @ difference),
        "weighted_sign_error_delta": float(mass @ (signs[0] - signs[1])),
        "crucial_bucket_mid_delta": float((mass[crucial] @ difference[crucial]) / 0.5),
        "candidate_finite_fraction": float(np.isfinite(predictions[0]).mean()),
        "reference_finite_fraction": float(np.isfinite(predictions[1]).mean()),
        "sequence_deltas": sequence_delta,
        "sequence_wins": sum(delta < 0 for delta in sequence_delta.values()),
        "sequence_count": len(sequence_delta),
        "query_count": len(current),
        "gate_authorized": False,
    }
