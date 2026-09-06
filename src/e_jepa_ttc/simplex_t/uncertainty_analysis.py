"""Paired full-cohort uncertainty using the historical sequence/whole-track recipe."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.evaluation.risk_geometry_v10 import hierarchical_losses
from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass


def paired_uncertainty(
    frames: dict[str, pd.DataFrame],
    *,
    reference: str,
    output: Path,
    resource_check: Callable[[], None],
) -> dict[str, Any]:
    """Save shared draws and all named contrasts; never filter cold-start queries.

    Callers must have verified frozen phase endpoints, role authority and source
    identities. This is exploratory OLD-development analysis, not confirmation.
    Existing output (including interrupted draws) is never overwritten. A failed
    analysis retains its partial evidence and can be retried in a new directory.
    """
    if reference not in frames or len(frames) < 2:
        raise ValueError("named reference and at least one comparison required")
    if output.exists():
        raise FileExistsError("uncertainty output already exists; preserve prior evidence")
    resource_check()
    names = sorted(frames)
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    aligned = {}
    for name in names:
        frame = frames[name]
        if (
            not {*identity, "loss"} <= set(frame)
            or len(frame) != 8192
            or frame.sample_token.duplicated().any()
            or frame.loc[:, identity].isna().to_numpy().any()
            or len(frame.sequence_id.unique()) != 9
            or set(frame.outer_fold) != {0, 1, 2}
        ):
            raise ValueError("complete nine-sequence canonical OLD cohort required")
        aligned[name] = frame.sort_values("sample_token").reset_index(drop=True)
    base = aligned[reference]
    for frame in aligned.values():
        if not frame.loc[:, identity].equals(base.loc[:, identity]):
            raise ValueError("paired uncertainty query/target identities differ")
    losses = np.column_stack([aligned[name].loss.to_numpy(np.float64) for name in names])
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("finite nonnegative losses required")
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    # Source routine resamples sequences then complete tracks with shared draws
    # across columns; it does not reconstruct or average any signed TTC values.
    bootstrap_frame = base.rename(columns={"target_ttc": "target_ttc_s"})
    output.mkdir(parents=True)
    sampled, bootstrap = hierarchical_losses(bootstrap_frame, losses, output, resource_check)
    reference_index = names.index(reference)
    sequence_names = sorted(base.sequence_id.unique())
    sequence_scores = []
    for sequence in sequence_names:
        mask = base.sequence_id.to_numpy() == sequence
        sequence_scores.append((mass[mask] / mass[mask].sum()) @ losses[mask])
    per_sequence = np.stack(sequence_scores)
    scores = mass @ losses
    comparisons = {}
    for index, name in enumerate(names):
        resource_check()
        delta = sampled[:, index] - sampled[:, reference_index]
        sequence_delta = per_sequence[:, index] - per_sequence[:, reference_index]
        comparisons[name] = {
            "score": float(scores[index]),
            "point_delta": float(scores[index] - scores[reference_index]),
            "hierarchical_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
            "hierarchical_fraction_negative": float(np.mean(delta < 0)),
            "sequence_only": exact_sequence_diagnostic(sequence_delta),
            "sequence_scores": per_sequence[:, index].tolist(),
        }
    report = {
        "schema": "simplex_t_paired_uncertainty_v1",
        "reference": reference,
        "column_order": names,
        "sequence_order": sequence_names,
        "bootstrap": bootstrap,
        "comparisons": comparisons,
        "scope": "REUSED_OLD_DEVELOPMENT_NOT_CONFIRMATORY",
        "fraction_negative_is_posterior_probability": False,
        "optimizer_updates": 0,
        "status": "COMPLETE_PAIRED_UNCERTAINTY",
    }
    write_new_json(output / "PAIRED_UNCERTAINTY.json", report)
    return report
