"""Three-seed loss diagnostics: optimization seeds do not create new scenes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass


def three_seed_losses(
    arms: dict[tuple[str, int], pd.DataFrame], *, family: str, pool: str
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Align the six verified canonical frames and average losses, never TTC.

    Caller verifies actual sealed publications and independent OLD identities.
    Return identity/target/loss-only frames for shared sequence/track bootstrap.
    Sample standard deviations describe the three optimization seeds, not a
    confidence interval over independent acquisitions or an ensemble predictor.
    """
    if family not in {"TPR", "LATENT"} or pool not in {"D0", "D1"}:
        raise ValueError("canonical family and primary pool required")
    seeds = (7, 13, 23)
    names = [f"{family}-{pool}-H{h}-C160" for h in (1, 8)]
    expected = {(name, seed) for name in names for seed in seeds}
    if set(arms) != expected:
        raise ValueError("exactly canonical H1/H8 seeds7/13/23 required")
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    aligned = {}
    for (name, seed), frame in arms.items():
        if (
            not {*identity, "arm", "seed", "loss"} <= set(frame)
            or len(frame) != 8192
            or frame.sample_token.duplicated().any()
            or frame.sequence_id.nunique() != 9
            or set(frame.outer_fold) != {0, 1, 2}
            or frame.loc[:, identity].isna().to_numpy().any()
            or not frame.arm.eq(name).all()
            or not frame.seed.eq(seed).all()
            or not np.isfinite(frame.loss.to_numpy()).all()
            or frame.loss.lt(0).any()
        ):
            raise ValueError("complete canonical OLD prediction/loss frames required")
        aligned[name, seed] = frame.sort_values("sample_token").reset_index(drop=True)
    base = aligned[names[0], 7].loc[:, identity]
    if any(not frame.loc[:, identity].equals(base) for frame in aligned.values()):
        raise ValueError("replicate query/target identities differ")
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    mean_frames, scores = {}, {}
    for name in names:
        losses = np.column_stack([aligned[name, seed].loss.to_numpy(np.float64) for seed in seeds])
        mean_frames[name] = base.assign(loss=losses.mean(axis=1))
        scores[name] = mass @ losses
    delta = scores[names[1]] - scores[names[0]]
    report = {
        "schema": "simplex_t_three_seed_loss_summary_v1",
        "seeds": list(seeds),
        "reference": names[0],
        "candidate": names[1],
        "scores": {
            name: {
                "per_seed": scores[name].tolist(),
                "mean": float(scores[name].mean()),
                "sample_std": float(scores[name].std(ddof=1)),
            }
            for name in names
        },
        "paired_delta": {
            "per_seed": delta.tolist(),
            "mean": float(delta.mean()),
            "sample_std": float(delta.std(ddof=1)),
        },
        "interpretation": "Mean of paired losses across optimization seeds, not a TTC ensemble",
        "scene_count": 9,
        "seed_count_creates_new_scenes": False,
        "sequence_id_establishes_independent_acquisition": False,
        "optimizer_updates": 0,
    }
    return mean_frames, report
