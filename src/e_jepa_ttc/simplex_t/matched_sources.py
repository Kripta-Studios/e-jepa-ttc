"""Registered matched-count views of validated caches with newly fitted TRAIN mass."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc

from .cache import CachedQueries, fit_normalizer, training_mass
from .training import state_digest


def matched_source_view(
    sources: dict[str, CachedQueries],
    *,
    selected_rows: np.ndarray,
    train_target_ttc: np.ndarray,
    train_sequences: np.ndarray,
    pool: str,
    pool_manifest_sha256: str,
) -> dict[str, CachedQueries]:
    """Reuse exact producer outputs, not a larger pool's fitted normalization.

    Caller verifies token order, role/producer lineage and matched-count equality.
    Original TTC is required for bucket weighting: inversion of rounded cached
    phase is not an acceptable replacement near target bucket boundaries.
    No raw expert inference or optimizer work occurs here.
    """
    if pool not in {"DENSE_OLD", "DIVERSE_MATCHED"} or set(sources) != {"inner_oof", "outer_dev"}:
        raise ValueError("registered matched pool and separate OLD roles required")
    if len(pool_manifest_sha256) != 64 or set(pool_manifest_sha256) - set("0123456789abcdef"):
        raise ValueError("exact matched pool manifest pin required")
    train, dev = sources["inner_oof"], sources["outer_dev"]
    if any(s.control != "NONE" or s.zero_latent or s.length != 8 for s in sources.values()):
        raise ValueError("subset base sources before arm controls")
    if train.features is not dev.features or train.anchor_us is not dev.anchor_us:
        raise ValueError("one validated observation universe required")
    if train.available_us is not dev.available_us or train.features.shape[1] != 17:
        raise ValueError("matched controls require shared scalar feature/timing schema")
    if (
        selected_rows.ndim != 1
        or selected_rows.dtype != np.int64
        or not len(selected_rows)
        or len(np.unique(selected_rows)) != len(selected_rows)
        or (selected_rows < 0).any()
        or (selected_rows >= train.population).any()
    ):
        raise ValueError("matched selection must contain unique in-range query rows")
    if (
        train_target_ttc.shape != (train.population,)
        or train_sequences.shape != train_target_ttc.shape
    ):
        raise ValueError("full TRAIN target and sequence alignment required")
    if not np.array_equal(
        phase_from_ttc(train_target_ttc).astype(np.float32), train.target_phase.astype(np.float32)
    ):
        raise ValueError("original TTC labels differ from source supervision")
    history = train.history[selected_rows]
    if history.shape[1] != 16 or not (history[:, -1] >= 0).all():
        raise ValueError("complete current input and H16-shaped histories required")
    if (history < -1).any() or (history >= len(train.features)).any():
        raise ValueError("matched history outside observation universe")
    allowed = np.zeros(len(train.features), bool)
    allowed[history[history >= 0]] = True
    dev_ids = dev.history[dev.history >= 0]
    if not dev_ids.size or dev_ids.max() >= len(allowed) or allowed[dev_ids].any():
        raise ValueError("OLD_DEV observations enter matched TRAIN normalization")
    normalizer = fit_normalizer(train.features, history, allowed)
    mass = training_mass(train_target_ttc[selected_rows], train_sequences[selected_rows])
    result = {}
    for role in sources:
        identity = state_digest(
            {
                "namespace": "SIMPLEX_T_MATCHED_POOL_VIEW",
                "pool": pool,
                "pool_manifest": pool_manifest_sha256,
                "role": role,
                "train_parent": train.identity_sha256,
                "dev_parent": dev.identity_sha256,
                "selected_rows": torch.from_numpy(selected_rows.copy()),
                "train_mass": torch.from_numpy(mass),
                "normalizer_mean": torch.from_numpy(normalizer.mean),
                "normalizer_scale": torch.from_numpy(normalizer.scale),
                "normalizer_ids": normalizer.consumed_ids_sha256,
            }
        )
        result[role] = CachedQueries(
            train.features,
            train.anchor_us,
            train.available_us,
            history if role == "inner_oof" else dev.history.copy(),
            train.target_phase[selected_rows] if role == "inner_oof" else dev.target_phase.copy(),
            mass if role == "inner_oof" else dev.mass.copy(),
            normalizer,
            identity,
        )
    return result
