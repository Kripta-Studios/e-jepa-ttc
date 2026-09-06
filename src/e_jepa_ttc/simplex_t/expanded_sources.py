"""Combine validated D0 and expansion inputs without changing experts or OLD queries."""

from __future__ import annotations

import numpy as np
import torch

from .cache import CachedQueries, fit_normalizer
from .training import state_digest


def merge_validated_sources(
    d0: dict[str, CachedQueries],
    expansion: CachedQueries,
    *,
    original_train_sequences: np.ndarray,
    expansion_train_sequences: np.ndarray,
) -> dict[str, CachedQueries]:
    """Fit one D1 TRAIN normalizer; retain OLD_DEV queries, targets and experts.

    Callers must first verify both cache lineages and selected query identities.
    Each source already has sequence-balanced, present-bucket-renormalized mass.
    Since the sequence sets are disjoint, scaling by their group counts preserves
    exactly that recipe on the union without rereading targets to form history.
    """
    if set(d0) != {"inner_oof", "outer_dev"}:
        raise ValueError("separate original TRAIN and OLD_DEV sources required")
    train, dev = d0["inner_oof"], d0["outer_dev"]
    for source in (train, dev, expansion):
        if source.control != "NONE" or source.zero_latent or source.length != 8:
            raise ValueError("merge unbound base sources before resolving candidate controls")
        if source.features.dtype != np.float32 or source.features.shape[1] not in {17, 145}:
            raise ValueError("FP32 registered feature schema required")
        if source.history.shape[1] != 16 or not (source.history[:, -1] >= 0).all():
            raise ValueError("complete-shaped H16 source with every current query required")
        if (source.history < -1).any() or (source.history >= len(source.features)).any():
            raise ValueError("source history index out of bounds")
    if train.features is not dev.features or train.anchor_us is not dev.anchor_us:
        raise ValueError("original roles must share one validated compiled observation universe")
    if train.available_us is not dev.available_us:
        raise ValueError("original role timing universes differ")
    if train.features.shape[1] != expansion.features.shape[1]:
        raise ValueError("D0 and expansion feature schemas differ")
    groups = []
    for source, sequences in (
        (train, original_train_sequences),
        (expansion, expansion_train_sequences),
    ):
        if sequences.shape != (source.population,):
            raise ValueError("query sequence identity alignment mismatch")
        unique = np.unique(sequences)
        if not len(unique) or any(not str(value) for value in unique):
            raise ValueError("missing TRAIN sequence identities")
        if any(
            not np.isclose(
                source.mass[sequences == value].sum(), 1 / len(unique), atol=1e-10, rtol=1e-7
            )
            for value in unique
        ):
            raise ValueError("source mass is not sequence balanced")
        groups.append(set(unique))
    if groups[0] & groups[1]:
        raise ValueError("expansion overlaps original TRAIN groups")
    offset = len(train.features)
    shifted = np.where(expansion.history >= 0, expansion.history + offset, -1)
    history = np.concatenate((train.history, shifted))
    features = np.concatenate((train.features, expansion.features))
    anchors = np.concatenate((train.anchor_us, expansion.anchor_us))
    available = np.concatenate((train.available_us, expansion.available_us))
    allowed = np.zeros(len(features), bool)
    ids = history[history >= 0]
    if ids.size == 0 or ids.max() >= len(features) or (history < -1).any():
        raise ValueError("TRAIN history is outside the combined observation universe")
    allowed[ids] = True
    dev_ids = dev.history[dev.history >= 0]
    if dev_ids.max() >= offset or allowed[dev_ids].any():
        raise ValueError("OLD_DEV observations enter TRAIN normalization")
    normalizer = fit_normalizer(features, history, allowed)
    total_groups = len(groups[0]) + len(groups[1])
    mass = np.concatenate(
        (train.mass * len(groups[0]) / total_groups, expansion.mass * len(groups[1]) / total_groups)
    )
    result = {}
    for role in ("inner_oof", "outer_dev"):
        identity = state_digest(
            {
                "namespace": "SIMPLEX_T_D1_COMBINED_TRAIN_UNCHANGED_OLD_DEV",
                "role": role,
                "original_train": train.identity_sha256,
                "original_dev": dev.identity_sha256,
                "expansion_train": expansion.identity_sha256,
                "original_sequences": original_train_sequences.tolist(),
                "expansion_sequences": expansion_train_sequences.tolist(),
                "normalizer_mean": torch.from_numpy(normalizer.mean),
                "normalizer_scale": torch.from_numpy(normalizer.scale),
                "normalizer_ids": normalizer.consumed_ids_sha256,
            }
        )
        result[role] = CachedQueries(
            features,
            anchors,
            available,
            history if role == "inner_oof" else dev.history.copy(),
            np.concatenate((train.target_phase, expansion.target_phase))
            if role == "inner_oof"
            else dev.target_phase.copy(),
            mass if role == "inner_oof" else dev.mass.copy(),
            normalizer,
            identity,
        )
    return result
