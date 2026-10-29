"""Replace original TRAIN by its registered dense pool while keeping OLD_DEV fixed."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc

from .cache import CachedQueries, fit_normalizer, training_mass
from .training import state_digest


def dense_source_pair(
    original: dict[str, CachedQueries],
    dense: CachedQueries,
    *,
    original_tokens: np.ndarray,
    original_sequences: np.ndarray,
    original_target_ttc: np.ndarray,
    dev_sequences: np.ndarray,
    dense_tokens: np.ndarray,
    dense_sequences: np.ndarray,
    dense_target_ttc: np.ndarray,
) -> dict[str, CachedQueries]:
    """Use dense TRAIN once, not D0 plus duplicated dense queries.

    Callers verify file hashes, registered pool order and producer ancestry first.
    Exact D0 overlap parity is checked again here. Targets are supervision only;
    they never select queries, histories, ROIs or producer families.
    """
    if set(original) != {"inner_oof", "outer_dev"}:
        raise ValueError("separate original TRAIN and OLD_DEV required")
    train, dev = original["inner_oof"], original["outer_dev"]
    for source in (train, dev, dense):
        if (
            source.control != "NONE"
            or source.zero_latent
            or source.length != 8
            or source.features.dtype != np.float32
            or source.features.shape[1] != 17
            or source.history.shape[1] != 16
            or not (source.history[:, -1] >= 0).all()
            or (source.history < -1).any()
            or (source.history >= len(source.features)).any()
        ):
            raise ValueError("unbound scalar H16 sources with complete current input required")
    if (
        train.features is not dev.features
        or train.anchor_us is not dev.anchor_us
        or train.available_us is not dev.available_us
    ):
        raise ValueError("original roles must share one validated observation universe")
    for source, tokens, sequences in (
        (train, original_tokens, original_sequences),
        (dense, dense_tokens, dense_sequences),
    ):
        if (
            tokens.shape != (source.population,)
            or sequences.shape != tokens.shape
            or len(np.unique(tokens)) != len(tokens)
            or any(not str(value) for value in tokens)
            or any(not str(value) for value in sequences)
        ):
            raise ValueError("TRAIN query identity alignment mismatch")
    if dev_sequences.shape != (dev.population,) or any(not str(v) for v in dev_sequences):
        raise ValueError("OLD_DEV sequence identity alignment mismatch")
    if (
        original_target_ttc.shape != (train.population,)
        or not np.isfinite(original_target_ttc).all()
    ):
        raise ValueError("original TRAIN TTC identity alignment mismatch")
    if set(original_sequences) != set(dense_sequences) or set(dense_sequences) & set(dev_sequences):
        raise ValueError("dense groups differ from original TRAIN or overlap OLD_DEV")
    if dense_target_ttc.shape != (dense.population,) or not np.array_equal(
        phase_from_ttc(dense_target_ttc.astype(np.float32)), dense.target_phase
    ):
        raise ValueError("dense original TTC supervision mismatch")
    positions = {str(token): row for row, token in enumerate(dense_tokens)}
    canonical_phase = dense.target_phase.copy()
    for row, token in enumerate(original_tokens):
        other = positions.get(str(token))
        if other is None or original_sequences[row] != dense_sequences[other]:
            raise ValueError("dense TRAIN must retain each original TRAIN query")
        old_ids, dense_ids = train.history[row], dense.history[other]
        mask = old_ids >= 0
        if not np.array_equal(mask, dense_ids >= 0):
            raise ValueError("original TRAIN context support changed")
        for field in ("features", "anchor_us", "available_us"):
            if getattr(train, field)[old_ids[mask]].tobytes() != (
                getattr(dense, field)[dense_ids[mask]].tobytes()
            ):
                raise ValueError("original TRAIN expert context changed")
        if np.float32(original_target_ttc[row]) != np.float32(
            dense_target_ttc[other]
        ) or np.float32(train.target_phase[row]) != np.float32(dense.target_phase[other]):
            raise ValueError("original TRAIN supervision changed")
        # Retain the historical source-phase bytes for overlap. A 1-ULP FP64
        # implementation difference is irrelevant to the FP32 training gather.
        canonical_phase[other] = train.target_phase[row]
    original_train_ids = np.unique(train.history[train.history >= 0])
    dev_ids = np.unique(dev.history[dev.history >= 0])
    if np.intersect1d(original_train_ids, dev_ids).size:
        raise ValueError("original TRAIN and OLD_DEV observation overlap")
    dense_ids = np.unique(dense.history[dense.history >= 0])
    if not np.array_equal(dense_ids, np.arange(len(dense.features))):
        raise ValueError("dense compiled universe contains unconsumed observations")
    # Retain only OLD_DEV observations from D0; shared TRAIN is already in dense.
    offset = len(dense.features)
    mapping = np.full(len(dev.features), -1, np.int64)
    mapping[dev_ids] = np.arange(len(dev_ids)) + offset
    dev_history = np.full_like(dev.history, -1)
    mask = dev.history >= 0
    dev_history[mask] = mapping[dev.history[mask]]
    features = np.concatenate((dense.features, dev.features[dev_ids]))
    anchors = np.concatenate((dense.anchor_us, dev.anchor_us[dev_ids]))
    available = np.concatenate((dense.available_us, dev.available_us[dev_ids]))
    allowed = np.arange(len(features)) < offset
    normalizer = fit_normalizer(features, dense.history, allowed)
    mass = training_mass(dense_target_ttc, dense_sequences)
    result = {}
    for role in ("inner_oof", "outer_dev"):
        identity = state_digest(
            {
                "namespace": "SIMPLEX_T_DENSE_TRAIN_UNCHANGED_OLD_DEV",
                "role": role,
                "dense": dense.identity_sha256,
                "original_dev": dev.identity_sha256,
                "original_train": train.identity_sha256,
                "target_phase": torch.from_numpy(canonical_phase),
                "tokens": dense_tokens.tolist(),
                "sequences": dense_sequences.tolist(),
                "dev_sequences": dev_sequences.tolist(),
                "mass": torch.from_numpy(mass),
                "dev_observations": torch.from_numpy(dev_ids),
                "normalizer_mean": torch.from_numpy(normalizer.mean),
                "normalizer_scale": torch.from_numpy(normalizer.scale),
                "normalizer_ids": normalizer.consumed_ids_sha256,
            }
        )
        result[role] = CachedQueries(
            features,
            anchors,
            available,
            dense.history.copy() if role == "inner_oof" else dev_history,
            canonical_phase.copy() if role == "inner_oof" else dev.target_phase.copy(),
            mass if role == "inner_oof" else dev.mass.copy(),
            normalizer,
            identity,
        )
    return result
