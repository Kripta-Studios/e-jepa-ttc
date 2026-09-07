"""Full population cached-input QA, without model inference or optimizer updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from .cache import CachedQueries


def audit_cached_source(source: CachedQueries, *, boundary: Callable[[], None]) -> dict:
    """Check every minibatch against the same source without registered controls.

    Caller must first validate role, producer, data and normalization lineage.
    This tests the actual gather path for any registered pool, not only D0.
    """
    reference = replace(source, control="NONE", zero_latent=False)
    features = source.features.shape[1]
    slots = 0
    for offset in range(0, source.population, 128):
        boundary()
        ids = torch.arange(offset, min(offset + 128, source.population))
        batch, plain = source.gather(ids), reference.gather(ids)
        x, timing, mask, experts, target, mass = batch
        if not all(torch.isfinite(value).all() for value in batch):
            raise ValueError("nonfinite cached head input")
        if x.shape != (len(ids), source.length, features):
            raise ValueError("cached feature shape differs from registered input")
        if not mask[:, -1].all() or x[~mask].count_nonzero():
            raise ValueError("current or missing history mask changed")
        if any(
            not torch.equal(value, original)
            for value, original in zip(batch[1:], plain[1:], strict=True)
        ):
            raise ValueError("control changed timing, experts, targets or mass")
        if not torch.equal(x[:, -1, :17], plain[0][:, -1, :17]):
            raise ValueError("control changed current scalar inputs")
        if source.zero_latent and x[:, :, 17:].count_nonzero():
            raise ValueError("latent-zero control retains latent values")
        slots += int(mask.sum())
    boundary()
    return {
        "schema": "simplex_t_full_population_gather_qa_v1",
        "queries": source.population,
        "valid_slots": slots,
        "history_length": source.length,
        "feature_count": features,
        "source_sha256": source.identity_sha256,
        "normalizer_ids_sha256": source.normalizer.consumed_ids_sha256,
        "model_inference": False,
        "optimizer_updates": 0,
    }
