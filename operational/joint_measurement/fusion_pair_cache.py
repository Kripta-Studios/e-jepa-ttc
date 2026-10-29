"""Bounded production-shaped extraction for an externally admitted frozen P2 parent.

The caller selects and verifies endpoint weights, cohorts and sensor inputs, and
owns the writer and any GPU scope. This module does not admit parents, inspect
accuracy, fit normalizers or substitute failed RGB reads with missing data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.range_transfer.common import sha

from .common import require_resources
from .fusion_cache import FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_pair_extract import FEATURE_NAMES, SCHEMA_SHA256, clock_pairs, extract_pair


class PairSource(Protocol):
    """A native sensor reader with an externally verified ordered population."""

    rows: pd.DataFrame
    population: int
    role: str

    def batch(self, indices: list[int]) -> ObjectEventV4Batch: ...


def extract_shards(
    directory: Path,
    model: torch.nn.Module,
    dataset: PairSource,
    identity: FeatureIdentity,
    *,
    manifest_sha256: str,
    available: np.ndarray,
    input_binding: dict,
    batch_size: int,
    chunk_size: int,
    max_new_chunks: int,
    device: torch.device,
) -> dict:
    """Resume exact shards, retaining every query and declared missing RGB row.

    Only pixels and their native delta enter the model. A read failure propagates;
    missingness must already be declared in the admitted availability vector.
    Batch size is bound because GPU reduction numerics can depend on batch shape.
    The source and private model remain caller-owned across bounded invocations.
    """
    identity.validate()
    if (
        not input_binding
        or identity.schema_sha256 != SCHEMA_SHA256
        or dataset.role not in {"P", "H", "V"}
        or dataset.population != len(dataset.rows)
        or available.dtype != np.bool_
        or available.shape != (dataset.population,)
        or (identity.modality == "event" and not available.all())
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError("Expected admitted frozen P2 parent, cohort and availability")
    for value in (batch_size, chunk_size, max_new_chunks):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Positive integer extraction bounds required")
    if any(value.device != device for value in model.state_dict().values()):
        raise ValueError("Frozen parent must already be on the caller-owned device")
    tokens = dataset.rows.sample_token.to_numpy(dtype=str)
    available = available.copy()
    sources = {
        name: sha(Path(__file__).with_name(name))
        for name in (
            "fusion_pair_cache.py",
            "fusion_pair_extract.py",
            "fusion_cache.py",
            "fusion_features.py",
        )
    }
    sink = FusionCacheWriter(
        directory,
        identity,
        role=dataset.role,
        manifest_sha256=manifest_sha256,
        sample_tokens=tokens,
        feature_count=len(FEATURE_NAMES),
        chunk_size=chunk_size,
        input_binding={
            "caller": input_binding,
            "sources": sources,
            "availability_sha256": hashlib.sha256(available.tobytes()).hexdigest(),
            "batch_size": batch_size,
            "device": str(device),
        },
    )
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    written = 0
    for index in range(sink.chunks):
        if sink.read_chunk(index) is not None:
            continue
        if written >= max_new_chunks:
            return {"status": "PAUSED", "new_chunks": written, "binding": sink.key}
        require_resources()
        start, stop = sink.bounds(index)
        selected = available[start:stop]
        features = np.full((stop - start, len(FEATURE_NAMES)), np.nan, np.float32)
        prediction = np.full(stop - start, np.nan, np.float32)
        deltas = np.full(stop - start, np.nan, np.float32)
        local_indices = np.flatnonzero(selected)
        for offset in range(0, len(local_indices), batch_size):
            local = local_indices[offset : offset + batch_size]
            global_indices = (local + start).tolist()
            batch = dataset.batch(global_indices)
            if batch.sample_tokens != tokens[global_indices].tolist():
                raise ValueError("Sensor reader reordered or substituted query tokens")
            expected_channels = 3 if identity.modality == "rgb" else 12
            if batch.events.shape[2] != expected_channels:
                raise ValueError("Sensor channels differ from declared modality")
            values, estimates = extract_pair(
                model, batch.events.to(device), batch.delta_t_s.to(device)
            )
            features[local], prediction[local] = values, estimates
            deltas[local] = batch.delta_t_s.detach().cpu().numpy()
        clocks = clock_pairs(
            dataset.rows.iloc[start:stop],
            modality=identity.modality,
            delta_t_s=deltas,
            available=selected,
        )
        if not all(
            torch.equal(initial[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        ):
            raise ValueError("Frozen producer state changed during feature extraction")
        sink.write_chunk(
            index,
            FeatureBlock(
                identity,
                dataset.role,
                manifest_sha256,
                tokens[start:stop],
                features,
                prediction,
                selected,
                clocks,
            ),
        )
        written += 1
    return {**sink.complete(), "new_chunks": written}
