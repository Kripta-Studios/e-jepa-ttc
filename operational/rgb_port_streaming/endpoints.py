"""Admit only completed native V13 teachers and modality-bound normalization."""

from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.rgb_port.infer_experts import _parents, _producer
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port.train_heads import load_head_endpoint

from .runtime import NativeExperts, NativeStream


def load_stream(
    run: Path,
    config: Path,
    p_manifest: Path,
    normalizer: Path,
    *,
    modality: str,
    device: str = "cuda",
    **options: Any,  # noqa: ANN401
) -> NativeStream:
    """Reject incomplete endpoints, V12 weights, or a different normalization binding."""
    if modality not in {"event", "rgb"}:
        raise ValueError("Explicit native modality required")
    ids = (
        ["E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED"]
        if modality == "event"
        else ["R_A5", "R_C2F", "PAIR_R"]
    )
    parents = _parents(run, ids)
    models = [_producer(run, config, p_manifest, name, device) for name in ids[:2]]
    pair = load_head_endpoint(run / "fits" / ids[2], ids[2], torch.device(device))
    experts = NativeExperts(
        models[0],
        models[1],
        pair,
        modality=modality,
        parent_sha256=canonical_sha256(parents),
        **options,
    )
    return NativeStream(experts, FrozenNormalizer.load(normalizer))
