"""V13-bound A5 student fitting on role H, with sequence-disjoint imitation checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256, RGB_PHASE17_SHA256
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.streaming_revision.distill import TARGETS, FeatureStudent


def fit(
    features: np.ndarray,
    sequences: np.ndarray,
    output: Path,
    *,
    modality: str,
    parent_sha256: str,
    fit_role: str,
    updates: int = 600,
    seed: int = 7,
) -> None:
    """Fit fixed CPU updates without V, Dev32, FCWD or test12 selection.

    Caller supplies verified native H observations, once per observation ID.
    Checkpoint binds exact teacher feature bytes and parent endpoints.
    """
    if fit_role != "H" or modality not in {"event", "rgb"} or updates < 1:
        raise ValueError("Native modality, role H and positive updates required")
    array = np.ascontiguousarray(features, dtype=np.float32)
    seq = np.asarray(sequences).astype(str)
    if array.shape != (len(seq), 17) or not np.isfinite(array).all() or len(set(seq)) < 2:
        raise ValueError("Finite PHASE17 and at least two sequences required")
    if len(parent_sha256) != 64 or any(c not in "0123456789abcdef" for c in parent_sha256):
        raise ValueError("Native parent SHA256 required")
    ordered = sorted(set(seq), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    holdout = set(ordered[: max(1, len(ordered) // 5)])
    mask = np.isin(seq, list(holdout))
    training, validation = torch.from_numpy(array[~mask]), torch.from_numpy(array[mask])
    output.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FeatureStudent(training.mean(0), training.std(0, unbiased=False).clamp_min(1e-5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(updates):
            raw = training[torch.randint(len(training), (min(512, len(training)),))]
            error = (model(raw)[:, list(TARGETS)] - raw[:, list(TARGETS)]) / model.scale[
                list(TARGETS)
            ]
            loss = torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            error = (
                (model(validation)[:, list(TARGETS)] - validation[:, list(TARGETS)]).abs().mean(0)
            )
    binding = {
        "modality": modality,
        "parent_sha256": parent_sha256,
        "fit_role": "H",
        "schema_sha256": EVENT_PHASE17_SHA256 if modality == "event" else RGB_PHASE17_SHA256,
    }
    checkpoint = {
        "schema": "rgb_port_student_v1",
        "binding": binding,
        "state": model.state_dict(),
        "mean": model.mean,
        "scale": model.scale,
        "updates": updates,
        "seed": seed,
    }
    torch.save(checkpoint, output / "student.pt")
    atomic_write_json(
        output / "RESULT.json",
        {
            **binding,
            "status": "COMPLETE_IMITATION_PILOT",
            "updates": updates,
            "seed": seed,
            "training_sequences": sorted(set(seq) - holdout),
            "validation_sequences": sorted(holdout),
            "feature_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "sequence_sha256": hashlib.sha256("\n".join(seq).encode()).hexdigest(),
            "validation_feature_mae": error.tolist(),
            "student_sha256": sha256_file(output / "student.pt"),
            "scope": "imitation validation; downstream TTC and latency still require measurement",
            "gpu_seconds": 0,
        },
    )


def load(path: Path, *, int8_cpu: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    """Reject legacy V12 students; optional actual dynamic INT8 CPU linear layers."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "rgb_port_student_v1":
        raise ValueError("Native V13 student identity required")
    binding = payload["binding"]
    if binding.get("fit_role") != "H":
        raise ValueError("Student was not trained on H")
    model = FeatureStudent(payload["mean"], payload["scale"])
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    if int8_cpu:
        from torch.ao.quantization import quantize_dynamic

        model.net = quantize_dynamic(model.net, {nn.Linear}, dtype=torch.qint8)
    return model, binding
