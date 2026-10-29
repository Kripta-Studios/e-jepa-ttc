"""Fixed H diagnostic controls using only admitted caches and native final weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc

from .cache import HeadCache
from .common import csv_write, freeze, read, sha, write


def trace_h(cache: HeadCache, owner: Path, run: Path) -> None:
    """Trace 32 preselected H controls without optimizer steps or V access."""
    directory = owner / "fits/E_CTX_MATCHED"
    checkpoint = directory / "checkpoint_last.pt"
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 2500:
        raise ValueError("Native final context head required for H diagnostic controls")
    if sha(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Native diagnostic head checkpoint changed")
    for name, expected in read(directory / "ENDPOINT_FREEZE.json")["files"].items():
        if sha(directory / name) != expected:
            raise ValueError("Native diagnostic head endpoint freeze mismatch")
    selected = pd.read_csv(run / "H_DIAGNOSTIC_CONTROLS.csv").sample_token.astype(str).tolist()
    indices = {str(token): i for i, token in enumerate(cache.values["sample_token"])}
    ids = torch.tensor([indices[token] for token in selected])
    output = run / "matched_H_controls"
    freeze(
        output / "BINDING.json",
        {
            "native_head_sha256": receipt["checkpoint_sha256"],
            "H_cache_sha256": sha(owner / "features/H/E_HISTORY.npz"),
            "normalizer_sha256": sha(owner / "features/H/E_NORMALIZER.json"),
            "selected_ids": selected,
            "scope": "H_native_matched_diagnostic_not_new_control_fit",
        },
    )
    model = TemporalRefiner(TemporalConfig(hidden=160)).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        prediction = cache.forward(model, ids)
        ttc = phase_to_ttc(prediction["raw_location"])
        experts = cache.tensor("expert_phase", ids, torch.device("cpu"))
        suppressed = phase_to_ttc(experts.median(dim=1).values)
    rows = [
        {
            "sample_token": token,
            "group_id": str(cache.values["group_id"][index]),
            "target_ttc": float(cache.target_ttc[index]),
            "native_head_ttc": float(ttc[j]),
            "residual_suppressed_ttc": float(suppressed[j]),
            "raw_residual": float(prediction["raw_residual"][j]),
            "raw_location": float(prediction["raw_location"][j]),
            "emitted_phase": float(prediction["point_phase"][j]),
            "aux_cost_index_not_router": int(prediction["expert_index"][j]),
            **{f"expert_phase_{k}": float(experts[j, k]) for k in range(3)},
        }
        for j, (token, index) in enumerate(zip(selected, ids.tolist(), strict=True))
    ]
    csv_write(output / "HEAD_TRACES.csv", rows)
    normalizer = read(owner / "features/H/E_NORMALIZER.json")
    mean, scale = np.asarray(normalizer["mean"]), np.asarray(normalizer["scale"])
    observations = []
    for token, index in zip(selected, ids.tolist(), strict=True):
        for slot in range(cache.values["features"].shape[1]):
            valid = bool(cache.values["valid"][index, slot])
            raw = cache.values["features"][index, slot] * scale + mean
            observations.append(
                {
                    "sample_token": token,
                    "observation_slot": slot,
                    "valid": valid,
                    **{
                        f"timing_{k}": float(v)
                        for k, v in enumerate(cache.values["timing"][index, slot])
                    },
                    **{f"phase17_{k}": float(v) if valid else None for k, v in enumerate(raw)},
                }
            )
    csv_write(output / "OBSERVATION_TRACES.csv", observations)
    write(
        output / "RESULT.json",
        {
            "status": "COMPLETE_CACHED_H_CONTROLS",
            "queries": len(selected),
            "optimizer_updates": 0,
            "V_labels_opened": False,
            "raw_internal_heights_and_transport_gate": "NOT_PRESENT_IN_NATIVE_H_FEATURE_CACHE",
            "new_scientific_endpoint": False,
        },
    )
