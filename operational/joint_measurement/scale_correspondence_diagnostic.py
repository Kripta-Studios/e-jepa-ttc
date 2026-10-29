"""P-only known texture expansion/contraction diagnostic, not physical event simulation."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
import torch
from torch.nn import functional as functional

from e_jepa_ttc.data.object_event_v4 import weak_box_masks
from e_jepa_ttc.joint_measurement.model import local_cosine_volume
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import digest, freeze, read, sha

from .common import ROOT, RUN, owner
from .data import SensorDataset, cache_root, role_rows
from .geometry_model import GeometricPairMeasurement
from .train import committed_sources, initialize

OUT = RUN / "diagnostics/known_scale_P"
MODELS = {
    "PAIR-CORR-FROZEN-S7": RUN,
    "PAIR-CORR-S7": RUN,
    "PAIR-CORR-GEOMETRY-S7": RUN / "adaptive/C1_geometry",
}
SCALES = (0.9, 1.1)


def warp(pixels: torch.Tensor, scale: float) -> torch.Tensor:
    """Current image samples previous at x/scale about the image centre."""
    if pixels.ndim != 4 or not 0 < scale < 2:
        raise ValueError("One finite positive texture scale and BCHW pixels required")
    theta = pixels.new_tensor([[1 / scale, 0, 0], [0, 1 / scale, 0]])[None]
    grid = functional.affine_grid(
        theta.expand(len(pixels), -1, -1), list(pixels.shape), align_corners=False
    )
    return functional.grid_sample(
        pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def matching(
    previous: torch.Tensor, current: torch.Tensor, scale: float, support: torch.Tensor
) -> dict:
    """Compare native discrete top-one offsets with the nearest known transformed cell."""
    if len(previous) != 1 or support.shape != previous.shape[-2:] or support.dtype != torch.bool:
        raise ValueError("One map pair and its boolean reporting support required")
    volume, valid = local_cosine_volume(previous, current)
    values, indices = volume.masked_fill(~valid, -torch.inf).topk(2, dim=1)
    height, width = support.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    true_x = (scale - 1) * (x.double() + 0.5 - width / 2)
    true_y = (scale - 1) * (y.double() + 0.5 - height / 2)
    dx, dy = indices[0, 0] % 5 - 2, indices[0, 0] // 5 - 2
    nearest_x, nearest_y = true_x.round().long(), true_y.round().long()
    supported = support & (nearest_x.abs() <= 2) & (nearest_y.abs() <= 2)
    nonzero = supported & ((nearest_x != 0) | (nearest_y != 0))
    strict = values[0, 0] - values[0, 1] > 1e-6
    correct = (dx == nearest_x) & (dy == nearest_y)
    error = ((dx - true_x).square() + (dy - true_y).square()).sqrt()
    oracle = ((nearest_x - true_x).square() + (nearest_y - true_y).square()).sqrt()
    return {
        "positions": int(supported.sum()),
        "nonzero_truth_positions": int(nonzero.sum()),
        "strict_nearest_accuracy": float((strict & correct)[supported].float().mean())
        if supported.any()
        else None,
        "strict_nonzero_accuracy": float((strict & correct)[nonzero].float().mean())
        if nonzero.any()
        else None,
        "mean_offset_error_cells": float(error[supported].mean()) if supported.any() else None,
        "nearest_grid_oracle_error_cells": float(oracle[supported].mean())
        if supported.any()
        else None,
    }


def run() -> dict:
    """Choose metadata-only queries and freeze every diagnostic choice before reading maps."""
    torch.set_num_threads(1)
    rows = role_rows("P").reset_index(drop=True)
    selected = []
    for group, subset in rows.groupby("group_id", sort=True):
        index = min(
            subset.index,
            key=lambda i: hashlib.sha256(str(rows.iloc[i].sample_token).encode()).hexdigest(),
        )
        selected.append(
            {"index": int(index), "group": str(group), "token": str(rows.iloc[index].sample_token)}
        )
    endpoints = {}
    for name, base in MODELS.items():
        folder = base / "fits" / name
        receipt = read(folder / "CHECKPOINT_RECEIPT.json")
        if (
            receipt["status"] != "COMPLETE"
            or receipt["completed_updates"] != 12000
            or sha(folder / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Known-scale diagnostic requires complete immutable endpoints")
        endpoints[name] = receipt["checkpoint_sha256"]
    sources = {
        relative: sha(ROOT / relative)
        for relative in (
            "operational/joint_measurement/scale_correspondence_diagnostic.py",
            "operational/joint_measurement/geometry_model.py",
            "src/e_jepa_ttc/joint_measurement/model.py",
        )
    }
    committed_sources(sources)
    record = {
        "sources": sources,
        "endpoints": endpoints,
        "role": "P",
        "samples": selected,
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_complete_sha256": sha(cache_root("P") / "COMPLETE.json"),
        "scales": list(SCALES),
        "transform": "bilinear grid_sample, align_corners=False, zero fill, centred scale",
        "support": "fixed8-cell interior; separately latest P weak bbox pooled>=0.5; report only",
        "null": "spatial permutation of transformed features, CPU seed7",
        "truth": (
            "nearest transformed feature-cell centre, with discrete-grid oracle error reported"
        ),
        "label_use": "No TTC targets used or synthesized; P bbox used only for reporting support",
        "optimizer_updates": 0,
        "device": "cpu",
        "threads": 1,
        "decision": (
            "Descriptive input for remaining family B after D1; "
            "no automatic training, selection or promotion"
        ),
        "limitations": [
            "Known texture warp, not event camera physics",
            "one fixed query per P group",
            "convolutional texture matching is not real temporal object motion",
            "neither H nor V opened",
        ],
    }
    freeze(OUT / "REGISTRATION.json", record)
    binding = digest(record)
    if (OUT / "RESULT.json").exists():
        result = read(OUT / "RESULT.json")
        if result["binding"] != binding:
            raise ValueError("Known scale registration changed")
        return result
    encoders = {}
    for name, base in MODELS.items():
        model = (
            GeometricPairMeasurement(cast(CausalScaleTTC, initialize("A5-P2-FROZEN")))
            if "GEOMETRY" in name
            else initialize(name)
        )
        payload = torch.load(
            base / "fits" / name / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        encoders[name] = model.encoder.eval().requires_grad_(False)
        del payload, model
    source = SensorDataset("P")
    measurements = []
    try:
        with torch.inference_mode():
            for sample in selected:
                row = source._row(sample["index"])
                pixels = torch.from_numpy(row["events"][-1:])
                box, _ = weak_box_masks(torch.from_numpy(row["boxes"])[None], height=128, width=128)
                weak = functional.avg_pool2d(box[0, -1:], 4)[0, 0] >= 0.5
                interior = torch.zeros(32, 32, dtype=torch.bool)
                interior[8:-8, 8:-8] = True
                permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
                for name, encoder in encoders.items():
                    previous = encoder(pixels)
                    for scale in SCALES:
                        current = encoder(warp(pixels, scale))
                        null = current.flatten(2)[:, :, permutation].reshape_as(current)
                        measurements.append(
                            {
                                **sample,
                                "model": name,
                                "scale": scale,
                                "supports": {
                                    support_name: {
                                        "warped": matching(previous, current, scale, support),
                                        "permuted": matching(previous, null, scale, support),
                                    }
                                    for support_name, support in (
                                        ("interior", interior),
                                        ("weak_bbox", interior & weak),
                                    )
                                },
                            }
                        )
    finally:
        source.close()
    aggregate = {}
    for name in MODELS:
        aggregate[name] = {}
        for support in ("interior", "weak_bbox"):
            aggregate[name][support] = {}
            for condition in ("warped", "permuted"):
                vals = [
                    m["supports"][support][condition]["strict_nonzero_accuracy"]
                    for m in measurements
                    if m["model"] == name
                ]
                defined = [value for value in vals if value is not None]
                aggregate[name][support][condition] = {
                    "defined_query_scales": len(defined),
                    "mean_strict_nonzero_accuracy": float(np.mean(defined)) if defined else None,
                }
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "binding": binding,
        "aggregate": aggregate,
        "measurements": measurements,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "TTC_accuracy_measured": False,
        "automatic_family_B_admission": False,
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print({key: value for key, value in result.items() if key != "measurements"})
