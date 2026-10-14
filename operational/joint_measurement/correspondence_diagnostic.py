"""CPU diagnostic of known translations of P event textures, without TTC labels.

This tests translation matching, not physical event simulation or real temporal
correspondence. It cannot alone enable a geometric TTC readout or prove accuracy.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.joint_measurement.model import local_cosine_volume
from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, owner
from .data import SensorDataset, cache_root, role_rows
from .evaluate import freeze_primary
from .train import initialize

MODELS = ("PAIR-CORR-FROZEN-S7", "PAIR-CORR-S7")
OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2))


def translate(value: Tensor, dy: int, dx: int) -> Tensor:
    """Translate an image with zero fill, without wraparound or interpolation."""
    height, width = value.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("Translation must leave support")
    result = torch.zeros_like(value)
    result[..., max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)] = value[
        ..., max(-dy, 0) : height - max(dy, 0), max(-dx, 0) : width - max(dx, 0)
    ]
    return result


def matching(previous: Tensor, current: Tensor, dy: int, dx: int, margin: int = 8) -> dict:
    """Report exact matching with strict tie handling on a fixed interior support."""
    if dy not in range(-2, 3) or dx not in range(-2, 3):
        raise ValueError("Offset outside the native radius-two volume")
    if margin < 2 or min(previous.shape[-2:]) <= 2 * margin:
        raise ValueError("Insufficient fixed interior support")
    volume, valid = local_cosine_volume(previous, current)
    volume = volume[:, :, margin:-margin, margin:-margin]
    valid = valid[:, :, margin:-margin, margin:-margin]
    if not valid.all() or not torch.isfinite(volume).all():
        raise ValueError("Diagnostic interior must contain all finite offsets")
    truth = (dy + 2) * 5 + dx + 2
    prediction = volume.argmax(1)
    other = volume.clone()
    other[:, truth] = -torch.inf
    gap = volume[:, truth] - other.max(1).values
    error = (
        (prediction // 5 - 2 - dy).float().square() + (prediction % 5 - 2 - dx).float().square()
    ).sqrt()
    return {
        "positions": gap.numel(),
        "strict_correct": int((gap > 1e-6).sum()),
        "strict_accuracy": float((gap > 1e-6).float().mean()),
        "argmax_accuracy": float((prediction == truth).float().mean()),
        "mean_margin": float(gap.mean()),
        "mean_offset_error_cells": float(error.mean()),
    }


def run() -> dict:
    """Freeze metadata-only sample choice before inspecting any feature matches."""
    torch.set_num_threads(1)
    directory = RUN / "diagnostics/known_translation_P"
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
    endpoints = freeze_primary()["endpoints"]
    registration = {
        "schema": "v18_known_texture_translation_diagnostic_v1",
        "models": {name: endpoints[name] for name in MODELS},
        "role": "P",
        "samples": selected,
        "sample_rule": "minimum SHA256(sample_token) within each P acquisition group",
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_complete_sha256": sha(cache_root("P") / "COMPLETE.json"),
        "source_sha256": sha(Path(__file__)),
        "correlation_source_sha256": sha(ROOT / "src/e_jepa_ttc/joint_measurement/model.py"),
        "offsets_cells": [list(value) for value in OFFSETS],
        "pixels_per_cell": 4,
        "interior_margin_cells": 8,
        "input": "latest native P window translated spatially; no temporal or TTC targets",
        "null": "spatial permutation of translated features, CPU generator seed 7",
        "strict_margin_min": 1e-6,
        "decision": "descriptive mechanism evidence only; no automatic adaptive fit or promotion",
        "limitations": [
            "synthetic texture warp, not event physics",
            "one query per group, not a population accuracy estimate",
            "convolution translation equivariance is not learned real motion",
            "no foreground/background separation",
        ],
        "device": "cpu",
        "threads": 1,
        "optimizer_updates": 0,
    }
    freeze(directory / "REGISTRATION.json", registration)
    binding = digest(registration)
    if (directory / "RESULT.json").exists():
        result = read(directory / "RESULT.json")
        if result["binding"] != binding:
            raise ValueError("Diagnostic registration changed")
        return result
    dataset = SensorDataset("P")
    encoders = {}
    for name in MODELS:
        model = initialize(name)
        payload = torch.load(
            RUN / "fits" / name / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        encoders[name] = model.encoder.eval().requires_grad_(False)
        del model, payload
    started = time.perf_counter()
    measurements = []
    try:
        with torch.inference_mode():
            for sample in selected:
                pixels = torch.from_numpy(dataset._row(sample["index"])["events"][-1:])
                for name, encoder in encoders.items():
                    previous = encoder(pixels)
                    if previous.shape != (1, 64, 32, 32):
                        raise ValueError("Unexpected native spatial feature shape")
                    permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
                    for dy, dx in OFFSETS:
                        current = encoder(translate(pixels, dy * 4, dx * 4))
                        null = current.flatten(2)[:, :, permutation].reshape_as(current)
                        measurements.append(
                            {
                                **sample,
                                "model": name,
                                "dy": dy,
                                "dx": dx,
                                "translated": matching(previous, current, dy, dx),
                                "permuted": matching(previous, null, dy, dx),
                            }
                        )
                print(f"CORRESPONDENCE P {sample['group']} complete", flush=True)
    finally:
        dataset.close()
    aggregate = {
        name: {
            condition: float(
                np.mean(
                    [
                        item[condition]["strict_accuracy"]
                        for item in measurements
                        if item["model"] == name
                    ]
                )
            )
            for condition in ("translated", "permuted")
        }
        for name in MODELS
    }
    result = {
        "status": "COMPLETE",
        "binding": binding,
        "utc": utc(),
        "seconds_cpu_wall_after_model_load": time.perf_counter() - started,
        "group_macro_strict_accuracy": aggregate,
        "measurements": measurements,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "ttc_accuracy_measured": False,
        "real_temporal_correspondence_measured": False,
    }
    freeze(directory / "RESULT.json", result)
    return result


if __name__ == "__main__":
    outcome = run()
    print({key: value for key, value in outcome.items() if key != "measurements"})
