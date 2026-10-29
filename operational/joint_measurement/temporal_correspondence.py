"""Registered P-only real-window correspondence diagnostic, without TTC fitting."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from e_jepa_ttc.data.object_event_v4 import weak_box_masks
from e_jepa_ttc.joint_measurement.model import local_cosine_volume
from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, owner
from .data import SensorDataset, cache_root, role_rows
from .evaluate import freeze_primary
from .train import initialize, sources

MODELS = ("PAIR-CORR-FROZEN-S7", "PAIR-CORR-S7")


def coherence(previous: Tensor, current: Tensor, support: Tensor) -> dict:
    """Count strict, reciprocal top-one matches; constant-map ties never count."""
    if previous.shape[0] != 1 or support.shape != previous.shape[-2:]:
        raise ValueError("Diagnostic requires one pair and its spatial support")
    if support.dtype != torch.bool:
        raise ValueError("Diagnostic support must be boolean")
    forward, valid = local_cosine_volume(previous, current)
    reverse, reverse_valid = local_cosine_volume(current, previous)
    if not torch.isfinite(forward).all() or not torch.isfinite(reverse).all():
        raise ValueError("Nonfinite temporal correspondence")
    values, index = forward.masked_fill(~valid, -torch.inf).topk(2, dim=1)
    reverse_values, reverse_index = reverse.masked_fill(~reverse_valid, -torch.inf).topk(2, dim=1)
    height, width = support.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    dx, dy = index[0, 0] % 5 - 2, index[0, 0] // 5 - 2
    destination = (yy + dy).clamp(0, height - 1) * width + (xx + dx).clamp(0, width - 1)
    reverse_choice = reverse_index[0, 0].flatten()[destination]
    gap = values[0, 0] - values[0, 1]
    reverse_gap = (reverse_values[0, 0] - reverse_values[0, 1]).flatten()[destination]
    strict = (gap > 1e-6) & (reverse_gap > 1e-6)
    cycle_x = dx + reverse_choice % 5 - 2
    cycle_y = dy + reverse_choice // 5 - 2
    mutual = strict & (cycle_x == 0) & (cycle_y == 0)
    nonzero = (dx != 0) | (dy != 0)
    count = int(support.sum())
    return {
        "positions": count,
        "strict_mutual": int((mutual & support).sum()),
        "strict_mutual_nonzero": int((mutual & nonzero & support).sum()),
        "strict_mutual_fraction": float(mutual[support].float().mean()) if count else None,
        "strict_mutual_nonzero_fraction": (
            float((mutual & nonzero)[support].float().mean()) if count else None
        ),
        "mean_top_margin": float(gap[support].mean()) if count else None,
        "mean_cycle_error_cells": (
            float((cycle_x.float().square() + cycle_y.float().square()).sqrt()[support].mean())
            if count
            else None
        ),
        "displacement_counts": torch.bincount(index[0, 0][support], minlength=25).tolist(),
    }


def run() -> dict:
    """Register fixed queries, support and null before observing any matches."""
    torch.set_num_threads(1)
    directory = RUN / "diagnostics/real_temporal_P"
    rows = role_rows("P").reset_index(drop=True)
    samples = []
    for group, subset in rows.groupby("group_id", sort=True):
        indices = sorted(
            subset.index,
            key=lambda i: hashlib.sha256(str(rows.iloc[i].sample_token).encode()).hexdigest(),
        )[:4]
        for index in indices:
            samples.append(
                {
                    "index": int(index),
                    "group": str(group),
                    "token": str(rows.iloc[index].sample_token),
                }
            )
    endpoints = freeze_primary()["endpoints"]
    existing = directory / "REGISTRATION.json"
    source_commit = (
        read(existing)["git_commit"]
        if existing.exists()
        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    )
    record = {
        "schema": "v18_real_temporal_correspondence_diagnostic_v1",
        "models": {name: endpoints[name] for name in MODELS},
        "role": "P",
        "samples": samples,
        "sample_rule": "four smallest SHA256(sample_token) per P group; no target filtering",
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_complete_sha256": sha(cache_root("P") / "COMPLETE.json"),
        "sources": {
            **sources(),
            **{
                relative: sha(ROOT / relative)
                for relative in (
                    "operational/joint_measurement/temporal_correspondence.py",
                    "src/e_jepa_ttc/joint_measurement/model.py",
                    "src/e_jepa_ttc/data/object_event_v4.py",
                )
            },
        },
        "git_commit": source_commit,
        "hypothesis": (
            "Real temporal nonzero reciprocal correspondence could justify paired geometric "
            "readout versus the failed free readout; consistency alone is not accurate motion"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "input": "both real native P event windows in their common ROI",
        "support": "all cells with radius-two interior; separately previous bbox pooled >=0.5",
        "bbox_use": "P-only weak foreground reporting mask after encoder; never model input",
        "matching": "native radius-two cosine, unique top-one forward/reverse margin >1e-6",
        "null": "current spatial features permuted with CPU seed7; no resampling",
        "enable_geometry_followup": {
            "minimum_groups_with_bbox_support": 18,
            "fraction_groups_mutual_advantage_at_least_0_10": 0.75,
            "group_macro_mutual_nonzero_fraction_min": 0.10,
            "scope": "mechanism signal only; numeric paired geometry recipe still required",
        },
        "limitations": [
            "bbox is weak foreground, not object correspondences",
            "mutual consistency is not motion ground truth",
            "zero displacement alone does not establish scale information",
            "small P-only diagnostic is not TTC generalization",
        ],
        "device": "cpu",
        "threads": 1,
        "optimizer_updates": 0,
    }
    freeze(directory / "REGISTRATION.json", record)
    binding = digest(record)
    if (directory / "RESULT.json").exists():
        result = read(directory / "RESULT.json")
        if result["binding"] != binding:
            raise ValueError("Temporal diagnostic registration changed")
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
    measurements = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for sample in samples:
                values = dataset._row(sample["index"])
                pixels = torch.from_numpy(values["events"])
                bbox, _ = weak_box_masks(
                    torch.from_numpy(values["boxes"])[None], height=128, width=128
                )
                interior = torch.zeros(32, 32, dtype=torch.bool)
                interior[2:-2, 2:-2] = True
                weak = functional.avg_pool2d(bbox[0, :1], 4)[0, 0] >= 0.5
                supports = {"interior": interior, "weak_bbox": interior & weak}
                permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
                for name, encoder in encoders.items():
                    dense = encoder(pixels)
                    if dense.shape != (2, 64, 32, 32):
                        raise ValueError("Unexpected native pair feature shape")
                    previous, current = dense[:1], dense[1:]
                    null = current.flatten(2)[:, :, permutation].reshape_as(current)
                    measurements.append(
                        {
                            **sample,
                            "model": name,
                            "delta_t_s": float(np.asarray(values["delta"]).reshape(-1)[-1]),
                            "supports": {
                                support_name: {
                                    "real": coherence(previous, current, mask),
                                    "permuted": coherence(previous, null, mask),
                                }
                                for support_name, mask in supports.items()
                            },
                        }
                    )
                print(f"TEMPORAL P {sample['group']} {sample['token']} complete", flush=True)
    finally:
        dataset.close()
    grouped = {}
    for name in MODELS:
        groups = {}
        for group in sorted({sample["group"] for sample in samples}):
            items = [
                item["supports"]["weak_bbox"]
                for item in measurements
                if item["model"] == name
                and item["group"] == group
                and item["supports"]["weak_bbox"]["real"]["positions"] > 0
            ]
            if items:
                groups[group] = {
                    "queries": len(items),
                    "mutual_advantage": float(
                        np.mean(
                            [
                                item["real"]["strict_mutual_fraction"]
                                - item["permuted"]["strict_mutual_fraction"]
                                for item in items
                            ]
                        )
                    ),
                    "real_mutual_nonzero": float(
                        np.mean([item["real"]["strict_mutual_nonzero_fraction"] for item in items])
                    ),
                }
        advantage = (
            float(np.mean([value["mutual_advantage"] >= 0.10 for value in groups.values()]))
            if groups
            else 0.0
        )
        nonzero = (
            float(np.mean([value["real_mutual_nonzero"] for value in groups.values()]))
            if groups
            else 0.0
        )
        grouped[name] = {
            "groups": groups,
            "groups_with_support": len(groups),
            "fraction_groups_with_advantage": advantage,
            "group_macro_mutual_nonzero": nonzero,
            "geometry_followup_signal": len(groups) >= 18 and advantage >= 0.75 and nonzero >= 0.10,
        }
    result = {
        "status": "COMPLETE",
        "binding": binding,
        "utc": utc(),
        "measurements": measurements,
        "grouped": grouped,
        "seconds_cpu_wall_after_model_load": time.perf_counter() - started,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "ttc_accuracy_measured": False,
        "motion_ground_truth_available": False,
        "automatic_training": False,
    }
    freeze(directory / "RESULT.json", result)
    return result


if __name__ == "__main__":
    outcome = run()
    print({key: value for key, value in outcome.items() if key != "measurements"})
