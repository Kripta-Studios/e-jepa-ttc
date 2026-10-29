"""Minimal synthetic inputs for representation-operator unit tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.range_readout.cache_readout import capture
from operational.range_transfer.common import sha
from operational.representation_audit.spatial_cache import pool


def install(root: Path, monkeypatch) -> dict[str, torch.Tensor]:
    """Install a self-contained parent, calibrations, and cached vectors."""
    from operational.representation_audit import (
        geometry,
        precision,
        spatial_model,
        temporal_geometry,
    )

    owner = root / "owner"
    old = root / "old"
    run = root / "run"
    checkpoint = owner / "fits/E_A5_MATCHED/checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.manual_seed(31)
    config = CausalScaleTTCConfig(transport_enabled=True, transport_radius=1, dropout=0.0)
    native = CausalScaleTTC(config).eval()
    torch.save(
        {
            "identity": {"model_config": dataclasses.asdict(config)},
            "model_state_dict": native.state_dict(),
        },
        checkpoint,
    )
    old.mkdir(parents=True)
    (old / "ADMISSION.json").write_text(
        json.dumps({"parents": {"E_A5_MATCHED": sha(checkpoint)}}), encoding="utf-8"
    )

    geometry_calibration = run / "geometry.json"
    temporal_calibration = run / "temporal.json"
    geometry_calibration.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "role": "P",
        "mean": [0.0] * 128,
        "scale": [1.0] * 128,
        "coefficient": np.linspace(-0.002, 0.002, 128).tolist(),
        "intercept": 0.01,
    }
    geometry_calibration.write_text(json.dumps(base), encoding="utf-8")
    temporal_calibration.write_text(
        json.dumps({**base, "coefficient": np.linspace(0.003, -0.003, 128).tolist()}),
        encoding="utf-8",
    )
    spatial_dir = run / "spatial_geometry"
    spatial_dir.mkdir(parents=True)
    for arm in ("SP0", "SP1"):
        calibration = {
            "role": "P",
            "arm": arm,
            "mean": [0.0] * 512,
            "scale": [1.0] * 512,
            "coefficient": np.linspace(-0.001, 0.001, 512).tolist(),
            "intercept": 0.0,
            "variance_height_target": 0.2,
            "variance_temporal_target": 0.1,
        }
        (spatial_dir / f"{arm}_CALIBRATION.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )

    monkeypatch.setattr(precision, "OWNER", owner)
    monkeypatch.setattr(precision, "OLD", old)
    monkeypatch.setattr(geometry, "CALIBRATION", geometry_calibration)
    monkeypatch.setattr(spatial_model, "RUN", run)
    monkeypatch.setattr(temporal_geometry, "CONTROL_CALIBRATION", geometry_calibration)
    monkeypatch.setattr(temporal_geometry, "CALIBRATION", temporal_calibration)

    events = torch.randn(8, 3, 12, 128, 128)
    batch = SimpleNamespace(events=events, delta_t_s=torch.full((8,), 0.1))
    with torch.no_grad():
        values, _ = capture(native, batch, labels=False)
        dense = native.encoder.features(events.reshape(24, 12, 128, 128))
        values["height_features"] = pool(dense, 8, 3)
    height = values["visible_height"].clamp_min(0.05)
    values.update(
        target_height_normalized=height * 1.02,
        target_width_normalized=values["width_normalized"].clamp_min(0.05),
        target_centroid_x_normalized=values["centroid_x_normalized"],
        target_centroid_y_normalized=values["centroid_y_normalized"],
        target_valid=torch.ones(8, 3, dtype=torch.bool),
        target_ttc=torch.tensor([0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0]),
        dino_numerator=torch.zeros(8),
        dino_count=torch.ones(8),
    )
    return values
