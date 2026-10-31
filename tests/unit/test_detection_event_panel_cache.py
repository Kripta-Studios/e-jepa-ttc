from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from e_jepa_ttc.detection.eap_rgb_data import sha256_file
from e_jepa_ttc.detection.event_campaign import EventCampaignExtension
from e_jepa_ttc.detection.event_data import CausalEventConfig
from e_jepa_ttc.detection.event_panel_cache import CausalEventPanelCache, seal_event_panel
from e_jepa_ttc.detection.temporal_campaign import TemporalCampaignBatch
from e_jepa_ttc.detection.temporal_training import TemporalTrainingConfig


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def prepare(root: Path) -> CausalEventPanelCache:
    config = asdict(CausalEventConfig(width=160, height=96, max_open_handles=1))
    (root / "data").mkdir()
    pd.DataFrame({"sample_token": ["s:f"], "rgb_exposure_end_timestamp_us": [100000]}).to_parquet(
        root / "data/train.parquet", index=False
    )
    write(
        root / "CONTRACT.json",
        {
            "schema": "eap_causal_event_panel_v1",
            "manifest_sha256": "a" * 64,
            "metadata_sha256": sha256_file(root / "data/train.parquet"),
            "config": config,
            "populations": {"train": ["s:f"], "selection": []},
        },
    )
    contract = sha256_file(root / "CONTRACT.json")
    voxel = np.zeros((10, 96, 160), dtype=np.float32)
    np.savez_compressed(root / "frame.npz", voxel=voxel)
    digest = sha256_file(root / "frame.npz")
    write(
        root / "frame.json",
        {
            "identity": {
                "sample_token": "s:f",
                "role": "train",
                "contract_sha256": contract,
                "rgb_exposure_end_timestamp_us": 100000,
            },
            "file_sha256": digest,
            "representation": {
                "window_end_us_exclusive": 100000,
                "window_start_us_inclusive": 50000,
                "uses_future_events": False,
                "config": config,
                "event_count": 3,
                "voxel_sha256_float32_le": hashlib.sha256(voxel.tobytes()).hexdigest(),
            },
        },
    )
    write(
        root / "INDEX.json",
        [{"sample_token": "s:f", "role": "train", "path": "frame.npz", "sha256": digest}],
    )
    write(
        root / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": 1,
            "contract_sha256": contract,
            "index_sha256": sha256_file(root / "INDEX.json"),
        },
    )
    admission = seal_event_panel(root)
    assert seal_event_panel(root) == admission
    return CausalEventPanelCache(
        root,
        contract_sha256=contract,
        index_sha256=admission["sealed_index_sha256"],
        manifest_sha256="a" * 64,
    )


def test_event_cache_preserves_order_and_rejects_role_and_cutoff(tmp_path: Path) -> None:
    cache = prepare(tmp_path)
    batch = cache.load_batch([("s:f", 100000)], role="train", device=torch.device("cpu"))
    assert batch.tokens == ("s:f",)
    assert batch.voxel.shape == (1, 10, 96, 160)
    assert batch.observables.raw_event_count.tolist() == [3.0]
    with pytest.raises(ValueError, match="causal window"):
        cache.load_batch([("s:f", 100001)], role="train", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="role"):
        cache.load_batch([("s:f", 100000)], role="selection", device=torch.device("cpu"))


@pytest.mark.parametrize("file", ["frame.npz", "frame.json"])
def test_event_cache_detects_post_admission_mutation(tmp_path: Path, file: str) -> None:
    cache = prepare(tmp_path)
    (tmp_path / file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        cache.load_batch([("s:f", 100000)], role="train", device=torch.device("cpu"))


def test_event_cache_cannot_seal_incomplete_producer(tmp_path: Path) -> None:
    prepare(tmp_path)
    status = json.loads((tmp_path / "STATUS.json").read_text())
    status["status"] = "RUNNING"
    write(tmp_path / "STATUS.json", status)
    with pytest.raises(ValueError, match="incomplete"):
        seal_event_panel(tmp_path)


@pytest.mark.parametrize("mode", ["events", "rgb_capacity"])
def test_sensor_extension_binds_exposure_and_keeps_capacity_control_rgb_only(
    tmp_path: Path, mode: str
) -> None:
    cache = prepare(tmp_path)
    config = cast(
        TemporalTrainingConfig,
        SimpleNamespace(
            baseline=SimpleNamespace(
                data=SimpleNamespace(root=str(tmp_path), expected_manifest_sha256="a" * 64),
                evaluation=SimpleNamespace(full_selection_updates=()),
                seed=7,
            ),
            temporal=SimpleNamespace(
                history_limit=0, producer_version="test", maximum_history_age_us=500000
            ),
        ),
    )
    raw = {
        "schema": "eap_detection_event_screen_v1",
        "control": "F01",
        "input_mode": mode,
        "cache_root": str(tmp_path),
        "cache_contract_sha256": cache.contract_sha256,
        "cache_index_sha256": sha256_file(tmp_path / "SEALED_INDEX.json"),
    }
    path = tmp_path / "sensor.yaml"
    path.write_text(yaml.safe_dump(raw))
    extension = EventCampaignExtension(path, config)
    images = torch.randn(1, 3, 16, 24)
    batch = cast(
        TemporalCampaignBatch,
        SimpleNamespace(
            current_images=images,
            temporal_context=SimpleNamespace(current=(SimpleNamespace(frame_id="s:f"),)),
        ),
    )
    if mode == "rgb_capacity":
        # The capacity control must work without reading any event payload.
        (tmp_path / "frame.npz").unlink()
    arguments = extension.model_inputs(batch, role="train")
    assert arguments["event_voxel"].shape == (1, 10, 96, 160)
    assert arguments["event_tokens"] == ("s:f",)
    assert arguments["event_observables"].raw_event_count.tolist() == (
        [3.0] if mode == "events" else [1.0]
    )
    with pytest.raises(ValueError, match="population role"):
        extension.model_inputs(batch, role="selection")
    state = torch.get_rng_state()
    extension.build_adapter(torch.nn.Identity())
    assert torch.equal(state, torch.get_rng_state())
    raw["control"] = "F11"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="history config"):
        EventCampaignExtension(path, config)
