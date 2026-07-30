from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.data.carla_looming import (
    CARLA_LOOMING_EVENT_DTYPE,
    scan_carla_looming_root,
)
from e_jepa_ttc.training.carla_jepa import (
    EVTTC_BASE_EVENT_CHANNELS,
    EVTTC_BASE_INPUT_CHANNELS,
    CarlaJEPATrainerConfig,
    CarlaJEPAVoxelDataset,
)


def _write_sequence(root: Path) -> None:
    directory = root / "example_0"
    directory.mkdir(parents=True)
    timestamps = np.arange(0, 200, 10, dtype=np.uint32)
    events = np.zeros(timestamps.size, dtype=CARLA_LOOMING_EVENT_DTYPE)
    events["t"] = timestamps
    events["x"] = np.arange(timestamps.size, dtype=np.uint16) * 20
    events["y"] = np.arange(timestamps.size, dtype=np.uint16) * 10
    events["p"] = np.arange(timestamps.size, dtype=np.uint16) % 2
    np.save(directory / "events.npy", events, allow_pickle=False)
    np.savez(
        directory / "sim_data.npz",
        coll_type=np.asarray("none"),
        t_end=np.asarray(200),
        dt=np.asarray(10.0),
        vel=np.asarray(5.0),
        diameter_object=np.asarray(None, dtype=object),
    )


def test_carla_jepa_voxels_match_evttc_base_contract(tmp_path: Path) -> None:
    _write_sequence(tmp_path)
    sequences = scan_carla_looming_root(tmp_path, context_ms=20)
    config = CarlaJEPATrainerConfig(
        epochs=1,
        batch_size=1,
        gradient_accumulation=1,
        num_workers=0,
        context_ms=20,
        stride_ms=20,
        horizons_ms=(10, 20),
        future_window_ms=20,
        max_windows_per_sequence=2,
        width=32,
        height=24,
        early_stopping_min_epochs=1,
        early_stopping_patience=0,
    )

    dataset = CarlaJEPAVoxelDataset(tmp_path, sequences, config)
    context, future, valid, synthetic_ttc, has_ttc = dataset[0]

    assert context.shape == (EVTTC_BASE_INPUT_CHANNELS, 24, 32)
    assert future.shape == (2, EVTTC_BASE_INPUT_CHANNELS, 24, 32)
    assert valid.tolist() == [True, True]
    assert bool(context[:EVTTC_BASE_EVENT_CHANNELS].abs().sum() > 0)
    assert bool(future[:, :EVTTC_BASE_EVENT_CHANNELS].abs().sum() > 0)
    assert bool((context[EVTTC_BASE_EVENT_CHANNELS:] == 0).all())
    assert bool((future[:, EVTTC_BASE_EVENT_CHANNELS:] == 0).all())
    assert synthetic_ttc.item() == 0.0
    assert has_ttc.item() is False


def test_carla_jepa_rejects_non_base_event_bin_count() -> None:
    with pytest.raises(ValueError, match="exactly five event bins"):
        CarlaJEPATrainerConfig(bins=4)


def test_carla_jepa_rejects_negative_synthetic_ttc_weight() -> None:
    with pytest.raises(ValueError, match="synthetic_ttc_loss_weight"):
        CarlaJEPATrainerConfig(synthetic_ttc_loss_weight=-1.0)
