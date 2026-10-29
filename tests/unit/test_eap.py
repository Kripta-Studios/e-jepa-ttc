from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.eap import (
    EAPEventReader,
    box_corners_ego,
    build_eap_object_windows,
    project_box_3d_to_event,
    reconstruct_eap_object_states,
)
from e_jepa_ttc.data.eap_representation import (
    base_compatible_voxel_chunks,
    base_compatible_voxel_windows_chunks,
)
from e_jepa_ttc.training import eap_jepa as eap_jepa_module
from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig


def test_eap_ssl_default_chunk_size_matches_memory_audit() -> None:
    assert EAPJEPATrainerConfig().event_chunk_size == 250_000


def test_box_projection_returns_camera_geometry() -> None:
    box = np.asarray([0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0])
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    corners, projected, nearest_depth, visible_height = project_box_3d_to_event(
        box,
        intrinsic,
        np.eye(4),
        image_size=(100, 80),
    )

    assert box_corners_ego(box).shape == (8, 3)
    assert corners.shape == (8, 3)
    assert nearest_depth == 9.0
    assert projected[0] < 50.0 < projected[2]
    assert projected[1] < 40.0 < projected[3]
    assert visible_height > 20.0


def test_reconstructed_ttc_is_signed_depth_over_depth_velocity() -> None:
    tokens = ["sequence:0", "sequence:1", "sequence:2", "sequence:3"]
    event_from_ego = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    media = pd.DataFrame(
        {
            "sample_token": tokens,
            "split": ["train"] * 4,
            "sequence_id": ["sequence"] * 4,
            "rgb_shard_path": ["rgb.tar"] * 4,
            "rgb_member_path": ["rgb/image.png"] * 4,
            "events_path": ["events.h5"] * 4,
            "labels_path": ["labels.parquet"] * 4,
            "rgb_exposure_start_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
            "rgb_exposure_end_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
            "K_event": [intrinsic] * 4,
            "T_event_ego": [event_from_ego] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": ["sequence"] * 4,
            "track_id": ["track"] * 4,
            "category": ["car"] * 4,
            "bbox_3d_ego": [
                np.asarray([10.0 - index, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]) for index in range(4)
            ],
        }
    )

    states = reconstruct_eap_object_states(
        media,
        labels,
        derivative_radius=1,
        maximum_gap_s=1.1,
    )

    assert len(states) == 4
    assert np.allclose([state.depth_velocity_mps for state in states], -1.0)
    assert np.allclose([state.ttc_s for state in states], [9.0, 8.0, 7.0, 6.0])
    assert all(state.ttc_source.startswith("reconstructed_public") for state in states)
    windows = build_eap_object_windows(
        states,
        history_frames=2,
        horizons_ms=(1000,),
        maximum_slop_ms=1,
        maximum_history_gap_ms=1001,
        ttc_range_s=(0.0, 20.0),
    )
    assert len(windows) == 2
    assert windows[0].future[0][1].timestamp_us > windows[0].target.timestamp_us
    assert windows[0].ego_action_valid is False


def test_eap_event_reader_applies_exact_half_open_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "events.h5"
    with h5py.File(source, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.asarray([1, 2, 3, 4, 5], dtype=np.uint16))
        events.create_dataset("y", data=np.asarray([6, 7, 8, 9, 10], dtype=np.uint16))
        events.create_dataset("t", data=np.asarray([1000, 1500, 1999, 2000, 2500]))
        events.create_dataset("p", data=np.asarray([0, 1, 0, 1, 1], dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=np.asarray([0, 0, 3, 5], dtype=np.uint64))

    events = EAPEventReader(source).read_window(1500, 2500)

    assert events["t"].tolist() == [1500, 1999, 2000]
    assert events["x"].tolist() == [2, 3, 4]
    assert all(array.shape == (3,) for array in events.values())


def test_eap_event_reader_chunks_match_whole_window(tmp_path: Path) -> None:
    source = tmp_path / "events.h5"
    with h5py.File(source, "w") as handle:
        events = handle.create_group("events")
        values = np.arange(10, dtype=np.int32)
        events.create_dataset("x", data=values)
        events.create_dataset("y", data=values + 10)
        events.create_dataset("t", data=np.arange(1000, 2000, 100, dtype=np.int64))
        events.create_dataset("p", data=np.ones(10, dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=np.arange(0, 11, dtype=np.uint64))

    reader = EAPEventReader(source)
    whole = reader.read_window(1100, 1800)
    chunks = list(reader.iter_window_chunks(1100, 1800, chunk_events=2))
    combined = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in whole}
    assert all(np.array_equal(combined[key], whole[key]) for key in whole)


def test_multiwindow_voxelization_matches_separate_windows() -> None:
    events = {
        "x": np.asarray([10, 20, 30, 40, 50], dtype=np.int32),
        "y": np.asarray([10, 20, 30, 40, 50], dtype=np.int32),
        "t": np.asarray([5, 15, 25, 35, 45], dtype=np.int64),
        "p": np.asarray([1, -1, 1, -1, 1], dtype=np.int8),
    }
    windows = ((0, 20), (20, 40), (40, 60))
    separate = torch.stack(
        [
            base_compatible_voxel_chunks(
                [events],
                sequence_id="multi",
                start_us=start_us,
                end_us=end_us,
                width=32,
                height=32,
                bins=5,
            )
            for start_us, end_us in windows
        ]
    )
    combined = base_compatible_voxel_windows_chunks(
        [events],
        windows=windows,
        sequence_id="multi",
        width=32,
        height=32,
        bins=5,
    )
    assert torch.equal(separate, combined)


def test_ssl_temporal_voxel_cache_reuses_endpoint_tensors(monkeypatch) -> None:
    config = EAPJEPATrainerConfig(
        horizons_ms=(100,),
        max_windows_per_sequence=1,
        reuse_temporal_voxel_cache=True,
    )
    dataset = object.__new__(eap_jepa_module.EAPOnDemandJEPADataset)
    dataset.config = config
    dataset.samples = [eap_jepa_module._EAPJEPASample(sequence_id="seq", timestamp_us=1_000_000)]
    dataset._readers = {}
    dataset._voxel_cache = eap_jepa_module.OrderedDict()

    class Reader:
        def iter_window_chunks(self, start_us, end_us, *, chunk_events):
            yield {
                "x": np.asarray([0], dtype=np.int32),
                "y": np.asarray([0], dtype=np.int32),
                "t": np.asarray([start_us], dtype=np.int64),
                "p": np.asarray([1], dtype=np.int8),
            }

        def close(self):
            return None

    dataset._readers["seq"] = Reader()
    calls: list[tuple[tuple[int, int], ...]] = []

    def fake_voxelizer(chunks, *, windows, **_kwargs):
        calls.append(tuple(windows))
        list(chunks)
        return torch.zeros(len(windows), 21, 90, 160)

    monkeypatch.setattr(eap_jepa_module, "base_compatible_voxel_windows_chunks", fake_voxelizer)
    dataset[0]
    dataset[0]

    assert calls == [((900_000, 1_000_000), (1_000_000, 1_100_000))]


def test_ssl_temporal_voxel_cache_can_be_disabled(monkeypatch) -> None:
    config = EAPJEPATrainerConfig(
        horizons_ms=(100,),
        max_windows_per_sequence=1,
        reuse_temporal_voxel_cache=False,
    )
    dataset = object.__new__(eap_jepa_module.EAPOnDemandJEPADataset)
    dataset.config = config
    dataset.samples = [eap_jepa_module._EAPJEPASample(sequence_id="seq", timestamp_us=1_000_000)]
    dataset._readers = {}
    dataset._voxel_cache = eap_jepa_module.OrderedDict()

    class Reader:
        def iter_window_chunks(self, start_us, end_us, *, chunk_events):
            yield {
                "x": np.asarray([0], dtype=np.int32),
                "y": np.asarray([0], dtype=np.int32),
                "t": np.asarray([start_us], dtype=np.int64),
                "p": np.asarray([1], dtype=np.int8),
            }

        def close(self):
            return None

    dataset._readers["seq"] = Reader()
    calls = 0

    def fake_voxelizer(chunks, *, windows, **_kwargs):
        nonlocal calls
        calls += 1
        list(chunks)
        return torch.zeros(len(windows), 21, 90, 160)

    monkeypatch.setattr(eap_jepa_module, "base_compatible_voxel_windows_chunks", fake_voxelizer)
    dataset[0]
    dataset[0]

    assert calls == 2
