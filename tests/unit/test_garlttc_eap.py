"""Unit tests for GarlTTC ↔ eAP data loader logic and index validation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.garlttc_eap import (
    GarlTTCBatch,
    GarlTTCEAPDataset,
    GarlTTCEAPIndex,
    bbox_to_patch_mask,
    collate_garlttc,
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    resolve_eap_events_path,
    validate_garlttc_train_index,
)
from e_jepa_ttc.models import TubeletTokenGeometry
from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig


@pytest.fixture(autouse=True)
def mock_hdf5plugin():
    with patch("e_jepa_ttc.data.eap._require_hdf5plugin") as m:
        yield m


def make_dummy_events_h5(
    path: Path,
    num_events: int = 100,
    t_start_us: int = 0,
    t_end_us: int = 5000000,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Helper to create a fully valid eAP HDF5 event file using h5py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["width"] = width
        f.attrs["height"] = height
        f.attrs["t_start_us"] = t_start_us
        f.attrs["t_end_us"] = t_end_us

        events_grp = f.create_group("events")
        if num_events > 0:
            x = np.random.randint(0, width, size=num_events, dtype=np.int16)
            y = np.random.randint(0, height, size=num_events, dtype=np.int16)
            t = np.sort(np.random.randint(t_start_us, t_end_us, size=num_events, dtype=np.int64))
            p = np.random.choice([-1, 1], size=num_events).astype(np.int8)
        else:
            x = np.empty(0, dtype=np.int16)
            y = np.empty(0, dtype=np.int16)
            t = np.empty(0, dtype=np.int64)
            p = np.empty(0, dtype=np.int8)

        events_grp.create_dataset("x", data=x)
        events_grp.create_dataset("y", data=y)
        events_grp.create_dataset("t", data=t)
        events_grp.create_dataset("p", data=p)
        events_grp.attrs["num_events"] = num_events
        events_grp.attrs["t_start_us"] = t_start_us
        events_grp.attrs["t_end_us"] = t_end_us
        events_grp.attrs["width"] = width
        events_grp.attrs["height"] = height

        max_ms = t_end_us // 1000 + 10
        ms_idx = np.zeros(max_ms, dtype=np.int64)
        if num_events > 0:
            t_ms = t // 1000
            for ms in range(max_ms):
                ms_idx[ms] = np.searchsorted(t_ms, ms, side="left")
        f.create_dataset("ms_to_idx", data=ms_idx)


@pytest.fixture
def mock_garlttc_dir(tmp_path: Path) -> Path:
    root = tmp_path / "garlttc"
    data_dir = root / "data"
    ann_dir = root / "annotations"
    data_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    df_data = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq2"],
            "sample_token": ["token1", "token2", "token3"],
            "track_id": ["t1", "t2", "t1"],
            "public_track_id": ["pt1", "pt2", "pt1"],
            "timestamp_us": [2000000, 2000000, 3000000],
            "events_path": ["seq1/events.h5", "seq1/events.h5", "seq2/events.h5"],
            "event_windows_us": [[[1900000, 2000000]], [[1900000, 2000000]], [[2900000, 3000000]]],
        }
    )
    df_data.to_parquet(data_dir / "train.parquet")

    df_ann = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq2"],
            "sample_token": ["token1", "token2", "token3"],
            "track_id": ["t1", "t2", "t1"],
            "public_track_id": ["pt1", "pt2", "pt1"],
            "timestamp_us": [2000000, 2000000, 3000000],
            "boxes_xyxy": ["[[10, 20, 100, 200]]", "[[50, 60, 150, 250]]", "[[0, 0, 50, 50]]"],
            "ttc": [1.5, -0.5, 2.0],
        }
    )
    df_ann.to_parquet(ann_dir / "train.parquet")
    return root


# --- 1. Box normalization tests ---


def test_normalize_boxes_flat_box():
    res = normalize_boxes_xyxy([10, 20, 30, 40])
    assert res == [(10.0, 20.0, 30.0, 40.0)]


def test_normalize_boxes_temporal_sequence():
    res = normalize_boxes_xyxy([[10, 20, 30, 40], [15, 25, 35, 45]])
    assert res == [(10.0, 20.0, 30.0, 40.0), (15.0, 25.0, 35.0, 45.0)]


def test_normalize_boxes_ndarray_object():
    arr = np.array([[10, 20, 30, 40]], dtype=object)
    res = normalize_boxes_xyxy(arr)
    assert res == [(10.0, 20.0, 30.0, 40.0)]


class DummyPyArrow:
    def __init__(self, val):
        self._val = val

    def as_py(self):
        return self._val


def test_normalize_boxes_pyarrow_as_py():
    pa = DummyPyArrow([[10, 20, 30, 40]])
    res = normalize_boxes_xyxy(pa)
    assert res == [(10.0, 20.0, 30.0, 40.0)]


def test_normalize_boxes_string():
    res = normalize_boxes_xyxy("[[10, 20, 30, 40]]")
    assert res == [(10.0, 20.0, 30.0, 40.0)]


def test_normalize_boxes_rejects_nan_and_inf():
    with pytest.raises(ValueError, match="not finite"):
        normalize_boxes_xyxy([10, float("nan"), 30, 40])
    with pytest.raises(ValueError, match="not finite"):
        normalize_boxes_xyxy([[10, 20, 30, float("inf")]])


def test_normalize_boxes_rejects_bad_length():
    with pytest.raises(ValueError, match="exactly 4 coordinates"):
        normalize_boxes_xyxy([10, 20, 30])


# --- 2. Index loading & dataclass tests ---


def test_load_garlttc_train_index_valid(mock_garlttc_dir: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1", "seq2"])
    assert len(idx.merged) == 3
    assert "boxes_xyxy" in idx.merged.columns
    assert "ttc" in idx.merged.columns


def test_load_garlttc_train_index_duplicate_data(mock_garlttc_dir: Path):
    data_p = mock_garlttc_dir / "data" / "train.parquet"
    df = pd.read_parquet(data_p)
    df = pd.concat([df, df.iloc[[0]]])
    df.to_parquet(data_p)
    with pytest.raises(ValueError, match="Duplicate join keys found in data_df"):
        load_garlttc_train_index(mock_garlttc_dir, ["seq1"])


def test_load_garlttc_train_index_duplicate_annotations(mock_garlttc_dir: Path):
    ann_p = mock_garlttc_dir / "annotations" / "train.parquet"
    df = pd.read_parquet(ann_p)
    df = pd.concat([df, df.iloc[[0]]])
    df.to_parquet(ann_p)
    with pytest.raises(ValueError, match="Duplicate join keys found in ann_df"):
        load_garlttc_train_index(mock_garlttc_dir, ["seq1"])


def test_load_garlttc_train_index_null_join_key(mock_garlttc_dir: Path):
    data_p = mock_garlttc_dir / "data" / "train.parquet"
    df = pd.read_parquet(data_p)
    df.loc[0, "timestamp_us"] = None
    df.to_parquet(data_p)
    with pytest.raises(ValueError, match="Null values found in data_df join key"):
        load_garlttc_train_index(mock_garlttc_dir, ["seq1"])


def test_load_garlttc_train_index_left_only_unlinked(mock_garlttc_dir: Path):
    data_p = mock_garlttc_dir / "data" / "train.parquet"
    df = pd.read_parquet(data_p)
    df.loc[0, "timestamp_us"] = 999999
    df.to_parquet(data_p)
    with pytest.raises(ValueError, match="Outer merge revealed unlinked rows"):
        load_garlttc_train_index(mock_garlttc_dir, ["seq1"])


def test_frozen_dataclass_replacement(mock_garlttc_dir: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1", "seq2"])
    updated_idx = replace(idx, train_sequences=["seq1"], validation_sequences=["seq2"])
    assert updated_idx.train_sequences == ["seq1"]
    assert updated_idx.validation_sequences == ["seq2"]


def test_validate_garlttc_train_index_nan_ttc(mock_garlttc_dir: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    idx.merged.loc[0, "ttc"] = float("nan")
    with pytest.raises(ValueError, match="null TTC values"):
        validate_garlttc_train_index(idx, expected_rows=3, allow_version_change=True)


# --- 3. Bbox patch mask geometry tests ---


def test_bbox_patch_mask_scaling_1280x720_to_160x90():
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=1,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    m, _ = bbox_to_patch_mask(
        (0, 0, 16, 16),
        original_width=1280,
        original_height=720,
        input_width=160,
        input_height=90,
        geometry=geometry,
    )
    assert m.shape == (5, 10)
    assert m[0, 0].item() is True
    assert m.sum().item() == 1


def test_bbox_patch_mask_bottom_strip_80_90():
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=1,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    m, _ = bbox_to_patch_mask(
        (0, 640, 1280, 720),
        original_width=1280,
        original_height=720,
        input_width=160,
        input_height=90,
        geometry=geometry,
    )
    assert m.shape == (5, 10)
    assert m.sum().item() == 1
    assert m[4, 4].item() is True
    assert m[:4].any().item() is False


def test_bbox_patch_mask_tiny_bbox_fallback():
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=1,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    m, _ = bbox_to_patch_mask(
        (10, 10, 11, 11),
        original_width=1280,
        original_height=720,
        input_width=160,
        input_height=90,
        geometry=geometry,
    )
    assert m.shape == (5, 10)
    assert m.sum().item() == 1


# --- 4. Dataset & HDF5 tests ---


def test_resolve_eap_events_path(tmp_path: Path):
    eap = tmp_path / "eAP"
    eap.mkdir()
    f = eap / "data" / "train" / "seq1" / "events.h5"
    f.parent.mkdir(parents=True)
    f.touch()

    res = resolve_eap_events_path(eap, "seq1/events.h5")
    assert res == f.resolve()


def test_dataset_zero_event_window(mock_garlttc_dir: Path, tmp_path: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    eap = tmp_path / "eAP"
    make_dummy_events_h5(eap / "seq1" / "events.h5", num_events=0, t_start_us=0, t_end_us=5000000)

    config = EAPJEPATrainerConfig(horizons_ms=(100,))
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    ds = GarlTTCEAPDataset(eap, idx, ["seq1"], config, geometry=geometry)
    item = ds[0]
    # Context voxel should be zero tensor of shape [21, 90, 160]
    ctx = item[0]
    assert ctx.shape == (21, 90, 160)
    assert torch.all(ctx == 0)
    ds.close()


def test_dataset_missing_event_file(mock_garlttc_dir: Path, tmp_path: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    eap = tmp_path / "eAP"
    config = EAPJEPATrainerConfig(horizons_ms=(100,))
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    with pytest.raises(FileNotFoundError):
        GarlTTCEAPDataset(eap, idx, ["seq1"], config, geometry=geometry)


def test_dataset_window_out_of_bounds(mock_garlttc_dir: Path, tmp_path: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    eap = tmp_path / "eAP"
    # Event file ends at 100000 us, but context timestamp is 2000000 us
    make_dummy_events_h5(eap / "seq1" / "events.h5", num_events=10, t_start_us=0, t_end_us=100000)

    config = EAPJEPATrainerConfig(horizons_ms=(100,))
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    with pytest.raises(ValueError, match="No GarlTTC-eAP samples satisfy the protocol."):
        GarlTTCEAPDataset(eap, idx, ["seq1"], config, geometry=geometry)


def test_collate_garlttc_metadata(mock_garlttc_dir: Path, tmp_path: Path):
    idx = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    eap = tmp_path / "eAP"
    make_dummy_events_h5(eap / "seq1" / "events.h5", num_events=500, t_start_us=0, t_end_us=5000000)

    config = EAPJEPATrainerConfig(horizons_ms=(100,))
    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )
    ds = GarlTTCEAPDataset(eap, idx, ["seq1"], config, geometry=geometry)
    batch = collate_garlttc([ds[0]])

    assert isinstance(batch, GarlTTCBatch)
    assert batch.context.shape == (1, 21, 90, 160)
    assert len(batch.events_paths) == 1
    assert len(batch.original_bboxes) == 1
    assert len(batch.transformed_bboxes) == 1
    assert batch.timestamp_us.shape == (1,)
    ds.close()


@dataclass
class ValidGarlTTCFixture:
    eap_root: Path
    index: GarlTTCEAPIndex
    config: EAPJEPATrainerConfig
    geometry: TubeletTokenGeometry


@pytest.fixture
def valid_garlttc_fixture(mock_garlttc_dir, tmp_path):
    eap_root = tmp_path / "eap"
    eap_root.mkdir()

    seq1_dir = eap_root / "seq1"
    seq1_dir.mkdir()
    events_h5 = seq1_dir / "events.h5"

    make_dummy_events_h5(
        path=events_h5,
        num_events=100,
        t_start_us=0,
        t_end_us=5000000,
    )

    index = load_garlttc_train_index(mock_garlttc_dir, ["seq1"])
    index = replace(
        index,
        source_data_row_count=2,
        source_annotation_row_count=2,
        source_merged_row_count=2,
        selected_row_count=2,
    )

    config = EAPJEPATrainerConfig(
        event_window_ms=100, horizons_ms=(500,), max_windows_per_sequence=10
    )

    from e_jepa_ttc.models import TubeletTokenGeometry

    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )

    return ValidGarlTTCFixture(eap_root=eap_root, index=index, config=config, geometry=geometry)


def test_dataset_constructor_initializes_reader_cache_before_use(
    valid_garlttc_fixture,
):
    dataset = GarlTTCEAPDataset(
        eap_root=valid_garlttc_fixture.eap_root,
        index=valid_garlttc_fixture.index,
        sequence_ids=["seq1"],
        config=valid_garlttc_fixture.config,
        geometry=valid_garlttc_fixture.geometry,
    )

    assert len(dataset) > 0
    assert isinstance(dataset._readers, dict)
    assert isinstance(dataset._voxel_cache, OrderedDict)

    dataset.close()
