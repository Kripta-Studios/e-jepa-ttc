from pathlib import Path

import numpy as np

from e_jepa_ttc.data.evttc import (
    count_events_window,
    discover_event_layout,
    read_events_window,
    read_manifest,
    scan_evttc_root,
    validate_manifest,
    write_manifest,
)
from e_jepa_ttc.data.index import build_temporal_index, write_index
from e_jepa_ttc.data.ml_cache import build_voxel_cache
from e_jepa_ttc.data.split import create_sequence_splits, validate_split_groups
from e_jepa_ttc.data.synthetic import generate_synthetic_sequence, write_synthetic_hdf5
from e_jepa_ttc.utils.io import write_structured


def _write_sequence(root: Path, family: str, speed: str) -> None:
    sequence_dir = root / family / speed / "overlap-100"
    sequence_dir.mkdir(parents=True)
    sequence = generate_synthetic_sequence(windows=16, seed=len(speed))
    write_synthetic_hdf5(sequence_dir / f"{speed}.hdf5", sequence)
    write_synthetic_hdf5(sequence_dir / "gt.hdf5", sequence)
    rows = []
    for frame_id, timestamp_s, distance, relative_speed, ttc_s in zip(
        sequence.frame_id,
        sequence.timestamp_s,
        sequence.distance,
        sequence.relative_speed,
        sequence.ttc_s,
        strict=True,
    ):
        rows.append(f"{frame_id} {timestamp_s:.6f} {distance:.6f} {relative_speed:.6f} {ttc_s:.6f}")
    (sequence_dir / "ttc.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    label_dir = sequence_dir / "leftlabel"
    label_dir.mkdir()
    (label_dir / "0000.json").write_text('{"objects": []}', encoding="utf-8")


def test_scan_validate_index_and_split(tmp_path: Path) -> None:
    root = tmp_path / "evttc"
    _write_sequence(root, "CCRs-1", "low-100")
    _write_sequence(root, "CCRs-1", "medium-100")
    _write_sequence(root, "CCRs-1", "high-100")

    sequences = scan_evttc_root(root)
    assert [sequence.speed_bucket for sequence in sequences] == ["high", "low", "medium"]

    manifest = tmp_path / "manifest.yaml"
    write_manifest(manifest, sequences)
    loaded = read_manifest(manifest)
    assert len(loaded) == 3

    report = validate_manifest(manifest)
    assert report["sequence_count"] == 3
    assert report["sequences"][0]["event_layout"]["kind"] == "separate"
    assert report["sequences"][0]["event_layout"]["ms_map_idx"].endswith("ms_map_idx")
    assert report["sequences"][0]["event_layout"]["width"] == 64
    assert report["sequences"][0]["event_layout"]["height"] == 48

    entries = build_temporal_index(
        manifest_path=manifest,
        context_ms=20,
        stride_ms=20,
        horizons_ms=(20,),
        clip_ttc_seconds=(0.1, 10.0),
    )
    assert entries
    first = entries[0]
    assert first.horizons_us[20][0] == first.context_end_us + 20_000
    assert first.horizons_us[20][1] == first.context_end_us + 40_000

    splits = create_sequence_splits(loaded, seed=42)
    validate_split_groups(loaded, splits)
    assert splits["train"] == ["CCRs-1-low-100-overlap-100"]
    assert splits["validation"] == ["CCRs-1-medium-100-overlap-100"]
    assert splits["test"] == ["CCRs-1-high-100-overlap-100"]

    index_path = tmp_path / "index.json"
    split_path = tmp_path / "split.yaml"
    cache_path = tmp_path / "voxel_cache.npz"
    write_index(index_path, entries)
    write_structured(split_path, {"splits": splits})
    summary = build_voxel_cache(
        manifest_path=manifest,
        split_path=split_path,
        index_path=index_path,
        output_path=cache_path,
        width=16,
        height=12,
        bins=2,
        normalize=False,
        metadata_channels=True,
        navigation_channels=True,
        limit=3,
    )
    assert summary["shape"] == [3, 15, 12, 16]
    assert summary["normalize"] is False
    assert summary["metadata_channels"] is True
    assert summary["navigation_channels"] is True
    assert len(summary["navigation_feature_names"]) == 9
    cache = np.load(cache_path, allow_pickle=False)
    assert bool(cache["normalize"]) is False
    assert bool(cache["metadata_channels"]) is True
    assert bool(cache["navigation_channels"]) is True
    assert np.array_equal(cache["timestamp_us"], cache["context_end_us"])
    assert np.all(cache["context_start_us"] < cache["context_end_us"])
    assert str(cache["future_window_semantics"]) == (
        "disjoint_window_start_after_context_plus_horizon"
    )
    assert np.all(cache["x"][:, -11:-9].astype(np.float32) > 0.0)
    assert np.all(cache["x"][:, -9:].astype(np.float32) == 0.0)


def test_scan_uses_bbox_segmentation_when_leftlabel_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "evttc"
    _write_sequence(root, "CPLA", "high")
    sequence_dir = root / "CPLA" / "high" / "overlap-100"
    (sequence_dir / "leftlabel").rename(sequence_dir / "bbox_segmentation")

    sequences = scan_evttc_root(root)

    assert len(sequences) == 1
    assert sequences[0].label_dir == "bbox_segmentation"
    assert sequences[0].target_type == "pedestrian"
    assert sequences[0].extra["label_count"] == 1


def test_discover_event_layout_on_fixture(tmp_path: Path) -> None:
    sequence = generate_synthetic_sequence(windows=4, seed=1)
    path = tmp_path / "events.hdf5"
    write_synthetic_hdf5(path, sequence)

    layout = discover_event_layout(path)

    assert layout is not None
    assert layout.x == "events/x"
    assert layout.t == "events/t"
    assert layout.ms_map_idx == "events/ms_map_idx"

    window = read_events_window(
        path,
        t_start_us=20_000,
        t_end_us=60_000,
        sequence_id="fixture",
    )
    assert window.num_events > 0
    assert window.width == 64
    assert window.height == 48
    assert int(window.t_us[0]) >= 20_000
    assert int(window.t_us[-1]) < 60_000

    counted = count_events_window(
        path,
        t_start_us=20_000,
        t_end_us=60_000,
    )
    assert counted == window.num_events
