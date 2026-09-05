from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.data.raw_event_binding import (
    EXPECTED_SEQUENCES,
    ReadOnlyTrainAccess,
    _reject_external_links,
    build_raw_window_bindings,
    exact_window_bounds,
    select_hash_probe_tokens,
)
from e_jepa_ttc.data.raw_temporal_cache import rasterize_bound_window, temporal_bin_edges


def _event_file(path: Path, *, external: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.array([10_000, 10_000, 10_999, 11_000, 11_500, 11_999, 12_000])
    with h5py.File(path, "w") as handle:
        if external:
            target = path.with_name("external.h5")
            with h5py.File(target, "w") as other:
                other.create_dataset("x", data=np.arange(len(timestamps)))
            events = handle.create_group("events")
            events["x"] = h5py.ExternalLink(target.name, "x")
        else:
            events = handle.create_group("events")
            events.create_dataset("x", data=np.arange(len(timestamps)) + 100)
        events = handle["events"]
        events.create_dataset("y", data=np.arange(len(timestamps)) + 100)
        events.create_dataset("t", data=timestamps)
        events.create_dataset("p", data=np.arange(len(timestamps), dtype=np.uint8) % 2)
        handle.create_dataset("ms_to_idx", data=np.searchsorted(timestamps, np.arange(14) * 1000))


def test_exact_h5_bounds_handles_offset_duplicates_and_bad_hint(tmp_path: Path) -> None:
    path = tmp_path / "events.h5"
    _event_file(path)
    with h5py.File(path, "r") as handle:
        timestamps = handle["events/t"]
        broken_hint = np.full(14, len(timestamps), dtype=np.int64)
        actual = exact_window_bounds(
            timestamps, 10_000, 11_000, ms_to_idx=broken_hint, index_origin_us=0
        )
        assert actual == tuple(np.searchsorted(timestamps[:], [10_000, 11_000]))


def test_binding_is_two_rows_per_token_and_common_roi(tmp_path: Path) -> None:
    raw_root = tmp_path / "eap" / "data" / "train"
    stage_rows = []
    source_rows = []
    for index, sequence in enumerate(sorted(EXPECTED_SEQUENCES)):
        _event_file(raw_root / sequence / "events.h5")
        token = f"{sequence}_token"
        stage_rows.append(
            {
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": f"{sequence}_track",
                "outer_fold": index % 3,
            }
        )
        source_rows.append(
            {
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": f"{sequence}_track",
                "timestamp_us": 2_011_999,
                "frame_timestamps_us": [2_010_999, 2_011_999],
                "events_path": f"data/train/{sequence}/events.h5",
                "event_windows_us": [[10_000, 11_000], [11_000, 12_000]],
                "boxes_xyxy": [[90, 90, 120, 120], [92, 92, 125, 125]],
            }
        )
    stage = tmp_path / "stage.csv"
    parquet = tmp_path / "train.parquet"
    pd.DataFrame(stage_rows).to_csv(stage, index=False)
    pd.DataFrame(source_rows).to_parquet(parquet, index=False)
    access = ReadOnlyTrainAccess(raw_root, parquet)
    binding, manifest = build_raw_window_bindings(
        stage_metadata_path=stage,
        train_parquet=parquet,
        raw_train_root=raw_root,
        access=access,
        hash_progress_path=tmp_path / "hashes.json",
        expected_tokens=9,
    )
    assert len(binding) == 18
    assert binding.groupby("sample_token")["roi_transform_sha256"].nunique().eq(1).all()
    assert binding["h5_file_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert binding["frame_to_event_clock_offset_drift_us"].eq(0).all()
    assert binding["target_anchor_event_clock_us"].eq(12_000).all()
    assert manifest["common_roi"]["same_transform_for_both_windows"]
    assert len(select_hash_probe_tokens(binding, 9)) == 9


def test_path_escape_and_external_link_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "train"
    parquet = tmp_path / "train.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(parquet)
    _event_file(root / sorted(EXPECTED_SEQUENCES)[0] / "events.h5", external=True)
    access = ReadOnlyTrainAccess(root, parquet)
    with pytest.raises(ValueError):
        access.event_path(sorted(EXPECTED_SEQUENCES)[0], "../events.h5", "test")
    event_path = access.event_path(
        sorted(EXPECTED_SEQUENCES)[0],
        f"data/train/{sorted(EXPECTED_SEQUENCES)[0]}/events.h5",
        "external_link_test",
    )
    with h5py.File(event_path, "r") as handle, pytest.raises(ValueError, match="external"):
        _reject_external_links(handle, event_path)


def test_rasterization_is_half_open_and_polarity_safe() -> None:
    binding = pd.Series(
        {
            "event_x_offset_px": 0.0,
            "polarity_encoding": "zero_one",
            "window_start_us": 0,
            "window_end_us": 16,
            "roi_x0": 0.0,
            "roi_y0": 0.0,
            "roi_x1": 64.0,
            "roi_y1": 64.0,
        }
    )
    counts, roi = rasterize_bound_window(
        x=np.array([0, 1]),
        y=np.array([0, 1]),
        t_us=np.array([0, 15]),
        polarity=np.array([0, 1], dtype=np.uint8),
        binding=binding,
    )
    assert roi == 2
    assert counts[0, 0, 0, 0] == 1
    assert counts[15, 1, 1, 1] == 1


@pytest.mark.parametrize("duration", [16, 17, 31, 100001, 100003])
def test_every_integer_boundary_uses_the_stored_bin_duration(duration: int) -> None:
    start = 9_000_000_000
    edges = temporal_bin_edges(start, start + duration)
    binding = pd.Series(
        {
            "event_x_offset_px": 0.0,
            "polarity_encoding": "signed",
            "window_start_us": start,
            "window_end_us": start + duration,
            "roi_x0": 0.0,
            "roi_y0": 0.0,
            "roi_x1": 64.0,
            "roi_y1": 64.0,
        }
    )
    # Both ends of every nonempty half-open bin must map to that same bin.
    for expected_bin in range(16):
        timestamps = np.array([edges[expected_bin], edges[expected_bin + 1] - 1])
        counts, accepted = rasterize_bound_window(
            x=np.zeros(2),
            y=np.zeros(2),
            t_us=timestamps,
            polarity=np.ones(2, dtype=np.int8),
            binding=binding,
        )
        assert accepted == 2
        assert counts[expected_bin, 1, 0, 0] == 2
        assert counts.sum() == 2
    assert np.diff(edges).sum() == duration
    with pytest.raises(ValueError, match="half-open"):
        rasterize_bound_window(
            x=np.zeros(1),
            y=np.zeros(1),
            t_us=edges[-1:],
            polarity=np.ones(1, dtype=np.int8),
            binding=binding,
        )


def test_raw16_rejects_zero_width_temporal_bins() -> None:
    with pytest.raises(ValueError, match="16 microseconds"):
        temporal_bin_edges(0, 15)


def test_access_journal_survives_reader_recreation(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    parquet = tmp_path / "train.parquet"
    pd.DataFrame({"sequence_id": []}).to_parquet(parquet)
    journal = tmp_path / "reads.jsonl"
    first = ReadOnlyTrainAccess(root, parquet, journal_path=journal)
    first.parquet_path("attempt_one")
    second = ReadOnlyTrainAccess(root, parquet, journal_path=journal)
    second.parquet_path("attempt_two")
    assert [record.purpose for record in second.records] == ["attempt_one", "attempt_two"]
    audit = second.audit_summary()
    assert audit["recorded_reads"] == 2
    assert audit["recorded_paths_pass_policy"]
    assert not audit["exhaustive_process_access_audit"]
    assert audit["journal_sha256"]


def test_torn_access_journal_cannot_be_reported_as_complete(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    parquet = tmp_path / "train.parquet"
    pd.DataFrame({"sequence_id": []}).to_parquet(parquet)
    journal = tmp_path / "reads.jsonl"
    reader = ReadOnlyTrainAccess(root, parquet, journal_path=journal)
    reader.parquet_path("completed_record")
    prior = journal.read_bytes()
    journal.write_bytes(prior + b'{"path":')
    with pytest.raises(ValueError):
        ReadOnlyTrainAccess(root, parquet, journal_path=journal)
    assert journal.read_bytes() == prior + b'{"path":'
