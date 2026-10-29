from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.detection.event_data import (
    CausalEventConfig,
    CausalEventVoxelizer,
    encode_causal_event_chunks,
    resize_event_coordinates_pixel_center,
)


def chunk(
    timestamps: list[int],
    *,
    x: list[int] | None = None,
    y: list[int] | None = None,
    polarity: list[int] | None = None,
) -> dict[str, np.ndarray]:
    count = len(timestamps)
    return {
        "x": np.asarray(x if x is not None else [0] * count, dtype=np.uint16),
        "y": np.asarray(y if y is not None else [0] * count, dtype=np.uint16),
        "t": np.asarray(timestamps, dtype=np.int64),
        "p": np.asarray(polarity if polarity is not None else [1] * count, dtype=np.int8),
    }


def raw_config(**changes: object) -> CausalEventConfig:
    values = {
        "width": 4,
        "height": 2,
        "window_ms": 50,
        "bins_per_polarity": 5,
        "chunk_events": 10,
        "max_open_handles": 2,
        "normalize": False,
        **changes,
    }
    return CausalEventConfig(**values)  # type: ignore[arg-type]


def test_pixel_center_mapping_matches_rgb_resize_boundaries():
    x, y = resize_event_coordinates_pixel_center(
        np.asarray([0, 1, 2, 1278, 1279], dtype=np.uint16),
        np.asarray([0, 1, 2, 718, 719], dtype=np.uint16),
        output_size=(640, 384),
    )
    assert x.tolist() == [0, 0, 1, 639, 639]
    assert y.tolist() == [0, 0, 1, 383, 383]


def test_temporal_interpolation_polarity_and_half_open_boundaries():
    result = encode_causal_event_chunks(
        [
            chunk(
                [100_000, 106_250, 125_000, 149_999],
                x=[0, 0, 1279, 1279],
                y=[0, 0, 719, 719],
                polarity=[1, 1, 0, 1],
            )
        ],
        sequence_id="s",
        cutoff_us=150_000,
        config=raw_config(),
    )
    voxel = result.voxel
    assert voxel.shape == (10, 2, 4)
    assert voxel[0, 0, 0] == pytest.approx(1.5)
    assert voxel[1, 0, 0] == pytest.approx(0.5)
    assert voxel[7, 1, 3] == pytest.approx(1.0)
    assert voxel[:5].sum() == pytest.approx(3.0)
    assert voxel[5:].sum() == pytest.approx(1.0)
    assert result.receipt["window_start_us_inclusive"] == 100_000
    assert result.receipt["window_end_us_exclusive"] == 150_000
    assert result.receipt["events_discarded_by_cap"] == 0

    with pytest.raises(ValueError, match="outside the causal"):
        encode_causal_event_chunks(
            [chunk([150_000])], sequence_id="s", cutoff_us=150_000, config=raw_config()
        )


def test_chunk_partition_is_deterministic_and_empty_is_zero():
    all_events = chunk([100_000, 106_250, 125_000, 149_999], polarity=[1, 0, 1, 0])
    partitioned = [
        {key: value[:2] for key, value in all_events.items()},
        {key: value[2:] for key, value in all_events.items()},
    ]
    combined = encode_causal_event_chunks(
        [all_events], sequence_id="s", cutoff_us=150_000, config=raw_config()
    )
    split = encode_causal_event_chunks(
        partitioned, sequence_id="s", cutoff_us=150_000, config=raw_config()
    )
    assert np.array_equal(combined.voxel, split.voxel)
    assert combined.receipt["voxel_sha256_float32_le"] == split.receipt["voxel_sha256_float32_le"]
    empty = encode_causal_event_chunks(
        [], sequence_id="s", cutoff_us=150_000, config=raw_config(normalize=True)
    )
    assert not np.any(empty.voxel)
    assert empty.receipt["event_count"] == 0
    assert empty.receipt["nonempty_chunk_count"] == 0


@pytest.mark.parametrize("normalize", [False, True])
def test_large_cross_bin_stream_is_exactly_partition_independent(normalize: bool):
    timestamps = np.repeat(
        np.asarray([100_001, 112_347, 137_891, 149_999], dtype=np.int64),
        [25_001, 25_001, 25_001, 25_000],
    )
    events = {
        "x": np.zeros(len(timestamps), dtype=np.uint16),
        "y": np.zeros(len(timestamps), dtype=np.uint16),
        "t": timestamps,
        "p": np.resize(np.asarray([0, 1], dtype=np.int8), len(timestamps)),
    }
    partitions = [
        {key: value[start : start + 10_000] for key, value in events.items()}
        for start in range(0, len(timestamps), 10_000)
    ]
    config = raw_config(chunk_events=len(timestamps), normalize=normalize)
    combined = encode_causal_event_chunks(
        [events], sequence_id="s", cutoff_us=150_000, config=config
    )
    split = encode_causal_event_chunks(
        partitions, sequence_id="s", cutoff_us=150_000, config=config
    )
    assert np.array_equal(combined.voxel, split.voxel)
    assert combined.receipt["voxel_sha256_float32_le"] == split.receipt[
        "voxel_sha256_float32_le"
    ]


@pytest.mark.parametrize(
    ("bad_chunk", "message"),
    [
        (chunk([100_001, 100_000]), "rolled back within"),
        (chunk([100_000], polarity=[-1]), "raw values 0 and 1"),
        (chunk([99_999]), "outside the causal"),
        (chunk([100_000], x=[1280]), "outside the pinned"),
    ],
)
def test_invalid_event_contracts_fail(bad_chunk: dict[str, np.ndarray], message: str):
    with pytest.raises(ValueError, match=message):
        encode_causal_event_chunks(
            [bad_chunk], sequence_id="s", cutoff_us=150_000, config=raw_config()
        )


def test_rollback_across_chunks_and_chunk_bound_fail():
    with pytest.raises(ValueError, match="across chunks"):
        encode_causal_event_chunks(
            [chunk([100_100]), chunk([100_099])],
            sequence_id="s",
            cutoff_us=150_000,
            config=raw_config(),
        )
    with pytest.raises(ValueError, match="chunk event bound"):
        encode_causal_event_chunks(
            [chunk([100_000, 100_001])],
            sequence_id="s",
            cutoff_us=150_000,
            config=raw_config(chunk_events=1),
        )


class FakeReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.opened = False
        self.closed = False
        self.calls: list[tuple[int, int, int]] = []

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def iter_window_chunks(self, start_us: int, end_us: int, *, chunk_events: int):
        self.calls.append((start_us, end_us, chunk_events))
        return iter(())


def test_reader_lru_never_exceeds_two_and_uses_exposure_end(tmp_path: Path):
    created: list[FakeReader] = []
    live = 0
    peak_live = 0

    class CountingReader(FakeReader):
        def open(self) -> None:
            nonlocal live, peak_live
            super().open()
            live += 1
            peak_live = max(peak_live, live)

        def close(self) -> None:
            nonlocal live
            if not self.closed:
                live -= 1
            super().close()

    def factory(path: Path) -> CountingReader:
        reader = CountingReader(path)
        created.append(reader)
        return reader

    paths = []
    for name in ("a.h5", "b.h5", "c.h5"):
        path = tmp_path / name
        path.touch()
        paths.append(path)
    voxelizer = CausalEventVoxelizer(tmp_path, raw_config(), reader_factory=factory)
    for path in paths:
        result = voxelizer.encode(path.name, sequence_id=path.stem, rgb_exposure_end_us=150_000)
        assert result.receipt["cutoff_source"] == "rgb_exposure_end_timestamp_us"
        assert result.receipt["uses_future_events"] is False
    assert peak_live == 2
    assert live == 2
    assert len(voxelizer.open_reader_paths) == 2
    assert created[0].closed is True
    assert created[-1].calls == [(100_000, 150_000, 10)]
    voxelizer.close()
    assert not voxelizer.open_reader_paths
    assert all(reader.closed for reader in created)
    assert live == 0


def test_config_rejects_unapproved_temporal_and_resource_choices():
    with pytest.raises(ValueError, match="50 or 100"):
        raw_config(window_ms=75)
    with pytest.raises(ValueError, match="five bins"):
        raw_config(bins_per_polarity=4)
    with pytest.raises(ValueError, match="at most two"):
        raw_config(max_open_handles=3)
