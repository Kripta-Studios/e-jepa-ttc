"""Lossless cache parity and corruption guards independent of campaign data."""

from pathlib import Path

import numpy as np
import pytest

from operational.joint_measurement.mapped_data import MappedRows, write_pack


def records(count: int, start: int = 0) -> list[dict]:
    return [
        {
            "events": np.full((2, 3, 4, 4), i + start, dtype=np.float32),
            "delta": np.array(0.1, dtype=np.float32),
            "valid": np.array([True, False]),
            "token": np.array(f"query-{i + start}"),
            "ordinal": np.array(i + start, dtype=np.int64),
        }
        for i in range(count)
    ]


def test_exact_fields_duplicate_rows_and_eviction(tmp_path: Path) -> None:
    first, second = records(256), records(1, 256)
    write_pack(tmp_path / "pack_00000", first, {"source": "first"})
    write_pack(tmp_path / "pack_00001", second, {"source": "last"})
    reader = MappedRows(tmp_path, 257, max_open_packs=1)
    reader.admit_all()
    retained = reader.row(0)
    for index in [255, 256, 4, 4, 256, 0]:
        expected = (first + second)[index]
        actual = reader.row(index)
        assert set(actual) == set(expected)
        for key in actual:
            assert np.array_equal(actual[key], expected[key])
            assert not np.shares_memory(actual[key], retained[key])
        assert len(reader.maps) <= 1
    reader.close()
    assert np.array_equal(retained["events"], first[0]["events"])
    assert len(reader.verified) == 2
    assert np.array_equal(reader.row(256)["events"], second[0]["events"])
    reader.close()


def test_bytes_corruption_rejected_at_admission(tmp_path: Path) -> None:
    folder = tmp_path / "pack_00000"
    write_pack(folder, records(1), {"source": "x"})
    path = folder / "events.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="field changed"):
        MappedRows(tmp_path, 1).row(0)


def test_mutation_after_admission_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "pack_00000"
    write_pack(folder, records(1), {"source": "x"})
    reader = MappedRows(tmp_path, 1)
    reader.admit_all()
    with (folder / "events.npy").open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(ValueError, match="after admission"):
        reader.row(0)


def test_population_and_index_guards(tmp_path: Path) -> None:
    write_pack(tmp_path / "pack_00000", records(1), {"source": "x"})
    with pytest.raises(ValueError, match="population"):
        MappedRows(tmp_path, 2).row(0)
    for index in [-1, 1]:
        with pytest.raises(IndexError):
            MappedRows(tmp_path, 1).row(index)


def test_resume_rejects_different_source_binding(tmp_path: Path) -> None:
    rows = records(1)
    original = write_pack(tmp_path, rows, {"source": "x"})
    assert write_pack(tmp_path, rows, {"source": "x"}) == original
    with pytest.raises(ValueError, match="binding changed"):
        write_pack(tmp_path, rows, {"source": "y"})
