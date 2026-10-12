"""The optional storage adapter preserves the complete native batch contract."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement import data, mapped_runtime
from operational.joint_measurement.mapped_data import MappedRows, write_pack
from operational.joint_measurement.qa import exact
from operational.range_transfer.common import digest, sha, write


def fixture_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    source, destination = tmp_path / "source", tmp_path / "mapped"
    rows = []
    for i in range(2):
        rows.append(
            {
                "events": np.full((2, 12, 4, 4), i, dtype=np.float32),
                "delta": np.array(0.05, dtype=np.float32),
                "motion": np.array([i, 2], dtype=np.float32),
                "visible_heights": np.array([3, 4], dtype=np.float32),
                "target_ttc": np.array(2.0 + i, dtype=np.float32),
                "boxes": np.ones((2, 4), dtype=np.float32),
                "square": np.ones(4, dtype=np.float32),
                "relation_targets": np.ones((2, 3), dtype=np.float32),
                "relation_valid": np.array([True, False]),
                "token": np.array(f"query-{i}"),
            }
        )
    write(source / "COMPLETE.json", {"rows": 2, "pack_receipts_hash": "source receipts"})
    write(source / "pack_00000.json", {"source": "test"})
    binding = {
        "role": "P",
        "source_complete_sha256": sha(source / "COMPLETE.json"),
        "source_pack_receipts_hash": "source receipts",
    }
    write(destination / "BINDING.json", binding)
    pack_binding = {
        "campaign": digest(binding),
        "source_receipt_sha256": sha(source / "pack_00000.json"),
        "start": 0,
        "stop": 2,
    }
    receipt = write_pack(destination / "pack_00000", rows, pack_binding)
    write(
        destination / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "rows": 2,
            "packs": 1,
            "binding": digest(binding),
            "all_fields_exact": True,
            "pack_receipts_hash": digest([receipt]),
        },
    )
    monkeypatch.setattr(mapped_runtime, "mapped_root", lambda: destination)
    monkeypatch.setattr(mapped_runtime, "cache_root", lambda role: source)
    monkeypatch.setattr(data, "cache_root", lambda role: source)
    frame = pd.DataFrame(
        {
            "sample_token": ["query-0", "query-1"],
            "sequence_id": ["seq", "seq"],
            "track_id": ["track", "track"],
            "role": ["P", "P"],
        }
    )
    monkeypatch.setattr(data, "role_rows", lambda role: frame)
    return rows


def test_native_collation_metadata_and_duplicates_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fixture_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(data.SensorDataset, "_row", lambda self, i: rows[i])
    reader = mapped_runtime.admit_mapped()
    mapped = mapped_runtime.MappedSensorDataset("P", admission=reader)
    original = data.SensorDataset("P")
    left, right = original.batch([1, 0, 1]), mapped.batch([1, 0, 1])
    assert all(exact(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
    mapped.close()
    assert not reader.maps
    assert len(reader.verified) == 1
    original.close()


def test_source_ancestry_and_population_admission_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_cache(tmp_path, monkeypatch)
    unadmitted = MappedRows(tmp_path / "mapped", 2)
    with pytest.raises(ValueError, match="Admit all"):
        mapped_runtime.MappedSensorDataset("P", admission=unadmitted)
    with pytest.raises(ValueError, match="Only the P"):
        mapped_runtime.MappedSensorDataset("H", admission=unadmitted)
    write(tmp_path / "source/pack_00000.json", {"source": "changed"})
    with pytest.raises(ValueError, match="ancestry"):
        mapped_runtime.admit_mapped()


def test_token_alignment_is_not_shape_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_cache(tmp_path, monkeypatch)
    reader = mapped_runtime.admit_mapped()
    mapped = mapped_runtime.MappedSensorDataset("P", admission=reader)
    mapped.rows.loc[0, "sample_token"] = "different query"
    with pytest.raises(ValueError, match="token mismatch"):
        mapped.batch([0])
    mapped.close()
