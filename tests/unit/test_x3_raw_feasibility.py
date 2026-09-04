from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.data.x3_raw_feasibility import (
    _normalise_relative_path,
    build_x3_raw_binding,
)


def _write_event_file(path: Path) -> None:
    path.parent.mkdir(parents=True)
    timestamps = np.arange(0, 100_000, 100, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.arange(len(timestamps), dtype=np.uint16) % 1280)
        events.create_dataset("y", data=np.arange(len(timestamps), dtype=np.uint16) % 720)
        events.create_dataset("t", data=timestamps)
        events.create_dataset("p", data=np.arange(len(timestamps), dtype=np.int8) % 2)
        handle.create_dataset(
            "ms_to_idx",
            data=np.searchsorted(timestamps, np.arange(101, dtype=np.int64) * 1000),
        )


def test_build_x3_raw_binding_is_complete_and_signed(tmp_path: Path) -> None:
    eap_root = tmp_path / "eap"
    sequences = ["seq-a", "seq-b"]
    for sequence in sequences:
        _write_event_file(eap_root / "data" / "train" / sequence / "events.h5")

    rows = []
    stage_rows = []
    for index in range(4):
        sequence = sequences[index % 2]
        token = f"token-{index}"
        start = index * 20_000
        rows.append(
            {
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": f"track-{index}",
                "events_path": f"data/train/{sequence}/events.h5",
                "event_windows_us": [
                    [start, start + 10_000],
                    [start + 10_000, start + 20_000],
                ],
            }
        )
        stage_rows.append(
            {
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": f"track-{index}",
            }
        )
    stage_path = tmp_path / "stage.csv"
    source_path = tmp_path / "garl" / "data" / "train.parquet"
    source_path.parent.mkdir(parents=True)
    pd.DataFrame(stage_rows).to_csv(stage_path, index=False)
    pd.DataFrame(rows).to_parquet(source_path, index=False)

    binding_path = tmp_path / "X3_RAW_BINDING.csv"
    manifest_path = tmp_path / "X3_RAW_BINDING_MANIFEST.json"
    manifest, probe, proposal = build_x3_raw_binding(
        stage_metadata_path=stage_path,
        garl_train_parquet=source_path,
        eap_root=eap_root,
        binding_csv_path=binding_path,
        binding_manifest_path=manifest_path,
        code_commit="a" * 40,
        protocol_sha256="b" * 64,
        expected_tokens=4,
        read_probe_tokens=4,
        hash_event_files=True,
    )

    binding = pd.read_csv(binding_path)
    assert len(binding) == 4
    assert binding["raw_event_count"].tolist() == [200, 200, 200, 200]
    assert binding["events_file_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest["tokens"] == 4
    assert manifest["windows"] == 8
    assert manifest["forbidden_paths_opened"] is False
    assert verify_artifact_hash(manifest)
    assert probe["tokens"] == 4
    assert probe["events"] == 800
    assert proposal["microbin_us"] == 1000
    assert proposal["snapshot_interval_us"] == 5000


@pytest.mark.parametrize(
    "value",
    ["data/test/seq/events.h5", "../train/events.h5", "data/private/events.h5"],
)
def test_raw_binding_rejects_sealed_or_escaping_paths(value: str) -> None:
    with pytest.raises(ValueError):
        _normalise_relative_path(value)
