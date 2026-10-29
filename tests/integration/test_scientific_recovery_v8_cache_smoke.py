"""Small end-to-end checks for the isolated V8 temporal cache."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheConfig,
    ScientificRecoveryV8CacheDataset,
    collate_scientific_recovery_v8,
    scientific_recovery_v8_model_inputs,
    write_scientific_recovery_v8_cache_for_testing,
)


def _record(index: int, *, steps: int, channels: int) -> dict[str, object]:
    identity = ("sequence-a", f"token-{index}", "track-a", "public-a", str(100 + index))
    return {
        "representation": np.full((steps, channels, 8, 8), index + 0.25, dtype=np.float32),
        "endpoint_us": np.arange(steps, dtype=np.int64) * 100 + 1000 + index * 1_000,
        "sample_token": identity[1],
        "sequence_id": identity[0],
        "track_id": identity[2],
        "row_identity": list(identity),
        "target_ttc": np.float32(1.5 + index),
        "sample_weight": np.float32(0.5),
        "outer_fold": np.int64(index % 3),
        "common_roi_xyxy": np.asarray((1.0, 2.0, 10.0, 11.0), dtype=np.float32),
        "endpoint_boxes_xyxy": np.asarray(((1.0, 2.0, 10.0, 11.0),) * steps, dtype=np.float32),
        "visible_heights_px": np.asarray((9.0, 9.0), dtype=np.float32),
        "representation_source": "fixture",
        "endpoint_diagnostics": [{"event_count": 0.0}] * steps,
    }


@pytest.mark.parametrize(("representation", "channels"), (("timevol20", 20), ("exp6", 6)))
def test_v8_cache_float16_roundtrip_resume_and_collate(
    tmp_path, representation: str, channels: int
) -> None:
    config = ScientificRecoveryV8CacheConfig(
        representation=representation, roi_size=8, shard_size=1, expected_rows=None
    )
    records = [_record(index, steps=3, channels=channels) for index in range(2)]
    manifest = write_scientific_recovery_v8_cache_for_testing(
        records=records, output_dir=tmp_path / representation, config=config
    )
    assert manifest["split_counts"] == {"train": 2}
    assert manifest["model_input_fields"] == ["representation", "endpoint_us"]
    assert manifest["raw_materialization"] is False

    repeat = write_scientific_recovery_v8_cache_for_testing(
        records=records, output_dir=tmp_path / representation, config=config, resume=True
    )
    assert repeat["row_identity_sha256"] == manifest["row_identity_sha256"]

    dataset = ScientificRecoveryV8CacheDataset(tmp_path / representation / "manifest.json")
    assert len(dataset) == 2
    first = dataset[0]
    assert torch.as_tensor(first["representation"]).dtype == torch.float16
    assert torch.allclose(
        torch.as_tensor(first["representation"], dtype=torch.float32),
        torch.full((3, channels, 8, 8), 0.25),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_scientific_recovery_v8)))
    assert batch.representations.shape == (2, 3, channels, 8, 8)
    inputs = scientific_recovery_v8_model_inputs(batch)
    assert set(inputs) == {"representations", "endpoint_us"}
    assert all("target" not in key and "weight" not in key for key in inputs)
    assert not (tmp_path / representation / "spill").exists()


def test_v8_cache_spill_merge_sorts_interleaved_identities_and_flushes_runs(
    tmp_path, monkeypatch
) -> None:
    """Stage 30 cannot keep 8,192 tensors in RAM; spill runs must merge by identity."""

    from e_jepa_ttc.data import scientific_recovery_v8_cache as cache_mod

    monkeypatch.setattr(cache_mod, "_SPILL_FLUSH_BYTES", 1)
    config = ScientificRecoveryV8CacheConfig(
        representation="exp6", roi_size=8, shard_size=2, expected_rows=None
    )
    records = [_record(index, steps=3, channels=6) for index in (2, 0, 1, 4, 3)]
    output = tmp_path / "exp6"
    manifest = write_scientific_recovery_v8_cache_for_testing(
        records=records, output_dir=output, config=config
    )
    assert manifest["split_counts"] == {"train": 5}
    assert not (output / "spill").exists()
    dataset = ScientificRecoveryV8CacheDataset(output / "manifest.json")
    assert [str(dataset[index]["sample_token"]) for index in range(5)] == [
        "token-0",
        "token-1",
        "token-2",
        "token-3",
        "token-4",
    ]
    shard_paths = sorted((output / "train").glob("shard-*.pt"))
    assert [path.name for path in shard_paths] == [
        "shard-00000.pt",
        "shard-00001.pt",
        "shard-00002.pt",
    ]
    assert len(cache_mod._load_records(shard_paths[0])) == 2
    assert len(cache_mod._load_records(shard_paths[2])) == 1


def test_spill_resume_matches_bin_beside_meta_json_not_meta_bin(tmp_path) -> None:
    """Resume looked for seq-*.meta.bin and recoded every spilled sequence."""

    from e_jepa_ttc.data.scientific_recovery_v8_cache import (
        _flush_spill_run,
        _spill_sidecar_matches,
    )

    records = [_record(index, steps=2, channels=6) for index in range(3)]
    metadata = _flush_spill_run(records, tmp_path, "seq-deadbeef")
    sidecar = tmp_path / "seq-deadbeef.meta.json"
    bin_path = tmp_path / "seq-deadbeef.bin"
    assert sidecar.is_file() and bin_path.is_file()
    assert metadata["path"] == "spill/seq-deadbeef.bin"
    assert not (tmp_path / "seq-deadbeef.meta.bin").exists()
    assert sidecar.with_suffix(".bin") == tmp_path / "seq-deadbeef.meta.bin"
    identities = [list(record["row_identity"]) for record in records]
    assert _spill_sidecar_matches(sidecar, identities) is True
    assert _spill_sidecar_matches(sidecar, identities[:-1]) is False


def test_spill_assemble_keeps_protocol_row_identity_hash(tmp_path, monkeypatch) -> None:
    """Join-key spill hashes must not clobber the frozen contract row_identity_sha256."""

    from e_jepa_ttc.data.scientific_recovery_v8_cache import _write_records

    monkeypatch.setattr("e_jepa_ttc.data.scientific_recovery_v8_cache._SPILL_FLUSH_BYTES", 1)
    config = ScientificRecoveryV8CacheConfig(
        representation="exp6", steps=2, roi_size=8, shard_size=2, expected_rows=None
    )
    records = [_record(index, steps=2, channels=6) for index in range(3)]
    contract = "fe4ea01a" + "ab" * 28
    manifest = _write_records(
        records=records,
        output_dir=tmp_path / "cache",
        config=config,
        provenance={"raw_materialization": False, "row_identity_sha256": contract},
        resume=False,
    )
    assert manifest["row_identity_sha256"] == contract
    assert manifest["split_counts"] == {"train": 3}


def test_v8_cache_rejects_non_train_split_and_incomplete_resume(tmp_path) -> None:
    config = ScientificRecoveryV8CacheConfig(
        representation="exp6", steps=2, roi_size=8, shard_size=2, expected_rows=None
    )
    output = tmp_path / "cache"
    write_scientific_recovery_v8_cache_for_testing(
        records=[_record(0, steps=2, channels=6)], output_dir=output, config=config
    )
    with pytest.raises(ValueError, match="train only"):
        ScientificRecoveryV8CacheDataset(output / "manifest.json", split="validation")
    shard = output / "train" / "shard-00000.pt"
    shard.unlink()
    with pytest.raises(RuntimeError, match="integrity"):
        write_scientific_recovery_v8_cache_for_testing(
            records=[_record(0, steps=2, channels=6)], output_dir=output, config=config, resume=True
        )


def test_exp6_window_ingest_chunks_match_whole_read(tmp_path, monkeypatch) -> None:
    """A 373M-event gap must not be loaded as one EventBatch."""

    import e_jepa_ttc.data.scientific_recovery_v8_cache as cache_mod
    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.data.scientific_recovery_v8 import CausalExponentialStateRepresentation
    from e_jepa_ttc.data.scientific_recovery_v8_cache import (
        _ingest_exp6_window,
        _raw_event_batch,
    )

    timestamps = np.arange(0, 8_000, 200, dtype=np.int64)
    n = int(timestamps.size)
    source = tmp_path / "events.h5"
    with h5py.File(source, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.ones(n, dtype=np.int32))
        events.create_dataset("y", data=np.ones(n, dtype=np.int32))
        events.create_dataset("t", data=timestamps)
        events.create_dataset("p", data=np.ones(n, dtype=np.int8))
        handle.create_dataset(
            "ms_to_idx",
            data=np.searchsorted(timestamps, np.arange(0, 12) * 1_000).astype(np.uint64),
        )

    monkeypatch.setattr(cache_mod, "_EXP6_EVENT_CHUNK", 3)
    reader = EAPEventReader(source)
    reader.open()
    try:
        raw = reader.read_window(0, 7_001)
        whole = CausalExponentialStateRepresentation(target_size=(8, 8))
        whole.update(
            _raw_event_batch(raw, sequence_id="seq", start_us=0, end_us=7_000),
            7_000,
        )
        chunked = CausalExponentialStateRepresentation(target_size=(8, 8))
        count = _ingest_exp6_window(
            chunked, reader, start_us=0, endpoint_us=7_000, sequence_id="seq"
        )
    finally:
        reader.close()

    np.testing.assert_allclose(chunked._state, whole._state, atol=1e-6, rtol=1e-6)
    assert count == int(np.count_nonzero(raw["t"] < 7_000))
