"""Small end-to-end checks for the isolated V8 temporal cache."""

from __future__ import annotations

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
