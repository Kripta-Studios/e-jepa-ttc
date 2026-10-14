"""Feature extraction may resume only intact shards from the same producer and cohort."""

from dataclasses import replace

import numpy as np
import pytest

from operational.joint_measurement.fusion_cache import FusionCacheWriter
from operational.joint_measurement.fusion_features import FeatureBlock, FeatureIdentity


def source():
    return FeatureBlock(
        FeatureIdentity("rgb", "a" * 64, "b" * 64, "c" * 64),
        "H",
        "d" * 64,
        np.array(["q0", "q1", "q2"]),
        np.array([[1, 2], [np.nan, np.nan], [3, 4]], np.float32),
        np.array([1, np.nan, -3], np.float32),
        np.array([True, False, True]),
        np.array([[0, 100], [0, 0], [200, 300]], np.int64),
    )


def writer(path, value=None, **overrides):
    value = value or source()
    options = {
        "role": value.role,
        "manifest_sha256": value.manifest_sha256,
        "sample_tokens": value.sample_tokens,
        "feature_count": 2,
        "input_binding": {"sensor_cache_sha256": "e" * 64},
        "chunk_size": 2,
    }
    options.update(overrides)
    return FusionCacheWriter(path, value.identity, **options)


def part(value, start, stop):
    return replace(
        value,
        **{
            name: getattr(value, name)[start:stop]
            for name in (
                "sample_tokens",
                "features",
                "prediction_ttc",
                "available",
                "sensor_pair_us",
            )
        },
    )


@pytest.mark.parametrize("dtype", ["U2", "S2"])
def test_interrupted_extraction_resumes_losslessly_without_dropping_missing_rgb(tmp_path, dtype):
    value = replace(source(), sample_tokens=source().sample_tokens.astype(dtype))
    first = writer(tmp_path, value)
    first.write_chunk(0, part(value, 0, 2))
    with pytest.raises(ValueError, match="every feature shard"):
        first.complete()
    resumed = writer(tmp_path, value)
    assert resumed.read_chunk(1) is None
    np.testing.assert_array_equal(resumed.read_chunk(0).features, value.features[:2])
    resumed.write_chunk(1, part(value, 2, 3))
    receipt = resumed.complete()
    assert receipt["rows"] == 3 and receipt["available_rows"] == 2
    assert resumed.complete() == receipt
    final = resumed.load(tmp_path / "features.npz", receipt["sha256"], value.sample_tokens)
    for name in ("sample_tokens", "features", "prediction_ttc", "available", "sensor_pair_us"):
        np.testing.assert_array_equal(getattr(final, name), getattr(value, name))
    with np.load(tmp_path / "features.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "metadata",
            "sample_tokens",
            "features",
            "prediction_ttc",
            "available",
            "sensor_pair_us",
        }


@pytest.mark.parametrize("change", ["producer", "order", "clock_source", "feature_count"])
def test_resume_cannot_relabel_existing_features(tmp_path, change):
    value = source()
    writer(tmp_path, value).write_chunk(0, part(value, 0, 2))
    options = {}
    if change == "producer":
        value = replace(value, identity=replace(value.identity, endpoint_sha256="f" * 64))
    elif change == "order":
        value = replace(value, sample_tokens=value.sample_tokens[::-1])
    elif change == "clock_source":
        options["input_binding"] = {"sensor_cache_sha256": "f" * 64}
    else:
        options["feature_count"] = 3
    with pytest.raises((ValueError, RuntimeError)):
        writer(tmp_path, value, **options)


def test_corrupt_admitted_shard_cannot_be_overwritten_or_completed(tmp_path):
    value = source()
    sink = writer(tmp_path, value)
    sink.write_chunk(0, part(value, 0, 2))
    (tmp_path / "chunk_00000.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="bytes differ"):
        sink.read_chunk(0)
    with pytest.raises(ValueError, match="bytes differ"):
        sink.write_chunk(0, part(value, 0, 2))


def test_invalid_order_is_rejected_before_chunk_publication(tmp_path):
    value = source()
    sink = writer(tmp_path, value)
    chunk = part(value, 0, 2)
    with pytest.raises(ValueError, match="reordered"):
        sink.write_chunk(0, replace(chunk, sample_tokens=chunk.sample_tokens[::-1]))
    assert not (tmp_path / "chunk_00000.npz").exists()


def test_valid_committed_shard_is_immutable(tmp_path):
    value = source()
    sink = writer(tmp_path, value)
    sink.write_chunk(0, part(value, 0, 2))
    with pytest.raises(FileExistsError):
        sink.write_chunk(0, part(value, 0, 2))


def test_orphan_without_receipt_can_be_rebuilt_but_not_reused(tmp_path):
    value = source()
    sink = writer(tmp_path, value)
    (tmp_path / "chunk_00000.npz").write_bytes(b"interrupted publication")
    assert sink.read_chunk(0) is None
    sink.write_chunk(0, part(value, 0, 2))
    assert sink.read_chunk(0) is not None
