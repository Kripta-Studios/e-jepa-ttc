"""Exercise resumed sensor-to-cache inference, missing RGB and admission failures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.joint_measurement import fusion_pair_cache as runner
from operational.joint_measurement.fusion_features import FeatureIdentity, load_feature_cache
from operational.joint_measurement.fusion_pair_extract import SCHEMA_SHA256


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.03))
        self.calls = 0

    def forward(self, pixels, delta):
        self.calls += 1
        return {
            "descriptor": torch.cat(
                (pixels.new_zeros(len(pixels), 192), delta[:, None], delta.log()[:, None]), 1
            ),
            "raw_phase": pixels.mean((1, 2, 3, 4)) * self.scale,
        }


class Source:
    role = "H"
    population = 7

    def __init__(self):
        self.rows = pd.DataFrame(
            {
                "sample_token": [f"query-{i}" for i in range(7)],
                "producer_sensor_times_us": [
                    np.array([i * 200000, i * 200000 + 100003], np.int64) for i in range(7)
                ],
            }
        )
        self.calls = []
        self.wrong_token = False
        self.wrong_clock = False

    def batch(self, indices):
        self.calls.extend(indices)
        assert not {2, 3}.intersection(indices), "Missing RGB must not enter the reader"
        pixels = torch.tensor([(-1) ** i * (i + 1) for i in indices], dtype=torch.float32)
        return SimpleNamespace(
            events=pixels[:, None, None, None, None].expand(-1, 2, 3, 4, 4),
            delta_t_s=torch.full((len(indices),), 0.1 if self.wrong_clock else 0.100003),
            sample_tokens=["wrong" if self.wrong_token else f"query-{i}" for i in indices],
        )


def options():
    return {
        "manifest_sha256": "d" * 64,
        "available": np.array([True, True, False, False, True, True, True]),
        "input_binding": {"sensor_manifest_sha256": "e" * 64, "parent_admission": "fixture"},
        "batch_size": 2,
        "chunk_size": 2,
        "max_new_chunks": 1,
        "device": torch.device("cpu"),
    }


def identity():
    return FeatureIdentity("rgb", "a" * 64, SCHEMA_SHA256, "c" * 64)


def test_bounded_resume_matches_direct_inference_and_keeps_all_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "require_resources", lambda: None)
    model, source, kwargs = Parent().eval().requires_grad_(False), Source(), options()
    result = runner.extract_shards(tmp_path, model, source, identity(), **kwargs)
    assert result["status"] == "PAUSED" and source.calls == [0, 1]
    first_bytes = (tmp_path / "chunk_00000.npz").read_bytes()
    result = runner.extract_shards(tmp_path, model, source, identity(), **kwargs)
    assert result["status"] == "PAUSED" and source.calls == [0, 1]
    kwargs["max_new_chunks"] = 4
    result = runner.extract_shards(tmp_path, model, source, identity(), **kwargs)
    assert result["status"] == "COMPLETE" and result["rows"] == 7
    assert result["available_rows"] == 5 and source.calls == [0, 1, 4, 5, 6]
    assert (tmp_path / "chunk_00000.npz").read_bytes() == first_bytes
    block = load_feature_cache(
        tmp_path / "features.npz",
        expected_sha256=result["sha256"],
        expected_identity=identity(),
        role="H",
        manifest_sha256="d" * 64,
        sample_tokens=source.rows.sample_token.to_numpy(dtype=str),
        feature_count=196,
    )
    np.testing.assert_array_equal(block.available, kwargs["available"])
    assert np.isnan(block.features[~block.available]).all()
    np.testing.assert_array_equal(block.sensor_pair_us[~block.available], np.zeros((2, 2)))
    indices = np.flatnonzero(block.available)
    batch = source.batch(indices.tolist())
    expected = phase_to_ttc(model(batch.events, batch.delta_t_s)["raw_phase"]).numpy()
    np.testing.assert_array_equal(block.prediction_ttc[indices], expected)
    np.testing.assert_array_equal(block.features[indices, 192], np.full(5, 0.100003, np.float32))
    assert model.scale.grad is None
    calls = model.calls
    assert runner.extract_shards(tmp_path, model, source, identity(), **kwargs)["new_chunks"] == 0
    assert model.calls == calls


@pytest.mark.parametrize("change", ["availability", "batch_size", "corrupt_shard"])
def test_resume_rejects_changed_extraction_or_bytes(tmp_path, monkeypatch, change):
    monkeypatch.setattr(runner, "require_resources", lambda: None)
    model, source, kwargs = Parent().eval().requires_grad_(False), Source(), options()
    runner.extract_shards(tmp_path, model, source, identity(), **kwargs)
    if change == "availability":
        kwargs["available"][0] = False
    elif change == "batch_size":
        kwargs["batch_size"] = 1
    else:
        (tmp_path / "chunk_00000.npz").write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError)):
        runner.extract_shards(tmp_path, model, source, identity(), **kwargs)
    assert source.calls == [0, 1]


@pytest.mark.parametrize("change", ["wrong_token", "wrong_clock"])
def test_input_error_cannot_be_relabelled_missing_rgb(tmp_path, monkeypatch, change):
    monkeypatch.setattr(runner, "require_resources", lambda: None)
    model, source = Parent().eval().requires_grad_(False), Source()
    setattr(source, change, True)
    with pytest.raises(ValueError):
        runner.extract_shards(tmp_path, model, source, identity(), **options())
    assert not (tmp_path / "chunk_00000.json").exists()


def test_event_cohort_uses_own_clock_and_requires_complete_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "require_resources", lambda: None)

    class EventSource(Source):
        def __init__(self):
            super().__init__()
            self.rows["producer_times_us"] = [
                np.array([i * 200000, i * 200000 + 100000], np.int64) for i in range(7)
            ]

        def batch(self, indices):
            return SimpleNamespace(
                events=torch.ones(len(indices), 2, 12, 4, 4),
                delta_t_s=torch.full((len(indices),), 0.1),
                sample_tokens=[f"query-{i}" for i in indices],
            )

    model, source, kwargs = Parent().eval().requires_grad_(False), EventSource(), options()
    producer = FeatureIdentity("event", "a" * 64, SCHEMA_SHA256, "c" * 64)
    with pytest.raises(ValueError, match="availability"):
        runner.extract_shards(tmp_path, model, source, producer, **kwargs)
    assert not (tmp_path / "BINDING.json").exists()
    kwargs.update(available=np.ones(7, np.bool_), max_new_chunks=4)
    result = runner.extract_shards(tmp_path, model, source, producer, **kwargs)
    assert result["status"] == "COMPLETE" and result["available_rows"] == 7
    with np.load(tmp_path / "features.npz") as cache:
        np.testing.assert_array_equal(cache["sensor_pair_us"][:, 1] % 200000, np.full(7, 100000))


def test_mutating_eval_producer_cannot_publish_features(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "require_resources", lambda: None)

    class MutableParent(Parent):
        def forward(self, pixels, delta):
            result = super().forward(pixels, delta)
            self.scale.add_(0.1)
            return result

    model = MutableParent().eval().requires_grad_(False)
    with pytest.raises(ValueError, match="producer state changed"):
        runner.extract_shards(tmp_path, model, Source(), identity(), **options())
    assert not (tmp_path / "chunk_00000.json").exists()
