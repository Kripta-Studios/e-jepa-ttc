"""Resident fusion inputs retain query order, producer identity and exact fallback."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from operational.joint_measurement.fusion_data import FusionTensorBank
from operational.joint_measurement.fusion_features import FeatureBlock, FeatureIdentity
from operational.joint_measurement.fusion_model import RGBAnchorComplement


def source(modality):
    return FeatureBlock(
        identity=FeatureIdentity(modality, "a" * 64, "b" * 64, "c" * 64),
        role="H",
        manifest_sha256="d" * 64,
        sample_tokens=np.array(["q0", "q1", "q2"]),
        features=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        prediction_ttc=np.array([1, -2, 3], dtype=np.float32),
        available=np.ones(3, dtype=np.bool_),
        sensor_pair_us=np.array([[0, 10000], [20000, 30000], [40000, 50000]], dtype=np.int64),
    )


def build(rgb=None, event=None, **overrides):
    rgb, event = rgb or source("rgb"), event or source("event")
    args = {
        "rgb_normalizer": rgb.fit_h_normalizer(),
        "event_normalizer": event.fit_h_normalizer(),
        "role": "H",
        "manifest_sha256": "d" * 64,
        "sample_tokens": np.array(["q0", "q1", "q2"]),
    }
    args.update(overrides)
    return FusionTensorBank(rgb, event, **args)


def test_resident_copy_matches_direct_normalization_and_preserves_sampling():
    rgb, event = source("rgb"), source("event")
    event.sensor_pair_us[:, 1] += 50000
    bank = build(rgb, event)
    rng = torch.get_rng_state().clone()
    batch = bank.batch(torch.tensor([2, 0, 2]))
    expected = torch.from_numpy(rgb.normalized(rgb.fit_h_normalizer())[[2, 0, 2]])
    torch.testing.assert_close(batch["rgb_features"], expected, rtol=0, atol=0)
    torch.testing.assert_close(batch["event_ttc"], torch.tensor([3.0, 1.0, 3.0]))
    assert torch.equal(rng, torch.get_rng_state())
    assert not np.array_equal(bank.sensor_pair_us["event"], bank.sensor_pair_us["rgb"])
    assert bank.storage_bytes == 3 * (2 * 2 * 4 + 2 * 4 + 1)
    assert set(batch) == {"rgb_features", "event_features", "rgb_ttc", "event_ttc", "rgb_available"}
    rgb.features[:] = 1000
    rgb.prediction_ttc[:] = 50
    event.sensor_pair_us[:] = -1
    again = bank.batch(torch.tensor([2, 0, 2]))
    for key in batch:
        torch.testing.assert_close(batch[key], again[key], rtol=0, atol=0)
    assert np.all(bank.sensor_pair_us["event"] >= 0)


def test_missing_rgb_survives_batching_and_is_exact_event_fallback():
    rgb = source("rgb")
    rgb.available[1] = False
    rgb.features[1] = np.nan
    rgb.prediction_ttc[1] = np.nan
    rgb.sensor_pair_us[1] = 0
    bank = build(rgb)
    batch = bank.batch(torch.tensor([1, 0, 2, 1]))
    assert torch.isfinite(batch["rgb_features"]).all()
    for mode in ("TRUE", "ZERO"):
        model = RGBAnchorComplement(2, 2, hidden=8, mode=mode)
        output = model(**batch)
        assert torch.equal(output["ttc"][[0, 3]], batch["event_ttc"][[0, 3]])
        assert output["used_rgb"].tolist() == [False, True, True, False]


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([]),
        torch.tensor([0.0]),
        torch.tensor([-1]),
        torch.tensor([3]),
        torch.tensor([[0]]),
    ],
)
def test_invalid_indices_cannot_silently_change_population(indices):
    with pytest.raises(ValueError, match="indices"):
        build().batch(indices)


@pytest.mark.parametrize("error", ["order", "role", "manifest", "producer", "shape", "scale"])
def test_misaligned_or_stale_inputs_fail_at_admission(error):
    rgb, event = source("rgb"), source("event")
    options = {}
    if error == "order":
        event = replace(event, sample_tokens=event.sample_tokens[::-1])
    elif error == "role":
        options["role"] = "V"
    elif error == "manifest":
        options["manifest_sha256"] = "e" * 64
    else:
        normalizer = rgb.fit_h_normalizer()
        if error == "producer":
            normalizer = replace(normalizer, producer_sha256="f" * 64)
        elif error == "shape":
            normalizer = replace(normalizer, mean=np.zeros(1, np.float32))
        else:
            normalizer = replace(normalizer, scale=np.zeros(2, np.float32))
        options["rgb_normalizer"] = normalizer
    with pytest.raises(ValueError):
        build(rgb, event, **options)


def test_v_uses_h_statistics_without_fitting_or_exposing_targets():
    rgb, event = source("rgb"), source("event")
    norms = [block.fit_h_normalizer() for block in (rgb, event)]
    rgb, event = replace(rgb, role="V"), replace(event, role="V")
    bank = FusionTensorBank(
        rgb, event, *norms, role="V", manifest_sha256="d" * 64, sample_tokens=rgb.sample_tokens
    )
    assert bank.role == "V"
    assert not hasattr(bank, "target_ttc")
    assert set(bank.batch(torch.tensor([0]))) == set(bank.inputs)
