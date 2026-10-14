"""Reject stale parents, cross-role statistics and misaligned multimodal queries."""

from dataclasses import replace

import numpy as np
import pytest

from operational.joint_measurement.fusion_features import FeatureBlock, FeatureIdentity


def block(modality="rgb", role="H"):
    return FeatureBlock(
        identity=FeatureIdentity(modality, "a" * 64, "b" * 64, "c" * 64),
        role=role,
        manifest_sha256="d" * 64,
        sample_tokens=np.array(["q0", "q1", "q2"]),
        features=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        prediction_ttc=np.array([1, -2, 3], dtype=np.float32),
        available=np.ones(3, dtype=np.bool_),
        sensor_pair_us=np.array([[0, 10000], [20000, 30000], [40000, 50000]], dtype=np.int64),
    )


def admit(value, expected=None):
    reference = expected or block(value.identity.modality)
    value.validate(
        expected_identity=reference.identity,
        role=reference.role,
        manifest_sha256=reference.manifest_sha256,
        sample_tokens=reference.sample_tokens,
        feature_count=2,
    )


@pytest.mark.parametrize("field", ["endpoint_sha256", "schema_sha256", "extractor_sha256"])
def test_same_shape_from_different_producer_or_extraction_is_rejected(field):
    value = block()
    changed = replace(value, identity=replace(value.identity, **{field: "e" * 64}))
    with pytest.raises(ValueError, match="identity changed"):
        admit(changed)


@pytest.mark.parametrize("error", ["reorder", "duplicate", "manifest", "role"])
def test_query_and_population_identity_cannot_be_repaired_silently(error):
    value = block()
    if error == "reorder":
        value = replace(value, sample_tokens=value.sample_tokens[::-1])
    elif error == "duplicate":
        value.sample_tokens[1] = value.sample_tokens[0]
    elif error == "manifest":
        value = replace(value, manifest_sha256="f" * 64)
    else:
        value = replace(value, role="V")
    with pytest.raises(ValueError):
        admit(value)


def test_each_modality_keeps_its_own_clock_and_reversed_pair_is_rejected():
    rgb, event = block(), block("event")
    event.sensor_pair_us[:, 1] += 50000
    admit(rgb)
    admit(event)
    assert not np.array_equal(rgb.sensor_pair_us, event.sensor_pair_us)
    event.sensor_pair_us[0] = [10, 0]
    with pytest.raises(ValueError, match="chronological"):
        admit(event)


def test_missing_rgb_does_not_pollute_h_statistics_and_event_still_covers_all():
    value = block()
    value.available[1] = False
    value.features[1] = np.nan
    value.prediction_ttc[1] = np.nan
    value.sensor_pair_us[1] = 0
    admit(value)
    normalizer = value.fit_h_normalizer()
    np.testing.assert_array_equal(normalizer.mean, [3, 4])
    np.testing.assert_array_equal(normalizer.scale, [2, 2])
    np.testing.assert_array_equal(value.normalized(normalizer), [[-1, -1], [0, 0], [1, 1]])
    event = block("event")
    event.available[1] = False
    with pytest.raises(ValueError, match="fallback"):
        admit(event)


@pytest.mark.parametrize("role", ["P", "V"])
def test_statistics_cannot_be_fit_outside_h(role):
    with pytest.raises(ValueError, match="role"):
        block(role=role).fit_h_normalizer()


@pytest.mark.parametrize("field", ["endpoint_sha256", "schema_sha256", "extractor_sha256"])
def test_h_transform_applies_to_v_without_refitting_and_rejects_stale_parent(field):
    normalizer = block().fit_h_normalizer()
    value = block(role="V")
    value.features[:] = 100
    expected = (value.features - normalizer.mean) / normalizer.scale
    np.testing.assert_array_equal(value.normalized(normalizer), expected)
    value = replace(value, identity=replace(value.identity, **{field: "f" * 64}))
    with pytest.raises(ValueError, match="identity mismatch"):
        value.normalized(normalizer)


def test_all_rgb_missing_cannot_create_nan_normalizer():
    value = block()
    value.available[:] = False
    value.features[:] = np.nan
    admit(value)
    with pytest.raises(ValueError, match="No available H"):
        value.fit_h_normalizer()


@pytest.mark.parametrize("error", ["float_clock", "float_features", "mask", "nan", "zero_ttc"])
def test_invalid_available_observations_fail_before_training(error):
    value = block()
    if error == "float_clock":
        value = replace(value, sensor_pair_us=value.sensor_pair_us.astype(np.float64))
    elif error == "float_features":
        value = replace(value, features=value.features.astype(np.float64))
    elif error == "mask":
        value = replace(value, available=value.available.astype(np.int8))
    elif error == "nan":
        value.features[0, 0] = np.nan
    else:
        value.prediction_ttc[0] = 0
    with pytest.raises(ValueError):
        admit(value)
