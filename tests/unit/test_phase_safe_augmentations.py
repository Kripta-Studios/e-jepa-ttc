"""Causal, deterministic perturbations of geometry observations only."""

import inspect

import numpy as np

from scripts.build_phase_safe_augmentations import (
    VARIANTS,
    GeometryQuery,
    build_geometry_cache,
    eligible_sensor_rows,
    perturb_history,
)


def track() -> tuple[np.ndarray, np.ndarray]:
    clocks = np.arange(0, 1_000_001, 50_000, dtype=np.int64)
    width = 40 + np.arange(len(clocks)) * 2
    boxes = np.column_stack((np.zeros(len(clocks)), np.zeros(len(clocks)), width, width))
    return boxes.astype(float), clocks


def test_causal_cutoff_applies_before_last_observations_and_randomness() -> None:
    boxes, clocks = track()
    for variant in VARIANTS:
        full = perturb_history(boxes, clocks, 400_000, "row", variant)
        prefix = perturb_history(boxes[:9], clocks[:9], 400_000, "row", variant)
        np.testing.assert_array_equal(full[0], prefix[0])
        np.testing.assert_array_equal(full[1], prefix[1])
        assert np.all(full[1] <= 400_000)
    last, times = perturb_history(boxes, clocks, 900_000, "row", "last2")
    np.testing.assert_array_equal(times, [850_000, 900_000])
    assert last.shape == (2, 4)


def test_jitter_is_bounded_and_deterministic_by_identity() -> None:
    boxes, clocks = track()
    left = perturb_history(boxes, clocks, 1_000_000, "group:row", "jitter2px")
    right = perturb_history(boxes, clocks, 1_000_000, "group:row", "jitter2px")
    np.testing.assert_array_equal(left[0], right[0])
    original = boxes[clocks >= 350_000]
    assert np.all(abs(left[0] - original) <= 2.0)
    assert np.any(left[0] != original)
    other = perturb_history(boxes, clocks, 1_000_000, "group:other", "jitter2px")
    assert not np.array_equal(left[0], other[0])


def test_only_eligible_group_sensor_fields_are_read_without_targets() -> None:
    class Excluded(dict):
        def __getitem__(self, key):
            if key != "sequence_id":
                raise AssertionError("Excluded-group field was accessed")
            return "excluded"

    included = {
        "sequence_id": "allowed",
        "query_id": "q",
        "anchor_us": 5,
        "rgb_timestamps_us": [1, 5],
        "rgb_boxes": [[0, 0, 2, 2], [0, 0, 3, 3]],
        "target": object(),
    }
    selected = eligible_sensor_rows([Excluded(), included], {"allowed"})
    assert len(selected) == 1 and "target" not in selected[0]
    assert set(selected[0]) == {
        "sequence_id",
        "query_id",
        "anchor_us",
        "rgb_timestamps_us",
        "rgb_boxes",
    }
    assert "target" not in inspect.signature(build_geometry_cache).parameters
    assert "target" not in inspect.signature(perturb_history).parameters


def test_shapes_missing_masks_and_group_identity_are_fresh() -> None:
    boxes, clocks = track()
    queries = [GeometryQuery("a:0", "a", 0), GeometryQuery("b:1", "b", 500_000)]
    tracks = {"a": (boxes, clocks), "b": (boxes * 2, clocks)}
    out = build_geometry_cache(queries, tracks)
    assert out["geometry"].shape == (4, 2)
    assert out["quality"].shape == (4, 2, 20)
    assert out["valid"].shape == (4, 2)
    assert not out["valid"][:, 0].any()
    assert np.isnan(out["geometry"][:, 0]).all()
    assert np.isfinite(out["quality"]).all()
    np.testing.assert_array_equal(out["valid"], np.isfinite(out["geometry"]))
    reordered = build_geometry_cache(list(reversed(queries)), tracks)
    np.testing.assert_array_equal(out["geometry"], reordered["geometry"][:, ::-1])
    np.testing.assert_array_equal(out["quality"], reordered["quality"][:, ::-1])


def test_dropout_can_remove_every_observation_without_endpoint_imputation() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    clocks = np.array([100_000], dtype=np.int64)
    results = [perturb_history(boxes, clocks, 100_000, str(i), "dropout50") for i in range(40)]
    assert any(len(times) == 0 for _, times in results)
    assert any(len(times) == 1 for _, times in results)
