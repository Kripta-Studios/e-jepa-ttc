"""Runtime admission must retain the original population, arithmetic scope and tolerances."""

import copy

import pytest

from operational.joint_measurement.fusion_cost_historical_shape import bind_cost, shape_map


def inventory():
    observation = {"id": "native", "input_frames": 3, "forward_batch": 29}
    return {
        "status": "COMPLETE_METADATA_AUDIT",
        "indices": [4, 8],
        "modalities": {
            modality: {
                "queries": [
                    {"index": index, "observations": [copy.deepcopy(observation)]}
                    for index in (4, 8)
                ]
            }
            for modality in ("E", "R")
        },
    }


def test_repeated_observation_requires_identical_shape():
    data = inventory()
    assert shape_map(data, [4, 8])["R"]["native"]["forward_batch"] == 29
    data["modalities"]["R"]["queries"][1]["observations"][0]["forward_batch"] = 30
    with pytest.raises(ValueError, match="Conflicting"):
        shape_map(data, [4, 8])


def test_no_dropping_or_reordering_cost_queries():
    with pytest.raises(ValueError, match="population"):
        shape_map(inventory(), [4])
    data = inventory()
    data["modalities"]["E"]["queries"].reverse()
    with pytest.raises(ValueError, match="order/coverage"):
        shape_map(data, [4, 8])


def test_binding_declares_backend_without_relaxing_parity():
    value = {"parity_rtol": 1e-5, "parity_atol": 1e-4, "indices": [4, 8], "batch_size": 1}
    original = copy.deepcopy(value)
    bind_cost(value, "runtime", "inventory")
    assert all(value[key] == item for key, item in original.items())
    assert "TF32 enabled" in value["precision"]
    assert "charge all work" in value["producer_batch_policy"]
