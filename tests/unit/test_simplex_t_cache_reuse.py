"""Content-bound reuse keeps frozen expert outputs byte-identical."""

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.cache_reuse import rebind_block
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc


def fixture():
    points = np.array([[1.0, 1.0, np.inf], [1.0, 1.0, 1.0]], np.float32)
    features = np.zeros((2, 145), np.float32)
    features[:, 8:11] = expert_phase_from_ttc(points)
    arrays = {
        "features145": features,
        "expert_ttc": points,
        "known": np.ones((2, 2), bool),
        "pair_features": np.zeros((2, 133), np.float32),
        "observation_ids": np.array([0, 1], np.int64),
        "anchor_us": np.array([100, 200], np.int64),
        "available_us": np.array([210, 210], np.int64),
    }
    arguments = dict(
        source_keys=np.array(["a" * 64, "b" * 64]),
        destination_keys=np.array(["c" * 64, "b" * 64, "a" * 64]),
        destination_ids=np.array([2, 1], np.int64),
        anchors_us=np.array([100, 200], np.int64),
        available_us=210,
    )
    return arrays, arguments


def test_rebinding_changes_only_observation_row_numbers():
    arrays, arguments = fixture()
    before = {name: value.tobytes() for name, value in arrays.items()}
    result = rebind_block(arrays, **arguments)
    assert result["observation_ids"].tolist() == [2, 1]
    for name in arrays:
        assert arrays[name].tobytes() == before[name]
        if name != "observation_ids":
            assert result[name] is arrays[name]
    arguments["destination_ids"][0] = 0
    assert result["observation_ids"][0] == 2


@pytest.mark.parametrize("failure", ["key", "bounds", "mask", "time", "roi", "duplicate", "hash"])
def test_changed_reuse_binding_rejected(failure):
    arrays, arguments = fixture()
    if failure == "key":
        arguments["destination_keys"][2] = "d" * 64
    elif failure == "bounds":
        arguments["destination_ids"][0] = 3
    elif failure == "mask":
        arguments["destination_ids"] = np.array([2], np.int64)
    elif failure == "time":
        arguments["anchors_us"][0] = 99
    elif failure == "roi":
        arguments["available_us"] = 211
    elif failure == "duplicate":
        arguments["destination_ids"][0] = 1
    else:
        arguments["source_keys"][0] = "bad"
        arguments["destination_keys"][2] = "bad"
    with pytest.raises(ValueError):
        rebind_block(arrays, **arguments)
