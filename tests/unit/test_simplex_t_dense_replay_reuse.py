"""Catalog-to-queue wiring using synthetic content identities."""

from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.dense_replay_reuse import DenseReplayReuse


@pytest.mark.parametrize("failure", ["", "identity", "missing", "family"])
def test_dense_catalog_bridge(tmp_path, failure):
    history = np.full((2, 16), -1, dtype=np.int64)
    history[:, -1] = [0, 1]
    path = tmp_path / "outer0.npz"
    np.savez(path, keys=np.array(["old", "new"]), history=history)

    class Histories:
        root = tmp_path
        records = [{"path": path.name, "sha256": sha256(path)}]

        def __call__(self, outer):
            assert outer == 0
            return history

    loads = []

    def load(record, **kwargs):
        loads.append(record)
        np.testing.assert_array_equal(kwargs["destination_ids"], [0])
        return {"observations": kwargs["destination_ids"]}

    catalog = SimpleNamespace(
        identity={"pin": "a"},
        outer=0,
        verify_recipe=lambda identity: None,
        plan=lambda tokens, families: {0: {"query": 99}},
        load=load,
    )
    index = {
        "tokens": np.array(["old", "new"]),
        "producer_family": np.array([[0, 0], [-1, -1], [-1, -1]]),
        "valid": history >= 0,
        "anchor_us": np.array([100, 200]),
        "lag_us": np.arange(15, -1, -1),
        "roi_available_us": np.array([100, 200]),
    }
    identities = {} if failure == "missing" else {0: {"pin": "b" if failure == "identity" else "a"}}
    reuse = DenseReplayReuse(
        catalog_loader=lambda outer: catalog,
        expected_identities=identities,
        histories=Histories(),
        index=index,
        extraction_identity={"pool": "DENSE_OLD"},
    )
    if failure:
        with pytest.raises(ValueError):
            reuse(1 if failure == "family" else 0, 0)
        assert not loads
    else:
        assert reuse(0, 0) is not None
        assert reuse(0, 1) is None
        assert len(loads) == 1
