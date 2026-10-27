from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.final_refit.materialize import (
    FragmentStore,
    materialize_query_pair_features,
    pipeline_status,
)
from operational.final_refit.recipe import ordered_row_identity_sha256

NATIVE_ROWS_SHA256 = "e" * 64


class _Source:
    population_size = 5
    _batch_keys = [("P", 3), ("P", 3), ("H", 2), ("H", 2), ("V", 3)]

    @property
    def batch_keys(self):
        return self._batch_keys

    def batch(self, indices, modality) -> ObjectEventV4Batch:
        assert modality == "rgb"
        steps = self.batch_keys[indices[0]][1]
        assert all(self.batch_keys[index] == self.batch_keys[indices[0]] for index in indices)
        return cast(
            ObjectEventV4Batch,
            SimpleNamespace(
                events=torch.full((len(indices), steps, 3, 2, 2), 0.25),
                delta_t_s=torch.full((len(indices), steps - 1), 0.1),
            ),
        )

    def row(self, index):
        return {
            "target_phase": np.float32(index / 10),
            "group_id": f"g{index % 2}",
            "sample_token": f"q{index}",
            "sequence_id": f"s{index % 2}",
        }


def _ordered_source_identity() -> str:
    source = _Source()
    rows = [source.row(index) for index in range(source.population_size)]
    return ordered_row_identity_sha256(
        [row["sample_token"] for row in rows], [row["group_id"] for row in rows]
    )


class _Producer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(modality="rgb")

    def forward(self, inputs: Tensor, delta: Tensor, return_dense_features: bool = False):
        del return_dense_features
        batch, steps = inputs.shape[:2]
        support = torch.linspace(0.4, 0.8, steps).expand(batch, -1)
        return SimpleNamespace(
            ttc_mean_seconds=torch.full((batch,), 2.0),
            log_height_ratio=torch.full((batch, steps), 0.1),
            sensor_support=support,
            pair_tokens=torch.arange(128, dtype=torch.float32).expand(batch, steps, -1),
            ttc_log_variance=torch.zeros(batch),
            known_mask=torch.ones((batch, steps), dtype=torch.bool),
            diagnostics={"transport_flow_magnitude": torch.ones((batch, steps))},
        )


def test_fragment_store_resume_rejects_changed_binding(tmp_path: Path) -> None:
    root = tmp_path / "parts"
    store = FragmentStore(root, {"stage": "x", "parent": "a" * 64})
    store.write(0, {"values": np.asarray([[1, 2]], np.float32)})
    np.testing.assert_array_equal(store.read(0)["values"], [[1, 2]])  # type: ignore[index]
    try:
        FragmentStore(root, {"stage": "x", "parent": "b" * 64})
    except ValueError as error:
        assert "binding changed" in str(error)
    else:
        raise AssertionError("changed fragment ancestry was accepted")


def test_query_pair_materializer_executes_models_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "PAIR_R.npz"
    first = materialize_query_pair_features(
        _Source(),
        _Producer(),
        output,
        modality="rgb",
        parents={"R_A5": "a" * 64},
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        expected_ordered_row_identity_sha256=_ordered_source_identity(),
        rows_per_fragment=2,
        batch_size=2,
    )
    assert first["rows"] == 5
    assert first["groups"] == 2
    with np.load(output, allow_pickle=False) as archive:
        assert archive["features"].shape == (5, 133)
        np.testing.assert_allclose(archive["features"][:, 128], 0.1)
        np.testing.assert_allclose(archive["features"][:, 130], 10.0)
        assert set(archive["role"]) == {"TRAIN"}
    second = materialize_query_pair_features(
        _Source(),
        _Producer(),
        output,
        modality="rgb",
        parents={"R_A5": "a" * 64},
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        expected_ordered_row_identity_sha256=_ordered_source_identity(),
    )
    assert second["output_sha256"] == first["output_sha256"]


def test_pipeline_status_is_read_only_and_reports_missing(tmp_path: Path) -> None:
    status = pipeline_status(tmp_path)
    assert status["complete"] is False
    assert {item["status"] for item in status["stages"].values()} == {"MISSING"}
    assert not (tmp_path / "features").exists()


def test_completed_cache_cannot_hide_a_changed_parent(tmp_path: Path) -> None:
    output = tmp_path / "PAIR_R.npz"
    materialize_query_pair_features(
        _Source(),
        _Producer(),
        output,
        modality="rgb",
        parents={"R_A5": "a" * 64},
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        expected_ordered_row_identity_sha256=_ordered_source_identity(),
    )
    with pytest.raises(ValueError, match="parent bindings differ"):
        materialize_query_pair_features(
            _Source(),
            _Producer(),
            output,
            modality="rgb",
            parents={"R_A5": "b" * 64},
            native_train40_rows_sha256=NATIVE_ROWS_SHA256,
            expected_ordered_row_identity_sha256=_ordered_source_identity(),
        )


def test_completed_cache_cannot_hide_changed_train40_population(tmp_path: Path) -> None:
    output = tmp_path / "PAIR_R.npz"
    materialize_query_pair_features(
        _Source(),
        _Producer(),
        output,
        modality="rgb",
        parents={"R_A5": "a" * 64},
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        expected_ordered_row_identity_sha256=_ordered_source_identity(),
    )
    with pytest.raises(ValueError, match="TRAIN40 population differs"):
        materialize_query_pair_features(
            _Source(),
            _Producer(),
            output,
            modality="rgb",
            parents={"R_A5": "a" * 64},
            native_train40_rows_sha256="f" * 64,
            expected_ordered_row_identity_sha256=_ordered_source_identity(),
        )


def test_orphan_fragment_is_committed_only_after_exact_recomputation(tmp_path: Path) -> None:
    store = FragmentStore(tmp_path, {"parent": "a" * 64})
    values = {"values": np.asarray([[1, 2]], np.float32)}
    np.savez_compressed(tmp_path / "part_000000.npz", **values)
    assert store.read(0) is None
    with pytest.raises(ValueError, match="uncommitted fragment differs"):
        store.write(0, {"values": np.asarray([[3, 4]], np.float32)})
    store.write(0, values)
    np.testing.assert_array_equal(store.read(0)["values"], values["values"])
