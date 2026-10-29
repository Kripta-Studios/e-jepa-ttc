"""Integrated companion boundaries; no real labels, media or optimizer updates."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.timeline import InputObservation, indices, validate_model_fields


def records():
    return [
        dict(
            token=str(i),
            sequence_id="s",
            acquisition_group="s",
            track_id="t",
            producer_family_sha256="a" * 64,
            label_anchor_us=10**16 + i * 100000,
            latest_sensor_dependency_us=10**16 + i * 100000 + 1000,
            latest_roi_dependency_us=10**16 + i * 100000,
            raw_binding_hash="b" * 64,
            role="TRAIN",
            ttc=float(i),
            depth=100.0,
            velocity=20.0,
        )
        for i in range(20)
    ]


def index(rows):
    return indices(
        rows,
        np.array([19]),
        length=8,
        allowed_groups={"s"},
        producer_fit_ancestors={"a" * 64: {"fit"}},
        outer_heldout={"outer"},
    )


def test_history_index_unchanged_by_all_numeric_targets():
    raw = records()
    before = index([InputObservation.project_trusted_record(r) for r in raw])
    for row in raw:
        row.update(ttc=float("nan"), depth=-100000.0, velocity=0.0)
    after = index([InputObservation.project_trusted_record(r) for r in raw])
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after[0], np.arange(12, 20))


def test_future_roi_is_not_available_history():
    rows = [InputObservation.project_trusted_record(r) for r in records()]
    rows[18] = replace(rows[18], latest_roi_dependency_us=rows[19].available_us + 1)
    assert 18 not in index(rows)


def test_ancestor_exclusions_are_transitive():
    rows = [InputObservation.project_trusted_record(r) for r in records()]
    with pytest.raises(ValueError, match="ancestor"):
        indices(
            rows,
            np.array([19]),
            length=8,
            allowed_groups={"s"},
            producer_fit_ancestors={"a" * 64: {"s"}},
            outer_heldout={"outer"},
        )


def test_model_rejects_privileged_fields():
    with pytest.raises(ValueError, match="model batch"):
        validate_model_fields(
            dict(features=None, timing=None, valid=None, current_expert_phase=None, ttc=1.0)
        )


@pytest.mark.parametrize(
    "features,hidden,backbone,count",
    [
        (17, 64, "gru", 51781),
        (17, 160, "gru", 313765),
        (17, 128, "transformer", 268677),
        (145, 160, "gru", 334245),
    ],
)
def test_registered_parameter_counts(features, hidden, backbone, count):
    model = TemporalRefiner(TemporalConfig(features, hidden, backbone))
    assert sum(p.numel() for p in model.parameters()) == count
    x = torch.zeros(2, 1, features)
    output = model(
        x, torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool), torch.full((2, 3), 0.02)
    )
    assert torch.isfinite(output["point_phase"]).all()


def test_maximum_graph_exactly_84_fits_210000_updates():
    graph = registered_graph(
        d1=True, density=True, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    assert len(graph) == 84
    assert sum(f.updates for f in graph) == 210000
    assert len({(f.name, f.fold, f.seed) for f in graph}) == 84


def test_d0_only_graph_preserves_controls():
    graph = registered_graph(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    assert len(graph) == 24
    assert all("D0" in f.name for f in graph)
