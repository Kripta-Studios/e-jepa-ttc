"""History coverage includes cold starts and must use canonical frozen TRAIN."""

from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.history_support import frozen_train_history_support, history_support
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


def test_support_includes_cold_starts_in_denominator():
    history = np.zeros((4, 16), dtype=np.int64)
    history[0, :-1] = -1
    history[1, :-8] = -1
    result = history_support(history)
    assert result == dict(queries=4, full_h8=3, fraction_train_h8=0.75, full_h16=2, cold_start_h8=1)


@pytest.mark.parametrize("problem", ["empty", "width", "dtype", "current", "hole", "sentinel"])
def test_support_rejects_invalid_history(problem):
    history = np.zeros((3, 16), dtype=np.int64)
    if problem == "empty":
        history = history[:0]
    elif problem == "width":
        history = history[:, :8]
    elif problem == "dtype":
        history = history.astype(float)
    elif problem == "current":
        history[0, -1] = -1
    elif problem == "hole":
        history[0, -2] = -1
    else:
        history[0, 0] = -2
    with pytest.raises(ValueError):
        history_support(history)


@pytest.mark.parametrize("problem", ["", "identity", "control", "primary"])
def test_frozen_support_uses_all_three_canonical_train_sources(problem):
    flags = dict(
        d1=True, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    graph = registered_graph(**flags)
    canonical = "TPR-D1-H8-C160"
    freeze = {
        "source_contract": {"availability": flags},
        "canonical_scalar": canonical,
        "source_identities": {fit_key(s): {"inner_oof": "a" * 64} for s in graph},
    }
    calls = []

    def source(spec, role):
        calls.append((spec.fold, spec.name, role))
        return SimpleNamespace(
            identity_sha256=("b" if problem == "identity" else "a") * 64,
            control="REPEAT_CURRENT" if problem == "control" else "NONE",
            length=8,
            history=np.zeros((3, 16), dtype=np.int64),
        )

    if problem == "primary":
        freeze["canonical_scalar"] = "TPR-D0-H8-C160"
    sources = SimpleNamespace(graph=graph, source=source)
    if problem:
        with pytest.raises(ValueError):
            frozen_train_history_support(sources, freeze, validate_frozen_sources=lambda: None)
    else:
        result = frozen_train_history_support(sources, freeze, validate_frozen_sources=lambda: None)
        assert result["fraction_train_h8"] == (1.0, 1.0, 1.0)
        assert calls == [(fold, canonical, "inner_oof") for fold in range(3)]
        assert not result["scientific_gate_authorized"]


def test_stage_wrapper_uses_derived_fractions(monkeypatch):
    from e_jepa_ttc.simplex_t import stage_gate

    freeze = {"source_contract": {"availability": {"fixture": True}}}
    sources = object()
    calls = []

    def validate():
        calls.append("validate")

    def support(actual_sources, actual_freeze, *, validate_frozen_sources):
        assert actual_sources is sources and actual_freeze is freeze
        assert validate_frozen_sources is validate
        validate_frozen_sources()
        return {"fraction_train_h8": (0.4, 0.6, 0.8)}

    def predictions(**kwargs):
        assert kwargs["fraction_train_h8"] == (0.4, 0.6, 0.8)
        assert kwargs["validate_frozen_publications_and_lineage"] is validate
        return "fixture callback"

    monkeypatch.setattr(stage_gate, "frozen_train_history_support", support)
    monkeypatch.setattr(stage_gate, "stage_gate_from_predictions", predictions)
    assert (
        stage_gate.stage_gate_from_frozen_sources(
            freeze=freeze,
            sources=sources,
            scalar=None,
            latent=None,
            risk17=None,
            validate_frozen_publications_and_lineage=validate,
        )
        == "fixture callback"
    )
    assert calls == ["validate"]
