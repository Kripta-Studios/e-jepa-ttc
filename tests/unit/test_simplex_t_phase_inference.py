"""Sealed inference wiring with mocked heads; zero optimizer updates."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import phase_inference
from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    availability = dict(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    graph = registered_graph(**availability)
    records = []
    for i, spec in enumerate(graph):
        path = tmp_path / f"fixture{i}.pt"
        path.write_bytes(b"mocked checkpoint, not scientific weights")
        records.append(
            dict(
                key=fit_key(spec),
                fit=asdict(spec),
                model=asdict(resolve_arm(spec, graph).model),
                checkpoint=path.name,
                checkpoint_sha256=sha256(path),
                train_source_sha256="a" * 64,
            )
        )
    data = dict(
        schema="simplex_t_phase_endpoints_v1",
        stage="T2",
        scientific_freeze_sha256="b" * 64,
        availability=availability,
        fits=records,
        optimizer_endpoint_updates=60000,
        scores_read_by_sealer=False,
    )
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = []
    monkeypatch.setattr(phase_inference, "load_endpoint", lambda *a, **k: loaded.append(a[0]))
    kwargs = dict(
        manifest_sha256=sha256(path), freeze_sha256="b" * 64, stage="T2", availability=availability
    )
    return path, data, graph, kwargs, loaded


def test_all_endpoints_validated_before_first_population_and_only_complete_outputs(
    sealed, monkeypatch
):
    path, _, graph, kwargs, loaded = sealed
    calls = []
    history = np.arange(16).reshape(1, 16)

    def source(spec):
        assert len(loaded) >= len(graph)
        calls.append(spec)
        return SimpleNamespace(identity_sha256="dev", history=history, length=8)

    monkeypatch.setattr(phase_inference, "predict_cached", lambda *a, **k: {"point": np.zeros(1)})
    results = list(
        phase_inference.iter_phase_predictions(
            path,
            path.parent,
            **kwargs,
            dev_source_hashes={fit_key(s): "dev" for s in graph},
            validate_prerequisites=lambda: None,
            dev_source_loader=source,
            resource_ok=lambda: True,
        )
    )
    assert calls == graph
    assert len(results) == 24
    assert all(np.array_equal(item[2], history[:, -8:]) for item in results)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_model", "escape", "changed_bytes"])
def test_corrupt_seal_is_rejected_before_any_population(sealed, kind):
    path, data, graph, kwargs, _ = sealed
    if kind == "missing":
        data["fits"].pop()
    elif kind == "duplicate":
        data["fits"][-1] = data["fits"][0]
    elif kind == "wrong_model":
        data["fits"][0]["model"]["hidden"] = 999
    elif kind == "escape":
        data["fits"][0]["checkpoint"] = str(path.resolve())
    else:
        (path.parent / data["fits"][0]["checkpoint"]).write_bytes(b"changed")
    path.write_text(json.dumps(data), encoding="utf-8")
    kwargs["manifest_sha256"] = sha256(path)

    def forbidden(spec):
        pytest.fail("population opened before full phase validation")

    with pytest.raises(ValueError):
        list(
            phase_inference.iter_phase_predictions(
                path,
                path.parent,
                **kwargs,
                dev_source_hashes={fit_key(s): "dev" for s in graph},
                validate_prerequisites=lambda: None,
                dev_source_loader=forbidden,
                resource_ok=lambda: True,
            )
        )


@pytest.mark.parametrize("failure", ["resource", "source_hash", "inference"])
def test_no_partial_fit_output_on_resource_or_identity_failure(sealed, monkeypatch, failure):
    path, _, graph, kwargs, _ = sealed

    def predict(*args, **kw):
        raise InterruptedError("interrupted minibatch; no partial output")

    monkeypatch.setattr(phase_inference, "predict_cached", predict)
    iterator = phase_inference.iter_phase_predictions(
        path,
        path.parent,
        **kwargs,
        dev_source_hashes={fit_key(s): "dev" for s in graph},
        validate_prerequisites=lambda: None,
        dev_source_loader=lambda s: SimpleNamespace(
            identity_sha256="changed" if failure == "source_hash" else "dev"
        ),
        resource_ok=lambda: failure != "resource",
    )
    with pytest.raises(ValueError if failure == "source_hash" else InterruptedError):
        next(iterator)


def test_gate_rejection_precedes_even_manifest_access(tmp_path):
    def gate():
        raise ValueError("gate not authorized")

    with pytest.raises(ValueError, match="gate not authorized"):
        list(
            phase_inference.iter_phase_predictions(
                tmp_path / "absent",
                tmp_path,
                manifest_sha256="x",
                freeze_sha256="y",
                stage="T3",
                availability={},
                dev_source_hashes={},
                validate_prerequisites=gate,
                dev_source_loader=lambda s: None,
                resource_ok=lambda: True,
            )
        )
