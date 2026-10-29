"""Preparation-to-queue integration with zero new optimizer updates."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.prepared_phase import run_prepared_phase
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.source_preparation import prepare_source_identities


@pytest.fixture
def prepared(tmp_path):
    flags = dict.fromkeys(
        ("d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"), False
    )
    prepared_flags = {**flags, "t3": True}
    calls, released = [], []

    def source(spec, role):
        calls.append((fit_key(spec), role))
        return SimpleNamespace(
            identity_sha256=hashlib.sha256(f"{fit_key(spec)}:{role}".encode()).hexdigest()
        )

    sources = SimpleNamespace(
        graph=registered_graph(**prepared_flags),
        source=source,
        release=lambda: released.append(True),
    )
    folder = tmp_path / "prepared"
    prepare_source_identities(
        folder,
        sources=sources,
        availability=prepared_flags,
        configuration_sha256="a" * 64,
        authority_sha256="b" * 64,
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
        resume=False,
    )
    calls.clear()
    released.clear()
    path = folder / "SOURCE_PREPARATION.json"
    args = dict(
        output=tmp_path / "execution",
        sources=sources,
        preparation=path,
        preparation_sha256=sha256(path),
        configuration_sha256="a" * 64,
        authority_sha256="b" * 64,
        freeze_sha256="c" * 64,
        stage="T2",
        availability=flags,
        technical_reserved=645,
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
        resume=False,
    )
    return args, calls, released


def test_real_queue_pauses_before_source_read_and_resumes(prepared):
    args, calls, released = prepared
    args["resource_ok"] = lambda: False
    state = run_prepared_phase(**args)
    assert state["status"] == "PAUSED_RESOURCE"
    assert not state["fits"]
    assert not calls
    assert released == [True]
    args["resume"] = True
    assert run_prepared_phase(**args) == state
    assert len(released) == 2


def test_both_roles_verified_before_entering_fit(prepared, monkeypatch):
    args, calls, released = prepared
    fitted = []

    def paused(*positional, **kwargs):
        assert [role for _, role in calls] == ["inner_oof", "outer_dev"]
        fitted.append(True)
        assert kwargs["journal"].budget.technical_reserved == 645
        return {"status": "PAUSED_RESOURCE", "completed_updates": 0}

    monkeypatch.setattr("e_jepa_ttc.simplex_t.queue.fit", paused)
    assert run_prepared_phase(**args)["status"] == "PAUSED_RESOURCE"
    assert fitted == released == [True]
    assert not (args["output"] / "T2_ENDPOINTS.json").exists()


def test_changed_old_dev_rejected_before_optimizer(prepared, monkeypatch):
    args, _, released = prepared
    original = args["sources"].source

    def altered(spec, role):
        result = original(spec, role)
        if role == "outer_dev":
            result.identity_sha256 = "f" * 64
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("changed OLD_DEV must never reach optimizer")

    args["sources"].source = altered
    monkeypatch.setattr("e_jepa_ttc.simplex_t.queue.fit", forbidden)
    with pytest.raises(ValueError, match="OLD_DEV source differs"):
        run_prepared_phase(**args)
    assert released == [True]


@pytest.mark.parametrize("change", ["incomplete", "pool", "authority", "bytes"])
def test_invalid_preparation_is_not_execution_authority(prepared, change):
    args, calls, released = prepared
    if change == "pool":
        args["availability"] = {**args["availability"], "d1": True}
    elif change == "authority":
        args["authority_sha256"] = "d" * 64
    else:
        state = json.loads(args["preparation"].read_text())
        state["status"] = "PAUSED_RESOURCE"
        args["preparation"].write_text(json.dumps(state), encoding="utf-8")
        if change == "incomplete":
            args["preparation_sha256"] = sha256(args["preparation"])
    with pytest.raises(ValueError):
        run_prepared_phase(**args)
    assert not calls
    assert not args["output"].exists()
    assert released == [True]


def test_missing_scientific_freeze_stops_before_preparation_read(prepared):
    args, calls, released = prepared

    def reject():
        raise ValueError("missing scientific freeze")

    args["validate_prerequisites"] = reject
    args["preparation"] = args["preparation"].with_name("does_not_exist.json")
    with pytest.raises(ValueError, match="scientific freeze"):
        run_prepared_phase(**args)
    assert not calls
    assert not args["output"].exists()
    assert released == [True]


def test_reserved_later_stage_is_not_enabled(prepared):
    args, calls, released = prepared
    args["stage"] = "T3"
    with pytest.raises(ValueError, match="complete registered stage"):
        run_prepared_phase(**args)
    assert not calls
    assert not args["output"].exists()
    assert released == [True]


def test_preparation_rechecked_after_resource_admission(prepared):
    args, calls, released = prepared

    def mutate_then_admit():
        args["preparation"].write_text("{}", encoding="utf-8")
        return True

    args["resource_ok"] = mutate_then_admit
    with pytest.raises(ValueError, match="changed during phase"):
        run_prepared_phase(**args)
    assert not calls
    assert not (args["output"] / "fits").exists()
    assert released == [True]
