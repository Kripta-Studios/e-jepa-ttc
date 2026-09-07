"""Source preparation must pause safely without fitting any model."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.source_preparation import prepare_source_identities


@pytest.mark.parametrize("fault", ["none", "source", "qa", "inventory", "resource", "missing"])
def test_read_only_verification_never_rewrites_preparation(setup, fault):
    output, kwargs, calls, released = setup
    kwargs["validate_loaded_source"] = lambda source: {"identity": source.identity_sha256}
    if fault != "missing":
        prepare_source_identities(output, **kwargs)
    if fault == "source":
        kwargs["sources"].source = lambda *_: SimpleNamespace(identity_sha256="c" * 64)
    elif fault == "qa":
        kwargs["validate_loaded_source"] = lambda source: {"identity": "changed"}
    elif fault == "inventory":
        path = output / "SOURCE_PREPARATION.json"
        state = json.loads(path.read_text())
        state["identities"].pop(next(iter(state["identities"])))
        path.write_text(json.dumps(state), encoding="utf-8")
    elif fault == "resource":
        kwargs["resource_ok"] = lambda: False
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*")}
    calls.clear()
    kwargs["verify_only"] = True
    if fault in {"source", "qa", "inventory", "resource"}:
        with pytest.raises((ValueError, InterruptedError)):
            prepare_source_identities(output, **kwargs)
    else:
        result = prepare_source_identities(output, **kwargs)
        assert result["status"] == (
            "SOURCE_IDENTITIES_INCOMPLETE"
            if fault == "missing"
            else "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
        )
        assert len(calls) == (0 if fault == "missing" else len(kwargs["sources"].graph) * 2)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*")} == before
    assert len(released) == (1 if fault == "missing" else 2)


@pytest.fixture
def setup(tmp_path):
    flags = dict.fromkeys(
        ("d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"), False
    )
    calls = []
    released = []

    def source(spec, role):
        calls.append((fit_key(spec), role))
        digest = hashlib.sha256(f"{fit_key(spec)}:{role}".encode()).hexdigest()
        return SimpleNamespace(identity_sha256=digest)

    adapter = SimpleNamespace(
        graph=registered_graph(**flags), source=source, release=lambda: released.append(True)
    )
    kwargs = {
        "sources": adapter,
        "availability": flags,
        "configuration_sha256": "a" * 64,
        "authority_sha256": "b" * 64,
        "validate_prerequisites": lambda: None,
        "resource_ok": lambda: True,
        "resume": False,
    }
    return tmp_path / "prepare", kwargs, calls, released


def test_complete_is_not_freeze_and_resume_reloads_every_pair(setup):
    output, kwargs, calls, released = setup
    result = prepare_source_identities(output, **kwargs)
    count = len(kwargs["sources"].graph)
    assert len(result["identities"]) == count
    assert len(calls) == count * 2
    assert result["optimizer_updates"] == 0
    assert result["scientific_freeze"] is False
    assert result["gates_enabled"] is False
    assert result["status"] == "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
    kwargs["resume"] = True
    assert prepare_source_identities(output, **kwargs) == result
    assert len(calls) == count * 4
    assert len(released) == 2
    assert not (output / "SOURCE_PREPARATION.lock").exists()


def test_loaded_qa_interruption_keeps_only_complete_pairs(setup):
    output, kwargs, _, released = setup
    checks = []

    def audit(source):
        checks.append(source.identity_sha256)
        if len(checks) == 4:
            raise InterruptedError("resource pause during gather")
        return {"source_sha256": source.identity_sha256}

    kwargs["validate_loaded_source"] = audit
    paused = prepare_source_identities(output, **kwargs)
    assert paused["status"] == "PAUSED_RESOURCE"
    assert len(paused["identities"]) == len(paused["source_qa"]) == 1
    assert set(next(iter(paused["source_qa"].values()))) == {"inner_oof", "outer_dev"}
    assert released == [True]
    kwargs.update(
        resume=True, validate_loaded_source=lambda s: {"source_sha256": s.identity_sha256}
    )
    complete = prepare_source_identities(output, **kwargs)
    assert set(complete["source_qa"]) == set(complete["identities"])


def test_pause_between_roles_never_persists_half_pair(setup):
    output, kwargs, calls, released = setup
    admission = iter((True, True, True, False))
    kwargs["resource_ok"] = lambda: next(admission)
    state = prepare_source_identities(output, **kwargs)
    assert state["status"] == "PAUSED_RESOURCE"
    assert len(state["identities"]) == 1
    assert len(calls) == 3
    assert all(set(pair) == {"inner_oof", "outer_dev"} for pair in state["identities"].values())
    assert released == [True]
    kwargs.update(resume=True, resource_ok=lambda: True)
    result = prepare_source_identities(output, **kwargs)
    assert len(result["identities"]) == len(kwargs["sources"].graph)


def test_missing_authority_creates_no_output_or_source_reads(setup):
    output, kwargs, calls, _ = setup

    def reject():
        raise ValueError("missing temporal authority")

    kwargs["validate_prerequisites"] = reject
    with pytest.raises(ValueError, match="temporal authority"):
        prepare_source_identities(output, **kwargs)
    assert not output.exists()
    assert not calls


def test_resume_rejects_changed_source(setup):
    output, kwargs, _, released = setup
    prepare_source_identities(output, **kwargs)
    kwargs["sources"].source = lambda *_: SimpleNamespace(identity_sha256="c" * 64)
    kwargs["resume"] = True
    with pytest.raises(ValueError, match="normalizer changed"):
        prepare_source_identities(output, **kwargs)
    assert len(released) == 2


def test_explicit_resume_and_identical_contract_required(setup):
    output, kwargs, calls, _ = setup
    prepare_source_identities(output, **kwargs)
    calls.clear()
    with pytest.raises(FileExistsError, match="explicit resume"):
        prepare_source_identities(output, **kwargs)
    kwargs.update(resume=True, configuration_sha256="c" * 64)
    with pytest.raises(ValueError, match="contract changed"):
        prepare_source_identities(output, **kwargs)
    assert not calls


def test_authority_rechecked_between_fits(setup):
    output, kwargs, calls, released = setup
    checks = []

    def validate():
        checks.append(True)
        if len(checks) == 3:
            raise ValueError("authority changed")

    kwargs["validate_prerequisites"] = validate
    with pytest.raises(ValueError, match="authority changed"):
        prepare_source_identities(output, **kwargs)
    assert len(calls) == 2
    assert released == [True]
    state = json.loads((output / "SOURCE_PREPARATION.json").read_text())
    assert len(state["identities"]) == 1
    assert state["scientific_freeze"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema", "foreign"), ("optimizer_updates", 1), ("scientific_freeze", True)],
)
def test_resume_cannot_claim_scientific_progress(setup, field, value):
    output, kwargs, calls, _ = setup
    state = prepare_source_identities(output, **kwargs)
    state[field] = value
    (output / "SOURCE_PREPARATION.json").write_text(json.dumps(state), encoding="utf-8")
    kwargs["resume"] = True
    calls.clear()
    with pytest.raises(ValueError, match="non-scientific"):
        prepare_source_identities(output, **kwargs)
    assert not calls
