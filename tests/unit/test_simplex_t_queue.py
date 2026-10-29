"""Finite queue ordering and refusal tests; training is never performed."""

import json

import pytest

from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.queue import run_phase
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.training import load_checkpoint

AVAILABILITY = dict(
    d1=False,
    density=False,
    t3=False,
    latent=False,
    replicate_scalar=False,
    replicate_latent=False,
)


class Source:
    identity_sha256 = "a" * 64
    population = 1

    def gather(self, ids):
        raise AssertionError("test must not execute a model update")


def arguments(tmp_path):
    return dict(
        output=tmp_path / "execution",
        stage="T2",
        availability=AVAILABILITY,
        freeze_sha256="b" * 64,
        train_source_hashes={fit_key(s): "a" * 64 for s in registered_graph(**AVAILABILITY)},
        technical_reserved=625,
        validate_prerequisites=lambda: None,
        source_loader=lambda _: Source(),
        resource_ok=lambda: True,
        resume=False,
    )


def test_missing_prerequisite_has_no_files_or_data_reads(tmp_path):
    args = arguments(tmp_path)

    def missing():
        raise ValueError("fixture missing replay parity")

    args["validate_prerequisites"] = missing
    with pytest.raises(ValueError, match="replay parity"):
        run_phase(**args)
    assert not args["output"].exists()


def test_prerequisites_rechecked_after_admission_before_loading(tmp_path):
    args = arguments(tmp_path)
    checks = []

    def changed_after_initial_check():
        checks.append(True)
        if len(checks) == 2:
            raise ValueError("frozen input changed after initial admission")

    def unread(_):
        raise AssertionError("changed prerequisites must not load TRAIN")

    args["validate_prerequisites"] = changed_after_initial_check
    args["source_loader"] = unread
    with pytest.raises(ValueError, match="changed after initial admission"):
        run_phase(**args)
    assert len(checks) == 2
    assert not (args["output"] / "fits").exists()
    assert not (args["output"] / "HEAD_WRITER.lock").exists()


def test_resource_pause_precedes_loading_and_preserves_resume_contract(tmp_path):
    args = arguments(tmp_path)
    args["resource_ok"] = lambda: False

    def unread(_):
        raise AssertionError("resource pause must not load data")

    args["source_loader"] = unread
    state = run_phase(**args)
    assert state["status"] == "PAUSED_RESOURCE"
    assert not state["fits"]
    args["resume"] = True
    assert run_phase(**args)["fits"] == {}
    args["freeze_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="contract changed"):
        run_phase(**args)


def test_paused_fit_stops_queue_before_any_sealing(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []

    def paused(*positional, **kwargs):
        calls.append(kwargs)
        assert "stop_after" not in kwargs  # Scientific queue cannot shorten the recipe.
        assert any(key.startswith("T3/") for key in kwargs["journal"].budget.graph)
        assert kwargs["journal"].key.startswith("T2/")
        return {"status": "PAUSED_RESOURCE", "completed_updates": 0, "scientific_endpoint": False}

    monkeypatch.setattr("e_jepa_ttc.simplex_t.queue.fit", paused)
    state = run_phase(**args)
    assert len(calls) == len(state["fits"]) == 1
    assert not (args["output"] / "T2_ENDPOINTS.json").exists()
    assert state == json.loads((args["output"] / "T2_STATE.json").read_text())


def test_changed_source_refused_before_fit(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args["train_source_hashes"] = dict.fromkeys(args["train_source_hashes"], "c" * 64)

    def uncalled(*args, **kwargs):
        raise AssertionError("must not train mismatched source")

    monkeypatch.setattr("e_jepa_ttc.simplex_t.queue.fit", uncalled)
    with pytest.raises(ValueError, match="frozen TRAIN source"):
        run_phase(**args)


def test_actual_engine_zero_update_pause_through_queue(tmp_path):
    args = arguments(tmp_path)
    checks = iter([True, False])
    args["resource_ok"] = lambda: next(checks)
    state = run_phase(**args)
    assert state["status"] == "PAUSED_RESOURCE"
    key = next(iter(state["fits"]))
    checkpoint = load_checkpoint(args["output"] / "fits" / key / "checkpoint_last.pt")
    assert checkpoint["completed_updates"] == 0
    assert checkpoint["optimizer"]["state"] == {}
    assert not (args["output"] / "HEAD_WRITER.lock").exists()
