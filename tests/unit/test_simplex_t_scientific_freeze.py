"""Synthetic publication checks; no scientific authority or optimizer is created."""

import json
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.freeze_integrity import REQUIRED_CATEGORIES, FrozenFile
from e_jepa_ttc.simplex_t.frozen_phase import run_frozen_phase
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.scientific_freeze import (
    publish_scientific_freeze,
    read_scientific_freeze,
)


def fixture_inputs(root: Path, failure: str = "") -> dict:
    flags = dict(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    graph = registered_graph(**flags)
    values = {category: {"synthetic": category} for category in REQUIRED_CATEGORIES}

    def digest(value: dict) -> str:
        return sha256(json.dumps(value).encode()).hexdigest()

    values["normalizers"] = {
        "schema": "simplex_t_source_preparation_v1",
        "status": "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE",
        "optimizer_updates": 0,
        "scientific_freeze": False,
        "gates_enabled": False,
        "contract": {
            "loaded_source_qa": "simplex_t_full_population_gather_qa_v1",
            "availability": flags,
            "configuration_sha256": digest(values["config"]),
            "authority_sha256": digest(values["roles"]),
            "fits": [
                {"fit": asdict(s), "model": asdict(resolve_arm(s, graph).model)} for s in graph
            ],
        },
        "identities": {fit_key(s): {"inner_oof": "a" * 64, "outer_dev": "b" * 64} for s in graph},
    }
    values["normalizers"]["source_qa"] = {
        fit_key(s): {
            role: {
                "schema": "simplex_t_full_population_gather_qa_v1",
                "source_sha256": digest_value,
                "history_length": resolve_arm(s, graph).history,
                "feature_count": resolve_arm(s, graph).model.feature_count,
                "queries": 1,
                "valid_slots": resolve_arm(s, graph).history,
                "model_inference": False,
                "optimizer_updates": 0,
                "normalizer_ids_sha256": "d" * 64,
            }
            for role, digest_value in (("inner_oof", "a" * 64), ("outer_dev", "b" * 64))
        }
        for s in graph
    }
    values["qa"] = {"schema": "simplex_t_technical_budget_v1", "reservations": {"probe": 20}}
    prepared = values["normalizers"]
    if failure == "missing_fit":
        prepared["identities"].pop(next(iter(prepared["identities"])))
    elif failure == "missing_role":
        next(iter(prepared["identities"].values())).pop("outer_dev")
    elif failure == "authority":
        prepared["contract"]["authority_sha256"] = "c" * 64
    elif failure == "budget":
        values["qa"]["reservations"]["probe"] = 1021
    elif failure == "candidate":
        prepared["contract"]["fits"].pop()
    elif failure == "missing_source_qa":
        prepared.pop("source_qa")
    elif failure == "source_qa_identity":
        next(iter(prepared["source_qa"].values()))["inner_oof"]["source_sha256"] = "c" * 64
    elif failure == "source_qa_normalizer":
        next(iter(prepared["source_qa"].values()))["outer_dev"]["normalizer_ids_sha256"] = "c" * 64
    elif failure == "source_qa_boolean":
        next(iter(prepared["source_qa"].values()))["inner_oof"]["history_length"] = True
    pins = []
    for category, value in sorted(values.items()):
        path = root / f"{category}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        pins.append(FrozenFile(category, "work", path.name, digest(value)))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)

    git("init", "--quiet")
    git("add", "code.json")
    git(
        "-c",
        "user.name=QA Fixture",
        "-c",
        "user.email=qa@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Synthetic freeze fixture",
    )
    commit = git("rev-parse", "HEAD").decode().strip()
    return dict(
        files=pins,
        roots={"work": root},
        preparation=next(p for p in pins if p.category == "normalizers"),
        technical_ledger=next(p for p in pins if p.category == "qa"),
        code_commit=commit,
    )


def test_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path)
    calls = []

    def validator() -> None:
        calls.append("synthetic validation")

    output = tmp_path / "freeze.json"
    record = publish_scientific_freeze(output, **inputs, validate_prerequisites=validator)
    assert len(calls) == 2
    assert record["holdout_authorized"] is False
    assert record["practical_stage_gates_enabled_by_freeze"] is False
    assert record["technical_reserved_not_execution_claim"] == 20
    pin = sha256(output.read_bytes()).hexdigest()
    assert (
        read_scientific_freeze(
            output, expected_sha256=pin, roots=inputs["roots"], validate_prerequisites=validator
        )
        == record
    )
    with pytest.raises(FileExistsError):
        publish_scientific_freeze(output, **inputs, validate_prerequisites=validator)
    assert sha256(output.read_bytes()).hexdigest() == pin
    (tmp_path / "roles.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        read_scientific_freeze(
            output, expected_sha256=pin, roots=inputs["roots"], validate_prerequisites=validator
        )


@pytest.mark.parametrize("change", ["invented_commit", "repinned_working_code", "untracked_other"])
def test_freeze_code_must_belong_to_declared_commit(tmp_path: Path, change: str) -> None:
    inputs = fixture_inputs(tmp_path)
    if change == "invented_commit":
        inputs["code_commit"] = "a" * 40
    elif change == "repinned_working_code":
        path = tmp_path / "code.json"
        path.write_text('{"different": true}', encoding="utf-8")
        inputs["files"] = [
            FrozenFile(
                pin.category, pin.root, pin.relative_path, sha256(path.read_bytes()).hexdigest()
            )
            if pin.category == "code"
            else pin
            for pin in inputs["files"]
        ]
    else:
        (tmp_path / "unrelated_proposal.md").write_text("preserve me", encoding="utf-8")
    output = tmp_path / "freeze.json"
    if change == "untracked_other":
        publish_scientific_freeze(output, **inputs, validate_prerequisites=lambda: None)
        assert output.exists()
    else:
        with pytest.raises(ValueError, match="commit"):
            publish_scientific_freeze(output, **inputs, validate_prerequisites=lambda: None)
        assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_fit",
        "missing_role",
        "authority",
        "budget",
        "candidate",
        "missing_source_qa",
        "source_qa_identity",
        "source_qa_normalizer",
        "source_qa_boolean",
    ],
)
def test_rejects_incomplete_or_changed_contract(tmp_path: Path, failure: str) -> None:
    inputs = fixture_inputs(tmp_path, failure)
    output = tmp_path / "freeze.json"
    with pytest.raises(ValueError):
        publish_scientific_freeze(output, **inputs, validate_prerequisites=lambda: None)
    assert not output.exists()


@pytest.mark.parametrize("reject_call", [1, 2])
def test_semantic_rejection_before_publication(tmp_path: Path, reject_call: int) -> None:
    inputs = fixture_inputs(tmp_path)
    count = 0

    def validate() -> None:
        nonlocal count
        count += 1
        if count == reject_call:
            raise ValueError("missing actual prerequisite")

    output = tmp_path / "freeze.json"
    with pytest.raises(ValueError, match="actual prerequisite"):
        publish_scientific_freeze(output, **inputs, validate_prerequisites=validate)
    assert not output.exists()
    assert not output.with_suffix(".lock").exists()


@pytest.mark.parametrize("mode", ["resource_pause", "gate", "ledger", "old_dev", "fit_boundary"])
def test_freeze_to_actual_queue(tmp_path: Path, monkeypatch, mode: str) -> None:
    inputs = fixture_inputs(tmp_path)
    path = tmp_path / "freeze.json"
    record = publish_scientific_freeze(path, **inputs, validate_prerequisites=lambda: None)
    availability = record["source_contract"]["availability"]
    released, loaded, fitted = [], [], []

    def source(spec, role):
        loaded.append(role)
        digest = "a" * 64 if role == "inner_oof" else "b" * 64
        if mode == "old_dev" and role == "outer_dev":
            digest = "c" * 64
        return SimpleNamespace(identity_sha256=digest)

    sources = SimpleNamespace(
        graph=registered_graph(**availability), source=source, release=lambda: released.append(True)
    )

    def gate(stage, flags):
        assert stage == "T2" and flags == availability
        if mode == "gate":
            raise ValueError("unresolved scientific gate")

    def resources():
        if mode == "ledger":
            (tmp_path / "qa.json").write_text("{}", encoding="utf-8")
        return mode != "resource_pause"

    def paused_fit(*args, **kwargs):
        assert mode == "fit_boundary"
        assert loaded == ["inner_oof", "outer_dev"]
        assert kwargs["journal"].budget.technical_reserved == 20
        fitted.append(True)
        return {"status": "PAUSED_RESOURCE", "completed_updates": 0}

    monkeypatch.setattr("e_jepa_ttc.simplex_t.queue.fit", paused_fit)
    options = dict(
        sources=sources,
        freeze=path,
        freeze_sha256=sha256(path.read_bytes()).hexdigest(),
        roots=inputs["roots"],
        stage="T2",
        availability=availability,
        validate_authority_and_qa=lambda: None,
        validate_stage_gate=gate,
        resource_ok=resources,
        resume=False,
    )
    output = tmp_path / "execution"
    if mode in {"gate", "ledger", "old_dev"}:
        with pytest.raises(ValueError):
            run_frozen_phase(output, **options)
        assert not fitted
    else:
        result = run_frozen_phase(output, **options)
        assert result["status"] == "PAUSED_RESOURCE"
        assert result["contract"]["technical_reserved"] == 20
        assert result["contract"]["freeze"] == options["freeze_sha256"]
    assert released == [True]
    if mode in {"resource_pause", "gate", "ledger"}:
        assert not loaded
    assert not (output / "T2_ENDPOINTS.json").exists()
