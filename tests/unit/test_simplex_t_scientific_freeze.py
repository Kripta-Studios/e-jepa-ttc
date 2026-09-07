"""Synthetic publication checks; no scientific authority or optimizer is created."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.freeze_integrity import REQUIRED_CATEGORIES, FrozenFile
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
            "availability": flags,
            "configuration_sha256": digest(values["config"]),
            "authority_sha256": digest(values["roles"]),
            "fits": [
                {"fit": asdict(s), "model": asdict(resolve_arm(s, graph).model)} for s in graph
            ],
        },
        "identities": {fit_key(s): {"inner_oof": "a" * 64, "outer_dev": "b" * 64} for s in graph},
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
        values["qa"]["reservations"]["probe"] = 1001
    elif failure == "candidate":
        prepared["contract"]["fits"].pop()
    pins = []
    for category, value in sorted(values.items()):
        path = root / f"{category}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        pins.append(FrozenFile(category, "work", path.name, digest(value)))
    return dict(
        files=pins,
        roots={"work": root},
        preparation=next(p for p in pins if p.category == "normalizers"),
        technical_ledger=next(p for p in pins if p.category == "qa"),
        code_commit="d" * 40,
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


@pytest.mark.parametrize(
    "failure", ["missing_fit", "missing_role", "authority", "budget", "candidate"]
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
