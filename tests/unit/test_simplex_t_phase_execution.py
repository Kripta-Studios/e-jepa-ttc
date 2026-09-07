"""Fit/seal/export orchestration without any optimizer or model execution."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import phase_execution as module
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.mark.parametrize("stage", ["T2", "T3", "T4", "T5"])
@pytest.mark.parametrize("mode", ["fresh", "sealed", "pause", "gate", "bad_seal", "cohort"])
def test_phase_transition(tmp_path: Path, monkeypatch, stage: str, mode: str):
    flags = dict(
        d1=False, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    graph = registered_graph(**flags)
    record = {
        "source_contract": {"availability": flags},
        "source_identities": {fit_key(spec): {"outer_dev": "a" * 64} for spec in graph},
    }
    calls, releases = [], []
    execution = tmp_path / "execution"
    publication = tmp_path / "publication"
    endpoints = execution / f"{stage}_ENDPOINTS.json"
    if mode in {"sealed", "bad_seal"}:
        execution.mkdir()
        endpoints.write_text("synthetic seal", encoding="utf-8")
    sources = SimpleNamespace(graph=graph, release=lambda: releases.append(True))

    def read(*args, **kwargs):
        kwargs["validate_prerequisites"]()
        return record

    def authority():
        calls.append("authority")

    def gate(selected_stage, availability):
        assert selected_stage == stage and availability == flags
        calls.append("gate")
        if mode == "gate":
            raise ValueError("practical gate refused")

    def fit(*args, **kwargs):
        calls.append("fit")
        sources.release()
        if mode == "pause":
            return {"status": "PAUSED_RESOURCE"}
        execution.mkdir()
        endpoints.write_text("synthetic seal", encoding="utf-8")
        return {"status": "ENDPOINTS_SEALED_NOT_EVALUATED"}

    def seal(*args, **kwargs):
        calls.append("seal_validation")
        if mode == "bad_seal":
            raise ValueError("invalid actual endpoint")

    def export(*args, **kwargs):
        calls.append("export")
        assert "seal_validation" in calls
        assert kwargs["dev_source_hashes"] == {
            fit_key(spec): "a" * 64 for spec in graph if spec.stage == stage
        }
        kwargs["validate_prerequisites"]()
        sources.release()
        return {"status": "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"}

    monkeypatch.setattr(module, "read_scientific_freeze", read)
    monkeypatch.setattr(module, "run_frozen_phase", fit)
    monkeypatch.setattr(module, "validated_phase", seal)
    monkeypatch.setattr(module, "export_phase", export)
    options: dict[str, Any] = dict(
        sources=sources,
        freeze=tmp_path / "freeze.json",
        freeze_sha256="f" * 64,
        roots={"work": tmp_path},
        stage=stage,
        availability=flags,
        expected_queries=pd.DataFrame(
            {
                "sample_token": [str(i) for i in range(8192)],
                "sequence_id": [str(i % 9) for i in range(8192)],
                "track_id": [str(i % 10) for i in range(8192)],
                "outer_fold": [i % 3 for i in range(8192)],
            }
        )
        if mode != "cohort"
        else pd.DataFrame({"fixture": [1]}),
        validate_authority_qa_and_cohort=authority,
        validate_stage_gate=gate,
        resource_ok=lambda: True,
        resume=mode in {"sealed", "bad_seal"},
    )
    if mode in {"gate", "bad_seal", "cohort"}:
        with pytest.raises(ValueError):
            module.run_and_publish_frozen_phase(execution, publication, **options)
        assert "fit" not in calls and "export" not in calls
    else:
        result = module.run_and_publish_frozen_phase(execution, publication, **options)
        if mode == "pause":
            assert result["status"] == "PAUSED_RESOURCE"
            assert result["publication_started"] is False
            assert "seal_validation" not in calls and "export" not in calls
        else:
            assert result["status"] == "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"
            assert result["next_stage_authorized"] is False
            assert calls.count("export") == 1
            assert calls.count("fit") == (1 if mode == "fresh" else 0)
    assert len(releases) == (2 if mode == "fresh" else 1)


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "missing_pin",
        "bytes",
        "receipt",
        "mutation",
        "resource",
        "integrated",
        "integrated_t5",
        "missing_ack",
    ],
)
def test_historical_cohort_phase_binding(tmp_path: Path, monkeypatch, failure: str):
    historical = tmp_path / "historical"
    (historical / "tables").mkdir(parents=True)
    paths = [
        historical / name
        for name in (
            "NESTED_ANCESTRY_AUDIT.json",
            "FROZEN_EXPERT_TABLE_INDEX.json",
            *[f"tables/outer{o}_outer_dev.{suffix}" for o in range(3) for suffix in ("csv", "npz")],
        )
    ]
    if failure == "integrated_t5":
        paths.extend(
            [
                tmp_path / "ACK.json",
                tmp_path / "artifacts/simplex_t/T1/risk17_frozen_replay/REPLAY.json",
                historical / "frozen_audit/extracted_input/run/stage65/ALL_RIDGE_FITS_FROZEN.json",
            ]
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    pins = [
        {"root": "work", "relative_path": str(p.relative_to(tmp_path)), "sha256": sha256(p)}
        for p in paths
    ]
    if failure == "missing_pin":
        pins.pop()
    if failure == "bytes":
        paths[-1].write_text("changed", encoding="utf-8")
    record = {"files": pins}
    calls, released = [], []
    cohort = pd.DataFrame({"fixture": [1]})
    sources = SimpleNamespace(
        historical_root=historical,
        ancestry_sha256=sha256(paths[0]),
        allowed_sequences={str(i) for i in range(9)},
        release=lambda: released.append(True),
    )

    def read(*args, **kwargs):
        kwargs["validate_prerequisites"]()
        return record

    def load(root, **kwargs):
        calls.append("cohort")
        assert root == historical
        assert kwargs["table_index_sha256"] == sha256(paths[1])
        return cohort, {
            "folds": [
                {
                    "outer_fold": fold,
                    "metadata_sha256": "0" * 64
                    if failure == "receipt"
                    else sha256(paths[2 + fold * 2]),
                    "arrays_sha256": sha256(paths[3 + fold * 2]),
                }
                for fold in range(3)
            ]
        }

    def execute(*args, **kwargs):
        calls.append("execute")
        try:
            assert kwargs["expected_queries"] is cohort
            if failure == "mutation":
                paths[-1].write_text("changed", encoding="utf-8")
            kwargs["validate_authority_qa_and_cohort"]()
            kwargs["validate_stage_gate"](kwargs["stage"], kwargs["availability"])
            return {"fixture": "paused before fit"}
        finally:
            sources.release()

    monkeypatch.setattr(module, "read_scientific_freeze", read)
    monkeypatch.setattr(module, "load_old_evaluation_cohort", load)
    monkeypatch.setattr(module, "run_and_publish_frozen_phase", execute)

    def risk(*args, **kwargs):
        calls.append("risk")
        assert kwargs["expected_identity"] is cohort
        assert kwargs["ack_sha256"] == sha256(tmp_path / "ACK.json")
        return cohort

    def gate_factory(**kwargs):
        calls.append("gate_factory")
        assert kwargs["expected_queries"] is cohort
        if failure == "integrated_t5":
            kwargs["load_verified_risk17"]()
        else:
            assert kwargs["load_verified_risk17"] is None

        def gate(*args):
            calls.append("gate")

        return gate

    monkeypatch.setattr(module, "load_acknowledged_risk17", risk)
    monkeypatch.setattr(module, "stage_gate_from_publications", gate_factory)
    options: dict[str, Any] = dict(
        sources=sources,
        freeze=tmp_path / "freeze.json",
        freeze_sha256="f" * 64,
        roots={"work": tmp_path},
        stage="T2",
        availability={},
        validate_authority_and_qa=lambda: None,
        validate_stage_gate=lambda *_: None,
        resource_ok=lambda: failure != "resource",
        resume=False,
    )
    if failure in {"integrated", "integrated_t5", "missing_ack"}:
        options["validate_stage_gate"] = None
        options["publications"] = {}
        if failure != "integrated":
            options["stage"] = "T5"
        if failure == "integrated_t5":
            options["risk17_ack"] = tmp_path / "ACK.json"
    if failure not in {"", "integrated", "integrated_t5"}:
        with pytest.raises(InterruptedError if failure == "resource" else ValueError):
            module.run_historical_cohort_phase(
                tmp_path / "execution", tmp_path / "publication", **options
            )
        if failure in {"missing_pin", "bytes", "resource"}:
            assert not calls
        if failure in {"receipt", "missing_ack"}:
            assert calls == ["cohort"]
    else:
        result = module.run_historical_cohort_phase(
            tmp_path / "execution", tmp_path / "publication", **options
        )
        assert result == {"fixture": "paused before fit"}
        assert calls == (
            ["cohort", "execute"]
            if not failure
            else ["cohort", "gate_factory", "risk", "execute", "gate"]
            if failure == "integrated_t5"
            else ["cohort", "gate_factory", "execute", "gate"]
        )
    assert released == [True]
