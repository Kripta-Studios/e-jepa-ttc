"""Fit/seal/export orchestration without any optimizer or model execution."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

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
