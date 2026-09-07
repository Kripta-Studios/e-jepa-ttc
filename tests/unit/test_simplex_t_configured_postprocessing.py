"""Configured T6 input ownership and frozen binding checks; no fabricated results."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import configured_postprocessing as module


@pytest.fixture
def configured(tmp_path, monkeypatch):
    historical = tmp_path / "historical"
    shared = tmp_path / "shared"
    names = [
        "historical/FROZEN_EXPERT_TABLE_INDEX.json",
        "historical/NESTED_ANCESTRY_AUDIT.json",
        "shared/SIMPLEX_T_STAGE70_ACK.json",
        "artifacts/simplex_t/T1/risk17_frozen_replay/REPLAY.json",
        "historical/frozen_audit/extracted_input/run/stage65/ALL_RIDGE_FITS_FROZEN.json",
        *[
            f"historical/tables/outer{o}_outer_dev.{ext}"
            for o in range(3)
            for ext in ("csv", "npz")
        ],
    ]
    pins = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        pins.append({"root": "work", "relative_path": name, "sha256": sha256(path)})
    record = {"files": pins}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(record), encoding="utf-8")
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps({"worktree": str(tmp_path), "shared_coordination": str(shared)}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(module, "frozen_history_pools", lambda *args, **kwargs: {})
    source = SimpleNamespace(
        historical_root=historical,
        ancestry_sha256=sha256(historical / "NESTED_ANCESTRY_AUDIT.json"),
        allowed_sequences={"old"},
        release=lambda: calls.append("release"),
    )
    monkeypatch.setattr(
        module, "validate_scientific_admission", lambda *a, **kw: calls.append("admit")
    )

    def read(path, **kwargs):
        kwargs["validate_prerequisites"]()
        assert sha256(path) == kwargs["expected_sha256"]
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(module, "read_scientific_freeze", read)

    def open_sources(*args):
        assert calls == ["admit"]
        calls.append("sources")
        return source, {}

    monkeypatch.setattr(module, "open_acknowledged_source_configuration", open_sources)
    cohort = pd.DataFrame({"sample_token": ["old-query"]})
    receipt = {
        "folds": [
            {
                "outer_fold": outer,
                "metadata_sha256": sha256(historical / "tables" / f"outer{outer}_outer_dev.csv"),
                "arrays_sha256": sha256(historical / "tables" / f"outer{outer}_outer_dev.npz"),
            }
            for outer in range(3)
        ]
    }
    monkeypatch.setattr(module, "load_old_evaluation_cohort", lambda *a, **kw: (cohort, receipt))

    def risk(*args, **kwargs):
        assert kwargs["expected_identity"] is cohort
        calls.append("risk")
        return cohort.copy()

    monkeypatch.setattr(module, "load_acknowledged_risk17", risk)

    def post(output, **kwargs):
        assert kwargs["expected_queries"] is cohort
        kwargs["validate_authority_and_qa"]()
        assert kwargs["load_verified_risk17"]().equals(cohort)
        calls.append("post")
        return {"status": "WIRED_NOT_SCIENTIFIC_RESULTS"}

    monkeypatch.setattr(module, "postprocess_completed_campaign", post)
    args = dict(
        local_paths=local,
        source_configuration=tmp_path / "sources.json",
        source_configuration_sha256="a" * 64,
        evidence_profile=tmp_path / "qa.json",
        evidence_profile_sha256="b" * 64,
        freeze=freeze,
        freeze_sha256=sha256(freeze),
        roots={"work": tmp_path},
        phases={},
        resource_ok=lambda: True,
    )
    return tmp_path / "output", args, record, receipt, calls


def test_configured_postprocessing_independently_loads_inputs_and_releases(configured):
    output, args, _, _, calls = configured
    assert (
        module.postprocess_configured_campaign(output, **args)["status"]
        == "WIRED_NOT_SCIENTIFIC_RESULTS"
    )
    assert calls[-3:] == ["risk", "post", "release"]
    assert not output.exists()


def test_admission_failure_precedes_sources(configured, monkeypatch):
    output, args, _, _, calls = configured

    def fail(*a, **kw):
        raise ValueError("H16 missing")

    monkeypatch.setattr(module, "validate_scientific_admission", fail)
    with pytest.raises(ValueError, match="H16"):
        module.postprocess_configured_campaign(output, **args)
    assert calls == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "digest", "receipt"])
def test_changed_historical_pins_never_start_postprocessing(configured, mutation):
    output, args, record, receipt, calls = configured
    if mutation == "missing":
        record["files"].pop(0)
    elif mutation == "duplicate":
        record["files"].append(record["files"][0].copy())
    elif mutation == "digest":
        record["files"][0]["sha256"] = "f" * 64
    else:
        receipt["folds"][0]["metadata_sha256"] = "f" * 64
    args["freeze"].write_text(json.dumps(record), encoding="utf-8")
    args["freeze_sha256"] = sha256(args["freeze"])
    with pytest.raises(ValueError):
        module.postprocess_configured_campaign(output, **args)
    assert calls[-1] == "release" and "post" not in calls


def test_resource_pause_releases_constructed_sources(configured):
    output, args, _, _, calls = configured
    args["resource_ok"] = lambda: False
    with pytest.raises(InterruptedError, match="PAUSED_RESOURCE"):
        module.postprocess_configured_campaign(output, **args)
    assert calls[-1] == "release" and "post" not in calls


def test_analysis_failure_releases_sources(configured, monkeypatch):
    output, args, _, _, calls = configured

    def fail(*a, **kw):
        raise ValueError("missing sealed phase")

    monkeypatch.setattr(module, "postprocess_completed_campaign", fail)
    with pytest.raises(ValueError, match="missing sealed"):
        module.postprocess_configured_campaign(output, **args)
    assert calls[-1] == "release"
