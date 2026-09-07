"""Graph closure under practical positive/negative branches; no fitted models."""

from itertools import product

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import campaign_completion as module
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize("t3,scalar,latent", list(product((False, True), repeat=3)))
def test_exact_graph_closure(tmp_path, monkeypatch, t3, scalar, latent):
    run(tmp_path, monkeypatch, t3, scalar, latent, "ok")


@pytest.mark.parametrize("fault", ["missing", "extra", "arm", "gate", "pause", "tamper"])
def test_refuses_false_completion(tmp_path, monkeypatch, fault):
    run(tmp_path, monkeypatch, False, True, False, fault)


def run(tmp_path, monkeypatch, t3, scalar, latent, fault):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic endpoint fixture")
    frozen = dict(
        d1=True, density=True, latent=True, t3=True, replicate_scalar=True, replicate_latent=True
    )
    resolved = dict(frozen, t3=t3, replicate_scalar=scalar, replicate_latent=latent)
    graph = registered_graph(**resolved)
    stages = {spec.stage for spec in graph}
    if fault == "missing":
        stages.remove("T5")
    if fault == "extra":
        stages.add("T3")
    bindings = {
        stage: CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, resolved)
        for stage in stages
    }
    monkeypatch.setattr(
        module,
        "read_scientific_freeze",
        lambda *a, **k: {"source_contract": {"availability": frozen}},
    )
    monkeypatch.setattr(
        module,
        "frozen_train_history_support",
        lambda *a, **k: {"fraction_train_h8": (1.0, 1.0, 1.0)},
    )

    def load(*args, stage, **kwargs):
        specs = registered_graph(**dict(resolved, t3=True)) if fault == "extra" else graph
        result = {(s.name, s.seed): pd.DataFrame() for s in specs if s.stage == stage}
        if fault == "arm" and stage == "T5":
            result.pop(next(iter(result)))
        return result

    monkeypatch.setattr(module, "load_sealed_analysis_arms", load)
    monkeypatch.setattr(
        module,
        "canonical_practical_decisions",
        lambda *a, family, **k: {
            "t3_practically_eligible": t3,
            "t5_practically_eligible": None
            if fault == "gate"
            else scalar
            if family == "TPR"
            else latent,
        },
    )
    if fault == "tamper":
        seal.write_text("changed")
    kwargs = dict(
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        sources=None,
        phases=bindings,
        expected_queries=pd.DataFrame(),
        load_verified_risk17=pd.DataFrame,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: fault != "pause",
    )
    if fault != "ok":
        with pytest.raises((ValueError, InterruptedError)):
            module.verify_completed_scientific_graph(**kwargs)
    else:
        result = module.verify_completed_scientific_graph(**kwargs)
        assert result["fits_completed"] == len(graph)
        assert result["scientific_updates_completed"] == len(graph) * 2500
        assert result["resolved_availability"] == resolved
        assert result["phase_fit_counts"]["T4"] == 9
        assert result["campaign_complete"] is False
