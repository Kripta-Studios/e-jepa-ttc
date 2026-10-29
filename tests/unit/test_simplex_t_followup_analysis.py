"""Fixed follow-up analysis groups; inference and bootstrap are mocked here."""

import json

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import followup_analysis as module
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize("stage", ["T3", "T4", "T5"])
@pytest.mark.parametrize("d1", [False, True])
@pytest.mark.parametrize("pause", [False, True])
def test_registered_same_seed_references(tmp_path, monkeypatch, stage, d1, pause):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic fixture")
    flags = dict(
        d1=d1, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    graph = registered_graph(**flags)

    def load(*args, stage, **kwargs):
        return {
            (spec.name, spec.seed): pd.DataFrame({"seed": [spec.seed], "arm": [spec.name]})
            for spec in graph
            if spec.stage == stage
        }

    calls = []

    def uncertainty(frames, *, reference, output, resource_check):
        assert reference.endswith("H1-C160") and reference in frames
        assert len({int(frame.seed.iloc[0]) for frame in frames.values()}) == 1
        calls.append((reference, sorted(frames)))
        resource_check()
        output.mkdir()
        (output / "PAIRED_UNCERTAINTY.json").write_text(json.dumps({"fixture": True}))

    monkeypatch.setattr(module, "load_sealed_analysis_arms", load)
    monkeypatch.setattr(module, "paired_uncertainty", uncertainty)
    monkeypatch.setattr(
        module,
        "sampled_temporal_diagnostics",
        lambda *args, **kwargs: (pd.DataFrame({"fixture": [1]}), pd.DataFrame({"fixture": [1]})),
    )
    monkeypatch.setattr(
        module, "summarize_diagnostics", lambda *args, **kwargs: pd.DataFrame({"fixture": [1]})
    )
    binding = CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, flags)
    kwargs = dict(
        stage=stage,
        output=tmp_path / "analysis",
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        expected_queries=pd.DataFrame(),
        scalar_seed7=binding if stage == "T3" else None,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: not pause,
    )
    if pause:
        with pytest.raises(InterruptedError):
            module.analyze_followup_phase(binding, **kwargs)
        assert not (tmp_path / "analysis").exists()
        return
    result = module.analyze_followup_phase(binding, **kwargs)
    assert len(calls) == (4 if stage == "T5" else 1)
    assert len(calls[0][1]) == {"T3": 5, "T4": 3, "T5": 2}[stage]
    assert result["optimizer_updates"] == 0
    assert result["controls_can_replace_primary"] is False
    assert result["ttc_ensembling_performed"] is False
