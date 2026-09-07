"""Post-fit orchestration tests with simulated sealed component outputs only."""

import json

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import campaign_postprocessing as module
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("fault", ["none", "missing", "pause", "changed"])
def test_assembly_preserves_all_required_components(tmp_path, monkeypatch, expanded, fault):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic fixture")
    flags = dict(replicate_scalar=expanded, replicate_latent=expanded)
    coverage = dict(
        resolved_availability=flags,
        fits_completed=84 if expanded else 24,
        scientific_updates_completed=210000 if expanded else 60000,
    )
    visits = []
    verifies = []

    def verify(**kwargs):
        verifies.append(True)
        if fault == "missing":
            raise ValueError("missing sealed fits")
        return (
            dict(coverage, changed=True) if fault == "changed" and len(verifies) > 1 else coverage
        )

    def publish(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": True}))

    def t2(binding, *, output, **kwargs):
        visits.append("T2")
        publish(output / "T2_ANALYSIS.json")

    def followup(binding, *, output, stage, scalar_seed7, **kwargs):
        assert (scalar_seed7 is not None) == (stage == "T3")
        visits.append(stage)
        publish(output / "FOLLOWUP_ANALYSIS.json")

    def compact(output, *, stage, resume, **kwargs):
        assert resume is False
        visits.append("weights_" + stage)
        publish(output / "COMPACT_PHASE.json")

    def seeds(first, second, *, output, family, **kwargs):
        visits.append("seeds_" + family)
        publish(output / "THREE_SEED_ANALYSIS.json")

    def interface(output, *, phases, **kwargs):
        assert set(phases) == ({"T2", "T4"} if expanded else {"T2"})
        visits.append("interface")
        publish(output)

    for name, method in (
        ("phase_bundle_members", lambda *args, **kwargs: {}),
        ("verify_completed_scientific_graph", verify),
        ("analyze_sealed_t2", t2),
        ("analyze_followup_phase", followup),
        ("export_compact_phase", compact),
        ("analyze_three_seed_family", seeds),
        ("publish_candidate_interface", interface),
    ):
        monkeypatch.setattr(module, name, method)
    phases = {
        stage: CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, flags)
        for stage in (["T2", "T3", "T4", "T5"] if expanded else ["T2"])
    }
    output = tmp_path / "assembled"
    kwargs = dict(
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        sources=None,
        phases=phases,
        expected_queries=pd.DataFrame(),
        load_verified_risk17=pd.DataFrame,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: fault != "pause",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            module.postprocess_completed_campaign(output, **kwargs)
        assert not (output / "POSTPROCESSING.json").exists()
        if fault == "missing":
            assert not output.exists()
    else:
        result = module.postprocess_completed_campaign(output, **kwargs)
        assert set(result["analyses"]) == set(phases) == set(result["compact_weights"])
        assert set(result["three_seed"]) == ({"TPR", "LATENT"} if expanded else set())
        assert result["optimizer_updates_executed"] == 0
        assert not result["campaign_complete"] and not result["transport_bundle_complete"]
        assert len(verifies) == 2 and visits[-1] == "interface"
        inventory = result["output_inventory"]
        assert inventory["files"] == (12 if expanded else 4)
        for name, pin in inventory["members"].items():
            assert sha256(output / name) == pin["sha256"]
            assert (output / name).stat().st_size == pin["bytes"]
