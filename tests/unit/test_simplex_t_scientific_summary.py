"""Report rendering from synthetic verified-result structures, without any fitting."""

import pytest

from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.scientific_summary import render_scientific_summary


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize(
    "fault", ["none", "count", "nan", "canonical", "family", "delta", "holdout"]
)
def test_summary_keeps_primary_identity_and_scope(expanded, fault):
    flags = dict.fromkeys(
        ("d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"), expanded
    )
    graph = registered_graph(**flags)
    pool = "D1" if expanded else "D0"
    families = ("TPR", "LATENT") if expanded else ("TPR",)
    decisions = {
        family: dict(
            canonical_candidate=f"{family}-{pool}-H8-C160",
            h1_control=f"{family}-{pool}-H1-C160",
            seed=7,
            versus_h1=dict(candidate_score=100.0, reference_score=120.0, point_delta=-20.0),
            versus_risk17=dict(candidate_score=100.0, reference_score=90.0, point_delta=10.0),
        )
        for family in families
    }
    coverage = dict(
        status="SCIENTIFIC_GRAPH_COVERAGE_VERIFIED_NOT_FINAL_DELIVERY",
        freeze_sha256="a" * 64,
        resolved_availability=flags,
        fits_completed=len(graph),
        scientific_updates_completed=2500 * len(graph),
        holdout_opened=False,
        decisions=decisions,
        phase_fit_counts={s.stage: 1 for s in graph},
    )
    accounting = dict(
        status="COMPLETE_REQUIRED_FITS_AND_WORK_ACCOUNTING_VERIFIED_NOT_FINAL_DELIVERY",
        freeze_sha256="a" * 64,
        scientific_fits_completed=len(graph),
        scientific_saved_updates=2500 * len(graph),
        campaign_complete=False,
        recorded_total_updates_lower=2500 * len(graph) + 745,
        recorded_total_updates_upper=2500 * len(graph) + 755,
        technical=dict(recorded_executed_updates=745, historical_noninstrumented_updates=85),
    )
    if fault == "count":
        accounting["scientific_fits_completed"] -= 1
    elif fault == "nan":
        decisions["TPR"]["versus_h1"]["candidate_score"] = float("nan")
    elif fault == "canonical":
        decisions["TPR"]["canonical_candidate"] = "best_control"
    elif fault == "family":
        decisions.pop("TPR")
    elif fault == "delta":
        decisions["TPR"]["versus_h1"]["point_delta"] = 20
    elif fault == "holdout":
        coverage["holdout_opened"] = True
    if fault != "none":
        with pytest.raises(ValueError):
            render_scientific_summary(coverage, accounting)
        return
    result = render_scientific_summary(coverage, accounting)
    assert "| 100 | 120 | -20 | 90 | 10 |" in result
    assert "no acredita seguimiento" in result
    assert "no ejecución observada adicional" in result
    assert "no certifica por sí solo" in result
    assert ("Tres semillas" in result) == expanded
