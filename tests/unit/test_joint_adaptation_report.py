"""The report must preserve rejection, missing evidence and comparison direction."""

from copy import deepcopy

import pytest

from operational.joint_measurement import adaptation_report as report


@pytest.fixture
def evidence():
    """Synthetic report inputs, never campaign results."""
    fit, control, base = report.FIT, report.CONTROL, report.BASELINE
    metric = {
        "num_rows": 2,
        "overall_MiD": 12.0,
        "bins": {band: {"group_macro_mid": 13.0} for band in report.BANDS},
        "diagnostics": {"mae_s": 4.0, "sign_error_rate": 0.2},
        "complete_prediction_and_mid_coverage": True,
    }
    decision = {
        "overall_ratio": 0.8,
        "ratios": {band: 1.2 for band in report.BANDS},
        "eligible": False,
        "checks": {"bands": False, "sign": False, "coverage": True},
    }
    return {
        "status": "A1_READY_FOR_INTERPRETATION",
        "system_promotion": False,
        "official_comparison": False,
        "cohorts": {f"{r}/{n}": deepcopy(metric) for r in ("P", "H") for n in (fit, control, base)},
        "endpoint_proofs": {n: {"status": "PASS"} for n in (fit, control)},
        "exports": {n: {"status": "COMPLETE"} for n in (fit, control)},
        "metric_proofs": {
            f"{r}/{n}": {"status": "PASS"} for r in ("P", "H") for n in (fit, control)
        },
        "selection": {
            "status": "NO_CANDIDATE_A1",
            "ranked": [],
            "decisions": {n: deepcopy(decision) for n in (fit, control)},
            "paired_mechanism": deepcopy(decision),
        },
        "bootstrap": {
            f"{role}/{candidate}_vs_{reference}": {
                "status": "COMPLETE",
                "intervals": {"delta_overall": {"ci95_if_all_defined": [-3, 2]}},
            }
            for role in ("P", "H")
            for candidate, reference in ((fit, control), (fit, base), (control, base))
        },
        "known_scale_diagnostic": {"status": "COMPLETE_DIAGNOSTIC", "measurements": []},
        "prepared_cost": {
            "status": "COMPLETE",
            "results": {
                n: {"p50_seconds": 0.01, "p95_seconds": 0.02} for n in (fit, control, base)
            },
        },
    }


def test_rejection_and_bad_bands_survive_good_aggregate(evidence):
    rendered = report.render(evidence)
    assert "**NO_CANDIDATE_A1**" in rendered
    assert "0.800000 | 1.200000" in rendered
    assert '"sign": false' in rendered
    assert f"H/{report.CONTROL}_vs_{report.BASELINE}" in rendered
    assert "[-3, 2]" in rendered
    assert "candidato menos referencia" in rendered


def test_admitted_development_result_does_not_claim_system_promotion(evidence):
    evidence["selection"]["decisions"][report.FIT]["eligible"] = True
    evidence["selection"].update(status="DEVELOPMENT_CANDIDATE", ranked=[report.FIT])
    rendered = report.render(evidence)
    assert "**DEVELOPMENT_CANDIDATE**" in rendered
    assert "No demuestra promoción del sistema ni SOTA" in rendered


@pytest.mark.parametrize(
    "section", ["cohorts", "exports", "endpoint_proofs", "metric_proofs", "bootstrap"]
)
def test_missing_control_or_evidence_is_not_silently_dropped(evidence, section):
    evidence[section].pop(next(iter(evidence[section])))
    with pytest.raises(ValueError):
        report.render(evidence)


@pytest.mark.parametrize("fault", ["status", "ranked", "promotion", "cost", "proof"])
def test_partial_or_inconsistent_evidence_cannot_produce_complete_report(evidence, fault):
    if fault == "status":
        evidence["status"] = "TRAINING"
    elif fault == "ranked":
        evidence["selection"]["ranked"] = [report.FIT]
    elif fault == "promotion":
        evidence["system_promotion"] = True
    elif fault == "cost":
        evidence["prepared_cost"]["status"] = "PARTIAL"
    else:
        evidence["metric_proofs"][f"P/{report.FIT}"]["status"] = "FAIL"
    with pytest.raises(ValueError):
        report.render(evidence)
