"""Stage admission from actual synthetic OLD-sized predictions, with no fits."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc
from e_jepa_ttc.simplex_t.stage_gate import stage_gate_from_predictions


def flags():
    return dict(
        d1=True, density=True, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )


def pair(family, improved=True):
    ids = np.arange(8192)
    target = np.resize([-2.0, 1.0, 4.0, 10.0], len(ids))
    frame = pd.DataFrame(
        dict(
            sample_token=[f"q{i}" for i in ids],
            sequence_id=[f"s{i % 9}" for i in ids],
            track_id=ids.astype(str),
            outer_fold=ids % 3,
            target_ttc=target,
            seed=7,
        )
    )
    phase = benchmark_phase(target)
    return (
        frame.assign(
            arm=f"{family}-D1-H8-C160",
            prediction_ttc_s=scientific_ttc(phase + (0.0001 if improved else 0.002)),
        ),
        frame.assign(arm=f"{family}-D1-H1-C160", prediction_ttc_s=scientific_ttc(phase + 0.001)),
    )


def make(*, scalar=None, latent=None, risk=None, validate=lambda: None):
    return stage_gate_from_predictions(
        frozen_availability=flags(),
        scalar=scalar,
        latent=latent,
        risk17=risk,
        fraction_train_h8=(1.0, 1.0, 1.0),
        validate_frozen_publications_and_lineage=validate,
    )


@pytest.mark.parametrize("stage", ["T2", "T4"])
def test_ungated_and_technical_stage_need_no_scalar_results(stage):
    make()(stage, flags())


@pytest.mark.parametrize("stage", ["T3", "T5"])
def test_missing_prediction_evidence_is_waiting(stage):
    with pytest.raises(ValueError, match="WAITING_PRACTICAL_EVIDENCE"):
        make()(stage, flags())


def test_t3_passes_without_risk_but_t5_requires_it():
    gate = make(scalar=pair("TPR"))
    gate("T3", flags())
    with pytest.raises(ValueError, match="WAITING_PRACTICAL_EVIDENCE"):
        gate("T5", flags())


def test_t5_independent_families_and_repeated_lineage_check():
    scalar, latent = pair("TPR"), pair("LATENT", improved=False)
    changed = []

    def validate():
        if changed:
            raise ValueError("publication bytes changed")

    gate = make(scalar=scalar, latent=latent, risk=scalar[1], validate=validate)
    selected = flags() | {"replicate_latent": False}
    gate("T5", selected)
    with pytest.raises(ValueError, match="PRACTICAL_GATE_NOT_PASSED: LATENT"):
        gate("T5", flags())
    gate("T4", flags())
    changed.append(True)
    with pytest.raises(ValueError, match="publication bytes changed"):
        gate("T5", selected)


@pytest.mark.parametrize("change", [{"d1": False}, {"density": False}, {"latent": 1}])
def test_execution_cannot_change_data_pools_or_boolean_schema(change):
    with pytest.raises(ValueError, match="frozen availability"):
        make()("T2", flags() | change)
