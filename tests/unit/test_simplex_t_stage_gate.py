"""Stage admission from actual synthetic OLD-sized predictions, with no fits."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc
from e_jepa_ttc.simplex_t import stage_gate as module
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


@pytest.mark.parametrize("stage", ["T2", "T3", "T4", "T5"])
def test_publication_gate_reads_only_required_families(tmp_path: Path, monkeypatch, stage: str):
    reads, changed = [], []
    scalar, latent = pair("TPR"), pair("LATENT")

    def load(*args, **kwargs):
        reads.append(kwargs["family"])
        kwargs["validate_authority"]()
        if changed:
            raise ValueError("publication changed")
        return scalar if kwargs["family"] == "TPR" else latent

    def risk():
        reads.append("RISK17")
        return scalar[1]

    def with_sources(**kwargs):
        # Synthetic TRAIN support; actual frozen-source support has separate QA.
        return stage_gate_from_predictions(
            frozen_availability=kwargs["freeze"]["source_contract"]["availability"],
            scalar=kwargs["scalar"],
            latent=kwargs["latent"],
            risk17=kwargs["risk17"],
            fraction_train_h8=(1.0, 1.0, 1.0),
            validate_frozen_publications_and_lineage=kwargs[
                "validate_frozen_publications_and_lineage"
            ],
        )

    monkeypatch.setattr(module, "load_canonical_publication", load)
    monkeypatch.setattr(module, "stage_gate_from_frozen_sources", with_sources)
    publications = {
        family: module.CanonicalPublication(
            tmp_path / family, "a" * 64, tmp_path / "endpoints", "b" * 64, tmp_path, flags()
        )
        for family in ("TPR", "LATENT")
    }
    gate = module.stage_gate_from_publications(
        stage=stage,
        availability=flags(),
        freeze={"source_contract": {"availability": flags()}},
        freeze_sha256="f" * 64,
        sources=cast(module.CampaignSources, SimpleNamespace()),
        publications=publications,
        expected_queries=scalar[0],
        load_verified_risk17=risk,
        validate_frozen_sources_cohort_and_lineage=lambda: None,
        resource_ok=lambda: True,
    )
    gate(stage, flags())
    assert set(reads) == (
        {"TPR"} if stage == "T3" else {"TPR", "LATENT", "RISK17"} if stage == "T5" else set()
    )
    with pytest.raises(ValueError, match="different execution allocation"):
        gate("T4" if stage != "T4" else "T2", flags())
    if stage in {"T3", "T5"}:
        changed.append(True)
        with pytest.raises(ValueError, match="publication changed"):
            gate(stage, flags())


@pytest.mark.parametrize("missing", ["publication", "risk17"])
def test_publication_gate_missing_evidence_cannot_be_negative(tmp_path: Path, missing: str):
    publications = (
        {}
        if missing == "publication"
        else {
            family: module.CanonicalPublication(
                tmp_path, "a" * 64, tmp_path, "b" * 64, tmp_path, flags()
            )
            for family in ("TPR", "LATENT")
        }
    )
    with pytest.raises(ValueError, match="WAITING_"):
        module.stage_gate_from_publications(
            stage="T5",
            availability=flags(),
            freeze={"source_contract": {"availability": flags()}},
            freeze_sha256="f" * 64,
            sources=cast(module.CampaignSources, SimpleNamespace()),
            publications=publications,
            expected_queries=pd.DataFrame(),
            load_verified_risk17=None,
            validate_frozen_sources_cohort_and_lineage=lambda: None,
            resource_ok=lambda: True,
        )


@pytest.mark.parametrize("change", [{"d1": False}, {"latent": 1}, {"t3": False}])
def test_publication_allocation_rejected_before_missing_results(change):
    with pytest.raises(ValueError, match="frozen execution allocation"):
        module.stage_gate_from_publications(
            stage="T3",
            availability=flags() | change,
            freeze={"source_contract": {"availability": flags()}},
            freeze_sha256="f" * 64,
            sources=cast(module.CampaignSources, SimpleNamespace()),
            publications={},
            expected_queries=pd.DataFrame(),
            load_verified_risk17=None,
            validate_frozen_sources_cohort_and_lineage=lambda: None,
            resource_ok=lambda: True,
        )
