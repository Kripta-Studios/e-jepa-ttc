from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from operational.independent_refit import producer_contract as subject

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent


def _plan() -> dict[str, Any]:
    return subject.build_producer_plan(
        component="E_A5",
        rotation=0,
        train40_protocol_path=(
            HANDOFF
            / "e-jepa-ttc-v12-efficient-context"
            / "artifacts/train40_system_20261005/TRAINING_PROTOCOL.json"
        ),
        rgb_recipe_path=ROOT / "configs/rgb_port/producers.json",
        fold_contract_path=(
            ROOT / "output/independent_phase/train24_population_v2/FOLD_CONTRACT.json"
        ),
        raw_history_receipt_path=(
            ROOT / "output/independent_phase/raw_history_index_v4/RECEIPT.json"
        ),
        h5_preflight_path=(
            ROOT / "output/independent_phase/raw_history_h5_metadata_preflight_v4/PREFLIGHT.json"
        ),
        teacher_manifest_path=(
            ROOT / "output/independent_phase/train24_teacher_cache_v2/MANIFEST.json"
        ),
        teacher_role_receipt_path=(
            ROOT / "output/independent_phase/foundation_smoke_v2/IDENTITY_SOURCE.json"
        ),
    )


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("plan_sha256", None)
    value["plan_sha256"] = subject.canonical_sha256(unsigned)
    return value


def test_plan_is_exact_e_a5_no_drop_historical_budget() -> None:
    plan = _plan()
    subject.verify_producer_plan(plan)
    assert plan["component"] == "E_A5"
    assert plan["producer_queries"] == 35_048
    assert plan["drop_last"] is False
    assert plan["final_batch_size"] == 8
    assert plan["updates_per_epoch"] == 1_096
    assert plan["fixed_updates"] == 19_728


@pytest.mark.parametrize(
    ("mutation",),
    [
        ("learning_rate",),
        ("parents",),
        ("supervision",),
        ("source",),
    ],
)
def test_rehashed_plan_cannot_relax_scientific_contract(mutation: str) -> None:
    plan = deepcopy(_plan())
    if mutation == "learning_rate":
        plan["recipe"]["learning_rate"] = 123.0
    elif mutation == "parents":
        plan["parent_paths"] = {}
        plan["parent_sha256"] = {}
    elif mutation == "supervision":
        plan["supervision"]["mode"] = "unknown"
    else:
        plan["executed_source_sha256"]["src/e_jepa_ttc/models/causal_scale_ttc.py"] = "0" * 64
    with pytest.raises(ValueError):
        subject.verify_producer_plan(_rehash(plan))


def test_rgb_plan_is_fresh_and_remains_source_blocked() -> None:
    kwargs = {
        "component": "R_A5",
        "rotation": 0,
        "train40_protocol_path": HANDOFF
        / "e-jepa-ttc-v12-efficient-context"
        / "artifacts/train40_system_20261005/TRAINING_PROTOCOL.json",
        "rgb_recipe_path": ROOT / "configs/rgb_port/producers.json",
        "fold_contract_path": ROOT
        / "output/independent_phase/train24_population_v2/FOLD_CONTRACT.json",
        "raw_history_receipt_path": ROOT
        / "output/independent_phase/raw_history_index_v4/RECEIPT.json",
        "h5_preflight_path": ROOT
        / "output/independent_phase/raw_history_h5_metadata_preflight_v4/PREFLIGHT.json",
        "teacher_manifest_path": ROOT
        / "output/independent_phase/train24_teacher_cache_v2/MANIFEST.json",
        "teacher_role_receipt_path": ROOT
        / "output/independent_phase/foundation_smoke_v2/IDENTITY_SOURCE.json",
    }
    plan = subject.build_producer_plan(**kwargs)  # type: ignore[arg-type]
    subject.verify_producer_plan(plan)
    assert plan["component"] == "R_A5"
    assert plan["recipe"]["modality"] == "rgb"
    assert plan["recipe"]["in_channels"] == 3
    assert plan["recipe"]["fresh_initialization"] is True
    assert plan["recipe"]["learned_parent_checkpoints"] == ()
    assert plan["recipe"]["model_config"]["deterministic_transport_sampling"] is True
    assert plan["normalization"]["learned_normalizers"] == "NONE"
    assert plan["training_admitted"] is False
