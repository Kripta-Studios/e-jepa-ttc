"""CPU checks that exact transport gates retain every producer replay failure."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
import torch

from operational.train40_runtime.measurement import prediction_parity
from scripts.benchmark_train40_compiled_pipeline import capture_replay_parity
from scripts.benchmark_train40_exact_transport import (
    check_exact_replay_coverage,
    replay_schedule,
)

VARIANTS = ("original", "reverse", "reverse-summaries")
FEATURES = ("event_raw", "event_experts", "rgb_raw", "rgb_experts")


def _exact_features() -> dict[str, Any]:
    return {
        "passed": True,
        "exact": True,
        "features": {name: {"passed": True, "exact": True} for name in FEATURES},
    }


def _prediction(value: float) -> dict[str, Any]:
    return {
        "query_id": "roi_00",
        "predictions": {
            "PHASE_ONLY_TRAIN40_S7": value,
            "E_CTX_TRAIN40": value,
            "R_CTX_TRAIN40": value,
        },
        "history_lengths": {"event": 4, "rgb": 5},
        "rgb_available": True,
        "rgb_unavailable_reason": "",
    }


def _receipts() -> list[dict[str, Any]]:
    return [
        {
            "variant": variant,
            "stage": stage,
            "iteration": iteration,
            "actual_ttc": [_prediction(1.0)],
            "parity": {"passed": True},
            "feature_parity": _exact_features(),
        }
        for variant, stage, iteration in replay_schedule(VARIANTS, warmup=1, repeats=2)
    ]


def test_schedule_covers_initial_calls_every_warmup_and_alternating_replay_order() -> None:
    assert replay_schedule(VARIANTS, warmup=2, repeats=3) == [
        ("original", "initial_call", 0),
        ("original", "warmup", 0),
        ("original", "warmup", 1),
        ("reverse", "initial_call", 0),
        ("reverse", "warmup", 0),
        ("reverse", "warmup", 1),
        ("reverse-summaries", "initial_call", 0),
        ("reverse-summaries", "warmup", 0),
        ("reverse-summaries", "warmup", 1),
        ("original", "measured", 0),
        ("reverse", "measured", 0),
        ("reverse-summaries", "measured", 0),
        ("reverse-summaries", "measured", 1),
        ("reverse", "measured", 1),
        ("original", "measured", 1),
        ("original", "measured", 2),
        ("reverse", "measured", 2),
        ("reverse-summaries", "measured", 2),
    ]


@pytest.mark.parametrize(
    "variants,warmup,repeats",
    [((), 1, 2), (("original", "original"), 1, 2), (VARIANTS, 0, 2), (VARIANTS, 1, 0)],
)
def test_schedule_rejects_empty_duplicate_or_absent_stages(
    variants: tuple[str, ...], warmup: int, repeats: int
) -> None:
    with pytest.raises(ValueError):
        replay_schedule(variants, warmup=warmup, repeats=repeats)


def test_exact_gate_accepts_only_complete_parity_receipts() -> None:
    receipts = _receipts()
    gate = check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)
    assert gate["passed"]
    assert gate["schedule_exact"]
    assert gate["observed_calls"] == gate["expected_calls"] == 12


@pytest.mark.parametrize("failed_stage", ["initial_call", "warmup", "measured"])
def test_exact_gate_keeps_early_ttc_failure_after_later_correct_calls(failed_stage: str) -> None:
    receipts = _receipts()
    expected = [_prediction(1.0)]
    for receipt in receipts:
        bad = (
            receipt["variant"] == "reverse"
            and receipt["stage"] == failed_stage
            and receipt["iteration"] == 0
        )
        actual = [_prediction(2.0 if bad else 1.0)]
        receipt["actual_ttc"] = actual
        receipt["parity"] = prediction_parity(expected, actual, ["roi_00"])
    assert all(receipt["parity"]["passed"] for receipt in receipts[-3:])
    assert all(receipt["feature_parity"]["exact"] for receipt in receipts)
    gate = check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)
    assert gate["schedule_exact"]
    assert not gate["passed"]


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("field", ["passed", "exact"])
def test_exact_gate_rejects_each_feature_failure_even_when_aggregate_claims_success(
    feature: str, field: str
) -> None:
    receipts = _receipts()
    receipts[0]["feature_parity"]["features"][feature][field] = False
    # In particular, a tolerance PASS with exact=False cannot pass this gate.
    assert receipts[0]["parity"]["passed"]
    assert receipts[0]["feature_parity"]["passed"]
    assert receipts[0]["feature_parity"]["exact"]
    gate = check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)
    assert not gate["passed"]


@pytest.mark.parametrize("feature", FEATURES)
def test_exact_gate_rejects_missing_feature_evidence(feature: str) -> None:
    receipts = _receipts()
    del receipts[0]["feature_parity"]["features"][feature]
    assert not check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)["passed"]


@pytest.mark.parametrize("field", ["passed", "exact"])
def test_exact_gate_preserves_aggregate_feature_rejection(field: str) -> None:
    receipts = _receipts()
    receipts[0]["feature_parity"][field] = False
    assert not check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)["passed"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "reordered"])
def test_exact_gate_rejects_missing_duplicate_and_reordered_calls(damage: str) -> None:
    receipts = _receipts()
    if damage == "missing":
        del receipts[1]
    elif damage == "duplicate":
        receipts.append(deepcopy(receipts[-1]))
    else:
        receipts[0], receipts[1] = receipts[1], receipts[0]
    gate = check_exact_replay_coverage(receipts, VARIANTS, warmup=1, repeats=2)
    assert not gate["schedule_exact"]
    assert not gate["passed"]


def test_replay_capture_owns_values_before_graph_storage_changes_and_returns_cpu_receipt() -> None:
    shared = torch.tensor([2.0])
    expected = [_prediction(2.0)]

    def predict(owned: Any) -> list[dict[str, Any]]:
        assert owned[0][0].untyped_storage().data_ptr() != shared.untyped_storage().data_ptr()
        shared.fill_(3.0)
        assert owned[1] == [(None, None)]
        return [_prediction(float(owned[0][0][0]))]

    receipt = capture_replay_parity(
        ((shared, shared), [(None, None)]),
        variant="reverse",
        stage="initial_call",
        iteration=0,
        predict=predict,
        compare_ttc=lambda actual: prediction_parity(expected, actual, ["roi_00"]),
        compare_features=lambda owned: _exact_features(),
    )
    shared.fill_(4.0)
    assert receipt["parity"]["passed"]
    assert receipt["actual_ttc"][0]["predictions"]["E_CTX_TRAIN40"] == 2.0
    assert json.loads(json.dumps(receipt)) == receipt
