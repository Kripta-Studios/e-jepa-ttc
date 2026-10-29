"""TDD contracts for the V8 D0--D4 attribution stage."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.scientific_recovery_v8_jepa_attribution import (
    JEPACausalGateConfig,
    classify_jepa_causal_gate,
    nested_low_label_tokens,
    validate_equal_compute,
)


def test_nested_low_label_tokens_are_track_balanced_nested_and_deterministic() -> None:
    rows = [
        {"sample_token": f"a-{index}", "sequence_id": "s0", "track_id": "a"} for index in range(20)
    ] + [
        {"sample_token": f"b-{index}", "sequence_id": "s1", "track_id": "b"} for index in range(20)
    ]
    first = nested_low_label_tokens(rows, fractions=(0.05, 0.25, 1.0), seed=7)
    second = nested_low_label_tokens(rows, fractions=(0.05, 0.25, 1.0), seed=7)
    assert first == second
    assert set(first[0.05]).issubset(first[0.25])
    assert set(first[0.25]).issubset(first[1.0])
    assert set(first[1.0]) == {row["sample_token"] for row in rows}
    assert {token.split("-")[0] for token in first[0.05]} == {"a", "b"}


def test_equal_compute_rejects_any_d2_d4_difference_except_pairing() -> None:
    d2 = {
        "seed": 7,
        "total_updates": 4,
        "batch_schedule_sha256": "a" * 64,
        "model_initialization_sha256": "b" * 64,
        "trainer_config_sha256": "c" * 64,
        "compute_manifest_sha256": "d" * 64,
        "shuffled_future": False,
    }
    d4 = {**d2, "shuffled_future": True}
    assert validate_equal_compute(d2, d4)["passed"] is True
    with pytest.raises(ValueError, match="batch_schedule_sha256"):
        validate_equal_compute(d2, {**d4, "batch_schedule_sha256": "e" * 64})


def test_jepa_causal_gate_requires_low_label_and_all_three_controls() -> None:
    config = JEPACausalGateConfig()
    common = {
        "low_label_auc_mid": 10.0,
        "scratch_low_label_auc_mid": 14.0,
        "random_frozen_low_label_auc_mid": 14.0,
        "shuffled_future_low_label_auc_mid": 14.0,
        "paired_ci95_high_vs_scratch": -0.01,
        "full_label_delta_mid_vs_scratch": 2.9,
        "all_finite": True,
        "failure_rate": 0.0,
    }
    passed = classify_jepa_causal_gate(common, config=config)
    assert passed["causally_positive"] is True
    failed = classify_jepa_causal_gate(
        {**common, "shuffled_future_low_label_auc_mid": 9.0}, config=config
    )
    assert failed["causally_positive"] is False
    assert failed["gates"]["better_than_d4"] is False


def test_jepa_aggregate_uses_signed_three_fold_oof_and_bootstrap_upper_bound(
    tmp_path: Path,
) -> None:
    fractions = ("0.01", "0.05", "0.1", "0.25", "1.0")
    for arm in ("D0", "D1", "D2", "D3", "D4"):
        for fold in range(3):
            rows = [
                {
                    "token_id": f"{fold}-{fraction}",
                    "sequence_id": f"sequence-{fold}",
                    "track_id": f"track-{fold}",
                    "outer_fold": fold,
                    "target_ttc": 2.0,
                    "sample_weight": 1.0,
                    "prediction_ttc": 2.0 if arm in {"D2", "D3"} else 2.2,
                    "finite": True,
                }
                for fraction in fractions
            ]
            payload = {
                "artifact_type": "scientific_recovery_v8_jepa_oof_predictions_v1",
                "status": "completed",
                "arm": arm,
                "fractions": {fraction: [rows[index]] for index, fraction in enumerate(fractions)},
            }

            def hash_value(value: object) -> str:
                return hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()

            def row_hashes(items: list[dict[str, object]]) -> dict[str, str]:
                ordered = sorted(items, key=lambda item: str(item["token_id"]))
                return {
                    "row_identity_sha256": hash_value(
                        [
                            (item["token_id"], item["sequence_id"], item["track_id"])
                            for item in ordered
                        ]
                    ),
                    "target_sha256": hash_value(
                        [(item["token_id"], item["target_ttc"]) for item in ordered]
                    ),
                    "fold_sha256": hash_value(
                        [(item["token_id"], item["outer_fold"]) for item in ordered]
                    ),
                    "sample_weight_sha256": hash_value(
                        [(item["token_id"], item["sample_weight"]) for item in ordered]
                    ),
                }

            payload["oof_contract_hashes"] = {
                fraction: row_hashes([rows[index]]) for index, fraction in enumerate(fractions)
            }
            sign_artifact(payload)
            destination = tmp_path / arm.lower() / f"fold{fold}" / "seed7"
            destination.mkdir(parents=True)
            (destination / "oof_predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    aggregate = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2] / "scripts/aggregate_scientific_recovery_v8_jepa.py"
        )
    )
    output = tmp_path / "aggregate.json"
    # The fixture has one track per sequence, which is valid for the hierarchical bootstrap.
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["aggregate", "--results-root", str(tmp_path), "--output", str(output)]
        assert aggregate["main"]() == 0
    finally:
        sys.argv = old_argv
    assert verify_artifact_hash(json.loads(output.read_text(encoding="utf-8")))
