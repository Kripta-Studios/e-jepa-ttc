from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.collision_clock_aggregate import aggregate_verified_frame
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    EXECUTABLE_ARMS,
    REFERENCE_FAMILIES,
    ROW_LEVEL_OOF_COLUMNS,
    canonical_records_hash,
    clipping_diagnostics,
    precheck_production_oof,
    production_sequence_macro_metrics,
)

ROW_COUNT = 8192
SEQUENCES = [f"sequence-{index}" for index in range(9)]
SEQUENCE_TO_FOLD = {sequence: index // 3 for index, sequence in enumerate(SEQUENCES)}
CHECKPOINTS = {fold: str(fold + 1) * 64 for fold in range(3)}
CONFIG_SHA256 = "c" * 64
CACHE_SHA256 = "a" * 64
SPLIT_SHA256 = "b" * 64


def _phase(ttc: np.ndarray) -> np.ndarray:
    return -np.log1p(-0.1 / ttc)


def _refresh_prediction_coordinates(frame: pd.DataFrame) -> None:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    raw = frame["predicted_ttc_raw"].to_numpy(dtype=np.float64)
    frame["target_benchmark_phase"] = _phase(target)
    frame["predicted_inverse_ttc_raw"] = 1.0 / raw
    frame["predicted_benchmark_phase"] = _phase(raw)
    frame["predicted_ttc_clipped"] = np.clip(raw, -60.0, 60.0)
    frame["is_clip_saturated"] = np.abs(raw) > 60.0
    frame["scientific_mid_per_row"] = 1.0e4 * np.abs(
        frame["target_benchmark_phase"] - frame["predicted_benchmark_phase"]
    )
    frame["scientific_failure"] = np.abs(raw) < 0.1


def _core_frame() -> pd.DataFrame:
    index = np.arange(ROW_COUNT)
    sequence = np.asarray([SEQUENCES[value % 9] for value in index], dtype=object)
    target_values = np.asarray([-1.0, 1.0, 4.0, 8.0], dtype=np.float64)
    target = target_values[(index // 9) % 4]
    raw = target * 1.01
    frame = pd.DataFrame(
        {
            "sample_token": [f"token-{value:05d}" for value in index],
            "sequence_id": sequence,
            "track_id": [f"track-{value % 37:02d}" for value in index],
            "outer_fold": np.asarray([SEQUENCE_TO_FOLD[value] for value in sequence]),
            "target_ttc_s": target,
            "predicted_ttc_raw": raw,
            "sample_weight": np.full(ROW_COUNT, 1.0 / ROW_COUNT, dtype=np.float64),
            "arm_id": "X0-BASE-U",
            "seed": np.full(ROW_COUNT, 7, dtype=np.int64),
            "checkpoint_sha256": [CHECKPOINTS[SEQUENCE_TO_FOLD[value]] for value in sequence],
            "config_sha256": CONFIG_SHA256,
            "protocol_sha256": "pending",
            "cache_manifest_sha256": CACHE_SHA256,
            "split_manifest_sha256": SPLIT_SHA256,
        }
    )
    _refresh_prediction_coordinates(frame)
    return frame


def _signed_contracts(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    bucket_names = pd.cut(
        frame["target_ttc_s"],
        bins=[-10.0, 0.0, 3.0, 6.0, 10.0],
        labels=["negative", "crucial", "small", "large"],
    ).astype(str)
    bucket_counts = {
        sequence: {
            name: int(count)
            for name, count in bucket_names[frame["sequence_id"] == sequence]
            .value_counts()
            .to_dict()
            .items()
        }
        for sequence in SEQUENCES
    }
    protocol = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_eclock_protocol_v2",
            "production_row_count": ROW_COUNT,
            "authorized_seed": 7,
            "canonical_sequence_ids": SEQUENCES,
            "canonical_sequence_to_fold": SEQUENCE_TO_FOLD,
            "canonical_bucket_counts_by_sequence": bucket_counts,
            "canonical_hashes": {
                "token_identity_sha256": canonical_records_hash(
                    frame, ("sample_token", "sequence_id", "track_id")
                ),
                "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc_s")),
                "fold_assignment_sha256": canonical_records_hash(
                    frame, ("sample_token", "sequence_id", "outer_fold")
                ),
                "sample_weight_sha256": canonical_records_hash(
                    frame, ("sample_token", "sample_weight")
                ),
            },
            "metric": {
                "metric_delta_t_s": 0.1,
                "deployment_ttc_clip_seconds": 60.0,
                "minimum_abs_prediction_ttc_s": 0.1,
            },
            "cache_binding": {"file_sha256": CACHE_SHA256},
            "split_binding": {"file_sha256": SPLIT_SHA256},
            "executable_arm_registry": list(EXECUTABLE_ARMS),
        }
    )
    reference = sign_artifact(
        {
            "artifact_type": "eclock_x0_reference_v2",
            "protocol": {"artifact_sha256": protocol["artifact_sha256"]},
            "reference_family_registry": list(REFERENCE_FAMILIES),
            "families": {name: {"reference_family": name} for name in REFERENCE_FAMILIES},
        }
    )
    bound = frame.copy()
    bound["protocol_sha256"] = protocol["artifact_sha256"]
    return protocol, reference, bound.loc[:, ROW_LEVEL_OOF_COLUMNS]


@pytest.fixture
def canonical_case() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    return _signed_contracts(_core_frame())


def _check(
    frame: pd.DataFrame, protocol: dict[str, Any], reference: dict[str, Any]
) -> pd.DataFrame:
    return precheck_production_oof(
        frame,
        protocol=protocol,
        reference=reference,
        arm_id="X0-BASE-U",
        config_sha256=CONFIG_SHA256,
        checkpoint_sha256_by_fold=CHECKPOINTS,
    )


def test_canonical_frame_passes_and_recomputes_metrics(canonical_case: Any) -> None:
    protocol, reference, frame = canonical_case
    checked = _check(frame, protocol, reference)
    metrics = production_sequence_macro_metrics(checked)
    assert len(checked) == ROW_COUNT
    assert np.isfinite(metrics["sequence_macro_paper_MiD_overall"])
    assert set(metrics["per_sequence"]) == set(SEQUENCES)


Mutation = Callable[[pd.DataFrame], None]


def _one_sequence(frame: pd.DataFrame) -> None:
    frame["sequence_id"] = SEQUENCES[0]


def _eight_sequences(frame: pd.DataFrame) -> None:
    frame.loc[frame["sequence_id"] == SEQUENCES[-1], "sequence_id"] = SEQUENCES[-2]


def _ten_sequences(frame: pd.DataFrame) -> None:
    frame.loc[0, "sequence_id"] = "sequence-extra"


def _float_folds(frame: pd.DataFrame) -> None:
    frame["outer_fold"] = frame["outer_fold"].astype(float)


def _folds_one_two_three(frame: pd.DataFrame) -> None:
    frame["outer_fold"] = frame["outer_fold"] + 1


def _folds_zero_one_three(frame: pd.DataFrame) -> None:
    frame.loc[frame["outer_fold"] == 2, "outer_fold"] = 3


def _swapped_sequence_fold(frame: pd.DataFrame) -> None:
    mask = frame["sequence_id"] == SEQUENCES[0]
    frame.loc[mask, "outer_fold"] = 1


def _extra_token(frame: pd.DataFrame) -> None:
    frame.loc[0, "sample_token"] = "token-extra"


def _missing_token(frame: pd.DataFrame) -> None:
    frame.loc[0, "sample_token"] = ""


def _duplicate_token(frame: pd.DataFrame) -> None:
    frame.loc[0, "sample_token"] = frame.loc[1, "sample_token"]


def _missing_bucket(frame: pd.DataFrame) -> None:
    mask = (frame["sequence_id"] == SEQUENCES[0]) & (frame["target_ttc_s"] == 8.0)
    frame.loc[mask, "target_ttc_s"] = 4.0
    frame.loc[mask, "predicted_ttc_raw"] = 4.04
    _refresh_prediction_coordinates(frame)


@pytest.mark.parametrize(
    "mutation",
    [
        _one_sequence,
        _eight_sequences,
        _ten_sequences,
        _float_folds,
        _folds_one_two_three,
        _folds_zero_one_three,
        _swapped_sequence_fold,
        _extra_token,
        _missing_token,
        _duplicate_token,
        _missing_bucket,
    ],
    ids=[
        "one-sequence",
        "eight-sequences",
        "ten-sequences",
        "float-fold",
        "folds-1-2-3",
        "folds-0-1-3",
        "sequence-swapped-fold",
        "token-extra",
        "token-missing",
        "token-duplicated",
        "bucket-missing",
    ],
)
def test_canonical_precheck_rejects_adversarial_identity(
    canonical_case: Any, mutation: Mutation
) -> None:
    protocol, reference, frame = canonical_case
    mutation(frame)
    with pytest.raises(ValueError):
        _check(frame, protocol, reference)


def test_self_consistent_attacker_hashes_are_not_accepted(canonical_case: Any) -> None:
    protocol, reference, frame = canonical_case
    frame["sequence_id"] = "attacker-sequence"
    frame["outer_fold"] = np.resize(np.asarray([10.5, 11.5, 12.5]), ROW_COUNT)
    attacker_claimed_hash = canonical_records_hash(
        frame, ("sample_token", "sequence_id", "track_id")
    )
    assert attacker_claimed_hash != protocol["canonical_hashes"]["token_identity_sha256"]
    with pytest.raises(ValueError):
        _check(frame, protocol, reference)


@pytest.mark.parametrize(
    ("column", "value"),
    [("seed", 23), ("protocol_sha256", "f" * 64)],
    ids=["wrong-seed", "wrong-protocol"],
)
def test_precheck_rejects_runtime_identity_mismatch(
    canonical_case: Any, column: str, value: Any
) -> None:
    protocol, reference, frame = canonical_case
    frame[column] = value
    with pytest.raises(ValueError):
        _check(frame, protocol, reference)


def test_clipping_does_not_reduce_scientific_mid(canonical_case: Any) -> None:
    protocol, reference, frame = canonical_case
    index = int(frame.index[frame["target_ttc_s"] == 8.0][0])
    frame.loc[index, "predicted_ttc_raw"] = 1000.0
    _refresh_prediction_coordinates(frame)
    checked = _check(frame, protocol, reference)
    diagnostics = clipping_diagnostics(checked)
    target_phase = float(_phase(np.asarray([8.0]))[0])
    expected_raw_mid = 1.0e4 * abs(target_phase - float(_phase(np.asarray([1000.0]))[0]))
    expected_clipped_mid = 1.0e4 * abs(target_phase - float(_phase(np.asarray([60.0]))[0]))
    assert checked.loc[index, "scientific_mid_per_row"] == pytest.approx(expected_raw_mid)
    assert expected_raw_mid > expected_clipped_mid
    assert diagnostics["rows_improved_by_clipping"] >= 1
    assert diagnostics["deployment_clipping_not_used_for_scientific_metric"] is True


def test_failure_is_recomputed_from_raw_prediction(canonical_case: Any) -> None:
    protocol, reference, frame = canonical_case
    frame.loc[0, "scientific_failure"] = True
    with pytest.raises(ValueError, match="scientific_failure"):
        _check(frame, protocol, reference)


def test_reference_protocol_binding_cannot_be_substituted(canonical_case: Any) -> None:
    protocol, reference, frame = canonical_case
    substituted = deepcopy(reference)
    substituted["protocol"]["artifact_sha256"] = "0" * 64
    sign_artifact(substituted)
    with pytest.raises(ValueError, match="different protocol"):
        _check(frame, protocol, substituted)


def _aggregate_case(
    canonical_case: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, reference, frame = canonical_case
    protocol["bootstrap"] = {
        "method": "paired_hierarchical_sequence_then_track_cluster_bootstrap",
        "seed": 20260814,
        "draws": 20,
    }
    protocol["gates"] = {
        "a5_replay_mid_tolerance": 1.0e-9,
    }
    protocol["official_a5_reference_family"] = "official_a5_oof"
    sign_artifact(protocol)
    frame["protocol_sha256"] = protocol["artifact_sha256"]
    reference["protocol"]["artifact_sha256"] = protocol["artifact_sha256"]
    sign_artifact(reference)
    checked = _check(frame, protocol, reference)
    official = checked[
        ["sample_token", "sequence_id", "track_id", "target_ttc_s", "predicted_ttc_raw"]
    ].rename(columns={"predicted_ttc_raw": "prediction_ttc_s"})
    official["prediction_ttc_s"] *= 1.02
    target = official["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = official["prediction_ttc_s"].to_numpy(dtype=np.float64)
    official_for_metric = official.copy()
    official_for_metric["scientific_mid_per_row"] = 1.0e4 * np.abs(
        _phase(target) - _phase(prediction)
    )
    official_for_metric["ttc_bucket"] = pd.cut(
        target,
        bins=[-10.0, 0.0, 3.0, 6.0, 10.0],
        labels=["negative", "crucial", "small", "large"],
    ).astype(str)
    official_mid = production_sequence_macro_metrics(official_for_metric)[
        "sequence_macro_paper_MiD_overall"
    ]
    official_family = reference["families"]["official_a5_oof"]
    official_family.update(
        {
            "prediction_sha256": canonical_records_hash(
                official, ("sample_token", "prediction_ttc_s")
            ),
            "recomputed_mid": official_mid,
            "physical_references": [
                {
                    "path": "artifacts/official-a5.csv",
                    "file_sha256": "a" * 64,
                    "bytes": 1,
                }
            ],
            "artifact_reference": {"artifact_sha256": "b" * 64},
        }
    )
    sign_artifact(reference)
    config = {
        "arm_id": "X0-BASE-U",
        "execution_authorized": True,
        "seed": 7,
        "folds": [0, 1, 2],
        "checkpoint_policy": "last_update_fixed_budget",
    }
    return frame, official, protocol, reference, config


def test_synthetic_aggregator_recomputes_and_signs_valid_evidence(
    canonical_case: Any,
) -> None:
    frame, official, protocol, reference, config = _aggregate_case(canonical_case)
    result = aggregate_verified_frame(
        frame,
        official,
        config=config,
        protocol=protocol,
        reference=reference,
        config_sha256=CONFIG_SHA256,
        checkpoint_sha256_by_fold=CHECKPOINTS,
        candidate_identity={
            "reference_family": "X0-BASE-U",
            "path": "runs/base",
            "file_sha256": "d" * 64,
            "artifact_sha256": "e" * 64,
        },
    )
    assert result["evidence_class"] == "scientific_oof"
    assert result["reference_family"] == "official_a5_oof"
    assert result["integrity_chain_complete"] is True
    assert result["bootstrap"]["paired_identical_draws"] is True


def test_a5_replay_verifies_legacy_vector_without_using_clipping_as_scientific_mid(
    canonical_case: Any,
) -> None:
    protocol, reference, frame = canonical_case
    frame.loc[0, "predicted_ttc_raw"] = 1000.0
    _refresh_prediction_coordinates(frame)
    frame, _unused, protocol, reference, config = _aggregate_case((protocol, reference, frame))
    config["arm_id"] = "X0-A5-REPLAY"
    frame["arm_id"] = "X0-A5-REPLAY"
    official = frame[
        [
            "sample_token",
            "sequence_id",
            "track_id",
            "target_ttc_s",
            "predicted_ttc_clipped",
        ]
    ].rename(columns={"predicted_ttc_clipped": "prediction_ttc_s"})
    official_metric = official.copy()
    official_metric["scientific_mid_per_row"] = 1.0e4 * np.abs(
        _phase(official_metric["target_ttc_s"].to_numpy(dtype=np.float64))
        - _phase(official_metric["prediction_ttc_s"].to_numpy(dtype=np.float64))
    )
    official_metric["ttc_bucket"] = pd.cut(
        official_metric["target_ttc_s"].to_numpy(dtype=np.float64),
        bins=[-10.0, 0.0, 3.0, 6.0, 10.0],
        labels=["negative", "crucial", "small", "large"],
    ).astype(str)
    family = reference["families"]["official_a5_oof"]
    family["prediction_sha256"] = canonical_records_hash(
        official, ("sample_token", "prediction_ttc_s")
    )
    family["recomputed_mid"] = production_sequence_macro_metrics(official_metric)[
        "sequence_macro_paper_MiD_overall"
    ]
    sign_artifact(reference)
    result = aggregate_verified_frame(
        frame,
        official,
        config=config,
        protocol=protocol,
        reference=reference,
        config_sha256=CONFIG_SHA256,
        checkpoint_sha256_by_fold=CHECKPOINTS,
        candidate_identity={
            "reference_family": "official_a5_oof",
            "path": "runs/a5-replay",
            "file_sha256": "d" * 64,
            "artifact_sha256": "e" * 64,
        },
    )
    gate = result["gate_decision"]
    assert gate["passed"] is True
    assert gate["scientific_mid_raw"] != pytest.approx(
        gate["legacy_official_replay_mid_clipped_diagnostic_only"]
    )
    assert gate["deployment_clipping_not_used_for_scientific_metric"] is True


@pytest.mark.parametrize("arm_id", ["X0-DYN-W", "unknown-arm"])
def test_aggregator_rejects_dyn_w_and_unknown_arm(canonical_case: Any, arm_id: str) -> None:
    frame, official, protocol, reference, config = _aggregate_case(canonical_case)
    config["arm_id"] = arm_id
    frame["arm_id"] = arm_id
    with pytest.raises(ValueError, match="closed scientific OOF registry"):
        aggregate_verified_frame(
            frame,
            official,
            config=config,
            protocol=protocol,
            reference=reference,
            config_sha256=CONFIG_SHA256,
            checkpoint_sha256_by_fold=CHECKPOINTS,
            candidate_identity={
                "reference_family": arm_id,
                "path": "runs/rejected",
                "file_sha256": "d" * 64,
                "artifact_sha256": "e" * 64,
            },
        )


def test_aggregator_rejects_incomplete_folds(canonical_case: Any) -> None:
    frame, official, protocol, reference, config = _aggregate_case(canonical_case)
    incomplete = frame.loc[frame["outer_fold"] != 2].copy()
    with pytest.raises(ValueError):
        aggregate_verified_frame(
            incomplete,
            official,
            config=config,
            protocol=protocol,
            reference=reference,
            config_sha256=CONFIG_SHA256,
            checkpoint_sha256_by_fold=CHECKPOINTS,
            candidate_identity={
                "reference_family": "X0-BASE-U",
                "path": "runs/incomplete",
                "file_sha256": "d" * 64,
                "artifact_sha256": "e" * 64,
            },
        )
