"""Focused synthetic tests for the isolated V8 delivery contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import e_jepa_ttc.evaluation.scientific_recovery_v8_runner as runner_module
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import (
    V8RobustnessSpec,
    apply_v8_robustness,
    benchmark_v8_delivery,
    evaluate_v8_calibration,
    evaluate_v8_robustness,
    v8_robustness_specs,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (
    FrozenV8Inputs,
    V8IntegrityError,
    assert_adaptive_gate,
)


class _TinyDeliveryModel(nn.Module):
    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=tuple(range(1, values.ndim)), keepdim=False).unsqueeze(-1) + 1.0
        return mean, torch.zeros_like(mean), torch.zeros_like(mean)


def _events() -> EventBatch:
    return EventBatch(
        x=np.asarray([0, 1, 2, 3], dtype=np.int32),
        y=np.asarray([0, 1, 2, 3], dtype=np.int32),
        t_us=np.asarray([10, 20, 30, 40], dtype=np.int64),
        polarity=np.asarray([1, -1, 1, -1], dtype=np.int8),
        width=4,
        height=4,
        sequence_id="synthetic-v8",
        t_start_us=0,
        t_end_us=50,
    )


def _representation(events: EventBatch) -> torch.Tensor:
    return torch.tensor([float(events.num_events), float(np.sum(events.polarity))])


def _records(prefix: str) -> dict[str, np.ndarray]:
    return {
        "sample_id": np.asarray([f"{prefix}-{index}" for index in range(8)]),
        "target": np.asarray([0.3, 0.5, 0.8, 1.2, 1.6, 0.7, 1.4, 2.0]),
        "prediction": np.asarray([0.35, 0.45, 0.9, 1.0, 1.7, 0.8, 1.1, 1.9]),
        "std": np.full(8, 0.2),
        "risk_logit": np.asarray([2.0, 1.0, 0.5, -0.2, -1.0, 0.9, -0.8, -1.2]),
    }


def test_v8_robustness_matrix_is_exact_and_long_windows_require_history() -> None:
    specs = v8_robustness_specs(7)
    assert len(specs) == 22
    assert {spec.kind for spec in specs} == {
        "event_dropout",
        "timestamp_jitter_us",
        "background_event_rate",
        "hot_pixel_fraction",
        "dead_pixel_fraction",
        "polarity_drop",
        "temporal_window_scale",
        "spatial_crop_fraction",
    }
    with pytest.raises(ValueError, match="temporal_history_provider"):
        apply_v8_robustness(
            _events(), V8RobustnessSpec("temporal_window_scale", 1.25, 7), sample_index=0
        )
    expanded = apply_v8_robustness(
        _events(),
        V8RobustnessSpec("temporal_window_scale", 1.25, 7),
        sample_index=0,
        temporal_history_provider=lambda events, _: events,
    )
    assert expanded.sequence_id == "synthetic-v8"


def test_robustness_preserves_targets_source_and_reports_uncertainty_delta() -> None:
    result = evaluate_v8_robustness(
        _TinyDeliveryModel(),
        [{"events": _events(), "target_ttc": 1.0, "token_id": "a"}],
        _representation,
        seed=7,
        temporal_history_provider=lambda events, _: events,
    )
    assert result["source_events_unchanged"] is True
    assert len(result["results"]) == 23
    assert all(item["target_preserved"] is True for item in result["results"])
    assert all("uncertainty_delta_s" in item for item in result["results"][1:])


def test_calibration_refuses_outer_dev_or_overlapping_ids_and_reports_metrics() -> None:
    with pytest.raises(ValueError, match="outer-dev"):
        evaluate_v8_calibration(_records("fit"), _records("eval"), fit_scope="outer_dev")
    overlap = _records("fit")
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_v8_calibration(_records("fit"), overlap, fit_scope="train")
    result = evaluate_v8_calibration(_records("fit"), _records("eval"), fit_scope="inner_oof")
    assert result["fit_evaluation_disjoint"] is True
    assert set(result["intervals"]) == {"50%", "80%", "95%"}
    assert {"ece_10_bins", "brier", "auroc", "auprc", "false_negative_rate"}.issubset(
        result["risk"]
    )


def test_benchmark_has_separated_stage_percentiles() -> None:
    raw = np.asarray([1.0, 2.0], dtype=np.float32)

    def tensorize(values: np.ndarray, _: torch.device) -> tuple[tuple[torch.Tensor, ...], int]:
        return (torch.from_numpy(values).unsqueeze(0),), 4

    result = benchmark_v8_delivery(
        _TinyDeliveryModel(),
        lambda: raw.copy(),
        tensorize,
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert result["batch_size"] == 1
    assert set(result["stages"]) == {"read", "tensorization", "inference", "total"}
    assert result["stages"]["total"]["p99_ms"] >= result["stages"]["total"]["median_ms"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_inputs() -> FrozenV8Inputs:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "configs/protocol/scientific_recovery_v8_temporal.json"
    manifest_path = (
        root / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json"
    )
    return FrozenV8Inputs(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        protocol=json.loads(protocol_path.read_text()),
        manifest=json.loads(manifest_path.read_text()),
    )


def _write_signed(path: Path, payload: dict[str, object]) -> None:
    sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bound_payload(frozen: FrozenV8Inputs, *, artifact_type: str) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "protocol_artifact_sha256": frozen.protocol["artifact_sha256"],
        "protocol_file_sha256": _sha(frozen.protocol_path),
        "sample_contract": copy.deepcopy(frozen.protocol["sample_contract"]),
        "closed_evaluation": copy.deepcopy(frozen.protocol["closed_evaluation"]),
    }


def _expected_coverage(frozen: FrozenV8Inputs) -> dict[str, list[str]]:
    return {
        str(item["fold"]): sorted(item["dev_sequence_ids"])
        for item in frozen.protocol["sample_contract"]["fold_definitions"]
    }


def _valid_sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _aggregate_contract_payload(
    frozen: FrozenV8Inputs,
    *,
    coverage: dict[str, list[str]],
    config_sha256: dict[str, str],
) -> dict[str, object]:
    counts = frozen.protocol["sample_contract"]["row_count_contract"]
    prediction_sha256 = {fold: _valid_sha(f"prediction-{fold}") for fold in coverage}
    checkpoint_sha256 = {fold: _valid_sha(f"checkpoint-{fold}") for fold in coverage}
    return {
        "schema_version": frozen.protocol["schema_version"],
        "git_commit": frozen.protocol["git_base_commit"],
        "protocol_sha256": frozen.protocol["artifact_sha256"],
        "config_sha256": config_sha256,
        "row_count": 8192,
        "row_identity_sha256": frozen.protocol["sample_contract"]["row_identity_sha256"],
        "target_sha256": frozen.protocol["sample_contract"]["target_sha256"],
        "prediction_sha256": prediction_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "folds": {
            fold: {
                "status": "completed",
                "row_count": counts["by_outer_fold"][fold],
                "prediction_sha256": prediction_sha256[fold],
                "checkpoint_sha256": checkpoint_sha256[fold],
                "sequence_ids": coverage[fold],
            }
            for fold in coverage
        },
        "metrics": {
            "mid_macro_sequence": 150.0,
            "delta_mid_vs_a5": -3.1,
            "finite_fraction": 1.0,
            "failure_rate": 0.0,
            "coverage_drop_max_pp": 0.0,
        },
        "per_sequence": {
            sequence: {
                "mid_macro_sequence": 150.0,
                "delta_mid_vs_a5": -3.1,
                "row_count": counts["by_sequence"][sequence],
            }
            for values in coverage.values()
            for sequence in values
        },
        "per_bucket": {
            bucket: {
                "mid_macro_sequence": 150.0,
                "delta_mid_vs_a5": -3.1,
                "row_count": counts["by_bucket"][bucket],
            }
            for bucket in ("crucial", "small", "large", "negative")
        },
        "bootstrap": {
            "probability_delta_lt_zero": 0.91,
            "ci95_low": -7.0,
            "ci95_high": -2.0,
            "resamples": 5000,
        },
        "integrity_checks": {"rows": True, "folds": True, "targets": True},
    }


def _autopsy_factorial_cell(
    frozen: FrozenV8Inputs, coverage: dict[str, list[str]], seed: str
) -> dict[str, object]:
    counts = frozen.protocol["sample_contract"]["row_count_contract"]
    return {
        "row_count": 8192,
        "row_identity_sha256": frozen.protocol["sample_contract"]["row_identity_sha256"],
        "target_sha256": frozen.protocol["sample_contract"]["target_sha256"],
        "prediction_sha256": _valid_sha(seed),
        "metrics": {
            "mid_macro_sequence": 150.0,
            "delta_mid_vs_a5": -3.1,
            "delta_residual_vs_analytic": -1.0,
            "delta_transport_vs_without_transport": -1.0,
            "delta_history_vs_without_history": -1.0,
        },
        "per_sequence": {
            sequence: {
                "mid_macro_sequence": 150.0,
                "delta_mid_vs_a5": -3.1,
                "row_count": counts["by_sequence"][sequence],
            }
            for values in coverage.values()
            for sequence in values
        },
        "per_bucket": {
            bucket: {
                "mid_macro_sequence": 150.0,
                "delta_mid_vs_a5": -3.1,
                "row_count": counts["by_bucket"][bucket],
            }
            for bucket in ("crucial", "small", "large", "negative")
        },
        "coverage": {"outer_folds": [0, 1, 2], "sequences_by_outer_fold": coverage},
        "integrity_checks": {"rows": True, "predictions": True},
    }


def _write_c1_artifacts(
    tmp_path: Path,
    frozen: FrozenV8Inputs,
    *,
    route: str = "router_regime",
) -> Path:
    root = tmp_path
    diagnostics = root / "artifacts/scientific_recovery_v8/diagnostics"
    plan_ref = copy.deepcopy(frozen.manifest["c1_analysis_plans"][route])
    source_root = Path(__file__).resolve().parents[2]
    source_plan_path = source_root / plan_ref["path"]
    plan_path = root / plan_ref["path"]
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(source_plan_path.read_bytes())
    plan = json.loads(plan_path.read_text())
    source_contract = plan["source_aggregate_contract"]
    coverage = _expected_coverage(frozen)
    all_sequences = [sequence for values in coverage.values() for sequence in values]
    evidence = _bound_payload(frozen, artifact_type="scientific_recovery_v8_regime_evidence_v1")
    evidence["coverage"] = {"outer_folds": [0, 1, 2], "sequences_by_outer_fold": coverage}
    if route == "router_regime":
        evidence["stable_temporal_density_feature_dependence"] = {
            "features": ["event_rate", "flow"],
            "stable_by_outer_fold": {"0": True, "1": True, "2": True},
            "stable_by_sequence": {sequence: True for sequence in all_sequences},
        }
    elif route == "autopsy_h3":
        evidence["mechanism_decision"] = "H3"
    else:
        evidence["exp6_stable_regime_heterogeneity"] = {
            "preregistered": True,
            "stable_by_outer_fold": {"0": True, "1": True, "2": True},
            "stable_by_sequence": {sequence: True for sequence in all_sequences},
        }
    evidence_path = diagnostics / "c1_evidence.json"
    _write_signed(evidence_path, evidence)
    invariance = _bound_payload(frozen, artifact_type="scientific_recovery_v8_causal_invariance_v1")
    invariance["passed"] = True
    invariance_path = diagnostics / "c1_causality.json"
    _write_signed(invariance_path, invariance)
    aggregate = _bound_payload(frozen, artifact_type=source_contract["artifact_type"])
    aggregate.update(source_contract)
    aggregate["status"] = "completed"
    aggregate["coverage"] = {
        "outer_folds": [0, 1, 2],
        "sequences_by_outer_fold": coverage,
    }
    config_sha256 = source_contract.get(
        "config_sha256_by_fold", {fold: _valid_sha(f"config-{route}-{fold}") for fold in coverage}
    )
    aggregate.update(
        _aggregate_contract_payload(
            frozen,
            coverage=coverage,
            config_sha256=config_sha256,
        )
    )
    if route == "autopsy_h3":
        aggregate["mechanism_decision"] = "H3"
    else:
        aggregate["gate_decision"] = {"passed": True}
    if route == "autopsy_h3":
        factorial = _bound_payload(
            frozen, artifact_type="scientific_recovery_v8_autopsy_factorial_replay_v1"
        )
        factorial["status"] = "completed"
        factorial["factorial_cells"] = {}
        definitions = source_contract["factorial_replay_schema"]["combination_definitions"]
        for combination, settings in definitions.items():
            cell = _autopsy_factorial_cell(frozen, coverage, combination)
            cell["settings"] = settings
            factorial["factorial_cells"][combination] = cell
        factorial["output_hashes"] = {"replay_predictions": _valid_sha("factorial-output")}
        factorial_path = diagnostics / "c1_factorial_replay.json"
        _write_signed(factorial_path, factorial)
        diagnostic = _bound_payload(
            frozen, artifact_type="scientific_recovery_v8_autopsy_diagnostic_v1"
        )
        diagnostic["status"] = "completed"

        def diagnostic_record() -> dict[str, object]:
            return {"effect_size": 1.0, "evidence_present": True, "stable": True}

        diagnostic["by_ttc_bucket"] = {
            bucket: diagnostic_record() for bucket in ("crucial", "small", "large", "negative")
        }
        diagnostic["by_sequence"] = {sequence: diagnostic_record() for sequence in all_sequences}
        diagnostic["by_event_density"] = {
            label: diagnostic_record() for label in ("low", "medium", "high")
        }
        diagnostic["by_movement"] = {
            label: diagnostic_record() for label in ("low", "medium", "high")
        }
        diagnostic["by_sign"] = {label: diagnostic_record() for label in ("negative", "positive")}
        diagnostic["decision_inputs"] = {
            name: name
            in {
                "complementarity_present",
                "causal_regime_predictability_passed",
                "stable_across_outer_folds",
                "stable_across_sequences",
                "innocuous_change_invariance_passed",
            }
            for name in source_contract["diagnostic_schema"]["decision_rule"]["inputs"]
        }
        diagnostic["decision_rule_output"] = "H3"
        diagnostic["final_decision"] = "H3"
        diagnostic["integrity_checks"] = {"replay_binding": True, "coverage": True}
        diagnostic["output_hashes"] = {"diagnostic_table": _valid_sha("diagnostic-output")}
        diagnostic_path = diagnostics / "c1_autopsy_diagnostic.json"
        _write_signed(diagnostic_path, diagnostic)
        aggregate["autopsy_outputs"] = {
            "factorial_replay": {
                "path": factorial_path.relative_to(root).as_posix(),
                "sha256": _sha(factorial_path),
                "artifact_sha256": factorial["artifact_sha256"],
            },
            "diagnostic": {
                "path": diagnostic_path.relative_to(root).as_posix(),
                "sha256": _sha(diagnostic_path),
                "artifact_sha256": diagnostic["artifact_sha256"],
            },
        }
    aggregate_path = diagnostics / "c1_source_aggregate.json"
    _write_signed(aggregate_path, aggregate)
    opening = _bound_payload(frozen, artifact_type="scientific_recovery_v8_c1_opening_decision_v1")
    opening["arm"] = source_contract["arm"]
    opening["opening_route"] = route
    opening["evidence_refs"] = {
        "analysis_plan": plan_ref,
        "source_aggregate": {
            "path": aggregate_path.relative_to(root).as_posix(),
            "sha256": _sha(aggregate_path),
        },
        "regime_evidence": {
            "path": evidence_path.relative_to(root).as_posix(),
            "sha256": _sha(evidence_path),
        },
        "causal_invariance": {
            "path": invariance_path.relative_to(root).as_posix(),
            "sha256": _sha(invariance_path),
        },
    }
    opening_path = root / "results/opening.json"
    _write_signed(opening_path, opening)
    return opening_path


def _refresh_ref(opening_path: Path, key: str, artifact_path: Path) -> None:
    opening = json.loads(opening_path.read_text())
    root = opening_path.parents[1]
    opening["evidence_refs"][key] = {
        "path": artifact_path.relative_to(root).as_posix(),
        "sha256": _sha(artifact_path),
    }
    _write_signed(opening_path, opening)


@pytest.mark.parametrize("route", ["autopsy_h3", "exp6_regime", "router_regime"])
def test_adaptive_gate_accepts_exact_bound_preregistered_evidence(
    tmp_path, monkeypatch, route: str
) -> None:
    frozen = _frozen_inputs()
    _write_c1_artifacts(tmp_path, frozen, route=route)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    assert_adaptive_gate(results_root=tmp_path / "results", frozen=frozen)


def test_adaptive_gate_rejects_runtime_preregistration_without_frozen_plan(
    tmp_path, monkeypatch
) -> None:
    frozen = _frozen_inputs()
    opening_path = _write_c1_artifacts(tmp_path, frozen)
    runtime_plan = _bound_payload(
        frozen, artifact_type="scientific_recovery_v8_preregistered_analysis_plan_v1"
    )
    runtime_plan.update({"plan_id": "router_regime", "preregistered": True})
    runtime_plan_path = tmp_path / "artifacts/scientific_recovery_v8/diagnostics/runtime_plan.json"
    _write_signed(runtime_plan_path, runtime_plan)
    _refresh_ref(opening_path, "analysis_plan", runtime_plan_path)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    with pytest.raises(V8IntegrityError, match="C1/adaptive is closed"):
        assert_adaptive_gate(results_root=tmp_path / "results", frozen=frozen)


def test_runner_exposes_multiseed_replication_without_confirmation_alias() -> None:
    assert "multiseed_replication" in runner_module.STAGES
    assert "confirm" not in runner_module.STAGES
    command = runner_module._script_command(
        "multiseed_replication", device="cpu", candidate="frozen-a5"
    )
    assert "scripts/run_scientific_recovery_v8_multiseed_replication.py" in command


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "tampered",
        "wrong_protocol",
        "wrong_sequences",
        "extra_fold",
        "malformed_fold_key",
        "source_failed",
        "source_wrong_arm",
        "source_wrong_seed",
        "source_incomplete",
        "source_missing_metrics",
        "source_nonfinite",
        "source_failing_numbers",
        "source_bad_integrity",
        "source_bad_coverage",
        "source_minimal",
        "source_missing_schema",
        "source_row_count",
        "source_missing_fold",
        "source_missing_sequence",
        "source_missing_bucket",
        "source_invalid_hash",
        "source_invalid_checkpoint",
        "source_wrong_config",
        "source_coverage_drop",
        "source_wrong_fold_distribution",
        "source_empty_metric",
        "source_string_metric",
        "source_empty_group_record",
        "absolute_ref",
    ],
)
def test_adaptive_gate_rejects_unbound_or_tampered_evidence(
    tmp_path, monkeypatch, failure: str
) -> None:
    frozen = _frozen_inputs()
    opening_path = _write_c1_artifacts(tmp_path, frozen)
    opening = json.loads(opening_path.read_text())
    evidence_path = tmp_path / opening["evidence_refs"]["regime_evidence"]["path"]
    source_path = tmp_path / opening["evidence_refs"]["source_aggregate"]["path"]
    if failure == "missing":
        (tmp_path / frozen.manifest["c1_analysis_plans"]["router_regime"]["path"]).unlink()
    elif failure.startswith("source_"):
        source = json.loads(source_path.read_text())
        if failure == "source_failed":
            source["gate_decision"] = {"passed": False}
        elif failure == "source_wrong_arm":
            source["arm"] = "wrong-arm"
        elif failure == "source_wrong_seed":
            source["seed"] = 13
        elif failure == "source_incomplete":
            source["status"] = "running"
        elif failure == "source_missing_metrics":
            source.pop("metrics")
        elif failure == "source_nonfinite":
            source["metrics"]["delta_mid_vs_a5"] = float("nan")
        elif failure == "source_failing_numbers":
            source["metrics"]["delta_mid_vs_a5"] = -2.9
        elif failure == "source_bad_integrity":
            source["sample_contract"]["row_identity_sha256"] = "0" * 64
        elif failure == "source_bad_coverage":
            source["coverage"]["sequences_by_outer_fold"]["0"] = ["wrong-sequence"]
        elif failure == "source_minimal":
            source = _bound_payload(frozen, artifact_type=source["artifact_type"])
            source.update({"stage": "router", "arm": "router", "candidate_id": "R", "seed": 7})
        elif failure == "source_missing_schema":
            source.pop("metrics")
        elif failure == "source_row_count":
            source["row_count"] = 8191
        elif failure == "source_missing_fold":
            source["folds"].pop("2")
        elif failure == "source_missing_sequence":
            source["per_sequence"].pop(next(iter(source["per_sequence"])))
        elif failure == "source_missing_bucket":
            source["per_bucket"].pop("negative")
        elif failure == "source_invalid_hash":
            source["prediction_sha256"]["0"] = "not-a-sha"
        elif failure == "source_invalid_checkpoint":
            source["checkpoint_sha256"]["0"] = "not-a-sha"
        elif failure == "source_wrong_config":
            source["config_sha256"]["0"] = _valid_sha("wrong-config")
        elif failure == "source_wrong_fold_distribution":
            source["folds"]["0"]["row_count"] = 1
            source["folds"]["1"]["row_count"] = 1
            source["folds"]["2"]["row_count"] = 8190
        elif failure == "source_empty_metric":
            source["metrics"] = {}
        elif failure == "source_string_metric":
            source["metrics"]["mid_macro_sequence"] = "150"
        elif failure == "source_empty_group_record":
            source["per_bucket"]["negative"] = {}
        else:
            source["metrics"]["coverage_drop_max_pp"] = 1.1
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    elif failure == "absolute_ref":
        opening["evidence_refs"]["source_aggregate"]["path"] = str(source_path.resolve())
        _write_signed(opening_path, opening)
    else:
        evidence = json.loads(evidence_path.read_text())
        if failure == "tampered":
            evidence["stable_temporal_density_feature_dependence"]["features"] = ["uncertainty"]
        elif failure == "wrong_protocol":
            evidence["protocol_artifact_sha256"] = "0" * 64
        elif failure == "wrong_sequences":
            evidence["coverage"]["sequences_by_outer_fold"]["0"] = ["invented-sequence"]
        elif failure == "extra_fold":
            evidence["coverage"]["sequences_by_outer_fold"]["3"] = ["invented-sequence"]
        else:
            values = evidence["coverage"]["sequences_by_outer_fold"].pop("0")
            evidence["coverage"]["sequences_by_outer_fold"]["00"] = values
        _write_signed(evidence_path, evidence)
        _refresh_ref(opening_path, "regime_evidence", evidence_path)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    with pytest.raises(V8IntegrityError, match="C1/adaptive is closed"):
        assert_adaptive_gate(results_root=tmp_path / "results", frozen=frozen)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "tampered",
        "missing_cell",
        "missing_dimension",
        "decision_mismatch",
        "empty_factorial",
        "empty_diagnostic",
        "wrong_settings",
        "string_effect",
        "rule_input_mismatch",
        "wrong_factorial_row_count",
    ],
)
def test_adaptive_gate_rejects_autopsy_without_bound_factorial_outputs(
    tmp_path, monkeypatch, failure: str
) -> None:
    frozen = _frozen_inputs()
    opening_path = _write_c1_artifacts(tmp_path, frozen, route="autopsy_h3")
    opening = json.loads(opening_path.read_text())
    source_path = tmp_path / opening["evidence_refs"]["source_aggregate"]["path"]
    source = json.loads(source_path.read_text())
    replay_path = tmp_path / source["autopsy_outputs"]["factorial_replay"]["path"]
    diagnostic_path = tmp_path / source["autopsy_outputs"]["diagnostic"]["path"]
    if failure == "missing":
        replay_path.unlink()
    elif failure == "tampered":
        replay = json.loads(replay_path.read_text())
        replay["status"] = "failed"
        _write_signed(replay_path, replay)
    elif failure == "missing_cell":
        replay = json.loads(replay_path.read_text())
        replay["factorial_cells"].pop("analytic_transport")
        _write_signed(replay_path, replay)
        source["autopsy_outputs"]["factorial_replay"].update(
            {"sha256": _sha(replay_path), "artifact_sha256": replay["artifact_sha256"]}
        )
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    elif failure == "empty_factorial":
        replay = json.loads(replay_path.read_text())
        replay["factorial_cells"] = {}
        _write_signed(replay_path, replay)
        source["autopsy_outputs"]["factorial_replay"].update(
            {"sha256": _sha(replay_path), "artifact_sha256": replay["artifact_sha256"]}
        )
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    elif failure == "wrong_settings":
        replay = json.loads(replay_path.read_text())
        replay["factorial_cells"]["analytic_only"]["settings"]["analytic"] = False
        _write_signed(replay_path, replay)
        source["autopsy_outputs"]["factorial_replay"].update(
            {"sha256": _sha(replay_path), "artifact_sha256": replay["artifact_sha256"]}
        )
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    elif failure == "wrong_factorial_row_count":
        replay = json.loads(replay_path.read_text())
        replay["factorial_cells"]["analytic_only"]["per_bucket"]["negative"]["row_count"] += 1
        _write_signed(replay_path, replay)
        source["autopsy_outputs"]["factorial_replay"].update(
            {"sha256": _sha(replay_path), "artifact_sha256": replay["artifact_sha256"]}
        )
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    else:
        diagnostic = json.loads(diagnostic_path.read_text())
        if failure == "missing_dimension":
            diagnostic.pop("by_sign")
        elif failure == "decision_mismatch":
            diagnostic["final_decision"] = "H2"
        elif failure == "string_effect":
            diagnostic["by_sign"]["positive"]["effect_size"] = "1.0"
        elif failure == "rule_input_mismatch":
            diagnostic["decision_inputs"]["complementarity_present"] = False
        else:
            diagnostic["by_sequence"] = {}
        _write_signed(diagnostic_path, diagnostic)
        source["autopsy_outputs"]["diagnostic"].update(
            {"sha256": _sha(diagnostic_path), "artifact_sha256": diagnostic["artifact_sha256"]}
        )
        _write_signed(source_path, source)
        _refresh_ref(opening_path, "source_aggregate", source_path)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    with pytest.raises(V8IntegrityError, match="C1/adaptive is closed"):
        assert_adaptive_gate(results_root=tmp_path / "results", frozen=frozen)
