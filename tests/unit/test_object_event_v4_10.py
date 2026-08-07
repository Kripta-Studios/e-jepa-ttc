from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.aggregate_object_event_v4_10_multiseed import aggregate
from scripts.prepare_object_event_v4_10_seed_configs import materialize
from scripts.preflight_object_event_v4_10 import (
    CRITICAL_V49_HASHES,
    _sha256,
    assess_v42_replication_baseline,
)
from scripts.preflight_object_event_v4_9 import assess_v42_fusion_baseline
from e_jepa_ttc.object_event_v4_9 import FixedFusionConfig
from e_jepa_ttc.object_event_v4_10 import align_seed_fusions, pairwise_seed_metrics


def _v42_payload(*, screen_passed: bool, failed_gates: list[str] | None = None) -> dict[str, object]:
    failed = set(failed_gates or [])
    gate_names = (
        "pearson",
        "pearson_lower_ci",
        "sequence_macro_pearson",
        "all_sequences_positive",
        "balanced_sign",
        "negative_accuracy",
        "expansion_mae",
        "saturation",
        "zero_event_dependence",
        "shuffled_event_dependence",
        "shuffled_event_change",
    )
    return {
        "screen_passed": screen_passed,
        "selection_gates": {name: name not in failed for name in gate_names},
        "validation_metrics": {
            "event": {
                "pearson": 0.598,
                "balanced_sign_accuracy": 0.706,
                "negative_accuracy": 0.540,
                "expansion_mae": 0.0159,
                "ttc_saturation_rate": 0.0308,
            },
            "per_sequence": {"minimum_pearson": 0.468},
            "event_dependence": {
                "zero_event_pearson_drop": 0.598,
                "shuffled_event_pearson_drop": 0.606,
            },
        },
    }


def test_v42_replication_accepts_originally_passed_seed() -> None:
    result = assess_v42_replication_baseline(_v42_payload(screen_passed=True))
    assert result["accepted_for_replication"] is True
    assert result["acceptance_reason"] == "screen_passed"


def test_v42_replication_accepts_negative_accuracy_only_marginal_seed() -> None:
    result = assess_v42_replication_baseline(
        _v42_payload(screen_passed=False, failed_gates=["negative_accuracy"])
    )
    assert result["accepted_for_replication"] is True
    assert result["original_screen_passed"] is False
    assert result["acceptance_reason"] == "marginal_negative_accuracy_only"


def test_v42_replication_rejects_multiple_failures_or_below_chance_negative_accuracy() -> None:
    multiple = assess_v42_replication_baseline(
        _v42_payload(
            screen_passed=False,
            failed_gates=["negative_accuracy", "balanced_sign"],
        )
    )
    assert multiple["accepted_for_replication"] is False

    payload = _v42_payload(screen_passed=False, failed_gates=["negative_accuracy"])
    payload["validation_metrics"]["event"]["negative_accuracy"] = 0.49  # type: ignore[index]
    below_chance = assess_v42_replication_baseline(payload)  # type: ignore[arg-type]
    assert below_chance["accepted_for_replication"] is False


def test_v49_preflight_keeps_marginal_exception_explicit_and_narrow() -> None:
    marginal = _v42_payload(
        screen_passed=False, failed_gates=["negative_accuracy"]
    )
    strict = assess_v42_fusion_baseline(
        marginal, allow_marginal_negative_accuracy_only=False
    )
    replication = assess_v42_fusion_baseline(
        marginal, allow_marginal_negative_accuracy_only=True
    )
    assert strict["accepted_for_fusion"] is False
    assert replication["accepted_for_fusion"] is True
    assert replication["original_screen_passed"] is False
    assert replication["acceptance_reason"] == "marginal_negative_accuracy_only"

    multiple = assess_v42_fusion_baseline(
        _v42_payload(
            screen_passed=False,
            failed_gates=["negative_accuracy", "balanced_sign"],
        ),
        allow_marginal_negative_accuracy_only=True,
    )
    assert multiple["accepted_for_fusion"] is False


def test_v49_preflight_hash_contract_matches_current_file() -> None:
    relative = "scripts/preflight_object_event_v4_9.py"
    assert CRITICAL_V49_HASHES[relative] == _sha256(Path(relative))


def test_v49_analyzer_hash_contract_matches_current_file() -> None:
    relative = "scripts/analyze_object_event_v4_9_fixed_fusion.py"
    assert CRITICAL_V49_HASHES[relative] == _sha256(Path(relative))


def test_v49_analyzer_uses_same_explicit_marginal_contract() -> None:
    analyzer = Path("scripts/analyze_object_event_v4_9_fixed_fusion.py").read_text(
        encoding="utf-8"
    )
    assert "assess_v42_fusion_baseline" in analyzer
    assert '"--allow-marginal-v42-negative-accuracy-only"' in analyzer
    assert 'if not bool(v42_summary.get("screen_passed"))' not in analyzer


def test_runner_tolerates_only_diagnostic_v46_v47_overfit_failures() -> None:
    script = Path("scripts/run_object_event_v4_10_multiseed.ps1").read_text(
        encoding="utf-8"
    )
    assert (
        'Invoke-PythonChecked -Arguments $V46OverfitArgs -AllowedExitCodes @(0, 2)'
        in script
    )
    assert (
        'Invoke-PythonChecked -Arguments $V47OverfitArgs -AllowedExitCodes @(0, 2)'
        in script
    )
    assert (
        'Invoke-PythonChecked -Arguments $V48OverfitArgs -AllowedExitCodes @(0, 2)'
        not in script
    )
    assert (
        'Invoke-PythonChecked -Arguments $V48ScreenArgs -AllowedExitCodes @(0, 2)'
        not in script
    )
    assert script.count('"--allow-marginal-v42-negative-accuracy-only"') == 2
    assert (
        'Invoke-PythonChecked -Arguments $V49Args -AllowedExitCodes @(0, 2)'
        in script
    )
    assert (
        '$AllowedStatuses = @("fusion_screen_passed", "fusion_screen_failed")'
        in script
    )
    assert '"train_predictions.csv", "validation_predictions.csv"' in script


def _frame(offset: float = 0.0, count: int = 30) -> pd.DataFrame:
    target = np.linspace(-0.04, 0.06, count)
    prediction = target * 0.9 + offset
    return pd.DataFrame(
        {
            "sequence_id": [f"seq-{index % 3}" for index in range(count)],
            "sample_token": [f"sample-{index:03d}" for index in range(count)],
            "track_id": [f"track-{index % 5}" for index in range(count)],
            "delta_t_s": np.full(count, 0.05),
            "target_ttc_s": np.where(np.abs(target) > 1.0e-5, 0.05 / target, 60.0),
            "target_expansion": target,
            "fused_prediction_expansion": prediction,
            "fused_zero_events_expansion": np.zeros(count),
            "fused_shuffled_mean_expansion": prediction[::-1],
        }
    )


def test_align_seed_fusions_averages_by_identity() -> None:
    frames = {7: _frame(-0.001), 13: _frame(0.0), 23: _frame(0.001)}
    aligned = align_seed_fusions(frames, split_name="validation")
    expected = np.mean(
        np.stack(
            [
                frame.sort_values(
                    ["sequence_id", "sample_token", "track_id"], kind="stable"
                )["fused_prediction_expansion"].to_numpy()
                for frame in frames.values()
            ]
        ),
        axis=0,
    )
    assert np.allclose(aligned["fused_prediction_expansion"], expected)
    assert np.all(aligned["seed_prediction_std"] > 0.0)


def test_pairwise_seed_metrics_reports_all_pairs() -> None:
    frames = {7: _frame(-0.001), 13: _frame(0.0), 23: _frame(0.001)}
    metrics = pairwise_seed_metrics(frames)
    assert len(metrics) == 3
    assert np.allclose(metrics["prediction_pearson"], 1.0)


def test_materialize_changes_only_train_seed(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump({"model": {"width": 8}, "train": {"seed": 7, "epochs": 2}}),
        encoding="utf-8",
    )
    result = materialize(
        seed=13,
        sources={"v46": source, "v47": source, "v48": source},
        output_dir=tmp_path / "out",
    )
    output = yaml.safe_load((tmp_path / "out" / "v46_seed_13.yaml").read_text())
    assert output == {"model": {"width": 8}, "train": {"seed": 13, "epochs": 2}}
    assert result["scientific_contract"]["only_train_seed_is_overridden"] is True


def test_aggregate_true_seed_smoke(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    fixed = FixedFusionConfig()
    for seed, offset in ((7, -0.001), (13, 0.0), (23, 0.001)):
        seed_root = run_root / f"seed-{seed}"
        seed_root.mkdir(parents=True)
        summary = {
            "artifact_type": "object_event_v4_9_fixed_event_fusion",
            "status": "fusion_screen_passed",
            "passed": True,
            "gates": {"negative_accuracy": True},
            "fusion_config": asdict(fixed),
        }
        (seed_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        frame = _frame(offset)
        frame.to_csv(seed_root / "train_predictions.csv", index=False)
        frame.to_csv(seed_root / "validation_predictions.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aggregate": {
                    "seeds": [7, 13, 23],
                    "mean_seed_pearson_gate": 0.5,
                    "worst_seed_pearson_gate": 0.5,
                    "seed_pearson_std_gate": 0.5,
                    "mean_seed_balanced_sign_gate": 0.5,
                    "worst_seed_balanced_sign_gate": 0.5,
                    "mean_seed_negative_accuracy_gate": 0.5,
                    "ensemble_pearson_gate": 0.5,
                    "ensemble_track_bootstrap_lower_gate": 0.4,
                    "ensemble_weighted_mid_gate": 1000.0,
                    "ensemble_balanced_sign_gate": 0.5,
                    "ensemble_negative_accuracy_gate": 0.5,
                    "ensemble_expansion_mae_gate": 0.1,
                    "ensemble_min_sequence_pearson_gate": 0.5,
                    "ensemble_min_sequence_negative_accuracy_gate": 0.5,
                    "pairwise_prediction_pearson_gate": 0.5,
                    "mean_sample_prediction_std_gate": 0.1,
                    "zero_event_pearson_drop_gate": 0.5,
                    "shuffled_event_pearson_drop_gate": 0.5,
                    "per_sequence_negative_min_count": 1,
                    "track_bootstrap_repeats": 100,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = aggregate(
        run_root=run_root,
        config_path=config_path,
        output_dir=tmp_path / "out",
    )
    assert result["passed"] is True
    assert result["scientific_contract"]["true_seed_specific_training"] is True
    assert (tmp_path / "out" / "pairwise_seed_metrics.csv").is_file()


def test_aggregate_records_completed_failed_seed_without_relabelling(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    fixed = FixedFusionConfig()
    for seed, offset in ((7, -0.001), (13, 0.0), (23, 0.001)):
        seed_root = run_root / f"seed-{seed}"
        seed_root.mkdir(parents=True)
        passed = seed != 23
        summary = {
            "artifact_type": "object_event_v4_9_fixed_event_fusion",
            "status": "fusion_screen_passed" if passed else "fusion_screen_failed",
            "passed": passed,
            "gates": {
                "pearson_improvement": True,
                "negative_accuracy": passed,
            },
            "fusion_config": asdict(fixed),
        }
        (seed_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        frame = _frame(offset)
        frame.to_csv(seed_root / "train_predictions.csv", index=False)
        frame.to_csv(seed_root / "validation_predictions.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aggregate": {
                    "seeds": [7, 13, 23],
                    "require_all_seed_screens": True,
                    "mean_seed_pearson_gate": 0.5,
                    "worst_seed_pearson_gate": 0.5,
                    "seed_pearson_std_gate": 0.5,
                    "mean_seed_balanced_sign_gate": 0.5,
                    "worst_seed_balanced_sign_gate": 0.5,
                    "mean_seed_negative_accuracy_gate": 0.5,
                    "ensemble_pearson_gate": 0.5,
                    "ensemble_track_bootstrap_lower_gate": 0.4,
                    "ensemble_weighted_mid_gate": 1000.0,
                    "ensemble_balanced_sign_gate": 0.5,
                    "ensemble_negative_accuracy_gate": 0.5,
                    "ensemble_expansion_mae_gate": 0.1,
                    "ensemble_min_sequence_pearson_gate": 0.5,
                    "ensemble_min_sequence_negative_accuracy_gate": 0.5,
                    "pairwise_prediction_pearson_gate": 0.5,
                    "mean_sample_prediction_std_gate": 0.1,
                    "zero_event_pearson_drop_gate": 0.5,
                    "shuffled_event_pearson_drop_gate": 0.5,
                    "per_sequence_negative_min_count": 1,
                    "track_bootstrap_repeats": 100,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = aggregate(
        run_root=run_root,
        config_path=config_path,
        output_dir=tmp_path / "out-failed-seed",
    )
    assert result["passed"] is False
    assert result["gates"]["all_seed_screens"] is False
    assert result["seed_screen_status"]["23"]["passed"] is False
    assert result["seed_screen_status"]["23"]["failed_gates"] == ["negative_accuracy"]
    assert result["per_seed"][2]["v49_status"] == "fusion_screen_failed"
    assert result["scientific_contract"][
        "completed_failed_seed_screens_are_aggregated_not_relabelled"
    ] is True
    assert (tmp_path / "out-failed-seed" / "summary.json").is_file()
