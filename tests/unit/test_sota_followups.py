from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operational.sota_eval import followups


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wait_receipt_requires_exact_campaign_entrypoint() -> None:
    receipt = {
        "pid": 123,
        "create_time": 100.5,
        "command": ["python.exe", "-m", "operational.sota_eval.campaign", "--device", "cuda"],
        "optimizer_updates": 0,
    }
    assert followups.validate_wait_receipt(receipt)[:2] == (123, 100.5)
    receipt["command"] = [
        "python.exe",
        "-m",
        "operational.sota_eval.full_prefetch",
        "--note",
        "operational.sota_eval.campaign",
    ]
    with pytest.raises(ValueError, match="expanded campaign"):
        followups.validate_wait_receipt(receipt)


def test_campaign_result_verifies_all_bound_bytes(tmp_path: Path) -> None:
    paths = {
        "campaign_freeze_sha256": tmp_path / "CAMPAIGN_FREEZE.json",
        "baseline_preservation_sha256": tmp_path / "dev32_expanded_rgb/BASELINE_PRESERVATION.json",
        "full_prediction_seal_sha256": tmp_path / "dev32_expanded_rgb/PREDICTIONS_SEALED.json",
        "scored_predictions_sha256": tmp_path / "dev32_expanded_rgb/SCORED_PREDICTIONS.csv",
        "metrics_sha256": tmp_path / "expanded_metrics/SHA256.json",
    }
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(index), encoding="utf-8")
    _write(paths["campaign_freeze_sha256"], {})
    result = {
        "status": "COMPLETE",
        "optimizer_updates": 0,
        **{key: _digest(path) for key, path in paths.items()},
    }
    _write(tmp_path / "CAMPAIGN_RESULT.json", result)
    assert followups.verify_campaign_result(tmp_path, lambda _root, _config: True) == result
    next(iter(paths.values())).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="binding failed"):
        followups.verify_campaign_result(tmp_path, lambda _root, _config: True)


def test_default_fcwd_paths_match_publication_plan() -> None:
    arguments = followups.parser().parse_args(["--wait-receipt", "future.json"])
    assert arguments.fcwd_manifest.as_posix().endswith(
        "sota_campaign_20261008/fcwd_population/QUERY_MANIFEST.json"
    )
    assert arguments.fcwd_output.as_posix().endswith("sota_campaign_20261008/fcwd_inference")
    assert not hasattr(arguments, "fcwd_dataset_root")


def test_fcwd_population_requires_existing_frozen_dependency_bindings(tmp_path: Path) -> None:
    population_root = tmp_path / "fcwd_population"
    manifest = population_root / "QUERY_MANIFEST.json"
    assets = tmp_path / "ASSET_MANIFEST.json"
    reference = tmp_path / "REFERENCE_INPUT_CONTRACT.json"
    _write(assets, {"status": "COMPLETE"})
    _write(reference, {"status": "COMPLETE"})
    population = {
        "status": "FROZEN_LABEL_FREE",
        "query_count": 630,
        "rows": [{} for _ in range(630)],
        "ttc_targets_read": False,
        "asset_manifest_path": str(assets),
        "reference_contract_path": str(reference),
    }
    _write(manifest, population)
    freeze = {
        "status": "FROZEN_LABEL_FREE",
        "query_count": 630,
        "manifest_sha256": _digest(manifest),
        "input_adapter_sha256": _digest(followups.ROOT / "operational/sota_eval/fcwd_inputs.py"),
        "asset_manifest_sha256": _digest(assets),
        "reference_contract_sha256": _digest(reference),
        "ttc_targets_read": False,
        "optimizer_updates": 0,
    }
    _write(population_root / "FCWD_INPUT_FREEZE.json", freeze)
    assert followups.verify_fcwd_population(manifest) == freeze
    assets.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="population freeze"):
        followups.verify_fcwd_population(manifest)


def test_fcwd_verification_requires_630_and_score_binding(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"rows": [{} for _ in range(630)]})
    output = tmp_path / "eval"
    seal = {"status": "COMPLETE", "queries": 630, "manifest_sha256": _digest(manifest)}
    _write(output / "PREDICTIONS_SEALED.json", seal)
    scoring = {
        "status": "COMPLETE",
        "queries": 630,
        "prediction_seal_sha256": _digest(output / "PREDICTIONS_SEALED.json"),
    }
    _write(output / "SCORING_COMPLETE.json", scoring)
    assert followups.verify_fcwd(output, manifest) == scoring
    scoring["queries"] = 629
    _write(output / "SCORING_COMPLETE.json", scoring)
    with pytest.raises(ValueError, match="score receipt"):
        followups.verify_fcwd(output, manifest)


def test_cost_verification_checks_both_freezes(tmp_path: Path) -> None:
    _write(tmp_path / "EXECUTION_FREEZE.json", {"status": "EXECUTION_FROZEN"})
    _write(tmp_path / "MODEL_FREEZE.json", {"status": "MODELS_FROZEN"})
    result = {
        "status": "COMPLETE",
        "query_count": 8,
        "targets_read": False,
        "optimizer_updates": 0,
        "execution_freeze_sha256": _digest(tmp_path / "EXECUTION_FREEZE.json"),
        "model_freeze_sha256": _digest(tmp_path / "MODEL_FREEZE.json"),
    }
    _write(tmp_path / "SYSTEM_COST_SUMMARY.json", result)
    assert followups.verify_cost(tmp_path) == result
    result["query_count"] = 7
    _write(tmp_path / "SYSTEM_COST_SUMMARY.json", result)
    with pytest.raises(ValueError, match="incomplete"):
        followups.verify_cost(tmp_path)


def test_independent_branch_failure_is_not_scientific_negative(tmp_path: Path) -> None:
    def fail() -> None:
        raise RuntimeError("operational failure")

    result = followups._branch(tmp_path, "r1", fail, lambda: {})
    assert result["status"] == "FAILED_PRESERVED"
    assert result["scientific_negative"] is False
    assert (tmp_path / "R1_BRANCH.json").is_file()


def test_stop_preserves_branch_without_calling_action(tmp_path: Path) -> None:
    (tmp_path / "STOP_REQUEST").write_text("stop", encoding="utf-8")
    called = False

    def action() -> None:
        nonlocal called
        called = True

    result = followups._branch(tmp_path, "cost", action, lambda: {})
    assert result["status"] == "PAUSED_PRESERVED"
    assert result["reason"] == "STOP_REQUEST"
    assert called is False
    assert (tmp_path / "COST_BRANCH.json").is_file()


def test_r1_noncomplete_state_is_operational_failure(tmp_path: Path) -> None:
    path = tmp_path / "STATE.json"
    _write(path, {"status": "ATTEMPTS_EXHAUSTED_PRESERVED"})
    with pytest.raises(RuntimeError, match="without COMPLETE"):
        followups.verify_r1_state(path)
