from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from operational.sota_eval.publication import inspect_branches, publish
from operational.sota_eval.scoring import run as score


@pytest.fixture(autouse=True)
def _canonical_campaign_validation_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fixtures tiny; campaign validators have their own fragment/model test suite."""
    from operational.sota_eval import campaign, fcwd_run

    monkeypatch.setattr(campaign, "_sealed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(campaign, "_scored", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fcwd_run, "verify_seal", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(
        fcwd_run,
        "source_binding",
        lambda manifest, *_args, **_kwargs: json.loads(
            (Path(manifest).parent / "SOURCE_FREEZE.json").read_text(encoding="utf-8")
        ),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _scoring(path: Path, input_path: Path | None = None) -> None:
    path.mkdir(parents=True)
    input_path = input_path or path / "SCORED_PREDICTIONS.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "query_id,sequence_id,truth_ttc_seconds,model\nq0,s0,2.0,2.5\n",
        encoding="utf-8",
    )
    score(input_path, path, ["model"], bootstrap_draws=10, seed=20261008)


def _minimum(root: Path, *, complete: bool) -> None:
    _scoring(root / "scoring_existing")
    _write_json(
        root / "fcwd/ASSET_MANIFEST.json",
        {
            "status": "COMPLETE",
            "missing_assets": [],
            "missing_calibration_folders": [],
            "targets_read": False,
            "dataset_root": "E:\\private-data",
        },
    )
    _write_json(root / "garl_parity/RESULT.json", {"status": "PASSED"})
    (root / "garl_parity/REPORT.md").write_text("passed\n", encoding="utf-8")
    _write_json(
        root / "garl_parity/AUDIT_BINDING.json",
        {"result_sha256": _digest(root / "garl_parity/RESULT.json")},
    )
    _write_json(
        root / "garl_parity/SHA256.json",
        {
            name: _digest(root / "garl_parity" / name)
            for name in ("AUDIT_BINDING.json", "REPORT.md", "RESULT.json")
        },
    )
    _write_json(root / "baselines/FAILURE_COUNTS.json", {"status": "COMPLETE"})
    _write_json(root / "official_contract/STATUS.json", {"status": "BLOCKED"})
    _write_json(root / "R1_STATUS.json", {"status": "RUNNING"})
    if complete:
        expanded = root / "expanded_metrics"
        full = root / "dev32_expanded_rgb"
        full.mkdir(parents=True)
        _scoring(expanded, full / "SCORED_PREDICTIONS.csv")
        _write_json(full / "QUERY_MANIFEST.json", {"rows": []})
        _write_json(full / "INFERENCE_FREEZE.json", {"optimizer_updates": 0})
        _write_json(
            full / "PREDICTIONS_SEALED.json",
            {
                "status": "COMPLETE",
                "manifest_sha256": _digest(full / "QUERY_MANIFEST.json"),
                "binding_sha256": _digest(full / "INFERENCE_FREEZE.json"),
                "optimizer_updates": 0,
            },
        )
        baseline = root / "dev32_expanded"
        _write_json(full / "BASELINE_PRESERVATION.json", {"status": "FROZEN"})
        _write_json(
            root / "CAMPAIGN_FREEZE.json",
            {
                "status": "FROZEN",
                "baseline": str(baseline.resolve()),
                "full": str(full.resolve()),
                "metrics": str(expanded.resolve()),
                "device": "cpu",
                "event_backend": "direct",
                "full_backend": "direct",
                "sources": {},
            },
        )
        campaign_result = {
            "status": "COMPLETE",
            "optimizer_updates": 0,
            "campaign_freeze_sha256": _digest(root / "CAMPAIGN_FREEZE.json"),
            "baseline_preservation_sha256": _digest(full / "BASELINE_PRESERVATION.json"),
            "full_prediction_seal_sha256": _digest(full / "PREDICTIONS_SEALED.json"),
            "scored_predictions_sha256": _digest(full / "SCORED_PREDICTIONS.csv"),
            "metrics_sha256": _digest(expanded / "SHA256.json"),
        }
        _write_json(root / "CAMPAIGN_RESULT.json", campaign_result)
        _write_json(
            root / "PIPELINE_STATE.json",
            {"status": "COMPLETE", "result_sha256": _digest(root / "CAMPAIGN_RESULT.json")},
        )

        fcwd = root / "fcwd_inference"
        scoring = fcwd / "scoring"
        _scoring(scoring)
        _write_json(fcwd / "QUERY_MANIFEST.json", {"rows": []})
        _write_json(
            fcwd / "SOURCE_FREEZE.json",
            {
                "status": "FROZEN",
                "campaign": str(root / "train40"),
                "public_full_dir": str(root / "public_full"),
                "code_root": str(root / "code"),
                "device": "cpu",
            },
        )
        _write_json(
            fcwd / "INFERENCE_FREEZE.json",
            {
                "status": "FROZEN",
                "source_freeze_sha256": _digest(fcwd / "SOURCE_FREEZE.json"),
                "manifest_sha256": _digest(fcwd / "QUERY_MANIFEST.json"),
                "optimizer_updates": 0,
                "targets_read": False,
            },
        )
        _write_json(
            fcwd / "PREDICTIONS_SEALED.json",
            {
                "status": "COMPLETE",
                "manifest_sha256": _digest(fcwd / "QUERY_MANIFEST.json"),
                "source_freeze_sha256": _digest(fcwd / "SOURCE_FREEZE.json"),
                "binding_sha256": _digest(fcwd / "INFERENCE_FREEZE.json"),
                "optimizer_updates": 0,
            },
        )
        _write_json(
            scoring / "MODEL_COVERAGE.json",
            {
                "head_selection_performed": False,
                "full_model": {
                    "status": "DEPENDENCY_UNAVAILABLE",
                    "not_a_negative_model_result": True,
                },
            },
        )
        contract = {
            "status": "COMPLETE",
            "prediction_seal_sha256": _digest(fcwd / "PREDICTIONS_SEALED.json"),
            "source_freeze_sha256": _digest(fcwd / "SOURCE_FREEZE.json"),
            "query_manifest_sha256": _digest(fcwd / "QUERY_MANIFEST.json"),
            "model_coverage_sha256": _digest(scoring / "MODEL_COVERAGE.json"),
            "joined_predictions_sha256": _digest(scoring / "SCORED_PREDICTIONS.csv"),
            "scoring_manifest_sha256": _digest(scoring / "SHA256.json"),
            "labels_opened_only_after_verified_prediction_seal": True,
        }
        _write_json(scoring / "TARGET_JOIN_CONTRACT.json", contract)
        _write_json(
            fcwd / "SCORING_COMPLETE.json",
            {
                "status": "COMPLETE",
                "queries": 630,
                "methods": ["a", "b", "c", "d"],
                "planned_methods": ["a", "b", "c", "d", "e"],
                "optimizer_updates": 0,
                "prediction_seal_sha256": _digest(fcwd / "PREDICTIONS_SEALED.json"),
                "scoring_manifest_sha256": _digest(scoring / "SHA256.json"),
                "target_join_contract_sha256": _digest(scoring / "TARGET_JOIN_CONTRACT.json"),
            },
        )

        cost_source = root / "cost_source.py"
        cost_source.write_text("# frozen cost source\n", encoding="utf-8")
        selected = [{"query_id": f"q{index}", "sequence_id": f"s{index}"} for index in range(8)]
        _write_json(
            root / "cost/EXECUTION_FREEZE.json",
            {
                "status": "EXECUTION_FROZEN",
                "source": str(cost_source),
                "source_sha256": _digest(cost_source),
                "selected_queries": selected,
            },
        )
        _write_json(
            root / "cost/MODEL_FREEZE.json",
            {
                "status": "MODELS_FROZEN_BEFORE_MEASUREMENT",
                "execution_freeze_sha256": _digest(root / "cost/EXECUTION_FREEZE.json"),
            },
        )
        for index in range(8):
            raw = root / "cost" / f"RAW_q{index}.csv"
            raw.write_text("milliseconds\n1.0\n", encoding="utf-8")
            _write_json(
                root / "cost" / f"FRAGMENT_q{index}.json",
                {
                    "status": "COMPLETE",
                    "query_id": f"q{index}",
                    "optimizer_updates": 0,
                    "execution_freeze_sha256": _digest(root / "cost/EXECUTION_FREEZE.json"),
                    "model_freeze_sha256": _digest(root / "cost/MODEL_FREEZE.json"),
                    "raw_csv": raw.name,
                    "raw_csv_sha256": _digest(raw),
                },
            )
        _write_json(
            root / "cost/SYSTEM_COST_SUMMARY.json",
            {
                "schema": "sota_system_cost_fixed8_v1",
                "status": "COMPLETE",
                "query_count": 8,
                "optimizer_updates": 0,
                "execution_freeze_sha256": _digest(root / "cost/EXECUTION_FREEZE.json"),
                "model_freeze_sha256": _digest(root / "cost/MODEL_FREEZE.json"),
                "metrics": {
                    "h8_system_three_heads": {},
                    "garl_event_only_shared_preparation": {},
                    "garl_full_rgb_event": {},
                },
            },
        )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partial_publication_is_sanitized_and_preserves_plan(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs"
    _minimum(artifacts, complete=False)
    output.mkdir()
    (output / "PLAN.json").write_text("plan", encoding="utf-8")
    (output / "unowned.txt").write_text("preserve", encoding="utf-8")
    (artifacts / "UNRELATED_ROOT_FILE.txt").write_text("do not recurse", encoding="utf-8")
    bundle = artifacts / "SOTA_ESSENTIAL.zip"
    result = publish(artifacts, output, bundle, require_complete=False, repo=Path.cwd())
    assert result["status"] == "PARTIAL_BLOCKED"
    assert (output / "PLAN.json").read_text(encoding="utf-8") == "plan"
    assert (output / "unowned.txt").read_text(encoding="utf-8") == "preserve"
    assert not (output / "evidence/UNRELATED_ROOT_FILE.txt").exists()
    assert "E:\\private-data" not in (output / "evidence/fcwd/ASSET_MANIFEST.json").read_text()
    assert "${DATA}" in (output / "evidence/fcwd/ASSET_MANIFEST.json").read_text()
    assert bundle.is_file() and bundle.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(bundle) as archive:
        assert "PLAN.json" not in archive.namelist()
        assert "unowned.txt" not in archive.namelist()


def test_require_complete_refuses_partial_without_touching_output(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs"
    _minimum(artifacts, complete=False)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expanded_metrics, fcwd_metrics, system_cost"):
        publish(artifacts, output, artifacts / "bundle.zip", require_complete=True, repo=Path.cwd())
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_complete_publication_regenerates_exact_table(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs"
    _minimum(artifacts, complete=True)
    result = publish(
        artifacts,
        output,
        artifacts / "SOTA_ESSENTIAL.zip",
        require_complete=True,
        repo=Path.cwd(),
    )
    assert result["status"] == "COMPLETE"
    state = json.loads((output / "NEXT_DECISION.json").read_text(encoding="utf-8"))
    assert state["blocked_branches"] == []
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(output / "regenerate.py"), "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_hash_manifest_tamper_makes_historical_branch_incomplete(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs"
    _minimum(artifacts, complete=False)
    (artifacts / "scoring_existing/REPORT.json").write_text('{"changed":true}', encoding="utf-8")
    publish(artifacts, output, artifacts / "bundle.zip", require_complete=False, repo=Path.cwd())
    state = json.loads((output / "NEXT_DECISION.json").read_text(encoding="utf-8"))
    historical = next(row for row in state["branches"] if row["name"] == "scoring_existing_946")
    assert historical["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    ("relative", "branch"),
    [
        ("CAMPAIGN_RESULT.json", "expanded_metrics"),
        ("fcwd_inference/scoring/TARGET_JOIN_CONTRACT.json", "fcwd_metrics"),
        ("cost/RAW_q0.csv", "system_cost"),
    ],
)
def test_bound_lineage_tamper_blocks_branch(tmp_path: Path, relative: str, branch: str) -> None:
    artifacts = tmp_path / "artifacts"
    _minimum(artifacts, complete=True)
    path = artifacts / relative
    path.write_bytes(path.read_bytes() + b"changed")
    status = {item.name: item.complete for item in inspect_branches(artifacts)}
    assert status[branch] is False


def test_canonical_paths_ignore_stale_nested_complete_outputs(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _minimum(artifacts, complete=False)
    _scoring(artifacts / "stale/expanded_metrics")
    _write_json(artifacts / "stale/fcwd_inference/SCORING_COMPLETE.json", {"status": "COMPLETE"})
    _write_json(
        artifacts / "stale/cost/SYSTEM_COST_SUMMARY.json",
        {"status": "COMPLETE", "query_count": 8},
    )
    status = {item.name: item.complete for item in inspect_branches(artifacts)}
    assert status["expanded_metrics"] is False
    assert status["fcwd_metrics"] is False
    assert status["system_cost"] is False


def test_sanitized_json_key_collision_is_rejected(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs"
    _minimum(artifacts, complete=False)
    _write_json(
        artifacts / "fcwd/COLLISION.json",
        {
            "C:\\first\\one\\same\\tail\\file.json": 1,
            "D:\\second\\one\\same\\tail\\file.json": 2,
        },
    )
    with pytest.raises(ValueError, match="Sanitization key collision"):
        publish(
            artifacts,
            output,
            artifacts / "bundle.zip",
            require_complete=False,
            repo=Path.cwd(),
        )
