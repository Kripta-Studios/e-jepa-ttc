from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from operational.sota_eval import publication as publication_module
from operational.sota_eval.publication import inspect_branches, publish
from operational.sota_eval.scoring import run as score


@pytest.fixture(autouse=True)
def _canonical_campaign_validation_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fixtures tiny; campaign validators have their own fragment/model test suite."""
    from operational.sota_eval import campaign, fcwd_run, followups

    monkeypatch.setattr(campaign, "_sealed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(campaign, "_scored", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fcwd_run, "verify_seal", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(followups, "verify_fcwd", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(followups, "verify_cost", lambda *_args, **_kwargs: {})
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
    _write_json(
        root / "baselines/CMAX_REFERENCE_DIAGNOSIS.json",
        {
            "status": "COMPLETE",
            "old_variant": {"equation": "wrong", "requested": 32, "successful": 0},
            "corrected_variant": {
                "equation": "affine",
                "requested": 32,
                "successful": 15,
                "failed_retained": 17,
                "success_fraction": 0.46875,
            },
        },
    )
    pilot = root / "baselines/pilot_scored/METHOD_METRICS.csv"
    pilot.parent.mkdir(parents=True)
    pilot.write_text(
        "method,cohort_queries,truth_exposed_queries,prediction_coverage_all_queries,"
        "prediction_coverage_on_exposed_truth,complete_exposed_cohort_status,"
        "conditional_support,conditional_mae_seconds,conditional_rmse_seconds,"
        "conditional_median_absolute_error_seconds,conditional_signed_bias_seconds,"
        "conditional_metrics_not_rankable\n"
        "cmax_reference,32,29,0.46875,0.517,N/A,15,3.7,6.2,1.5,0.3,True\n"
        "strttc_adapted,32,29,0.34375,0.379,N/A,11,6.5,9.5,3.4,-2.2,True\n",
        encoding="utf-8",
    )
    _write_json(root / "baselines/pilot_scored/REPORT.json", {"status": "COMPLETE"})
    (root / "baselines/pilot_scored/SCORED_ROWS.csv").write_text("query_id\nq0\n", encoding="utf-8")
    _write_json(
        root / "baselines/pilot_scored/SHA256.json",
        {
            name: _digest(root / "baselines/pilot_scored" / name)
            for name in ("METHOD_METRICS.csv", "REPORT.json", "SCORED_ROWS.csv")
        },
    )
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
                "methods": ["H8_seed7", "H8_seed13", "H8_seed23", "public_Garl_event_lhr"],
                "planned_methods": [
                    "H8_seed7", "H8_seed13", "H8_seed23", "public_Garl_event_lhr",
                    "public_Garl_rgb_event_full",
                ],
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


def _publication_repo(tmp_path: Path) -> Path:
    for name in (
        "publication.py",
        "scoring.py",
        "campaign.py",
        "fcwd_run.py",
        "fcwd_score.py",
        "cost.py",
    ):
        source = Path.cwd() / "operational/sota_eval" / name
        destination = tmp_path / "operational/sota_eval" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path


def test_partial_publication_is_sanitized_and_preserves_plan(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    output.mkdir(parents=True)
    (output / "PLAN.json").write_text("plan", encoding="utf-8")
    (output / "unowned.txt").write_text("preserve", encoding="utf-8")
    (artifacts / "UNRELATED_ROOT_FILE.txt").write_text("do not recurse", encoding="utf-8")
    bundle = artifacts / "SOTA_ESSENTIAL.zip"
    result = publish(
        artifacts,
        output,
        bundle,
        require_complete=False,
        repo=_publication_repo(tmp_path),
    )
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
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expanded_metrics, fcwd_metrics, system_cost"):
        publish(
            artifacts,
            output,
            artifacts / "bundle.zip",
            require_complete=True,
            repo=_publication_repo(tmp_path),
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_complete_publication_regenerates_exact_table(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=True)
    result = publish(
        artifacts,
        output,
        artifacts / "SOTA_ESSENTIAL.zip",
        require_complete=True,
        repo=_publication_repo(tmp_path),
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
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Para la rama `expanded_metrics`" in readme
    assert "no cubren las filas históricas ni FCWD" in readme
    assert "sustitución coordinada" in readme


def test_regeneration_verifies_top_level_snapshot_manifest(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=True)
    publish(
        artifacts,
        output,
        artifacts / "bundle.zip",
        require_complete=True,
        repo=_publication_repo(tmp_path),
    )
    (output / "README.md").write_text("tampered\n", encoding="utf-8")
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(output / "regenerate.py"), "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "snapshot hash mismatch: README.md" in completed.stderr


def test_nested_checksum_manifest_is_rehashed_after_sanitization(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    repo = _publication_repo(tmp_path)
    audit = artifacts / "root_qa/full_outlier_audit/AUDIT.json"
    payload = artifacts / "root_qa/full_outlier_audit/DETAIL.txt"
    _write_json(audit, {"repo_path": str(repo / "private/source.py")})
    payload.write_text("measured\n", encoding="utf-8")
    source_manifest = audit.parent / "SHA256SUMS.txt"
    source_manifest.write_text(
        f"{_digest(audit)}  AUDIT.json\n{_digest(payload)}  DETAIL.txt\n",
        encoding="utf-8",
    )
    source_manifest_sha = _digest(source_manifest)
    incomplete = artifacts / "root_qa/incomplete_manifest/SHA256SUMS.txt"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text(f"{'0' * 64}  missing.json\n", encoding="utf-8")
    publish(
        artifacts,
        output,
        artifacts / "bundle.zip",
        require_complete=False,
        repo=repo,
    )
    published_root = output / "evidence/root_qa/full_outlier_audit"
    assert "${REPO}" in (published_root / "AUDIT.json").read_text(encoding="utf-8")
    for line in (published_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert _digest(published_root / name) == expected
    inventory = json.loads((output / "SOURCE_INVENTORY.json").read_text(encoding="utf-8"))
    item = next(
        row
        for row in inventory["files"]
        if row["published_path"].endswith("full_outlier_audit/SHA256SUMS.txt")
    )
    assert item["source_sha256"] == source_manifest_sha
    assert item["published_sha256"] == _digest(published_root / "SHA256SUMS.txt")
    assert item["nested_checksum_manifest_mode"] == "REHASHED_PUBLISHED_BYTES"
    incomplete_published = output / "evidence/root_qa/incomplete_manifest/SHA256SUMS.txt"
    assert incomplete_published.read_text(encoding="utf-8").startswith(
        "# SOURCE_ONLY_MANIFEST: not locally verifiable"
    )
    incomplete_item = next(
        row
        for row in inventory["files"]
        if row["published_path"].endswith("incomplete_manifest/SHA256SUMS.txt")
    )
    assert incomplete_item["nested_checksum_manifest_mode"] == (
        "SOURCE_ONLY_NOT_LOCALLY_VERIFIABLE"
    )


def test_regeneration_never_overwrites_a_tampered_published_table(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=True)
    publish(
        artifacts,
        output,
        artifacts / "bundle.zip",
        require_complete=True,
        repo=_publication_repo(tmp_path),
    )
    table = output / "tables/metrics.csv"
    tampered = b"tampered\n"
    table.write_bytes(tampered)
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(output / "regenerate.py"), "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert table.read_bytes() == tampered


def test_publication_rejects_paths_outside_repo_destinations(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _minimum(artifacts, complete=False)
    repo = _publication_repo(tmp_path)
    with pytest.raises(ValueError, match="Unsafe publication output"):
        publish(
            artifacts,
            tmp_path / "outside",
            artifacts / "bundle.zip",
            require_complete=False,
            repo=repo,
        )
    with pytest.raises(ValueError, match="Unsafe publication bundle"):
        publish(
            artifacts,
            tmp_path / "docs/sota",
            tmp_path / "bundle.zip",
            require_complete=False,
            repo=repo,
        )


def test_owned_tree_commit_rolls_back_on_move_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    output.mkdir(parents=True)
    (output / "README.md").write_text("old readme\n", encoding="utf-8")
    (output / "SOURCE_INVENTORY.json").write_text("old inventory\n", encoding="utf-8")
    real_move = shutil.move

    def failing_move(source: str, destination: str | Path) -> str:
        path = Path(source)
        if path.name == "README.md" and path.parent.name == "publication":
            raise OSError("injected tree commit failure")
        return str(real_move(source, destination))

    monkeypatch.setattr(publication_module.shutil, "move", failing_move)
    with pytest.raises(OSError, match="injected tree commit failure"):
        publish(
            artifacts,
            output,
            artifacts / "bundle.zip",
            require_complete=False,
            repo=_publication_repo(tmp_path),
        )
    assert (output / "README.md").read_text(encoding="utf-8") == "old readme\n"
    assert (output / "SOURCE_INVENTORY.json").read_text(encoding="utf-8") == "old inventory\n"
    assert not (output / "evidence").exists()


def test_publication_includes_only_campaign_qa_and_preservation_evidence(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts/campaign"
    _minimum(artifacts, complete=False)
    wanted = (
        "root_qa/PYTEST_RECEIPT.json", "fcwd_runner_qa/QA.json",
        "prefetch/PILOT.json", "full_prefetch/PILOT.json", "ACCOUNTING.json",
        "dev32_expanded/PREFETCH_PRESERVATION_VERIFIED.json",
        "dev32_expanded/REUSE.json",
    )
    for name in wanted:
        _write_json(artifacts / name, {"test": True})
    excluded = (
        artifacts.parent / "root_qa/OLD.json",
        artifacts / "prefetch/predictions/query_00000.json",
        artifacts / "prefetch/model.pth",
        artifacts / "prefetch/raw.hdf5",
    )
    for path in excluded:
        _write_json(path, {"excluded": True})
    sources = publication_module._allowlisted_sources(artifacts, inspect_branches(artifacts))
    assert all(artifacts / name in sources for name in wanted)
    assert all(path not in sources for path in excluded)


@pytest.mark.parametrize(
    "branch,verifier", [("fcwd_metrics", "verify_fcwd"), ("system_cost", "verify_cost")]
)
def test_publication_requires_canonical_full_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch: str, verifier: str
) -> None:
    from operational.sota_eval import followups

    artifacts = tmp_path / "artifacts"
    _minimum(artifacts, complete=True)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ValueError("mutated canonical lineage")

    monkeypatch.setattr(followups, verifier, reject)
    assert not next(item for item in inspect_branches(artifacts) if item.name == branch).complete


def test_publication_rejects_wrong_fcwd_method_identity(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _minimum(artifacts, complete=True)
    path = artifacts / "fcwd_inference/SCORING_COMPLETE.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["methods"] = list(reversed(receipt["methods"]))
    _write_json(path, receipt)
    branch = next(item for item in inspect_branches(artifacts) if item.name == "fcwd_metrics")
    assert not branch.complete


def test_publication_hashes_survive_git_text_filters(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    repo = _publication_repo(tmp_path)
    publish(artifacts, output, artifacts / "bundle.zip", require_complete=False, repo=repo)
    (repo / ".gitattributes").write_bytes(b"* text=auto eol=lf\n")
    git = shutil.which("git")
    assert git is not None
    for args in (["init", "--quiet"], ["config", "core.autocrlf", "true"], ["add", "docs/sota"]):
        subprocess.run([git, "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [git, "-C", str(repo), "diff", "--cached", "--check"], check=True, capture_output=True
    )
    for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        staged = subprocess.run(  # noqa: S603
            [git, "-C", str(repo), "show", f":docs/sota/{name}"],
            check=True, capture_output=True,
        ).stdout
        assert hashlib.sha256(staged).hexdigest() == expected, name


def test_hash_manifest_tamper_makes_historical_branch_incomplete(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    (artifacts / "scoring_existing/REPORT.json").write_text('{"changed":true}', encoding="utf-8")
    publish(
        artifacts,
        output,
        artifacts / "bundle.zip",
        require_complete=False,
        repo=_publication_repo(tmp_path),
    )
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


@pytest.mark.parametrize(
    "relative",
    [
        "baselines/CMAX_REFERENCE_DIAGNOSIS.json",
        "baselines/pilot_scored/METHOD_METRICS.csv",
        "baselines/pilot_scored/SHA256.json",
    ],
)
def test_partial_publication_handles_missing_baseline_companion(
    tmp_path: Path, relative: str
) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
    _minimum(artifacts, complete=False)
    (artifacts / relative).unlink()
    result = publish(
        artifacts,
        output,
        artifacts / "bundle.zip",
        require_complete=False,
        repo=_publication_repo(tmp_path),
    )
    assert result["status"] == "PARTIAL_BLOCKED"
    assert not (output / "evidence/baselines").exists()
    assert (output / "tables/baseline_coverage.csv").read_text(encoding="utf-8").count("\n") == 1


def test_sanitized_json_key_collision_is_rejected(tmp_path: Path) -> None:
    artifacts, output = tmp_path / "artifacts", tmp_path / "docs/sota"
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
            repo=_publication_repo(tmp_path),
        )
