from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from operational.efficient_context.common import atomic_json, digest
from operational.sota_eval import campaign


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative in campaign.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    inventory = root / "data/manifests/evttc_all32_local.yaml"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text("sequences: []\n", encoding="utf-8")
    return root


def _arguments(tmp_path: Path, source_root: Path) -> dict:
    return {
        "output": tmp_path / "campaign-output",
        "campaign": tmp_path / "train40",
        "baseline": tmp_path / "event-only",
        "full": tmp_path / "full",
        "data_repository": tmp_path / "data-repository",
        "code_root": tmp_path / "garl-code",
        "public_full_source": tmp_path / "public-full",
        "device": "cuda",
        "source_root": source_root,
    }


def _sealed_scored_baseline(root: Path) -> None:
    root.mkdir(parents=True)
    atomic_json(root / "QUERY_MANIFEST.json", {"rows": []})
    atomic_json(root / "INFERENCE_FREEZE.json", {"optimizer_updates": 0})
    atomic_json(
        root / "PREDICTIONS_SEALED.json",
        {
            "status": "COMPLETE",
            "queries": 0,
            "fragments": {},
            "manifest_sha256": digest(root / "QUERY_MANIFEST.json"),
            "binding_sha256": digest(root / "INFERENCE_FREEZE.json"),
        },
    )
    atomic_json(
        root / "RESULT.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": digest(root / "QUERY_MANIFEST.json"),
            "prediction_seal_sha256": digest(root / "PREDICTIONS_SEALED.json"),
        },
    )
    (root / "SCORED_PREDICTIONS.csv").write_text("query_id\n", encoding="utf-8")


def test_zero_return_without_completion_seal_pauses_and_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    calls: list[list[str]] = []

    def fake(command: Sequence[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(list(command))
        return 0

    monkeypatch.setattr(campaign, "_active_heavy_pids", lambda: [])
    result = campaign.run(**_arguments(tmp_path, source_root), invoke=fake)
    assert result == {"status": "PAUSED_PRESERVED", "phase": "event_inference"}
    assert len(calls) == 1
    state = json.loads(
        (tmp_path / "campaign-output/PIPELINE_STATE.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "PAUSED_PRESERVED"
    assert state["optimizer_updates"] == 0


def test_resume_rejects_source_hash_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(campaign, "_active_heavy_pids", lambda: [])
    arguments = _arguments(tmp_path, source_root)
    campaign.run(**arguments, invoke=lambda *_: 0)
    changed = source_root / campaign.SOURCE_FILES[1]
    changed.write_text("# changed after freeze\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen campaign configuration changed"):
        campaign.run(**arguments, invoke=lambda *_: pytest.fail("child must not start"))


def test_active_heavy_process_blocks_before_child_or_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(
        campaign,
        "_active_heavy_pids",
        lambda: [{"pid": 321, "create_time": 7.0, "entrypoint": "r1_measure"}],
    )
    result = campaign.run(
        **_arguments(tmp_path, source_root),
        invoke=lambda *_: pytest.fail("blocked campaign must not invoke a child"),
    )
    assert result["status"] == "BLOCKED_ACTIVE_HEAVY"
    assert not (tmp_path / "campaign-output/CAMPAIGN_FREEZE.json").exists()


def test_live_campaign_writer_lease_rejects_concurrent_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    arguments = _arguments(tmp_path, source_root)
    output = arguments["output"]
    output.mkdir(parents=True)
    monkeypatch.setattr(campaign, "_active_heavy_pids", lambda: [])
    with campaign.Lease(output), pytest.raises(RuntimeError, match="live campaign writer"):
        campaign.run(**arguments, invoke=lambda *_: 0)


def test_baseline_preservation_detects_hash_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    full = tmp_path / "full"
    _sealed_scored_baseline(baseline)
    full.mkdir()
    monkeypatch.setattr(campaign, "_sealed", lambda *_: True)
    campaign._preserve_baseline(baseline, full, device="cuda", event_backend="direct")
    seal = json.loads((baseline / "PREDICTIONS_SEALED.json").read_text(encoding="utf-8"))
    atomic_json(baseline / "PREDICTIONS_SEALED.json", {**seal, "tampered": True})
    with pytest.raises(ValueError, match="Frozen campaign configuration changed"):
        campaign._preserve_baseline(baseline, full, device="cuda", event_backend="direct")


def test_public_assets_are_copied_once_and_existing_difference_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-assets"
    target = tmp_path / "target-assets"
    for relative in campaign.PUBLIC_FULL_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    expected = campaign._copy_public_full(source, target)
    assert expected == {
        relative: digest(source / relative) for relative in campaign.PUBLIC_FULL_FILES
    }
    (target / campaign.PUBLIC_FULL_FILES[0]).write_bytes(b"different")
    with pytest.raises(ValueError, match="Existing public model asset differs"):
        campaign._copy_public_full(source, target)


def test_sealed_phase_skip_rejects_tampered_prediction_fragment(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    (root / "predictions").mkdir(parents=True)
    row = {"query_id": "q0"}
    atomic_json(root / "QUERY_MANIFEST.json", {"rows": [row]})
    atomic_json(root / "INFERENCE_FREEZE.json", {"optimizer_updates": 0})
    binding = digest(root / "INFERENCE_FREEZE.json")
    fragment = root / "predictions/query_00000.json"
    atomic_json(fragment, {"query_id": "q0", "binding_sha256": binding, "optimizer_updates": 0})
    fragment.with_suffix(".sha256").write_text(digest(fragment) + "\n", encoding="ascii")
    atomic_json(
        root / "PREDICTIONS_SEALED.json",
        {
            "status": "COMPLETE",
            "queries": 1,
            "manifest_sha256": digest(root / "QUERY_MANIFEST.json"),
            "binding_sha256": binding,
            "fragments": {"predictions/query_00000.json": digest(fragment)},
            "optimizer_updates": 0,
        },
    )
    assert campaign._sealed(root)
    atomic_json(fragment, {"query_id": "q0", "binding_sha256": binding, "tampered": True})
    assert not campaign._sealed(root)


def test_metrics_phase_skip_rejects_output_or_input_hash_mutation(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    input_csv = tmp_path / "SCORED_PREDICTIONS.csv"
    input_csv.write_text("query_id\n", encoding="utf-8")
    atomic_json(metrics / "REPORT.json", {"cohort": {"eligible_queries": 1}})
    atomic_json(
        metrics / "METADATA.json",
        {"input_filename": input_csv.name, "input_sha256": digest(input_csv)},
    )
    for name in campaign.METRIC_FILES - {"REPORT.json", "METADATA.json"}:
        (metrics / name).write_text(name + "\n", encoding="utf-8")
    atomic_json(
        metrics / "SHA256.json",
        {name: digest(metrics / name) for name in campaign.METRIC_FILES},
    )
    assert campaign._metrics_complete(metrics, input_csv)
    atomic_json(metrics / "REPORT.json", {"cohort": {"eligible_queries": 2}})
    assert not campaign._metrics_complete(metrics, input_csv)
    atomic_json(metrics / "REPORT.json", {"cohort": {"eligible_queries": 1}})
    input_csv.write_text("query_id\nchanged\n", encoding="utf-8")
    assert not campaign._metrics_complete(metrics, input_csv)


def test_wait_receipt_command_must_bind_exact_event_only_invocation(tmp_path: Path) -> None:
    event_campaign = tmp_path / "train40"
    baseline = tmp_path / "expanded"
    command = [
        str(Path(campaign.sys.executable).resolve()),
        "-m",
        "operational.evttc_transfer.run",
        "--campaign",
        str(event_campaign.resolve()),
        "--output",
        str(baseline.resolve()),
        "--device",
        "cuda",
    ]
    campaign._validated_event_command(command, event_campaign, baseline, "cuda", "direct")
    command[-1] = "cpu"
    with pytest.raises(ValueError, match="unexpected entrypoint"):
        campaign._validated_event_command(command, event_campaign, baseline, "cuda", "direct")


def test_scoring_seal_hashes_csv_and_rejects_mutation(tmp_path: Path) -> None:
    output = tmp_path / "score"
    output.mkdir()
    atomic_json(output / "QUERY_MANIFEST.json", {"rows": [{"query_id": "q"}]})
    atomic_json(output / "PREDICTIONS_SEALED.json", {"status": "COMPLETE"})
    atomic_json(
        output / "RESULT.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": digest(output / "QUERY_MANIFEST.json"),
            "prediction_seal_sha256": digest(output / "PREDICTIONS_SEALED.json"),
        },
    )
    for name in campaign.SCORE_FILES[1:]:
        (output / name).write_text(name + "\n", encoding="utf-8")
    campaign._seal_score(output)
    assert campaign._scored(output)
    (output / "SCORED_PREDICTIONS.csv").write_text("changed\n", encoding="utf-8")
    assert not campaign._scored(output)


def test_prefetch_receipt_command_shape_is_accepted(tmp_path: Path) -> None:
    event_campaign = tmp_path / "train40"
    baseline = tmp_path / "expanded"
    command = [
        str(Path(campaign.sys.executable).resolve()),
        "-m",
        "operational.sota_eval.prefetch",
        "run",
        "--campaign",
        str(event_campaign.resolve()),
        "--output",
        str(baseline.resolve()),
        "--device",
        "cuda",
        "--workers",
        "2",
        "--max-ahead",
        "2",
    ]
    campaign._validated_event_command(command, event_campaign, baseline, "cuda", "prefetch")


def test_child_retries_only_known_windows_state_replace_race(tmp_path: Path) -> None:
    attempts = 0

    def invoke(_command: Sequence[str], log: Path, _environment: dict[str, str]) -> int:
        nonlocal attempts
        attempts += 1
        log.parent.mkdir(parents=True, exist_ok=True)
        if attempts == 1:
            log.write_text(
                "PermissionError: [WinError 5] os.replace pending -> STATE.json\n",
                encoding="utf-8",
            )
            return 1
        return 0

    result = campaign._run_child("event", ["python"], tmp_path, {}, invoke)
    assert result == (0, 2)


def test_child_does_not_retry_integrity_failure(tmp_path: Path) -> None:
    calls = 0

    def invoke(_command: Sequence[str], log: Path, _environment: dict[str, str]) -> int:
        nonlocal calls
        calls += 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ValueError: checkpoint binding changed\n", encoding="utf-8")
        return 1

    assert campaign._run_child("event", ["python"], tmp_path, {}, invoke) == (1, 1)
    assert calls == 1
