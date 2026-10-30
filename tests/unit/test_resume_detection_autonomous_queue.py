"""Regression checks for paused-segment lineage and unattended continuation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/resume_detection_autonomous_queue.py"
SPEC = importlib.util.spec_from_file_location("resume_queue", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_segment_boundaries_do_not_leak_into_confirmation_seeds():
    base = {"existing_r0": {"start_update": 3840, "resume": {"sha256": "r0"}}}
    pause = {"paused_job": "r2_seed7", "checkpoint": {"update": 5376, "sha256": "r2"}}
    assert MODULE.segment_identity("r0_seed7", base, pause) == (3840, "r0")
    assert MODULE.segment_identity("r1_seed7", base, pause) == (0, None)
    assert MODULE.segment_identity("r2_seed7", base, pause) == (5376, "r2")
    assert MODULE.segment_identity("r2_seed13", base, pause) == (0, None)


def test_resumed_evaluations_use_bound_parent_and_new_child(tmp_path):
    queue = MODULE.load_queue()
    parent = tmp_path / "parent4096.json"
    parent.write_text("parent")
    pause = {"preserved_full_evaluation_4096": queue.binding(parent)}
    seen = []
    helper = SimpleNamespace(verify_evaluation=lambda path, **kw: seen.append((path, kw)) or {})
    summary = {"evaluation_protocol_sha256": "protocol", "full_selection_count": 23958}
    result = MODULE.verify_full(helper, queue, tmp_path / "child", summary, pause)
    assert set(result) == {"4096", "8192"}
    assert [item[0] for item in seen] == [
        parent,
        tmp_path / "child/evaluation/full_selection_update_008192.json",
    ]
    assert all(item[1]["expected_sample_count"] == 23958 for item in seen)
    parent.write_text("changed")
    with pytest.raises(ValueError, match="Pinned file changed"):
        MODULE.verify_full(helper, queue, tmp_path / "child", summary, pause)
    assert len(seen) == 2


def test_queue_adopts_completed_jobs_and_resumes_only_r2(tmp_path, monkeypatch):
    queue = MODULE.load_queue()
    calls = []
    verified = []
    source = tmp_path / "source"
    source.write_text("frozen")
    bound = queue.binding(source)
    names = [f"{recipe}_seed{seed}" for recipe in ("r0", "r1", "r2") for seed in (7, 13, 23)]
    jobs = {name: {"config": bound, "output": str(tmp_path / name)} for name in names}
    base = {
        "existing_r0": {"start_update": 3840, "resume": {"sha256": "r0"}},
        "helper": bound,
        "original_plan": bound,
        "resource_wait_hours": 24,
        "poll_seconds": 0,
        "later_phases_not_queued": {"S2": "pending"},
    }
    pause = {
        "paused_job": "r2_seed7",
        "checkpoint": {**bound, "update": 5376},
        "selected_checkpoint": bound,
        "preserved_full_evaluation_4096": bound,
        "completed_jobs": {
            name: {"summary_sha256": name, "full_evaluations": {}} for name in names[:0]
        },
    }
    for name in ("r0_seed7", "r1_seed7"):
        pause["completed_jobs"][name] = {"summary_sha256": name, "full_evaluations": {}}
    base_path, pause_path = tmp_path / "base.json", tmp_path / "pause.json"
    queue.write(base_path, base)
    queue.write(pause_path, pause)
    plan = {
        "supervisor_source": bound,
        "original_supervisor_source": bound,
        "pause_manifest": queue.binding(pause_path),
        "base_plan": queue.binding(base_path),
        "jobs": jobs,
    }
    plan_path = tmp_path / "plan.json"
    queue.write(plan_path, plan)

    def verify_summary(output, config, **kwargs):
        name = output.name
        verified.append((name, kwargs))
        return {
            "summary": {
                "best_selection_metric": 0.2 if name.startswith("r2") else 0.1,
                "selected_checkpoint": {},
                "evaluation_protocol_sha256": "p",
                "full_selection_count": 2,
            },
            "summary_path": str(output / "summary.json"),
            "summary_sha256": name,
        }

    helper = SimpleNamespace(
        load_plan=lambda *a: {
            "python": {"path": "python"},
            "producer": {"path": "train"},
            "gpu": "0",
        },
        verify_all_bindings=lambda *a: None,
        load_campaign_config=lambda *a: SimpleNamespace(seed=7),
        verify_summary=verify_summary,
        verify_population_same=lambda *a: None,
        resource_preflight=lambda *a: {"status": "PASS"},
        run_child=lambda **kw: calls.append(kw),
        _resolve=lambda path, name: Path(path),
    )
    monkeypatch.setattr(queue, "load_helper", lambda *a: helper)
    monkeypatch.setattr(MODULE, "load_queue", lambda: queue)
    monkeypatch.setattr(MODULE, "assert_no_other_campaign", lambda: None)
    monkeypatch.setattr(MODULE, "verify_full", lambda *a: {"4096": {}, "8192": {}})
    monkeypatch.setattr(
        MODULE.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="<root/>")
    )
    monkeypatch.setattr(MODULE, "os", SimpleNamespace(name="test", getpid=lambda: 123))
    MODULE.run(plan_path, queue.sha(plan_path))
    assert [call["name"] for call in calls] == [
        "r2_seed7",
        "r0_seed13",
        "r2_seed13",
        "r0_seed23",
        "r2_seed23",
    ]
    assert calls[0]["resume"] == source
    assert calls[0]["stop_after_update"] == 8192
    assert all(call["resume"] is None for call in calls[1:])
    expected = {name: kw for name, kw in verified}
    assert expected["r2_seed7"]["expected_start_update"] == 5376
    assert expected["r2_seed13"]["expected_start_update"] == 0
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "COMPLETED_QUEUED_SCOPE"
    with pytest.raises(FileExistsError):
        MODULE.run(plan_path, queue.sha(plan_path))
    assert len(calls) == 5


def test_duplicate_campaign_is_rejected(monkeypatch):
    own = SimpleNamespace(parents=lambda: [])
    other = SimpleNamespace(
        pid=123456,
        info={
            "name": "python.exe",
            "cmdline": ["python.exe", "train_detection_campaign.py", "--config", "frozen.yaml"],
        },
    )
    monkeypatch.setattr(MODULE.psutil, "Process", lambda: own)
    monkeypatch.setattr(MODULE.psutil, "process_iter", lambda *a: [other])
    with pytest.raises(RuntimeError, match="already running"):
        MODULE.assert_no_other_campaign()
