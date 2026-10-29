"""Owned-marker and failure cleanup contracts for the authorized V13 pause."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.sota_evidence import finish_campaign as coordinator


def owner(tmp_path: Path) -> tuple[Path, list[Path]]:
    output = tmp_path / "output"
    output.mkdir()
    run = tmp_path / "v13" / "artifacts" / "run"
    paths = [run / "PAUSE", run / "fits/A/STOP_REQUEST", run / "fits/B/STOP_REQUEST"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owned", encoding="utf-8")
    (output / "V13_PAUSE_OWNERSHIP.json").write_text(
        json.dumps(
            {
                "run": str(run),
                "paths": list(map(str, paths)),
                "marker": "owned",
            }
        ),
        encoding="utf-8",
    )
    return output, paths


def test_changed_marker_prevents_any_removal(tmp_path: Path) -> None:
    output, paths = owner(tmp_path)
    paths[-1].write_text("different owner", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ownership changed"):
        coordinator.resume_v13(output)
    assert all(path.exists() for path in paths)


def test_resume_preserves_current_io_recovery_route(tmp_path: Path, monkeypatch) -> None:
    output, paths = owner(tmp_path)
    calls = []
    monkeypatch.setattr(coordinator.psutil, "process_iter", lambda _: [])

    def launch(command, **kwargs):
        assert not any(path.exists() for path in paths)
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(coordinator.subprocess, "Popen", launch)
    coordinator.resume_v13(output)
    assert calls[0][0][4] == "operational.rgb_port_io_recovery.queue"
    assert str(tmp_path / "v13" / "src") in calls[0][1]["env"]["PYTHONPATH"]
    assert coordinator.read(output / "V13_RESUME_REQUEST.json")["pid"] == 4321


def test_dead_preparation_still_releases_pause(tmp_path: Path, monkeypatch) -> None:
    output, _ = owner(tmp_path)
    resumed = []
    monkeypatch.setattr(coordinator.psutil, "pid_exists", lambda _: False)
    monkeypatch.setattr(coordinator, "resume_v13", lambda out: resumed.append(out))
    with pytest.raises(RuntimeError, match="CPU preparation exited"):
        coordinator.run(output, tmp_path, tmp_path, budget_seconds=100, prepare_pid=4321)
    assert resumed == [output]
    assert not (output / "FINISH_OWNER.json").exists()
    assert coordinator.read(output / "CAMPAIGN_STATUS.json")["status"] == "FAILED"
