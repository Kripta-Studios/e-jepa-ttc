"""The preparation tool adopts exactly one worker and never creates processes."""

from pathlib import Path

import pytest

from scripts import prepare_runpod_event_watchdog_v1 as prepare


def _identity(pid: int, rotation: int = 2, state: str = "S") -> dict:
    return {"pid": pid, "state": state, "argv": [
        "/opt/python", "/ops/event.py", "producer", "--run", "/runs",
        "--rotation", str(rotation), "--fit", "E_A5",
    ]}


def test_adoption_ignores_other_rotation_and_zombie(tmp_path: Path, monkeypatch) -> None:
    states = {11: _identity(11), 12: _identity(12, 3), 13: _identity(13, state="Z")}
    for pid in states:
        (tmp_path / str(pid)).mkdir()
    monkeypatch.setattr(prepare.watchdog, "proc_identity", lambda pid, _root: states[pid])
    task = {"id": "R2_E_A5", "rotation": 2, "fit": "E_A5", "entrypoint": "/ops/event.py"}
    assert prepare.find_worker(task, "/runs", tmp_path)["pid"] == 11


def test_duplicate_or_missing_live_owners_rejected(tmp_path: Path, monkeypatch) -> None:
    task = {"id": "R2_E_A5", "rotation": 2, "fit": "E_A5", "entrypoint": "/ops/event.py"}
    with pytest.raises(ValueError, match="found 0"):
        prepare.find_worker(task, "/runs", tmp_path)
    for pid in (11, 12):
        (tmp_path / str(pid)).mkdir()
    monkeypatch.setattr(prepare.watchdog, "proc_identity", lambda pid, _root: _identity(pid))
    with pytest.raises(ValueError, match="found 2"):
        prepare.find_worker(task, "/runs", tmp_path)


def test_environment_read_preserves_values_without_printing(tmp_path: Path) -> None:
    (tmp_path / "11").mkdir()
    (tmp_path / "11/environ").write_bytes(b"PYTHONPATH=/src:/repo\0A=B=C\0")
    assert prepare.process_environment(11, tmp_path) == {"PYTHONPATH": "/src:/repo", "A": "B=C"}
