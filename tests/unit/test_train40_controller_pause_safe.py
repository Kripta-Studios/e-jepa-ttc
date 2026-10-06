"""CPU-only persistent pause checks for the frozen overlap controller wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from operational.efficient_context.common import ROOT, digest
from operational.train40_system import controller_pause_freeze, controller_pause_safe
from operational.train40_system.durable_io import atomic_json


@pytest.mark.parametrize("name", controller_pause_safe.PAUSE_MARKERS)
def test_pause_marker_blocks_engine_proxy_without_a_live_process(
    tmp_path: Path, name: str
) -> None:
    calls: list[str] = []
    active = controller_pause_safe.active_for(
        tmp_path, lambda marker: calls.append(marker) is not None
    )
    assert not active(controller_pause_safe.ENGINE_MARKER)
    assert calls == [controller_pause_safe.ENGINE_MARKER]
    (tmp_path / name).write_text("{}", encoding="utf-8")
    assert active(controller_pause_safe.ENGINE_MARKER)
    assert calls == [controller_pause_safe.ENGINE_MARKER]


def test_removing_pause_resumes_frozen_active_route(tmp_path: Path) -> None:
    marker = tmp_path / "COORDINATION_PAUSE_REQUEST.json"
    marker.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    active = controller_pause_safe.active_for(
        tmp_path, lambda task: calls.append(task) is not None
    )
    assert active(controller_pause_safe.ENGINE_MARKER)
    marker.unlink()
    assert not active(controller_pause_safe.ENGINE_MARKER)
    assert calls == [controller_pause_safe.ENGINE_MARKER]


def test_pause_only_intercepts_engine_marker(tmp_path: Path) -> None:
    (tmp_path / "STOP_REQUEST").write_text("stop", encoding="utf-8")
    calls: list[str] = []
    active = controller_pause_safe.active_for(
        tmp_path, lambda task: calls.append(task) is not None
    )
    assert not active("operational.train40_system.verify_downloads")
    assert calls == ["operational.train40_system.verify_downloads"]


def test_run_wraps_only_active_and_restores_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("COORDINATION_FREEZE.json", "CONTROLLER_PAUSE_SAFE_FREEZE.json"):
        atomic_json(tmp_path / name, {"files": []})
    original_active = controller_pause_safe.controller_overlap.active
    original_launch = controller_pause_safe.controller_overlap.launch
    observed: list[object] = []

    def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
        observed.extend(
            [
                output,
                raw_root,
                teacher_path,
                controller_pause_safe.controller_overlap.active,
                controller_pause_safe.controller_overlap.launch,
            ]
        )

    monkeypatch.setattr(controller_pause_safe.controller_overlap, "run", run)
    raw, teacher = tmp_path / "raw", tmp_path / "teacher"
    controller_pause_safe.run(tmp_path, raw, teacher)
    assert observed[:3] == [tmp_path, raw, teacher]
    assert observed[3] is not original_active
    assert observed[4] is original_launch
    assert controller_pause_safe.controller_overlap.active is original_active
    assert controller_pause_safe.controller_overlap.launch is original_launch


def test_freeze_requires_cpu_receipts_and_binds_coordination_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, marker in controller_pause_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    coordination = tmp_path / "COORDINATION_FREEZE.json"
    atomic_json(
        coordination,
        {
            "files": [],
            "trainer_sha256": "trainer",
            "controller_sha256": "controller",
        },
    )
    monkeypatch.setattr(controller_pause_freeze, "dependency_files", lambda seeds: [])
    controller_pause_freeze.run(tmp_path)
    frozen = controller_pause_freeze.read(tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json")
    assert frozen["coordination_freeze_sha256"] == digest(coordination)
    assert frozen["source_sha256"] == digest(
        ROOT / "operational/train40_system/controller_pause_safe.py"
    )
    assert frozen["new_optimizer_updates_during_admission"] == 0
    (tmp_path / "CONTROLLER_PAUSE_SAFE_RUFF.txt").write_text("failed", encoding="utf-8")
    (tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json").unlink()
    with pytest.raises(ValueError, match="QA failed"):
        controller_pause_freeze.run(tmp_path)
