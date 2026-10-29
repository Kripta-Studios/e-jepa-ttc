from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_concurrent.contracts import FIT_IDS, _fit_origin
from operational.rgb_port_concurrent.queue import (
    CONCURRENT_MODULE,
    PIPELINE_MODULE,
    _augment_config,
    _detach_factory,
    _DetachedChildError,
    _route_command,
)


def test_routes_only_two_pipeline_event_producers(tmp_path: Path) -> None:
    freeze = tmp_path / "CONCURRENT_FREEZE.json"
    freeze.write_text("{}", encoding="utf-8")

    def resolve(command: list[Any], _repository: Path) -> list[str]:
        return ["python", "-m", PIPELINE_MODULE, "--fit-id", str(command[-1])]

    for fit_id in FIT_IDS:
        routed = _route_command(resolve, freeze, ["--fit-id", fit_id], tmp_path)
        assert routed[:3] == ["python", "-m", CONCURRENT_MODULE]
        assert routed[3:8] == [
            "--concurrent-freeze",
            str(freeze.resolve()),
            "--delegate-module",
            PIPELINE_MODULE,
            "--",
        ]
        assert routed[-2:] == ["--fit-id", fit_id]
    assert _route_command(resolve, freeze, ["--fit-id", "R_A5"], tmp_path)[2] == PIPELINE_MODULE


def test_in_memory_config_queues_every_other_heavy_task(tmp_path: Path) -> None:
    config = {
        "resources": {"heavy_command_markers": [], "project_command_markers": []},
        "package": {"members": []},
        "tasks": [
            {"id": FIT_IDS[0], "fit_id": FIT_IDS[0], "heavy": True},
            {"id": FIT_IDS[1], "fit_id": FIT_IDS[1], "heavy": True},
            {"id": "R_A5", "fit_id": "R_A5", "heavy": True},
            {"id": "CPU", "heavy": False},
        ],
    }
    value = _augment_config(config, tmp_path)
    assert value["tasks"][0].get("soft_depends") is None
    assert value["tasks"][1].get("soft_depends") is None
    assert value["tasks"][2]["soft_depends"] == list(FIT_IDS)
    assert value["tasks"][3].get("soft_depends") is None
    assert CONCURRENT_MODULE in value["resources"]["heavy_command_markers"]
    assert config["tasks"][2].get("soft_depends") is None


def test_new_allowed_child_detaches_after_pid_is_available() -> None:
    launched: list[list[str]] = []

    class Process:
        pid = 123

        def wait(self) -> int:
            return 9

    def popen(command: list[str], *_args: Any, **_kwargs: Any) -> Process:
        launched.append(command)
        return Process()

    launch = _detach_factory(popen)
    child = launch(["python", "-m", CONCURRENT_MODULE, "--fit-id", FIT_IDS[0]])
    assert child.pid == 123
    with pytest.raises(_DetachedChildError):
        child.wait()
    ordinary = launch(["python", "-m", "ordinary.module"])
    assert ordinary.wait() == 9
    assert len(launched) == 2


def test_origin_is_copied_before_live_versions_can_be_pruned(tmp_path: Path) -> None:
    fit = tmp_path / "fits" / FIT_IDS[1]
    versions = fit / "checkpoint_versions"
    versions.mkdir(parents=True)
    checkpoint = versions / "checkpoint_000575.pt"
    checkpoint.write_bytes(b"full C2F state")
    atomic_write_json(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": 575,
            "version": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "identity_sha256": "c2f-identity",
        },
    )
    origin = _fit_origin(tmp_path, FIT_IDS[1])
    checkpoint.unlink()
    assert origin["completed_updates"] == 575
    assert sha256_file(Path(origin["checkpoint_path"])) == origin["checkpoint_sha256"]
