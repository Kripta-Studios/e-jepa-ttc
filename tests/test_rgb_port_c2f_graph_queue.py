from __future__ import annotations

from pathlib import Path
from typing import Any

from operational.rgb_port_c2f_graph.queue import (
    C2F_FIT_ID,
    CONCURRENT_MODULE,
    GRAPH_MODULE,
    _augment_config,
    _detach_factory,
    _route_command,
)


def test_route_wraps_only_resolved_concurrent_c2f(tmp_path: Path) -> None:
    freeze = tmp_path / "C2F_GRAPH_FREEZE.json"
    freeze.write_text("{}", encoding="utf-8")

    def concurrent(command: list[Any], _repository: Path) -> list[str]:
        return [
            "python",
            "-m",
            CONCURRENT_MODULE,
            "--concurrent-freeze",
            "concurrent.json",
            "--fit-id",
            str(command[-1]),
        ]

    routed = _route_command(concurrent, freeze, ["--fit-id", C2F_FIT_ID], tmp_path)
    assert routed[:3] == ["python", "-m", GRAPH_MODULE]
    assert routed[3:8] == [
        "--graph-freeze",
        str(freeze.resolve()),
        "--delegate-module",
        CONCURRENT_MODULE,
        "--",
    ]
    assert routed[-2:] == ["--fit-id", C2F_FIT_ID]
    a5 = _route_command(concurrent, freeze, ["--fit-id", "E_A5_MATCHED"], tmp_path)
    assert a5[2] == CONCURRENT_MODULE


def test_overlay_adds_marker_and_runtime_members_without_mutating_config(tmp_path: Path) -> None:
    fit = tmp_path / "fits" / C2F_FIT_ID
    receipts = fit / "c2f_graph_checkpoints"
    receipts.mkdir(parents=True)
    (fit / "C2F_GRAPH_RUNTIME.json").write_text("{}", encoding="utf-8")
    (receipts / "checkpoint_000900.json").write_text("{}", encoding="utf-8")
    config = {
        "resources": {"heavy_command_markers": [], "project_command_markers": []},
        "package": {"members": []},
    }
    value = _augment_config(config, tmp_path)
    assert GRAPH_MODULE in value["resources"]["heavy_command_markers"]
    assert f"fits/{C2F_FIT_ID}/C2F_GRAPH_RUNTIME.json" in value["package"]["members"]
    assert (
        f"fits/{C2F_FIT_ID}/c2f_graph_checkpoints/checkpoint_000900.json"
        in value["package"]["members"]
    )
    assert config["resources"]["heavy_command_markers"] == []


def test_graph_child_detaches_with_the_concurrent_control_exception() -> None:
    class Process:
        pid = 321

    child = _detach_factory(lambda *_args, **_kwargs: Process())(
        ["python", "-m", GRAPH_MODULE, "--fit-id", C2F_FIT_ID]
    )
    assert child.pid == 321
    from operational.rgb_port_concurrent.queue import _DetachedChildError

    try:
        child.wait()
    except _DetachedChildError:
        pass
    else:
        raise AssertionError("Graph child did not detach from the blocking frozen queue")
