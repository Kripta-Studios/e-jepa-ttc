"""Verify that C2F graph routing preserves the original downstream task queue."""

from pathlib import Path

import pytest

from operational.train40_system import controller_c2f_graph as controller


@pytest.mark.parametrize("arm", ["a5", "c2f"])
def test_only_producer_implementation_changes(monkeypatch, tmp_path: Path, arm: str):
    calls = []
    sentinel = object()

    def launch(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(controller.controller_overlap, "_launch", launch)
    result = controller.launch(
        tmp_path, arm, "operational.train40_system.engine", ["--arm", arm]
    )
    expected = "engine_c2f_graph_replay" if arm == "c2f" else "engine_host_fast_4"
    assert result is sentinel
    assert calls == [(tmp_path, arm, f"operational.train40_system.{expected}", ["--arm", arm])]


def test_downstream_arguments_and_implementation_preserved(monkeypatch, tmp_path: Path):
    calls = []
    sentinel = object()

    def launch(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(controller, "_launch", launch)
    arguments = ["--kind", "H8"]
    module = "operational.train40_system.history_features"
    assert controller.launch(tmp_path, "h8_features", module, arguments) is sentinel
    assert calls == [(tmp_path, "h8_features", module, arguments)]


@pytest.mark.parametrize("tag,args", [("garl", ["--arm", "garl"]), ("c2f", [])])
def test_unapproved_producer_arguments_rejected(tmp_path: Path, tag: str, args: list[str]):
    with pytest.raises(ValueError, match="Unexpected"):
        controller.launch(tmp_path, tag, "operational.train40_system.engine", args)
