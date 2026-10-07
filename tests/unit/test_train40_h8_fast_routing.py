"""The optimized queue may change execution only for the existing H8 feature task."""

from pathlib import Path
from unittest.mock import patch

import pytest

from operational.train40_system import controller_c2f_graph, controller_h8_fast


def test_fast_controller_changes_only_h8_feature_dispatch(tmp_path: Path) -> None:
    calls = []

    def existing_run(output, raw_root, teacher_path):
        controller_c2f_graph._launch(
            output,
            "h8_features",
            "operational.train40_system.history_features",
            ["--raw-root", str(raw_root), "--kind", "H8"],
        )
        controller_c2f_graph._launch(
            output, "c2f", "operational.train40_system.engine", ["--arm", "c2f"]
        )

    def previous(*args):
        calls.append(args)

    with (
        patch("operational.train40_system.history_resources8_fast.verify_admission"),
        patch.object(controller_c2f_graph, "run", existing_run),
        patch.object(controller_c2f_graph, "_launch", previous),
        patch("operational.train40_system.controller_overlap._launch") as launcher,
    ):
        controller_h8_fast.run(tmp_path, tmp_path / "raw", tmp_path / "teacher")
        assert launcher.call_args.args[2] == "operational.train40_system.history_resources8_fast"
        assert calls[0][1:3] == ("c2f", "operational.train40_system.engine")
        assert controller_c2f_graph._launch is previous


def test_fast_controller_rejects_changed_h8_recipe(tmp_path: Path) -> None:
    def existing_run(output, raw_root, teacher_path):
        controller_c2f_graph._launch(
            output, "h8_features", "operational.train40_system.history_features", ["--kind", "H16"]
        )

    with (
        patch("operational.train40_system.history_resources8_fast.verify_admission"),
        patch.object(controller_c2f_graph, "run", existing_run),
        pytest.raises(ValueError, match="Unexpected H8"),
    ):
        controller_h8_fast.run(tmp_path, tmp_path, tmp_path)
