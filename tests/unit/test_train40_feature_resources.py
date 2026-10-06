"""The H8 resource variant cannot change any frozen mathematical expression."""

import ast
from pathlib import Path
from unittest.mock import patch

from operational.train40_system.feature_resources import expected_variant

DIRECTORY = Path(__file__).resolve().parents[2] / "operational/train40_system"


def test_only_four_resource_sites_differ_from_frozen_kernel() -> None:
    original = ast.parse((DIRECTORY / "history_features.py").read_text(encoding="utf-8"))
    variant = ast.parse((DIRECTORY / "history_resources8.py").read_text(encoding="utf-8"))
    restored = 0
    for node in ast.walk(variant):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ProcessPoolExecutor":
                keyword = next(k for k in node.keywords if k.arg == "max_workers")
                assert isinstance(keyword.value, ast.Constant) and keyword.value.value == 8
                keyword.value.value = 4
                restored += 1
            if node.func.id == "min" and isinstance(node.args[0], ast.BinOp):
                expression = node.args[0]
                if isinstance(expression.left, ast.Name) and expression.left.id == "row":
                    assert isinstance(expression.right, ast.Constant)
                    assert expression.right.value == 8
                    expression.right.value = 4
                    restored += 1
        if isinstance(node, ast.Constant) and node.value == "FEATURES_EIGHT_WORKER_FREEZE.json":
            node.value = "FEATURES_FREEZE.json"
            restored += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--kind"
            ):
                keyword = next(k for k in node.keywords if k.arg == "choices")
                assert isinstance(keyword.value, ast.Tuple)
                assert len(keyword.value.elts) == 1
                assert isinstance(keyword.value.elts[0], ast.Constant)
                assert keyword.value.elts[0].value == "H8"
                keyword.value.elts.insert(0, ast.Constant(value="PAIR"))
                restored += 1
    assert restored == 4
    assert ast.dump(variant, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_resource_generation_rejects_unexpected_kernel_change() -> None:
    original = (DIRECTORY / "history_features.py").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="frozen resource site changed"):
        expected_variant(original.replace("min(row + 4, len(jobs))", "min(row + 6, len(jobs))"))


def test_controller_changes_only_the_h8_feature_entrypoint() -> None:
    from operational.train40_system import controller_resources8

    output = Path("unused-no-subprocess")
    arguments = ["--raw-root", "unused", "--kind", "H8"]
    with patch.object(controller_resources8.controller_resume, "launch") as launcher:
        controller_resources8.launch(
            output, "h8_features", "operational.train40_system.history_features", arguments
        )
        launcher.assert_called_once_with(
            output, "h8_features", "operational.train40_system.history_resources8", arguments
        )
        launcher.reset_mock()
        controller_resources8.launch(
            output, "a5", "operational.train40_system.engine", ["--arm", "a5"]
        )
        launcher.assert_called_once_with(
            output, "a5", "operational.train40_system.engine", ["--arm", "a5"]
        )


def test_resource_variant_is_recognized_by_the_one_heavy_writer_gate() -> None:
    from operational.train40_system import controller_resources8

    with patch.object(
        controller_resources8,
        "_active",
        side_effect=lambda marker: marker == "operational.train40_system.history_resources8",
    ) as existing:
        assert controller_resources8.active("operational.train40_system.history_features")
        assert existing.call_count == 2
        existing.reset_mock()
        assert not controller_resources8.active("operational.train40_system.engine")
        existing.assert_called_once_with("operational.train40_system.engine")
