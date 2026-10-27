"""Scheduling preserves the frozen fitting block and rejects unsafe recovery."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runpod_event_context_overlap_v1 import (
    compile_training_block,
    extract_training_block,
    wait_guard,
)


def test_extracted_training_uses_refiner_only_and_original_fit():
    source = Path("operational/final_refit/scientific_oof_pipeline.py").read_text()
    selected = extract_training_block(source)
    tree = ast.parse(source)
    original = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "context_branch"
    )
    dumps = [ast.dump(n, include_attributes=False) for n in original.body]
    extracted = [ast.dump(n, include_attributes=False) for n in selected.body]
    offset = dumps.index(extracted[0])
    assert dumps[offset : offset + len(extracted)] == extracted
    raw_accesses = [
        n
        for n in ast.walk(selected)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "raws"
    ]
    assert raw_accesses
    assert all(
        isinstance(n.slice, ast.Constant) and n.slice.value == "refiner" for n in raw_accesses
    )
    assert isinstance(selected.body[-1], ast.Expr)
    assert selected.body[-1].value.func.id == "fit_head"


def test_modified_scientific_source_fails_closed(tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text("# changed\n")
    with pytest.raises(ValueError, match="Frozen context pipeline differs"):
        compile_training_block(SimpleNamespace(__file__=str(source)))


class PauseError(RuntimeError):
    def __init__(self, manual=False):
        self.manual = manual


def test_only_automatic_resources_retry():
    calls = []

    def guard():
        calls.append(1)
        if len(calls) < 3:
            raise PauseError()

    sleeps = []
    wait_guard(guard, PauseError, sleep=sleeps.append)()
    assert len(calls) == 3 and sleeps == [5, 5]


@pytest.mark.parametrize(
    "error", [PauseError(True), ValueError("integrity"), TimeoutError("deadline")]
)
def test_manual_deadline_and_integrity_propagate(error):
    def guard():
        raise error

    with pytest.raises(type(error)):
        wait_guard(guard, PauseError, sleep=lambda _: pytest.fail("Must not retry"))()
