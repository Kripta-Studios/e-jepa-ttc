from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.upload_rgb_sparse_frames_v6 import remote_staged_credit


def test_large_inventory_is_sent_on_stdin_not_command_line() -> None:
    files = [(f"data/train/s/rgb_shards/{index:05d}.tar", 100) for index in range(100)]
    observed: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.update(command=command, **kwargs)
        return SimpleNamespace(
            stdout=json.dumps({"credit_bytes": 0, "target_files": 0, "partial_files": 0})
        )

    credit, _ = remote_staged_credit(["ssh"], "/workspace/root", files, runner=runner)
    assert credit == 0
    command = observed["command"]
    assert isinstance(command, list) and len(str(command[-1])) < 4_000
    assert len(json.loads(str(observed["input"]))) == 100


def test_credit_bound_is_preserved() -> None:
    with pytest.raises(ValueError, match="outside pending"):
        remote_staged_credit(
            ["ssh"],
            "/workspace/root",
            [("data/train/s/rgb_shards/a.tar", 10)],
            runner=lambda *_args, **_kwargs: SimpleNamespace(
                stdout=json.dumps({"credit_bytes": 11, "target_files": 0, "partial_files": 1})
            ),
        )


def test_credit_semantics_include_target_and_partial() -> None:
    report = {"credit_bytes": 10, "target_files": 1, "partial_files": 1}
    credit, observed = remote_staged_credit(
        ["ssh"],
        "/workspace/root",
        [
            ("data/train/s/rgb_shards/a.tar", 6),
            ("data/train/s/rgb_shards/b.tar", 8),
        ],
        runner=lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(report)),
    )
    assert credit == 10 and observed == report


def test_unsafe_path_rejected_before_remote_call() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        remote_staged_credit(
            ["ssh"],
            "/workspace/root",
            [("../escape", 1)],
            runner=lambda *_args, **_kwargs: None,
        )
