"""A missing H16 report never starts inference in the read-only verifier."""

import json
import runpy
import sys
import uuid
from pathlib import Path

import pytest


def test_h16_cli_missing_report_has_no_model_or_output(tmp_path, monkeypatch):
    work = Path(__file__).resolve().parents[2]
    local = tmp_path / "paths.json"
    local.write_text(json.dumps(dict(worktree=str(work))), encoding="utf-8")
    output = work / "artifacts/simplex_t/T0" / ("h16_verify_fixture_" + uuid.uuid4().hex)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "h16",
            "--local-paths",
            str(local),
            "--preprocessing-manifest",
            str(tmp_path / "not_opened.json"),
            "--output",
            str(output),
            "--other-reserved-bytes",
            "0",
            "--verify-only",
        ],
    )
    module = runpy.run_path(str(work / "scripts/run_simplex_t_h16_replay_qa.py"))
    scope = module["main"].__globals__
    scope["admitted"] = lambda _: dict(
        has_headroom=True, written_volume_free_bytes=[100_000_000_000]
    )
    scope["execute_h16_qa"] = lambda *a, **k: pytest.fail("verification must not execute inference")
    with pytest.raises(SystemExit) as stopped:
        module["main"]()
    assert stopped.value.code == 10
    assert not output.exists()
