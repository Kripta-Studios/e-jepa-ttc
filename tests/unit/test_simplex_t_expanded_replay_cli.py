"""Resource interruption reporting must not disguise data failures or invent work."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def entry(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_expanded_replay.py"
    spec = importlib.util.spec_from_file_location("expanded_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path)}), encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--local-paths",
            str(local),
            "--config",
            "launch.json",
            "--config-sha256",
            "a" * 64,
            "--other-reserved-bytes",
            "20000000000",
            "--max-new-queries",
            "81920",
            "--report",
            str(report),
        ],
    )
    return module, report


@pytest.mark.parametrize(
    "error",
    [
        InterruptedError("PAUSED_RESOURCE: stream support"),
        RuntimeError("PAUSED_RESOURCE: historical bindings"),
        RuntimeError("RESOURCE_PAUSE: source compilation"),
    ],
)
def test_resource_interrupt_preserves_unknown_progress(entry, monkeypatch, error):
    module, report = entry

    def pause(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "run_configured_expanded_replay", pause)
    assert module.main() == 2
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "PAUSED_RESOURCE"
    assert result["new_blocks"] is None
    assert result["optimizer_updates"] == 0
    assert result["launch_configuration_sha256"] == "a" * 64
    assert result["inspect_only"] is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("checkpoint corruption"),
        InterruptedError("unclassified interrupted operation"),
        ValueError("raw index differs"),
        FileExistsError("another owner holds lease"),
    ],
)
def test_nonresource_errors_are_not_reported_as_pauses(entry, monkeypatch, error):
    module, report = entry

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "run_configured_expanded_replay", fail)
    with pytest.raises(type(error), match=str(error)):
        module.main()
    assert not report.exists()


def test_existing_report_is_preserved_before_execution(entry, monkeypatch):
    module, report = entry
    report.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(
        module, "run_configured_expanded_replay", lambda *a, **kw: pytest.fail("ran")
    )
    with pytest.raises(ValueError, match="new companion-local"):
        module.main()
    assert report.read_text(encoding="utf-8") == "preserved"


def test_normal_completion_retains_observed_count(entry, monkeypatch):
    module, report = entry
    monkeypatch.setattr(
        module,
        "run_configured_expanded_replay",
        lambda *a, **kw: {
            "status": "SLICE_COMPLETE",
            "new_blocks": 17,
            "optimizer_updates": 0,
        },
    )
    assert module.main() == 0
    assert json.loads(report.read_text(encoding="utf-8"))["new_blocks"] == 17
