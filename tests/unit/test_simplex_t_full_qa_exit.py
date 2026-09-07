"""Runner exit propagation with a mocked pytest session and synthetic budget only."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_full_qa_preserves_pytest_exit_code(tmp_path, monkeypatch, exit_code):
    script = Path(__file__).resolve().parents[2] / "scripts/run_simplex_t_full_unit_qa.py"
    spec = importlib.util.spec_from_file_location("full_qa_exit_fixture", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *_: {"has_headroom": True, "written_volume_free_bytes": [100_000_000_000]},
    )

    def reserve(self, key, updates):
        assert self.path.is_relative_to(tmp_path) and updates == 20
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"fixture": True}))

    monkeypatch.setattr(module.TechnicalBudget, "reserve", reserve)
    output = tmp_path / "artifacts/simplex_t/T0/fixture"
    monkeypatch.setattr(
        "sys.argv",
        ["qa", "--output", str(output), "--operation-id", "fixture", "--other-reserved-bytes", "0"],
    )

    def run(*args, plugins, **kwargs):
        plugin = plugins[0]
        plugin.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid="fixture")]))
        plugin.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="fixture",
                when="call",
                outcome="failed" if exit_code else "passed",
                duration=0.0,
                failed=bool(exit_code),
                longrepr="synthetic failure",
            )
        )
        return exit_code

    monkeypatch.setattr(pytest, "main", run)
    if exit_code:
        with pytest.raises(SystemExit) as caught:
            module.main()
        assert caught.value.code == exit_code
    else:
        module.main()
    result = json.loads((output / "RESULT.json").read_text())
    assert result["exit_code"] == exit_code
    assert result["completed_optimizer_updates"] == 0
