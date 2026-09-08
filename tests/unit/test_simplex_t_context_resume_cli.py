"""Technical resume CLI wiring; the optimizer engine is explicitly mocked."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_benchmark_does_not_overlap_replay_or_load_sources(tmp_path, monkeypatch, device):
    path = Path(__file__).resolve().parents[2] / "scripts/probe_simplex_t_context_resume.py"
    spec = importlib.util.spec_from_file_location("benchmark_lease_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path)}), encoding="utf-8")
    lock = tmp_path / "artifacts/simplex_t/T1/CURRENT_REPLAY.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("existing-owner", encoding="utf-8")
    monkeypatch.setattr(module, "run_probe", lambda args: pytest.fail("must not run"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--local-paths",
            str(local),
            "--compiled",
            "unused",
            "--compiled-sha256",
            "c" * 64,
            "--index",
            "unused",
            "--dedup",
            "unused",
            "--output",
            str(tmp_path / "probe"),
            "--benchmark",
            "--device",
            device,
            "--feature-count",
            "145",
        ],
    )
    with pytest.raises(FileExistsError):
        module.main()
    assert lock.read_text(encoding="utf-8") == "existing-owner"
    assert not (tmp_path / "probe").exists()


def test_real_history_probe_plan_reserves_twenty_once_without_running_optimizer(
    tmp_path, monkeypatch
):
    path = Path(__file__).resolve().parents[2] / "scripts/probe_simplex_t_context_resume.py"
    spec = importlib.util.spec_from_file_location("context_resume_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "configs/experiment"
    config.mkdir(parents=True)
    (config / "simplex_t_coordination.json").write_text(
        json.dumps(dict(ack_filename="ack", ack_sha256="fixture")), encoding="utf-8"
    )
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps(dict(worktree=str(tmp_path), shared_coordination=str(tmp_path))),
        encoding="utf-8",
    )
    output = tmp_path / "probe"
    monkeypatch.setattr(
        module, "admitted", lambda *a: dict(has_headroom=True, host_available_bytes=16 * 1024**3)
    )
    monkeypatch.setattr(
        module,
        "verified_ack",
        lambda *a: dict(
            resources=dict(cpu_overlap_authorized=True),
            producers=dict(
                authoritative_historical_manifest=dict(path="fixture", sha256="fixture")
            ),
            interfaces=dict(role_manifest=dict(roles=dict(original=["s"]))),
        ),
    )
    monkeypatch.setattr(module, "compute_file_hash", lambda *a: "d" * 64)
    monkeypatch.setattr(module.torch, "set_num_interop_threads", lambda *a: None)

    def gather(ids):
        return (
            torch.zeros(1, 2, 17),
            torch.tensor([[[0.05, 0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 0.0]]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 3),
            torch.zeros(1),
            torch.ones(1),
        )

    source = SimpleNamespace(
        identity_sha256="a" * 64, population=1, history=np.array([[0, 1]]), gather=gather
    )
    monkeypatch.setattr(module, "load_context_sources", lambda *a, **kw: {"inner_oof": source})
    calls = []

    def fake_fit(source, config, folder, **kwargs):
        calls.append((folder.name, kwargs["stop_after"], kwargs["resume"]))
        folder.mkdir(parents=True, exist_ok=True)
        state = dict(
            identity=dict(source=source.identity_sha256, freeze=kwargs["freeze_sha256"]),
            completed_updates=kwargs["stop_after"],
            status="TECHNICAL_PARTIAL",
        )
        (folder / "checkpoint_last.pt").write_text(json.dumps(state), encoding="utf-8")
        return dict(status="TECHNICAL_PARTIAL", completed_updates=kwargs["stop_after"])

    monkeypatch.setattr(module, "fit", fake_fit)
    monkeypatch.setattr(module, "load_checkpoint", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--local-paths",
            str(local),
            "--compiled",
            "unused",
            "--compiled-sha256",
            "c" * 64,
            "--index",
            "unused",
            "--dedup",
            "unused",
            "--output",
            str(output),
        ],
    )
    module.main()
    assert calls == [("continuous", 10, False), ("split", 5, False), ("split", 10, True)]
    result = json.loads((output / "RESUME_QA.json").read_text())
    assert result["exact_complete_state_match"]
    assert result["executed_technical_updates"] == 20  # Mocked metadata, not executed work.
    budget = json.loads((tmp_path / "artifacts/simplex_t/TECHNICAL_BUDGET.json").read_text())
    contract = json.loads((output / "CONTRACT.json").read_text())
    operation = "real_context_cpu_resume_" + module.state_digest(contract)
    assert budget["reservations"] == {operation: 20}
    with pytest.raises(FileExistsError):
        module.main()
