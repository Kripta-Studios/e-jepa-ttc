"""CLI observation receipts on success, resource pause and error; no scientific fits."""

import importlib.util
import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


@pytest.mark.parametrize(
    "outcome", ["PHASE_INCOMPLETE", "PHASE_PUBLICATION_VERIFIED_NOT_T6", "pause", "error"]
)
def test_read_only_phase_verifier_never_writes_receipt(entry, monkeypatch, outcome):
    module, _, _, argv, receipt = entry
    argv.append("--verify-only")
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"preserve existing attempt")
    before = (receipt.read_bytes(), receipt.stat().st_mtime_ns)

    def execute(**kwargs):
        assert kwargs["verify_only"] is True
        if outcome == "pause":
            raise InterruptedError("PAUSED_RESOURCE: verifier")
        if outcome == "error":
            raise ValueError("bad phase")
        return {"status": outcome}

    monkeypatch.setattr(module, "execute_configured_phase", execute)
    if outcome == "error":
        with pytest.raises(ValueError, match="bad phase"):
            module.main()
    else:
        assert module.main() == (
            3 if outcome == "pause" else 10 if outcome == "PHASE_INCOMPLETE" else 0
        )
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before


@pytest.fixture
def entry(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_frozen_phase.py"
    spec = importlib.util.spec_from_file_location("scientific_resource_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = tmp_path / "artifacts/simplex_t/resource_observations/attempt.json"
    config = dict(
        schema="simplex_t_frozen_phase_launch_v2",
        local_paths="local.json",
        source_configuration="source.json",
        source_configuration_sha256="a" * 64,
        evidence_profile="qa.json",
        evidence_profile_sha256="b" * 64,
        freeze="freeze.json",
        freeze_sha256="c" * 64,
        roots={"work": str(tmp_path)},
        execution=str(tmp_path / "fits"),
        publication=str(tmp_path / "predictions"),
        stage="T2",
        availability={},
        publications={},
        resource_receipt=str(receipt),
    )
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(config))
    argv = [
        "runner",
        "--launch",
        str(launch),
        "--launch-sha256",
        sha256(launch),
        "--other-reserved-bytes",
        "20000000000",
        "--own-reserved-bytes",
        "1000000000",
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(module.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: dict(
            has_headroom=True,
            process_tree_rss_bytes=200,
            host_available_bytes=10_000_000_000,
            written_volume_free_bytes=[80_000_000_000],
        ),
    )
    return module, config, launch, argv, receipt


@pytest.mark.parametrize("outcome", ["COMPLETE", "PAUSED_RESOURCE", "error"])
def test_attempt_observation_is_not_execution_accounting(entry, monkeypatch, outcome):
    module, config, _, _, receipt = entry

    def execute(**kwargs):
        assert kwargs["resource_ok"]() is True
        if outcome == "error":
            raise ValueError("fixture failure")
        return {"status": outcome}

    monkeypatch.setattr(module, "execute_configured_phase", execute)
    if outcome == "error":
        with pytest.raises(ValueError, match="fixture failure"):
            module.main()
    else:
        assert module.main() == (3 if outcome == "PAUSED_RESOURCE" else 0)
    result = json.loads(receipt.read_text())
    assert result["freeze_sha256"] == config["freeze_sha256"]
    assert result["admission_samples"] == 1
    assert result["observer_optimizer_updates"] == 0
    assert result["campaign_complete"] is False
    assert (result["failure"] is not None) == (outcome == "error")


@pytest.mark.parametrize("fault", ["existing", "outside", "old_schema", "reservation"])
def test_bad_receipt_or_contract_fails_before_execution(entry, monkeypatch, fault):
    module, config, launch, argv, receipt = entry
    if fault == "existing":
        receipt.parent.mkdir(parents=True)
        receipt.write_text("keep")
    elif fault == "outside":
        config["resource_receipt"] = str(launch.parent / "outside.json")
    elif fault == "old_schema":
        config["schema"] = "simplex_t_frozen_phase_launch_v1"
    else:
        argv[-1] = "1"
    launch.write_text(json.dumps(config))
    argv[4] = sha256(launch)
    monkeypatch.setattr(module, "execute_configured_phase", lambda **kw: pytest.fail("ran"))
    with pytest.raises((ValueError, SystemExit)):
        module.main()
    if fault == "existing":
        assert receipt.read_text() == "keep"
