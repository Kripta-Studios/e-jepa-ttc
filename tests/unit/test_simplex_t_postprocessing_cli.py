"""Pinned postprocessing CLI wiring and absolute resource admission."""

import importlib.util
import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


@pytest.fixture
def entry(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/postprocess_simplex_t_campaign.py"
    spec = importlib.util.spec_from_file_location("postprocessing_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "schema": "simplex_t_postprocessing_launch_v2",
        "accounting": {
            "journal": "fits/PHYSICAL_WORK.json",
            "journal_sha256": "1" * 64,
            "reconciliation": "reconciliation.json",
            "reconciliation_sha256": "2" * 64,
            "ledger": "ledger.json",
            "ledger_sha256": "3" * 64,
        },
        "local_paths": "local.json",
        "source_configuration": "sources.json",
        "source_configuration_sha256": "a" * 64,
        "evidence_profile": "qa.json",
        "evidence_profile_sha256": "b" * 64,
        "freeze": "freeze.json",
        "freeze_sha256": "c" * 64,
        "roots": {"work": str(tmp_path)},
        "output": str(tmp_path / "output"),
        "publications": {
            "T2": {
                "publication": "T2/publication.json",
                "publication_sha256": "d" * 64,
                "endpoints": "T2/endpoints.json",
                "endpoints_sha256": "e" * 64,
                "checkpoint_root": "T2/fits",
                "availability": {"d1": True},
            }
        },
    }
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(config), encoding="utf-8")
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
    monkeypatch.setattr(module.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(module.torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": True,
            "written_volume_free_bytes": [80_000_000_000],
        },
    )
    return module, launch, config, argv


def test_postprocessing_cli_passes_real_configured_inputs(entry, monkeypatch, capsys):
    module, _, config, _ = entry

    def post(output, **kwargs):
        assert output == Path(config["output"])
        assert kwargs["source_configuration_sha256"] == "a" * 64
        assert set(kwargs["phases"]) == {"T2"}
        assert kwargs["phases"]["T2"].endpoints_sha256 == "e" * 64
        assert kwargs["accounting_pins"].journal == Path("fits/PHYSICAL_WORK.json")
        assert kwargs["accounting_pins"].ledger_sha256 == "3" * 64
        assert kwargs["resource_ok"]()
        return {"status": "WIRED", "campaign_complete": False}

    monkeypatch.setattr(module, "postprocess_configured_campaign", post)
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["campaign_complete"] is False


@pytest.mark.parametrize("missing_attempt", [False, True])
def test_delivery_launch_requires_resource_pins(entry, monkeypatch, missing_attempt):
    module, launch, config, argv = entry
    config["schema"] = "simplex_t_postprocessing_launch_v3"
    config["delivery"] = dict(
        output="artifacts/delivery",
        analysis_commit="f" * 40,
        resource_attempts=[]
        if missing_attempt
        else [
            dict(
                launch="attempt.json",
                launch_sha256="4" * 64,
                receipt="receipt.json",
                receipt_sha256="5" * 64,
            )
        ],
    )
    launch.write_text(json.dumps(config), encoding="utf-8")
    argv[4] = sha256(launch)

    def post(*args, **kwargs):
        assert not missing_attempt
        delivery = kwargs["delivery"]
        assert delivery.output == Path("artifacts/delivery")
        assert delivery.analysis_commit == "f" * 40
        assert delivery.resource_attempts[0].receipt_sha256 == "5" * 64
        return {"status": "WIRED_NOT_EXECUTED"}

    monkeypatch.setattr(module, "postprocess_configured_campaign", post)
    if missing_attempt:
        with pytest.raises(ValueError, match="resource attempts"):
            module.main()
    else:
        assert module.main() == 0


@pytest.mark.parametrize("mutation", ["hash", "schema", "stage", "no_t2", "extra", "accounting"])
def test_invalid_launch_fails_before_postprocessing(entry, monkeypatch, mutation):
    module, launch, config, argv = entry
    if mutation == "hash":
        argv[4] = "f" * 64
    else:
        if mutation == "schema":
            config["schema"] = "wrong"
        elif mutation == "stage":
            config["publications"]["T7"] = config["publications"]["T2"]
        elif mutation == "no_t2":
            config["publications"] = {}
        elif mutation == "accounting":
            config["accounting"].pop("journal_sha256")
        else:
            config["caller_targets"] = "not allowed"
        launch.write_text(json.dumps(config), encoding="utf-8")
        argv[4] = sha256(launch)
    monkeypatch.setattr(
        module, "postprocess_configured_campaign", lambda *a, **kw: pytest.fail("ran")
    )
    with pytest.raises(ValueError):
        module.main()


def test_reservations_are_subtracted_before_postprocessing(entry, monkeypatch, capsys):
    module, _, _, _ = entry
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": True,
            "written_volume_free_bytes": [40_000_000_000],
        },
    )
    monkeypatch.setattr(
        module, "postprocess_configured_campaign", lambda *a, **kw: pytest.fail("ran")
    )
    assert module.main() == 3
    assert json.loads(capsys.readouterr().out)["status"] == "PAUSED_RESOURCE"


@pytest.mark.parametrize("resource", [True, False])
def test_only_resource_failure_is_translated_to_pause(entry, monkeypatch, resource):
    module, _, _, _ = entry

    def fail(*a, **kw):
        raise RuntimeError("PAUSED_RESOURCE: analysis" if resource else "bad sealed predictions")

    monkeypatch.setattr(module, "postprocess_configured_campaign", fail)
    if resource:
        assert module.main() == 3
    else:
        with pytest.raises(RuntimeError, match="bad sealed"):
            module.main()


@pytest.mark.parametrize("state", ["missing", "complete", "invalid", "legacy"])
def test_read_only_delivery_verifier_exit_contract(entry, monkeypatch, state):
    module, launch, config, argv = entry
    argv.append("--verify-only")
    output = launch.parent / "delivery"
    if state != "legacy":
        config["schema"] = "simplex_t_postprocessing_launch_v3"
        config["delivery"] = {
            "output": str(output),
            "analysis_commit": "f" * 40,
            "resource_attempts": [
                {
                    "launch": "attempt.json",
                    "launch_sha256": "4" * 64,
                    "receipt": "receipt.json",
                    "receipt_sha256": "5" * 64,
                }
            ],
        }
    launch.write_text(json.dumps(config), encoding="utf-8")
    argv[4] = sha256(launch)
    if state in {"complete", "invalid"}:
        output.mkdir()
        (output / "DELIVERY.json").write_text("{}")

    def verify(*a, **kwargs):
        assert state in {"complete", "invalid"}
        assert kwargs["verify_only"] is True
        if state == "invalid":
            raise ValueError("bad payload")
        return {"status": "TEST_WIRING_ONLY"}

    monkeypatch.setattr(module, "postprocess_configured_campaign", verify)
    if state in {"invalid", "legacy"}:
        with pytest.raises(ValueError, match="bad payload|requires launch v3"):
            module.main()
    else:
        assert module.main() == (10 if state == "missing" else 0)
