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
        "schema": "simplex_t_postprocessing_launch_v1",
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
        assert kwargs["resource_ok"]()
        return {"status": "WIRED", "campaign_complete": False}

    monkeypatch.setattr(module, "postprocess_configured_campaign", post)
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["campaign_complete"] is False


@pytest.mark.parametrize("mutation", ["hash", "schema", "stage", "no_t2", "extra"])
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
            "written_volume_free_bytes": [60_000_000_000],
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
