"""CLI preparation validates scope before loading caches or target arrays."""

import json
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import configured_preparation as module


@pytest.fixture
def prepared_config(tmp_path, monkeypatch):
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps({"worktree": str(tmp_path), "shared_coordination": str(tmp_path)}),
        encoding="utf-8",
    )
    config = tmp_path / "sources.json"
    config.write_text("{}", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    manifest = index / "INDEX_MANIFEST.json"
    ancestry = {"path": "fixture", "sha256": "a" * 64}
    manifest.write_text(json.dumps({"ancestry": ancestry}), encoding="utf-8")
    sources = SimpleNamespace(index_root=index, release=lambda: None)
    monkeypatch.setattr(
        module,
        "open_acknowledged_source_configuration",
        lambda *args: (sources, {"ack_sha256": "b" * 64}),
    )
    monkeypatch.setattr(
        module,
        "verified_ack",
        lambda *args: {"producers": {"authoritative_historical_manifest": ancestry}},
    )
    monkeypatch.setattr(
        module,
        "sha256",
        lambda path: (
            "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
            if path == manifest
            else sha256(path)
        ),
    )
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(module.torch, "get_num_interop_threads", lambda: 2)
    return local, config, sha256(config), tmp_path / "output"


@pytest.mark.parametrize("pool", ["expansion", "dense", "matched"])
def test_expanded_configuration_not_reduced_to_original(tmp_path, pool):
    local = tmp_path / "local.json"
    local.write_text("{}", encoding="utf-8")
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({pool: {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="WAITING_EXPANDED_TIME_RECOGNITION"):
        module.prepare_configured_sources(
            local,
            config,
            sha256(config),
            tmp_path / "output",
            other_reserved_bytes=0,
            resume=False,
        )
    assert not (tmp_path / "output").exists()


def test_disk_floor_includes_pending_outputs_and_preparation_overhead(prepared_config, monkeypatch):
    checks = []
    monkeypatch.setattr(
        module,
        "admitted",
        lambda _: {"has_headroom": True, "written_volume_free_bytes": [40_067_108_863]},
    )

    def prepare(*args, **kwargs):
        kwargs["validate_prerequisites"]()
        checks.append(kwargs["resource_ok"]())
        assert kwargs["authority_sha256"] == "b" * 64
        assert kwargs["availability"]["t3"] is True  # Preparation is not gate authority.
        return {"status": "PAUSED_RESOURCE"}

    monkeypatch.setattr(module, "prepare_source_identities", prepare)
    assert (
        module.prepare_configured_sources(
            *prepared_config, other_reserved_bytes=20_000_000_000, resume=False
        )["status"]
        == "PAUSED_RESOURCE"
    )
    assert checks == [False]  # One byte short after both reservations.


def test_authority_rechecked_by_preparation_callback(prepared_config, monkeypatch):
    calls = []

    def prepare(*args, **kwargs):
        def revoked(*_):
            raise ValueError("ACK bytes changed")

        monkeypatch.setattr(module, "verified_ack", revoked)
        kwargs["validate_prerequisites"]()
        calls.append(True)

    monkeypatch.setattr(module, "prepare_source_identities", prepare)
    with pytest.raises(ValueError, match="ACK bytes changed"):
        module.prepare_configured_sources(*prepared_config, other_reserved_bytes=0, resume=False)
    assert not calls


def test_output_cannot_escape_companion(prepared_config):
    local, config, digest, output = prepared_config
    with pytest.raises(ValueError, match="inside the companion"):
        module.prepare_configured_sources(
            local,
            config,
            digest,
            output.parent.parent / "outside",
            other_reserved_bytes=0,
            resume=False,
        )


def test_acknowledged_expansion_preserves_configured_pools(prepared_config, monkeypatch):
    local, config, _, output = prepared_config
    config.write_text(json.dumps({"expansion": {}, "dense": {}, "matched": {}}))
    calls = []
    monkeypatch.setattr(module, "verify_expansion_authority", lambda *a, **k: calls.append("ack"))

    def prepare(*args, **kwargs):
        kwargs["validate_prerequisites"]()
        assert kwargs["availability"]["d1"] is True
        assert kwargs["availability"]["density"] is True
        return {"status": "fixture_configuration_preserved"}

    monkeypatch.setattr(module, "prepare_source_identities", prepare)
    result = module.prepare_configured_sources(
        local, config, sha256(config), output, other_reserved_bytes=0, resume=False
    )
    assert result["status"] == "fixture_configuration_preserved"
    assert len(calls) >= 3
