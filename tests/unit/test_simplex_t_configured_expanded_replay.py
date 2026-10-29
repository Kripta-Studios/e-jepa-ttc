"""Configured launch checks and queue wiring without models or raw media."""

import json

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import configured_expanded_replay as module


@pytest.fixture
def launch(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    local = write(
        tmp_path / "paths.json",
        {
            "worktree": str(tmp_path),
            "shared_coordination": str(tmp_path / "shared"),
            "eap_root": str(tmp_path / "raw"),
        },
    )
    ancestry = {"path": str(tmp_path / "ancestry"), "sha256": "a" * 64}
    families = [{"outer_fold": 0, "role": "inner0", "experts": {}}]
    original = write(
        tmp_path / "original/INDEX_MANIFEST.json",
        {
            "ancestry": ancestry,
            "families": families,
        },
    )
    index = write(tmp_path / "index/INDEX_MANIFEST.json", {"queries": 1, "families": families})
    dedup = write(tmp_path / "dedup/DEDUP_MANIFEST.json", {})
    prep = write(tmp_path / "prep.json", {})
    config = {
        "schema": "simplex_t_expanded_replay_launch_v1",
        "pool": "D1",
        "index": "index",
        "dedup": "dedup",
        "dedup_sha256": sha256(dedup),
        "original_index": "original",
        "preprocessing": tmp_path.name + "/prep.json",
        "output": "artifacts/simplex_t/T1/expansion_context_features_fp32",
        "reserved_output_bytes": 2_000_000,
    }
    config_path = write(tmp_path / "launch.json", config)
    monkeypatch.setattr(module, "D0_INDEX_SHA256", sha256(original))
    monkeypatch.setattr(
        module, "POOL_INDEX_SHA256", {"D1": sha256(index), "DENSE_OLD": sha256(index)}
    )
    monkeypatch.setattr(
        module,
        "sha256",
        lambda p: (
            "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
            if p == prep
            else sha256(p)
        ),
    )
    ack = {
        "producers": {"authoritative_historical_manifest": ancestry},
        "interfaces": {"role_manifest": {"roles": {"expansion": ["seq"], "original": ["old"]}}},
    }
    monkeypatch.setattr(module, "verified_ack", lambda *a: ack)
    monkeypatch.setattr(
        module,
        "verify_expansion_authority",
        lambda *a, **kw: {
            "sha256": "b" * 64,
            "scope": {"D1_EXPANSION": {"sequences": ["seq"]}, "DENSE_OLD": {"sequences": ["old"]}},
        },
    )
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": True,
            "written_volume_free_bytes": [80_000_000_000],
        },
    )
    return local, config_path, config, write


def invoke(launch, *, inspect_only=True):
    local, path, _, _ = launch
    return module.run_configured_expanded_replay(
        local,
        path,
        sha256(path),
        max_new_queries=10,
        other_reserved_bytes=20_000_000_000,
        inspect_only=inspect_only,
    )


def test_inspection_never_opens_producers_or_queue(launch, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection opened producer or queue")

    monkeypatch.setattr(module, "verify_acknowledged_producers", forbidden)
    monkeypatch.setattr(module, "run_expanded_context_cache", forbidden)
    result = invoke(launch)
    assert result["allowed_sequences"] == ["seq"]
    assert result["models_loaded"] is False
    assert result["optimizer_updates"] == 0


@pytest.mark.parametrize("mutation", [None, "payload", "source", "memory", "speed"])
def test_query_major_requires_exact_pinned_qa(launch, monkeypatch, mutation):
    local, path, config, write = launch
    root = local.parent / "artifacts/simplex_t/T0/reuse_qa"
    dependency = write(root / "dependency.json", {"version": 1})
    contract = write(
        root / "CONTRACT.json",
        {
            "queries": [455, 1366, 2277, 3187, 4097, 5007],
            "pins": {str(dependency): sha256(dependency)},
        },
    )
    payload = {
        "status": "QUERY_MAJOR_INPUT_REUSE_EXACT_PASS",
        "compared_blocks_per_pass": 18,
        "optimizer_updates": 0,
        "sampled_rss_max_bytes": 2 * 1024**3,
        "results": [
            {"mode": "baseline", "preparations": 18, "hits": 0, "seconds": 100.0},
            {"mode": "query_major", "preparations": 6, "hits": 12, "seconds": 37.0},
        ],
    }
    if mutation == "memory":
        payload["sampled_rss_max_bytes"] = 5 * 1024**3
    if mutation == "speed":
        payload["results"][1]["seconds"] = 101.0
    qa = write(root / "QA.json", payload)
    ref = {
        "path": qa.relative_to(local.parent).as_posix(),
        "sha256": sha256(qa),
        "contract_sha256": sha256(contract),
    }
    config["query_major_qa"] = ref
    write(path, config)
    if mutation == "payload":
        write(qa, {})
    if mutation == "source":
        write(dependency, {"version": 2})
    if mutation is not None:
        with pytest.raises(ValueError):
            invoke(launch)
        return
    assert invoke(launch)["query_major_qa"] == ref
    monkeypatch.setattr(module, "verify_acknowledged_producers", lambda *a, **kw: None)

    def queue(output, **kwargs):
        assert kwargs["query_major"] is True
        assert kwargs["input_reuse_qa_binding"] == ref
        return {"status": "WIRED"}

    monkeypatch.setattr(module, "run_expanded_context_cache", queue)
    assert invoke(launch, inspect_only=False)["status"] == "WIRED"


def test_execution_reverifies_lineage_and_wires_real_validator(launch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module, "verify_acknowledged_producers", lambda *a, **kw: calls.append("lineage")
    )

    def queue(output, **kwargs):
        assert calls == ["lineage"]
        assert kwargs["pool"] == "D1"
        assert kwargs["max_new_queries"] == 10
        assert kwargs["allowed_sequences"] == {"seq"}
        assert kwargs["reuse_catalog_loader"] is None
        kwargs["validate_expanded_authority"]()
        assert kwargs["resource_ok"]()
        return {"status": "SLICE_COMPLETE", "new_blocks": 10, "optimizer_updates": 0}

    monkeypatch.setattr(module, "run_expanded_context_cache", queue)
    assert invoke(launch, inspect_only=False)["new_blocks"] == 10


@pytest.mark.parametrize("mutation", ["escape", "output", "reserve", "dedup", "reuse", "schema"])
def test_launch_rejects_invalid_configuration(launch, mutation):
    _, path, config, write = launch
    if mutation == "escape":
        config["index"] = "../index"
    elif mutation == "output":
        config["output"] = "artifacts/simplex_t/T1/context_features_fp32"
    elif mutation == "reserve":
        config["reserved_output_bytes"] = 1
    elif mutation == "dedup":
        config["dedup_sha256"] = "f" * 64
    elif mutation == "reuse":
        config["d0_reuse"] = {}
    else:
        config["schema"] = "unknown"
    write(path, config)
    with pytest.raises(ValueError):
        invoke(launch)


def test_resource_pause_before_authority(launch, monkeypatch):
    monkeypatch.setattr(module, "admitted", lambda *a: {"has_headroom": False})
    monkeypatch.setattr(module, "verified_ack", lambda *a: pytest.fail("authority opened"))
    assert invoke(launch)["status"] == "PAUSED_RESOURCE"


def test_ack_failure_never_reaches_execution(launch, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("missing supplementary authority")

    monkeypatch.setattr(module, "verify_expansion_authority", unavailable)
    with pytest.raises(ValueError, match="supplementary"):
        invoke(launch, inspect_only=False)


def test_changed_config_rejected_at_queue_boundary(launch, monkeypatch):
    monkeypatch.setattr(module, "verify_acknowledged_producers", lambda *a, **kw: None)

    def queue(output, **kwargs):
        launch[3](launch[1], {})
        kwargs["validate_expanded_authority"]()

    monkeypatch.setattr(module, "run_expanded_context_cache", queue)
    with pytest.raises(ValueError, match="configuration changed"):
        invoke(launch, inspect_only=False)


def test_dense_requires_three_real_catalogs(launch):
    _, path, config, write = launch
    config.update(
        pool="DENSE_OLD",
        d0_reuse={"0": {}},
        output="artifacts/simplex_t/T1/dense_context_features_fp32",
    )
    write(path, config)
    with pytest.raises(ValueError, match="all three"):
        invoke(launch)


@pytest.mark.parametrize("inspect_only", [True, False])
def test_dense_catalog_identities_reach_queue_without_fallback(launch, monkeypatch, inspect_only):
    _, path, config, write = launch
    config.update(
        pool="DENSE_OLD",
        output="artifacts/simplex_t/T1/dense_context_features_fp32",
        d0_reuse={
            str(outer): {
                "compiled": f"compiled/outer{outer}",
                "compiled_sha256": str(outer) * 64,
                "cache": "cache",
                "dedup": f"dedup/outer{outer}.npz",
            }
            for outer in range(3)
        },
    )
    write(path, config)
    loaded = []

    class Catalog:
        def __init__(self, **kwargs):
            loaded.append(kwargs["outer"])
            self.identity = {"compiled_sha256": kwargs["compiled_sha256"]}

    monkeypatch.setattr(module, "D0ReuseCatalog", Catalog)
    monkeypatch.setattr(module, "verify_acknowledged_producers", lambda *a, **kw: None)

    def queue(output, **kwargs):
        assert not inspect_only
        assert loaded == [0, 1, 2]
        assert kwargs["pool"] == "DENSE_OLD"
        assert kwargs["allowed_sequences"] == {"old"}
        for outer in range(3):
            assert (
                kwargs["reuse_catalog_loader"](outer).identity
                == (kwargs["reuse_expected_identities"][outer])
            )
        kwargs["validate_expanded_authority"]()
        return {"status": "SLICE_COMPLETE", "new_blocks": 0, "optimizer_updates": 0}

    monkeypatch.setattr(module, "run_expanded_context_cache", queue)
    result = invoke(launch, inspect_only=inspect_only)
    assert loaded == [0, 1, 2]
    if inspect_only:
        assert set(result["d0_reuse_identities"]) == {0, 1, 2}


def test_wrong_configuration_digest_rejected_before_authority(launch, monkeypatch):
    monkeypatch.setattr(module, "verified_ack", lambda *a: pytest.fail("authority opened"))
    with pytest.raises(ValueError, match="configuration changed"):
        module.run_configured_expanded_replay(
            launch[0],
            launch[1],
            "f" * 64,
            max_new_queries=1,
            other_reserved_bytes=0,
            inspect_only=True,
        )
