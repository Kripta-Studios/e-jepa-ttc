"""D1 configuration-to-queue integration with no producer or sensor inference."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.expanded_execution import run_d1_context_cache


@pytest.mark.parametrize("pool", ["D1", "DENSE_OLD"])
@pytest.mark.parametrize("failure", ["pause", "authority", "roles", "wiring"])
def test_d1_execution_checks_authority_before_inference(tmp_path, monkeypatch, failure, pool):
    import e_jepa_ttc.simplex_t.expanded_execution as module

    index, dedup = tmp_path / "index", tmp_path / "dedup"
    index.mkdir()
    dedup.mkdir()
    ancestry = tmp_path / "ancestry.json"
    ancestry.write_text(json.dumps({"input_bindings": {}}), encoding="utf-8")
    prep = tmp_path / "prep.json"
    prep.write_text(
        json.dumps({"config": {"roi_size": 2, "event_pixel_diff": 1}}), encoding="utf-8"
    )
    valid = np.zeros((1, 16), dtype=bool)
    valid[:, -1] = True
    np.savez(
        index / "query_context_index.npz",
        tokens=np.array(["q"]),
        sequences=np.array(["seq"]),
        producer_family=np.array([[0], [4], [8]]),
        valid=valid,
    )
    families = [
        {
            "outer_fold": o,
            "role": f"inner{s}" if s < 3 else "outer_dev",
            "experts": dict.fromkeys(("A5", "C2F", "PAIR"), "a" * 64),
        }
        for o in range(3)
        for s in range(4)
    ]
    manifest = {
        "status": "D1_INPUT_INDEX_PREPARED_PENDING_OWNER_TIME_ACK_AND_REPLAY",
        "queries": 1,
        "families": families,
        "ancestry": {"path": str(ancestry), "sha256": sha256(ancestry)},
        "index_sha256": sha256(index / "query_context_index.npz"),
    }
    if pool == "DENSE_OLD":
        manifest["status"] = "DENSE_INPUT_INDEX_PREPARED_PENDING_TIME_ACK_AND_REPLAY"
        del manifest["ancestry"]
    path = index / "INDEX_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    dedup_path = dedup / "DEDUP_MANIFEST.json"
    dedup_manifest = {
        "status": "D1_CONTENT_INDEX_READY_NOT_FEATURE_CACHE_OR_REPLAY_AUTHORIZATION",
        "identity": {"index_manifest_sha256": sha256(path)},
        "outputs": [{}, {}, {}],
    }
    if pool == "DENSE_OLD":
        dedup_manifest["status"] = "DENSE_CONTENT_INDEX_READY_PENDING_TIME_ACK_AND_FEATURE_REPLAY"
        dedup_manifest["input_manifest_sha256"] = sha256(path)
    dedup_path.write_text(
        json.dumps(dedup_manifest),
        encoding="utf-8",
    )
    # Only the frozen preprocessing fixture digest is substituted; all other pins are real.
    monkeypatch.setattr(
        module,
        "sha256",
        lambda p: (
            "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
            if p == prep
            else sha256(p)
        ),
    )
    monkeypatch.setattr(module.torch, "get_num_interop_threads", lambda: 2)
    monkeypatch.setattr(module, "verify_expanded_stream_support", lambda *a, **kw: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("resource pause must precede producer inference")

    monkeypatch.setattr(module, "expanded_inference_family", forbidden)
    checks = []

    def resources():
        checks.append(True)
        return len(checks) == 1

    def authority():
        if failure == "authority":
            raise ValueError("missing expanded ACK")

    args = dict(
        index_root=index,
        index_manifest_sha256=sha256(path),
        dedup_root=dedup,
        dedup_manifest_sha256=sha256(dedup_path),
        ancestry=ancestry,
        ancestry_sha256=sha256(ancestry),
        preprocessing=prep,
        raw_train_root=tmp_path,
        allowed_sequences={"other"} if failure == "roles" else {"seq"},
        validate_expanded_authority=authority,
        resource_ok=resources,
        max_new_queries=1,
        pool=pool,
    )
    if pool == "DENSE_OLD":
        args.update(
            reuse_catalog_loader=forbidden,
            reuse_expected_identities={o: {"compiled_sha256": str(o) * 64} for o in range(3)},
            authorized_families=families,
        )
    output = tmp_path / "cache"
    if failure == "wiring":

        def queue(destination, **kwargs):
            assert destination == output
            kwargs["validate_prerequisites"]()
            identity = kwargs["identity"]
            assert json.loads(json.dumps(identity)) == identity
            assert identity["pool"] == pool
            if pool == "DENSE_OLD":
                assert isinstance(kwargs["reuse_block"], module.DenseReplayReuse)
                assert set(identity["d0_reuse_identities"]) == {"0", "1", "2"}
            else:
                assert kwargs["reuse_block"] is None
            return {"status": "WIRED_WITHOUT_INFERENCE"}

        monkeypatch.setattr(module, "run_expanded_blocks", queue)
        assert run_d1_context_cache(output, **args)["status"] == "WIRED_WITHOUT_INFERENCE"
    elif failure == "pause":
        assert run_d1_context_cache(output, **args)["status"] == "PAUSED_RESOURCE"
    else:
        with pytest.raises(ValueError):
            run_d1_context_cache(output, **args)
    assert not output.exists()
    assert not (tmp_path / "CURRENT_REPLAY.lock").exists()


@pytest.mark.parametrize("identities", [None, {}, {0: {}}, {0: {}, 1: {}}])
def test_dense_execution_requires_all_reuse_folds(tmp_path, identities):
    with pytest.raises(ValueError, match="all D0 fold identities"):
        run_d1_context_cache(
            tmp_path,
            index_root=tmp_path,
            index_manifest_sha256="",
            dedup_root=tmp_path,
            dedup_manifest_sha256="",
            ancestry=tmp_path,
            ancestry_sha256="",
            preprocessing=tmp_path,
            raw_train_root=tmp_path,
            allowed_sequences=set(),
            validate_expanded_authority=lambda: None,
            resource_ok=lambda: True,
            max_new_queries=1,
            pool="DENSE_OLD",
            reuse_catalog_loader=lambda outer: None,
            reuse_expected_identities=identities,
            authorized_families=[],
        )
