"""D1 configuration-to-queue integration with no producer or sensor inference."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.expanded_execution import run_d1_context_cache


@pytest.mark.parametrize("failure", ["pause", "authority", "roles"])
def test_d1_execution_checks_authority_before_inference(tmp_path, monkeypatch, failure):
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
    path = index / "INDEX_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    dedup_path = dedup / "DEDUP_MANIFEST.json"
    dedup_path.write_text(
        json.dumps(
            {
                "status": "D1_CONTENT_INDEX_READY_NOT_FEATURE_CACHE_OR_REPLAY_AUTHORIZATION",
                "identity": {"index_manifest_sha256": sha256(path)},
                "outputs": [{}, {}, {}],
            }
        ),
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
    )
    output = tmp_path / "cache"
    if failure == "pause":
        assert run_d1_context_cache(output, **args)["status"] == "PAUSED_RESOURCE"
    else:
        with pytest.raises(ValueError):
            run_d1_context_cache(output, **args)
    assert not output.exists()
    assert not (tmp_path / "CURRENT_REPLAY.lock").exists()
