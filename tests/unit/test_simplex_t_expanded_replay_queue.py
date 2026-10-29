"""Synthetic expanded publication/resume without raw events or GPU work."""

from contextlib import contextmanager

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.expanded_replay_queue import run_expanded_blocks
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc


@pytest.fixture
def inputs():
    history = np.full((2, 16), -1, dtype=np.int64)
    history[:, -1] = [0, 1]
    index = {
        "tokens": np.array(["q0", "q1"]),
        "valid": history >= 0,
        "producer_family": np.array([[0, 0], [-1, -1], [-1, -1]]),
        "anchor_us": np.array([100, 200], dtype=np.int64),
        "roi_available_us": np.array([100, 200], dtype=np.int64),
        "lag_us": np.arange(15, -1, -1, dtype=np.int64),
    }
    families = [
        {
            "outer_fold": o,
            "role": f"inner{s}" if s < 3 else "outer_dev",
            "experts": dict.fromkeys(("A5", "C2F", "PAIR"), "a" * 64),
        }
        for o in range(3)
        for s in range(4)
    ]
    calls, releases = [], []

    @contextmanager
    def family(family_id):
        assert family_id == 0

        def infer(qi):
            calls.append(qi)
            experts = np.ones((1, 3), dtype=np.float32)
            features = np.zeros((1, 145), dtype=np.float32)
            features[:, 8:11] = expert_phase_from_ttc(experts)
            return {
                "features145": features,
                "expert_ttc": experts,
                "pair_features": np.zeros((1, 133), dtype=np.float32),
                "known": np.ones((1, 2), dtype=bool),
                "observation_ids": np.array([qi], dtype=np.int64),
                "anchor_us": index["anchor_us"][qi : qi + 1],
                "available_us": index["roi_available_us"][qi : qi + 1],
            }

        try:
            yield infer
        finally:
            releases.append(True)

    return (
        dict(
            identity={"pool": "D1"},
            index=index,
            families=families,
            history_loader=lambda outer: history,
            inference_family=family,
            validate_prerequisites=lambda: None,
            resource_ok=lambda: True,
            max_new_queries=1,
        ),
        calls,
        releases,
    )


def test_resume_skips_verified_blocks_and_releases_family(tmp_path, inputs):
    args, calls, releases = inputs
    output = tmp_path / "cache"
    assert run_expanded_blocks(output, **args)["status"] == "SLICE_COMPLETE"
    assert run_expanded_blocks(output, **args)["status"] == "SLICE_COMPLETE"
    result = run_expanded_blocks(output, **args)
    assert result["status"] == "ALL_EXPANDED_BLOCKS_COMPLETE_NOT_SCIENTIFIC_FREEZE"
    assert calls == [0, 1] and len(releases) == 2
    assert not (tmp_path / "CURRENT_REPLAY.lock").exists()


def test_query_major_retains_each_family_once_and_resumes_exact_blocks(tmp_path, inputs):
    args, _, _ = inputs
    args["index"]["producer_family"][1] = 4
    args["identity"]["input_reuse_order"] = "query_major_single_fp32_input_v1"
    original_factory = args["inference_family"]
    entered, exited, order = [], [], []

    @contextmanager
    def resident_factory(family):
        entered.append(family)
        with original_factory(0) as original:

            def infer(query):
                order.append((query, family))
                return original(query)

            try:
                yield infer
            finally:
                exited.append(family)

    args["inference_family"] = resident_factory
    args["max_new_queries"] = 3
    output = tmp_path / "cache"
    first = run_expanded_blocks(output, **args, query_major=True)
    assert first["status"] == "SLICE_COMPLETE"
    assert order == [(0, 0), (0, 4), (1, 0)]
    assert entered == [0, 4] and exited == [4, 0]
    second = run_expanded_blocks(output, **args, query_major=True)
    assert second["new_blocks"] == 1
    assert order[-1] == (1, 4)
    assert entered == [0, 4, 4]
    assert not (tmp_path / "CURRENT_REPLAY.lock").exists()


def test_d1_selection_only_infers_selected_queries_and_verifies(tmp_path, inputs):
    args, calls, _ = inputs
    args["identity"]["query_selection"] = {"sha256": "a" * 64}
    args["selected_queries"] = np.array([1], dtype=np.int64)
    output = tmp_path / "cache"
    run_expanded_blocks(output, **args)
    assert calls == [1]
    assert not (output / "family00_query00000.npz").exists()
    result = run_expanded_blocks(output, **args, verify_only=True)
    assert result["status"] == "EXPANDED_CACHE_VERIFIED_NOT_SCIENTIFIC_FREEZE"
    del args["selected_queries"]
    with pytest.raises(ValueError, match="explicit query subset"):
        run_expanded_blocks(output, **args, verify_only=True)


@pytest.mark.parametrize("rows", [[1, 1], [-1], [2], []])
def test_d1_selection_rejects_invalid_rows(tmp_path, inputs, rows):
    args, calls, _ = inputs
    args["identity"]["query_selection"] = {"sha256": "a" * 64}
    args["selected_queries"] = np.asarray(rows, dtype=np.int64)
    with pytest.raises(ValueError, match="invalid explicitly bound"):
        run_expanded_blocks(tmp_path / "cache", **args)
    assert not calls


@pytest.mark.parametrize("blocks", [0, 1, 2])
def test_read_only_verification_never_takes_lease_or_infers(tmp_path, inputs, blocks):
    args, calls, _ = inputs
    output = tmp_path / "cache"
    for _ in range(blocks):
        run_expanded_blocks(output, **args)
    lock = tmp_path / "CURRENT_REPLAY.lock"
    lock.write_text("another owner")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    args["inference_family"] = lambda _: pytest.fail("verifier opened inference")
    result = run_expanded_blocks(output, **args, verify_only=True)
    assert result["status"] == (
        "EXPANDED_CACHE_VERIFIED_NOT_SCIENTIFIC_FREEZE"
        if blocks == 2
        else "EXPANDED_CACHE_INCOMPLETE"
    )
    assert result["new_blocks"] == 0 and len(calls) == blocks
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("reject_second", [False, True])
def test_resident_query_major_validates_each_inference_without_duplicate(
    tmp_path, inputs, reject_second
):
    args, calls, _ = inputs
    args["identity"]["input_reuse_order"] = "query_major_single_fp32_input_v1"
    args["max_new_queries"] = 2
    checks = []

    def validate():
        checks.append(len(calls))
        if reject_second and calls:
            raise ValueError("authority changed after first inference")

    args["validate_prerequisites"] = validate
    if reject_second:
        with pytest.raises(ValueError, match="authority changed"):
            run_expanded_blocks(tmp_path / "cache", **args, query_major=True)
        assert calls == [0]
    else:
        result = run_expanded_blocks(tmp_path / "cache", **args, query_major=True)
        assert result["new_blocks"] == 2
        assert calls == [0, 1]
    # Entry, lease, first model load, first inference, second inference.
    assert checks == [0, 0, 0, 0, 1]


def test_read_only_verification_rejects_changed_block(tmp_path, inputs):
    args, _, _ = inputs
    output = tmp_path / "cache"
    run_expanded_blocks(output, **args)
    (output / "family00_query00000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="completed block changed"):
        run_expanded_blocks(output, **args, verify_only=True)


@pytest.mark.parametrize("failure", ["lease", "resource", "authority", "orphan"])
def test_no_inference_when_not_admitted(tmp_path, inputs, failure):
    args, calls, _ = inputs
    output = tmp_path / "cache"
    if failure == "lease":
        (tmp_path / "CURRENT_REPLAY.lock").write_text("another owner", encoding="utf-8")
    elif failure == "resource":
        args["resource_ok"] = lambda: False
    elif failure == "authority":

        def reject():
            raise ValueError("missing time authority")

        args["validate_prerequisites"] = reject
    else:
        output.mkdir()
        (output / "family00_query00000.partial").write_bytes(b"interrupted")
    if failure == "resource":
        assert run_expanded_blocks(output, **args)["status"] == "PAUSED_RESOURCE"
    else:
        with pytest.raises((ValueError, FileExistsError)):
            run_expanded_blocks(output, **args)
    assert calls == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_dense_reuse_validates_blocks_without_new_inference(tmp_path, inputs, corrupt):
    args, calls, releases = inputs
    original = tmp_path / "original"
    args["max_new_queries"] = 100
    run_expanded_blocks(original, **args)
    assert calls == [0, 1]
    args["identity"] = {"pool": "DENSE_OLD"}

    def reuse(family, query):
        with np.load(
            original / f"family{family:02d}_query{query:05d}.npz", allow_pickle=False
        ) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if corrupt:
            arrays["anchor_us"] += 1
        return arrays

    args["reuse_block"] = reuse
    output = tmp_path / "dense"
    if corrupt:
        with pytest.raises(ValueError, match="sensor times changed"):
            run_expanded_blocks(output, **args)
    else:
        result = run_expanded_blocks(output, **args)
        assert result["new_blocks"] == 0 and result["reused_blocks"] == 2
        assert "WITH_D0_REUSE" in result["status"]
        assert not list(output.glob("family*.npz"))
        seal_before = (output / "IDENTITY.json").read_bytes()
        verified = run_expanded_blocks(output, **args, verify_only=True)
        assert verified["status"] == "EXPANDED_CACHE_VERIFIED_NOT_SCIENTIFIC_FREEZE"
        assert verified["new_blocks"] == 0 and verified["reused_blocks"] == 2
        assert (output / "IDENTITY.json").read_bytes() == seal_before
        assert not list(output.glob("family*.npz"))
    assert calls == [0, 1]
    assert len(releases) == 1


def test_dense_reuse_cannot_be_silently_omitted(tmp_path, inputs):
    args, calls, _ = inputs
    args["identity"] = {"pool": "DENSE_OLD"}
    with pytest.raises(ValueError, match="explicit D0 reuse"):
        run_expanded_blocks(tmp_path / "dense", **args)
    assert calls == []
