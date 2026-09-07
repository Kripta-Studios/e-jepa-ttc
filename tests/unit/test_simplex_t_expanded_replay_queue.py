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
