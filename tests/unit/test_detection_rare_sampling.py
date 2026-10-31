from __future__ import annotations

import copy
import hashlib
import json

import pytest

from e_jepa_ttc.detection.rare_sampling import (
    ReplayScheduleSampler,
    load_schedule,
    make_schedule,
)
from scripts.materialize_detection_rare_sampling import rewrite


def fixture():
    tokens = [f"sequence:{i}" for i in range(16)]
    rows = {
        token: {
            "sample_token": token,
            "sequence_id": "sequence",
            "role": "train",
            "trainer_overflow": False,
            "trainable_identities": {"bicycle": ["shared"]} if i < 2 else {},
        }
        for i, token in enumerate(tokens)
    }
    baseline = list(range(16)) * 8
    return tokens, rows, baseline


def test_caps_and_diversity_are_enforced_on_actual_calendar():
    tokens, rows, baseline = fixture()
    plan = make_schedule(tokens, rows, baseline, seed=7)
    assert plan == make_schedule(tokens, rows, baseline, seed=7)
    assert plan["frame_exposures"][:2] == [32, 32]
    assert min(plan["frame_exposures"]) >= 1
    assert sum(plan["frame_exposures"]) == len(baseline)
    assert list(plan["identity_exposures"].values()) == [64]
    assert plan["audit"]["replaced_common_exposures"] == 48
    assert set(plan["indices"]) == set(range(16))


def test_non_train_metadata_and_missing_rows_are_rejected():
    tokens, rows, baseline = fixture()
    rows[tokens[0]]["role"] = "selection"
    with pytest.raises(ValueError, match="train token"):
        make_schedule(tokens, rows, baseline, seed=7)
    del rows[tokens[0]]
    with pytest.raises(ValueError, match="population"):
        make_schedule(tokens, rows, baseline, seed=7)


def test_no_silent_relaxation_if_caps_and_diversity_are_infeasible():
    tokens, rows, _ = fixture()
    for row in rows.values():
        row["trainable_identities"] = {"bicycle": ["shared"]}
    with pytest.raises(ValueError, match="Insufficient"):
        make_schedule(tokens, rows, list(range(16)) * 8, seed=7)


def test_exact_resume_and_exhaustion_reject_wrong_plan():
    tokens, rows, baseline = fixture()
    plan = make_schedule(tokens, rows, baseline, seed=7)
    uninterrupted = ReplayScheduleSampler(plan["indices"], 16, 7, "bound")
    uninterrupted.next_batch(31)
    state = uninterrupted.state_dict()
    resumed = ReplayScheduleSampler(plan["indices"], 16, 7, "bound")
    resumed.load_state_dict(state)
    assert resumed.next_batch(97) == uninterrupted.next_batch(97)
    assert resumed.epoch == 8
    with pytest.raises(ValueError, match="exhausted"):
        resumed.next_batch(1)
    bad = ReplayScheduleSampler(plan["indices"], 16, 7, "different")
    with pytest.raises(ValueError, match="another"):
        bad.load_state_dict(state)
    for value in (-1, 129, True, 2.5):
        with pytest.raises(ValueError, match="cursor"):
            resumed.load_state_dict(state | {"cursor": value})


def test_loader_recounts_calendar_and_rejects_tampering(tmp_path):
    tokens, rows, baseline = fixture()
    plan = make_schedule(tokens, rows, baseline, seed=7)
    plan["manifest_sha256"] = "manifest"
    path = tmp_path / "plan.json"

    def load(value):
        path.write_text(json.dumps(value))
        return load_schedule(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tokens=tokens,
            seed=7,
            updates=32,
            batch_size=4,
            manifest_sha256="manifest",
        )

    sampler, receipt = load(plan)
    assert sampler.next_batch(4) == plan["indices"][:4]
    assert receipt["maximum_multiplier"] == 4
    tuple_sampler, _ = load_schedule(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        tokens=tuple(tokens),
        seed=7,
        updates=32,
        batch_size=4,
        manifest_sha256="manifest",
    )
    assert tuple_sampler.next_batch(4) == plan["indices"][:4]
    bad = copy.deepcopy(plan)
    bad["indices"] = [0] * 128
    with pytest.raises(ValueError, match="cap"):
        load(bad)
    bad = copy.deepcopy(plan)
    bad["identity_exposures"] = {}
    with pytest.raises(ValueError, match="identity"):
        load(bad)
    with pytest.raises(ValueError, match="hash"):
        load_schedule(
            path,
            expected_sha256="wrong",
            tokens=tokens,
            seed=7,
            updates=32,
            batch_size=4,
            manifest_sha256="manifest",
        )


def test_materializer_fails_closed_on_unrecognized_producer():
    with pytest.raises(ValueError, match="anchor"):
        rewrite("print('unrecognized')", "a" * 64)
