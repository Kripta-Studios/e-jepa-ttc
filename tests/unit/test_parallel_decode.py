"""Exercise real original NPZ decoding, ordered consumption and unchanged resource ceilings."""

from __future__ import annotations

import io
import random
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.efficient_context import budget
from operational.efficient_context.common import atomic_json, digest
from operational.efficient_context.compressed_cache import CompressedInputCache
from operational.efficient_context.expanded_disk import install
from operational.efficient_context.parallel_decode import DecodingInputCache
from operational.efficient_context.parallel_inputs import ParallelInputCache


def fixture_campaign(tmp_path):
    out = tmp_path / "owned"
    raw = tmp_path / "raw"
    path = raw / "s/events.h5"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture raw identity; never read for events")
    policy = tmp_path / "policy.json"
    atomic_json(policy, {"fixture": True})
    atomic_json(out / "PROTOCOL.json", {"fixture": True})
    atomic_json(out / "garl/PROTOCOL.json", {"fixture": True})
    atomic_json(
        out / "DISK_RESOURCE_AUTHORIZATION.json",
        {
            "parent_protocol_sha256": digest(out / "PROTOCOL.json"),
            "parent_policy_sha256": digest(policy),
            "max_owned_bytes": 40_000_000_000,
            "max_native_cache_bytes": 32_000_000_000,
            "atomic_checkpoint_reservation_bytes": 1_000_000_000,
            "scientific_recipe_changed": False,
        },
    )
    return SimpleNamespace(
        out=out,
        raw=raw,
        policy_path=policy,
        policy={"max_tree_rss_gib": 16_000_000_000 / 1024**3},
        require_resources=lambda: None,
    )


def cache_fixture(tmp_path, count=8, *, ram=False):
    campaign = fixture_campaign(tmp_path)
    tokens = [f"token{i}" for i in range(count)]
    cache = DecodingInputCache(campaign, {t: {"sequence_id": "s"} for t in tokens})
    values = {}
    rng = np.random.default_rng(17)
    for i, token in enumerate(tokens):
        b = io.BytesIO()
        np.savez_compressed(
            b,
            events=rng.standard_normal((40, 3, 3)).astype(np.float32),
            visible=np.array([i, i + 1], np.float32),
            target=float(i - 4.125),
        )
        payload = b.getvalue()
        values[token] = CompressedInputCache.unpack(payload)
        if ram:
            cache.values[token] = payload
            cache.bytes += len(payload)
        else:
            key = cache.token_keys[token]
            (cache.root / key).write_bytes(payload)
            cache.files[key] = len(payload)
            cache.disk += len(payload)
    return cache, tokens, values


@pytest.mark.parametrize("ram", [False, True])
def test_original_values_and_parent_rng_are_exact(tmp_path, ram):
    cache, tokens, reference = cache_fixture(tmp_path, ram=ram)
    torch_state, np_state, python_state = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    order = [tokens[i] for i in (3, 1, 1, 7, 0, 5, 2, 4, 6)]
    cache.pending = deque((token, None) for token in order)
    cache.fill()
    try:
        for token in order:
            value = cache.get(token)
            assert torch.equal(value[0], reference[token][0])
            assert torch.equal(value[1], reference[token][1])
            assert value[2] == reference[token][2]
        assert cache.hits == len(order) == cache.decoded_hits
        assert cache.reads == 0
        assert not cache.pending
        assert torch.equal(torch.get_rng_state(), torch_state)
        after = np.random.get_state()
        assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1])
        assert after[2:] == np_state[2:]
        assert random.getstate() == python_state
    finally:
        cache.close()


def test_raw_identity_change_refused_before_consuming_batch(tmp_path):
    cache, tokens, _ = cache_fixture(tmp_path)
    cache.pending = deque((token, None) for token in tokens)
    cache.fill()
    (cache.c.raw / "s/events.h5").write_bytes(b"changed")
    try:
        with pytest.raises(ValueError, match="raw identity changed"):
            cache.get(tokens[0])
        assert len(cache.pending) == len(tokens)
        assert cache.hits == 0
    finally:
        cache.close()


def test_wrong_sampler_order_refused(tmp_path):
    cache, tokens, _ = cache_fixture(tmp_path)
    cache.pending = deque((token, None) for token in tokens)
    cache.fill()
    try:
        with pytest.raises(ValueError, match="original sampler"):
            cache.get(tokens[1])
    finally:
        cache.close()


@pytest.mark.parametrize("error", [FileNotFoundError("evicted"), OSError("transient")])
def test_cache_read_failure_preserves_pending_order(tmp_path, monkeypatch, error):
    cache, tokens, _ = cache_fixture(tmp_path)
    job = Future()
    job.set_exception(error)
    cache.decoding[tokens[0]] = job
    cache.pending = deque([(tokens[0], None)])
    calls = []
    monkeypatch.setattr(ParallelInputCache, "get", lambda self, token: calls.append(token))
    try:
        if isinstance(error, FileNotFoundError):
            cache.get(tokens[0])
            assert calls == [tokens[0]]
        else:
            with pytest.raises(OSError, match="transient"):
                cache.get(tokens[0])
            assert not calls
        assert cache.pending[0][0] == tokens[0]
    finally:
        cache.close()


def test_drain_cancels_only_speculative_cpu_work(tmp_path):
    cache, tokens, _ = cache_fixture(tmp_path)
    job = Future()
    cache.decoding[tokens[0]] = job
    cache.pending = deque([(tokens[0], None)])
    cache.drain()
    assert job.cancelled()
    assert not cache.decoding and not cache.pending
    assert cache.hits == cache.reads == 0
    cache.close()


def test_decoding_lookahead_is_bounded_even_for_an_oversized_fixture_queue(tmp_path):
    cache, tokens, _ = cache_fixture(tmp_path, count=80)
    cache.pending = deque((token, None) for token in tokens)
    cache.fill()
    try:
        assert len(cache.decoding) == 64
        for token in tokens:
            cache.get(token)
            assert len(cache.decoding) <= 64
    finally:
        cache.close()


def test_cache_capacity_uses_amended_disk_and_keeps_original_atomic_reservation(tmp_path):
    cache, _, _ = cache_fixture(tmp_path)
    try:
        assert cache.limit == 32_000_000_000
        assert cache.memory_limit == 8 * 1024**3
    finally:
        cache.close()


@pytest.mark.parametrize(
    "used", [0, 10_000_000_000, 10_000_000_001, 40_000_000_000, 40_000_000_001]
)
def test_only_explicit_disk_ceiling_changes(tmp_path, monkeypatch, used):
    c = fixture_campaign(tmp_path)
    calls = []
    c.require_resources = lambda: calls.append("original resource guard")
    monkeypatch.setattr(budget, "require", budget.require)
    monkeypatch.setattr(budget, "_artifact_scans", {str(c.out): (float("inf"), used)})
    install(c)
    if used > 40_000_000_000:
        with pytest.raises(InterruptedError, match="authorized40GB"):
            budget.require(c)
    else:
        budget.require(c)
    assert calls == ["original resource guard"]


@pytest.mark.parametrize("fault", ["RAM", "available", "commit", "disk_free"])
def test_expanded_disk_does_not_swallow_other_resource_errors(tmp_path, monkeypatch, fault):
    c = fixture_campaign(tmp_path)

    def reject():
        raise InterruptedError(fault)

    c.require_resources = reject
    monkeypatch.setattr(budget, "require", budget.require)
    install(c)
    with pytest.raises(InterruptedError, match=fault):
        budget.require(c)


@pytest.mark.parametrize(
    "family,maximum", [("wide", 22500), ("garl", 200000), ("garl_heads", 15000)]
)
def test_scientific_caps_still_enforced_above_old_disk_cap(tmp_path, monkeypatch, family, maximum):
    c = fixture_campaign(tmp_path)
    if family == "garl":
        atomic_json(
            c.out / "garl/PHYSICAL_WORK.json",
            {"saved_updates": maximum + 1, "uncertain_lost_upper": 0, "fits": {}},
        )
    else:
        path = c.out / (
            "PHYSICAL_WORK.json" if family == "wide" else "garl_heads/PHYSICAL_WORK.json"
        )
        atomic_json(
            path,
            {
                "accounting": {
                    "scientific_saved_updates": maximum + 1,
                    "scientific_uncertain_lost_upper": 0,
                },
                "fits": {},
            },
        )
    monkeypatch.setattr(budget, "require", budget.require)
    monkeypatch.setattr(budget, "_artifact_scans", {str(c.out): (float("inf"), 20_000_000_000)})
    install(c)
    with pytest.raises(ValueError, match="physical work ceiling"):
        budget.require(c)


@pytest.mark.parametrize(
    "invalid", ["parent_protocol_sha256", "max_owned_bytes", "scientific_recipe_changed"]
)
def test_disk_amendment_binding_cannot_be_weakened(tmp_path, invalid):
    from operational.efficient_context.common import read
    from operational.efficient_context.expanded_disk import authorization

    c = fixture_campaign(tmp_path)
    path = c.out / "DISK_RESOURCE_AUTHORIZATION.json"
    data = read(path)
    data[invalid] = True
    atomic_json(path, data)
    with pytest.raises(ValueError, match="authorization differs"):
        authorization(c)


@pytest.mark.parametrize("expired", [False, True])
def test_deadline_does_not_become_negative_or_shorten_a_fit(tmp_path, expired):
    from datetime import UTC, datetime, timedelta

    from operational.efficient_context.common import read
    from operational.efficient_context.deadline import permits

    c = fixture_campaign(tmp_path)
    deadline = datetime.now(UTC) + timedelta(hours=-1 if expired else 1)
    atomic_json(
        c.out / "EXECUTION_DEADLINE.json",
        {
            "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
            "deadline_utc": deadline.isoformat(),
        },
    )
    assert permits(c) is not expired
    path = c.out / "TIME_BUDGET_PAUSE.json"
    assert path.exists() is expired
    if expired:
        assert read(path)["scientific_negative"] is False
        assert read(path)["partial_producers_not_admissible_as_completed_comparators"]


@pytest.mark.parametrize("mismatch", ["time", "cwd", "module"])
def test_decode_handoff_refuses_unrelated_writer(tmp_path, monkeypatch, mismatch):
    import psutil

    from operational.efficient_context.common import ROOT
    from operational.efficient_context.decode_handoff import handoff

    atomic_json(tmp_path / "WRITER.lock", {"pid": 123, "create_time": 1.0})
    process = SimpleNamespace(
        create_time=lambda: 2.0 if mismatch == "time" else 1.0,
        cwd=lambda: str(tmp_path if mismatch == "cwd" else ROOT),
        cmdline=lambda: [
            "python",
            "-m",
            "other_module"
            if mismatch == "module"
            else "operational.efficient_context.garl_train_scandir",
        ],
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    with pytest.raises(ValueError, match="identify"):
        handoff(SimpleNamespace(out=tmp_path))
    assert not (tmp_path / "garl/DECODE_HANDOFF_INTENT.json").exists()


def test_controller_restores_every_binding_even_when_recipe_fails(tmp_path, monkeypatch):
    from operational.efficient_context import (
        garl_train_cached,
        garl_train_decode,
        garl_train_parallel,
        garl_train_scandir,
        parallel_inputs,
    )

    calls = []
    c = fixture_campaign(tmp_path)
    monkeypatch.setattr(garl_train_decode, "Campaign", lambda protocol: c)
    monkeypatch.setattr(garl_train_decode, "verify_admission", lambda c: {})
    monkeypatch.setattr(garl_train_decode.sys, "argv", ["fixture", "--resume"])
    monkeypatch.setattr(budget, "require", budget.require)
    original_cache = parallel_inputs.ParallelInputCache
    original_guard = garl_train_cached.GuardedCampaign
    original_parallel_guard = garl_train_parallel.GuardedCampaign

    def fail():
        assert parallel_inputs.ParallelInputCache is DecodingInputCache
        assert issubclass(garl_train_cached.GuardedCampaign, original_guard)
        assert garl_train_parallel.GuardedCampaign is garl_train_cached.GuardedCampaign
        calls.append("original unchanged recipe")
        raise OSError("injected resume failure")

    monkeypatch.setattr(garl_train_scandir, "main", fail)
    with pytest.raises(OSError, match="injected resume failure"):
        garl_train_decode.main()
    assert calls == ["original unchanged recipe"]
    assert parallel_inputs.ParallelInputCache is original_cache
    assert garl_train_cached.GuardedCampaign is original_guard
    assert garl_train_parallel.GuardedCampaign is original_parallel_guard


@pytest.mark.parametrize("complete", [False, True])
def test_partial_bundle_keeps_amended_persistent_cache_until_producers_complete(tmp_path, complete):
    from operational.efficient_context.package import release_cache_for_bundle

    c = fixture_campaign(tmp_path)
    path = c.out / ("garl/native_cache/" + "a" * 64 + ".npz")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"disposable fixture")
    if complete:
        atomic_json(c.out / "garl/ENDPOINTS.json", {"fixture": True})
    release_cache_for_bundle(c)
    assert path.exists() is not complete


def test_pending_decode_handoff_prevents_cache_cleanup(tmp_path):
    from operational.efficient_context.package import release_cache_for_bundle

    c = fixture_campaign(tmp_path)
    atomic_json(c.out / "garl/DECODE_HANDOFF_INTENT.json", {"status": "REQUESTED"})
    with pytest.raises(InterruptedError, match="preserve native input cache"):
        release_cache_for_bundle(c)
