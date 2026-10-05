"""Numerical contracts with zero optimizer updates."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.analytical import transport_ewma
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, encode_union, map_roi
from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS, WideSource
from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer
from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def test_native_garl_floor_boundaries_and_sensor_offset(tmp_path):
    import h5py

    from e_jepa_ttc.efficient_context.garl_input import read_native_window
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    path = tmp_path / "events.h5"
    t = np.array([0, 999, 1000, 1999, 2000, 2999, 3000], np.int64)
    with h5py.File(path, "w") as f:
        f["events/t"] = t
        f["events/x"] = np.array([0, 1, 2, 3, 1918, 65534, 4], np.uint16)
        f["events/y"] = np.zeros(7, np.uint16)
        f["events/p"] = np.ones(7, np.uint8)
        f["ms_to_idx"] = np.array([0, 2, 4, 6], np.int64)
    pool = ReaderPool()
    try:
        raw = read_native_window(pool, path, 1500, 2999)
        # The native reader floors both ms bounds and intentionally includes1000us.
        assert np.array_equal(raw["t"], [1000, 1999])
        assert np.array_equal(raw["x"], [7, 8])
        with pytest.raises(InterruptedError, match="bounded extraction"):
            read_native_window(pool, path, 0, 2999, cap=1)
        with pytest.raises(ValueError, match="support"):
            read_native_window(pool, path, 0, 4000)
    finally:
        pool.close()


def test_native_garl_global_time_origin_is_preserved():
    from e_jepa_ttc.efficient_context.garl_input import native_feature

    raw = {
        "x": np.array([99, 1, 1]),
        "y": np.array([99, 1, 1]),
        "t": np.array([100000, 108000, 109000], np.int64),
    }
    value = native_feature(raw, (0, 0, 4, 4), size=4)
    assert value.dtype == torch.float32
    assert value[0].sum() == 0  # ROI-first filtering would move these events into plane0.
    assert value[1].sum() > 0


def test_native_gru_contract_and_zero_cost_objective():
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadConfig, model, objective

    with pytest.raises(ValueError, match="fixed Garl"):
        NativeHeadConfig(hidden=64)
    head = model().eval()
    assert len(head.cells) == 2 and head.project[0].in_features == 7
    x = torch.ones(2, 8, 3)
    times = torch.zeros(2, 8, 4)
    mask = torch.ones(2, 8, dtype=torch.bool)
    experts = torch.tensor([[0.02, 0.02, 0.02], [0.1, 0.1, 0.1]])
    with torch.no_grad():
        output = head(x, times, mask, experts)
    y = torch.tensor([0.03, 0.09])
    first = objective(output, y, experts, torch.ones(2) / 2, 2)
    altered = {**output, "relative_cost": torch.full((2, 3), float("nan"))}
    assert torch.equal(first, objective(altered, y, experts, torch.ones(2) / 2, 2))
    assert torch.isfinite(first)


def test_native_source_current_phase_and_shared_statistics():
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource, normalize

    features = np.asarray([[0.01, 2, 3], [0.02, 4, 5], [0.03, 6, 7]], np.float32)
    history = np.asarray([[-1, -1, -1, -1, -1, -1, 0, 1], [-1, -1, -1, -1, -1, 0, 1, 2]], np.int64)
    stats = normalize(features, history)
    times = np.zeros((2, 8, 4), np.float32)
    times[:, -1, 2] = 0.05
    source = NativeHeadSource(
        features,
        times,
        history,
        np.ones(2, np.float32),
        np.ones(2, np.float64) / 2,
        stats,
        "a" * 64,
        1,
    )
    x, t, valid, anchor, _, _ = source.gather(torch.tensor([0, 1]))
    assert x.shape == (2, 1, 3) and valid.all() and (t[:, :, 2] == 0).all()
    assert np.allclose(stats.mean, features.mean(0))
    assert torch.equal(anchor[:, 0], torch.tensor([0.02, 0.03]))
    assert torch.equal(anchor[:, 0], anchor[:, 2])


@pytest.mark.parametrize("missing", [0, 1, 7, 15])
def test_wide_gaps_and_cold_starts(missing):
    history = np.arange(16)[None]
    history[:, :missing] = -1
    parent = CachedQueries(
        np.ones((16, 17), np.float32),
        np.arange(16, dtype=np.int64) * 50000,
        np.arange(16, dtype=np.int64) * 50000 + 100,
        history,
        np.ones(1, np.float32),
        np.ones(1, np.float32),
        Normalizer(np.zeros(17), np.ones(17), "a" * 64),
        "b" * 64,
        length=16,
    )
    x, t, v, e, y, m = WideSource(parent).gather(torch.tensor([0]))
    assert x.shape == (1, 8, 17)
    assert v[0, -1]
    assert torch.all(t[~v] == 0)
    expected = parent.anchor_us[np.maximum(history[:, WIDE_SLOTS], 0)]
    for i in range(1, 8):
        gap = (expected[0, i] - expected[0, i - 1]) / 1e6 if v[0, i - 1] and v[0, i] else 0
        assert t[0, i, 2].item() == np.float32(gap)
    assert torch.equal(e, parent.gather(torch.tensor([0]))[3])


@pytest.mark.parametrize(
    "roi", [(0, 0, 50, 50), (-10, -10, 20, 20), (2000, 2000, 2100, 2100), (12.3, 9.7, 88.1, 85.5)]
)
@pytest.mark.parametrize("offset", [0, 5, -5])
@pytest.mark.parametrize("anchor", [1000000, 2**53 + 100])
def test_mapping_bitexact(roi, offset, anchor):
    rng = np.random.default_rng(7)
    raw = {
        "x": rng.integers(-5, 100, 1500),
        "y": rng.integers(-5, 100, 1500),
        "t": np.sort(rng.integers(anchor, anchor + 100001, 1500)),
        "p": rng.integers(-1, 2, 1500),
    }
    raw["t"][:3] = anchor
    raw["t"][-3:] = anchor + 100000
    raw["t"].sort()
    mapped = map_roi(raw, roi, 16, offset)
    for start, end in [(anchor, anchor + 100000), (anchor + 50000, anchor + 100001)]:
        keep = (raw["t"] >= start) & (raw["t"] < end)
        reference = encode_query_window(
            {k: v[keep] for k, v in raw.items()},
            square_xyxy=roi,
            start_us=start,
            end_us=end,
            sequence_id="fixture",
            roi_size=16,
            bins_per_polarity=5,
            event_pixel_diff=offset,
        )
        actual = encode_mapped(mapped, start, end, 16, "fixture")
        assert torch.equal(reference, actual)


@pytest.mark.parametrize("cap", [1, 1000000])
def test_chunk_union_and_capacity_fallback(cap):
    class Reader:
        def iter_window_chunks(self, start, end, chunk_events):
            for a in range(start, end, 5000):
                t = np.arange(a, min(a + 5000, end), 100, dtype=np.int64)
                yield {
                    "x": np.ones(len(t), np.int32) * 10,
                    "y": np.ones(len(t), np.int32) * 10,
                    "t": t,
                    "p": np.ones(len(t), np.int8),
                }

    windows = np.array([[800000, 900000], [850000, 950000], [900000, 1000000]])
    lags = np.arange(15, -1, -1) * 50000
    valid = np.ones(16, bool)
    kwargs = dict(sequence_id="fixture", roi_size=8, event_pixel_diff=5, retained_bytes_max=cap)
    if cap == 1:
        with pytest.raises(RuntimeError, match="RESOURCE_PAUSE"):
            encode_union(Reader(), windows, lags, valid, (0, 0, 50, 50), **kwargs)
    else:
        assert torch.equal(
            encode_context_union(Reader(), windows, lags, valid, (0, 0, 50, 50), **kwargs),
            encode_union(Reader(), windows, lags, valid, (0, 0, 50, 50), **kwargs),
        )


def test_transport_exact_cv_and_zero():
    ages = torch.arange(7, -1, -1).double()[None] * 0.05
    phases = ttc_to_phase(2 + ages)
    point, diag = transport_ewma(phases, ages, torch.ones((1, 8), dtype=torch.bool))
    assert torch.allclose(phase_to_ttc(point), torch.tensor([2.0], dtype=torch.float64), atol=1e-12)
    point, diag = transport_ewma(torch.zeros_like(ages), ages, torch.ones((1, 8), dtype=torch.bool))
    assert phase_to_ttc(point).item() == 60


def test_transport_excludes_terms_not_queries():
    ages = torch.arange(7, -1, -1).double()[None] * 0.1
    point, diag = transport_ewma(
        ttc_to_phase(torch.ones_like(ages) * 0.2), ages, torch.ones((1, 8), dtype=torch.bool)
    )
    assert torch.isfinite(point).all()
    assert diag["rejected_terms"] > 0 and diag["queries"] == 1


def test_mapping_rejects_timestamp_rollback():
    with pytest.raises(ValueError, match="rollback"):
        map_roi(
            {"x": np.ones(2), "y": np.ones(2), "t": np.array([2, 1]), "p": np.ones(2)},
            (0, 0, 10, 10),
            8,
            0,
        )


def test_completed_checkpoint_supersedes_stale_progress(tmp_path):
    """Completion can precede the final heartbeat; do not count saved work as unsaved."""
    from types import SimpleNamespace

    from operational.efficient_context.budget import accounting
    from operational.efficient_context.common import atomic_json

    key = "WIDE/fold2/seed7"
    atomic_json(
        tmp_path / "PHYSICAL_WORK.json",
        {
            "accounting": {
                "scientific_saved_updates": 7500,
                "scientific_uncertain_lost_upper": 100,
            },
            "fits": {key: {"completed": 2500, "pending": None}},
        },
    )
    atomic_json(tmp_path / "UPDATE_PROGRESS.json", {"fit": key, "confirmed": 2475, "durable": 2400})
    atomic_json(
        tmp_path / "data_recovery/INTERRUPTION_ACCOUNTING.json",
        {"confirmed_lost_updates_lower": 50},
    )
    result = accounting(SimpleNamespace(out=tmp_path))
    assert result["unsaved_updates_confirmed_by_progress"] == 0
    assert result["physical_execution_lower"] == 7550
    assert result["physical_execution_upper"] == 7600


def test_pending_progress_has_observed_lower_and_reserved_upper(tmp_path):
    """A pending reservation is not evidence that all its optimizer calls executed."""
    from types import SimpleNamespace

    from operational.efficient_context.budget import accounting
    from operational.efficient_context.common import atomic_json

    key = "WIDE/fold2/seed7"
    atomic_json(
        tmp_path / "PHYSICAL_WORK.json",
        {
            "accounting": {"scientific_saved_updates": 6600, "scientific_uncertain_lost_upper": 0},
            "fits": {key: {"completed": 1600, "pending": [1600, 1700]}},
        },
    )
    atomic_json(tmp_path / "UPDATE_PROGRESS.json", {"fit": key, "confirmed": 1650, "durable": 1600})
    result = accounting(SimpleNamespace(out=tmp_path))
    assert result["unsaved_updates_confirmed_by_progress"] == 50
    assert result["physical_execution_lower"] == 6650
    assert result["physical_execution_upper"] == 6700


@pytest.mark.parametrize("changed_sha", [False, True])
def test_restored_raw_requires_verified_full_sha_receipt(tmp_path, changed_sha):
    """Disk recreation changes mtime, but must never silently change source bytes."""
    from types import SimpleNamespace

    from operational.efficient_context.common import atomic_json, digest
    from operational.efficient_context.profile import verify_raw

    path = tmp_path / "train/fixture/events.h5"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"authorized TRAIN fixture")
    sha = digest(path)
    stat = path.stat()
    out = tmp_path / "outputs"
    atomic_json(
        out / "data_recovery/DOWNLOAD_PLAN.json",
        {
            "revision": "pinned_public_revision",
            "files": [{"sequence_id": "fixture", "sha256": sha, "bytes": stat.st_size}],
        },
    )
    atomic_json(
        out / "data_recovery/files/fixture.json",
        {
            "status": "VERIFIED",
            "path": str(path),
            "sha256": "wrong" if changed_sha else sha,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    )
    historical = {"bytes": stat.st_size, "mtime_ns": 1}
    campaign = SimpleNamespace(out=out)
    if changed_sha:
        with pytest.raises(ValueError, match="pinned SHA receipt"):
            verify_raw(campaign, path, historical)
    else:
        result = verify_raw(campaign, path, historical)
        assert result["restored"] and result["sha256"] == sha
        assert result["historical_mtime_preserved_in_reference_only"] == 1


@pytest.mark.parametrize("ratio", [1.0, 2.0, -1.0])
def test_native_runtime_retains_signed_and_infinite_outputs(monkeypatch, ratio):
    """Native cost evaluation must not cap infinity or discard invalid queries."""
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource
    from e_jepa_ttc.simplex_t.cache import Normalizer
    from operational.efficient_context.garl_runtime import prepared_forward

    class Producer(torch.nn.Module):
        def forward(self, _):
            return torch.tensor([[ratio * 2, 2]]), None

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    native = NativeHeadSource(
        features=np.zeros((1, 3), np.float32),
        times=np.zeros((1, 1, 4), np.float32),
        history=np.array([[0]], np.int64),
        truth=np.zeros(1),
        mass=np.ones(1),
        normalizer=Normalizer(np.zeros(3), np.ones(3), "fixture"),
        identity_sha256="fixture",
        length=1,
    )
    args = (Producer(), [torch.zeros((40, 2, 2))], [], native, native.gather(torch.tensor([0]))[:4])
    if ratio < 0:
        with pytest.raises(ArithmeticError, match="no query dropping"):
            prepared_forward(*args, label="GARL_NATIVE", refiner=torch.nn.Identity())
    else:
        value, fresh = prepared_forward(*args, label="GARL_NATIVE", refiner=torch.nn.Identity())
        assert fresh is None
        assert value[0] == pytest.approx(-np.log(ratio), abs=1e-7)
        assert np.isposinf(value[1]) if ratio == 1 else value[1] == pytest.approx(-0.1)


def test_native_runtime_uses_fresh_sensor_scalars_and_frozen_normalizer(monkeypatch):
    """The measured raw route must reconstruct the trained three-feature head inputs."""
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource, model
    from e_jepa_ttc.simplex_t.cache import Normalizer
    from operational.efficient_context.garl_runtime import prepared_forward

    class Producer(torch.nn.Module):
        def forward(self, _):
            return torch.tensor([[4.0, 2.0]]), None

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    native = NativeHeadSource(
        features=np.array([[-np.log(2.0), 2.0, 3.0]], np.float32),
        times=np.zeros((1, 1, 4), np.float32),
        history=np.array([[0]], np.int64),
        truth=np.zeros(1),
        mass=np.ones(1),
        normalizer=Normalizer(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "fixture"),
        identity_sha256="fixture",
        length=1,
    )
    expected = native.gather(torch.tensor([0]))[:4]
    _, actual = prepared_forward(
        Producer(),
        [torch.zeros((40, 2, 2))],
        [np.array([2.0, 3.0])],
        native,
        expected,
        label="GARL_H1",
        refiner=model().eval(),
    )
    assert actual is not None
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("variant", ["complete", "tampered", "missing"])
def test_completed_wide_reuse_checks_all_sealed_bytes(tmp_path, variant):
    """A resumable queue must reuse finished work without accepting altered weights."""
    from types import SimpleNamespace

    from operational.efficient_context.common import atomic_json, digest
    from operational.efficient_context.queue import completed_wide

    atomic_json(tmp_path / "WIDE_REPLICATION_RESULTS.json", {"status": "COMPLETE"})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"sealed fixture weights")
    sha = digest(checkpoint)
    for seed in (7, 13, 23):
        atomic_json(
            tmp_path
            / ("H8_WIDE_RESULTS.json" if seed == 7 else f"H8_WIDE_RESULTS_seed{seed}.json"),
            {"status": "COMPLETE"},
        )
        if variant == "missing" and seed == 23:
            continue
        atomic_json(
            tmp_path / f"ENDPOINTS_seed{seed}.json",
            {
                "all_three_frozen_before_evaluation": True,
                "fits": [
                    {
                        "seed": seed,
                        "fold": fold,
                        "updates": 2500,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha,
                    }
                    for fold in range(3)
                ],
            },
        )
    if variant == "tampered":
        checkpoint.write_bytes(b"altered weights")
        with pytest.raises(ValueError, match="do not retrain"):
            completed_wide(SimpleNamespace(out=tmp_path))
    else:
        assert completed_wide(SimpleNamespace(out=tmp_path)) == (variant == "complete")


@pytest.mark.parametrize("compressed", [False, True])
def test_exclusive_cache_retains_disk_victims_without_changing_training_tensors(
    tmp_path, monkeypatch, compressed
):
    """RAM plus disk must cover the working set while preserving all three returned fields."""
    import io
    from types import SimpleNamespace

    from e_jepa_ttc.efficient_context import garl_input
    from operational.efficient_context.compressed_cache import CompressedInputCache
    from operational.efficient_context.exclusive_cache import ExclusiveInputCache

    out = tmp_path / "out"
    (out / "garl").mkdir(parents=True)
    (out / "garl/PROTOCOL.json").write_text('{"fixture":true}', encoding="utf-8")
    raw = tmp_path / "raw/sequence/events.h5"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"read-only sensor fixture")
    rows = {str(i): {"sequence_id": "sequence", "value": i} for i in range(5)}
    expected = {
        str(i): (
            torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + i,
            torch.tensor([i, i + 1], dtype=torch.float32),
            float(i) + 0.25,
        )
        for i in range(5)
    }

    def encode(row, _pool, _root):
        return expected[str(row["value"])]

    monkeypatch.setattr(garl_input, "encode_record", encode)
    c = SimpleNamespace(
        out=out,
        raw=raw.parent.parent,
        policy={"max_tree_rss_gib": 12},
        require_resources=lambda: None,
    )
    cache = (CompressedInputCache if compressed else ExclusiveInputCache)(c, rows)
    stream = io.BytesIO()
    v = expected["0"]
    np.savez_compressed(stream, events=v[0].numpy(), visible=v[1].numpy(), target=v[2])
    cache.limit = len(stream.getvalue()) * 2 + 128
    cache.memory_limit = 3 * cache.tensor_bytes(expected["0"])
    if compressed:
        sizes = []
        for value in expected.values():
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer, events=value[0].numpy(), visible=value[1].numpy(), target=value[2]
            )
            sizes.append(len(buffer.getvalue()))
        cache.limit = 2 * max(sizes) + 64
        cache.memory_limit = 3 * max(sizes) + 64
    rng_before = torch.get_rng_state().clone()
    for _ in range(3):
        for token, reference in expected.items():
            result = cache.get(token)
            assert torch.equal(result[0], reference[0])
            assert torch.equal(result[1], reference[1])
            assert result[2] == reference[2]
            assert cache.disk <= cache.limit and cache.bytes <= cache.memory_limit
    assert cache.reads == 5
    assert cache.preserved_disk_evictions > 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    raw.write_bytes(b"changed identity")
    with pytest.raises(ValueError, match="raw identity changed"):
        cache.get("0")
    cache.close()
