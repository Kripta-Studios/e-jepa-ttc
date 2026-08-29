"""Causal and state-contract tests for the isolated V8 temporal controls."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.event_v4_geometry import EVENT_V4_STEPS
from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_timevolume_roi_np,
)
from e_jepa_ttc.data.scientific_recovery_v8 import (
    EXP6_ALPHAS,
    EXP6_CONFIG_SOURCE_PATH,
    EXP6_CONFIG_SOURCE_SHA256,
    EXP6_INTERNAL_DT_MS,
    EXP6_OFFICIAL_COMMIT_SHA,
    EXP6_OUTPUT_INTERVAL_MS,
    EXP6_OUTPUT_TIME_BINS,
    EXP6_PROCESSOR_SOURCE_PATH,
    EXP6_PROCESSOR_SOURCE_SHA256,
    EXP6_RASTER_CONTRACT,
    CausalExponentialStateRepresentation,
    GarlTimeVolumeRepresentation,
    ScientificRecoveryV8Batch,
    TemporalEndpointRepresentation,
)
from e_jepa_ttc.data.types import EventBatch


def _events(
    *,
    sequence_id: str = "sequence-a",
    x: tuple[int, ...] = (1, 1, 3, 6, 2),
    y: tuple[int, ...] = (1, 1, 3, 6, 2),
    t_us: tuple[int, ...] = (0, 200, 400, 800, 1_000),
    polarity: tuple[int, ...] = (1, -1, 1, 1, -1),
    start_us: int = 0,
) -> EventBatch:
    event_count = len(t_us)
    x_values = np.asarray(x, dtype=np.int32)
    y_values = np.asarray(y, dtype=np.int32)
    polarity_values = np.asarray(polarity, dtype=np.int8)
    return EventBatch(
        x=np.resize(x_values, event_count),
        y=np.resize(y_values, event_count),
        t_us=np.asarray(t_us, dtype=np.int64),
        polarity=np.resize(polarity_values, event_count),
        width=8,
        height=8,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=max(t_us, default=start_us),
    )


def _official_exp6_snapshot_reference(
    *,
    events: EventBatch,
    state: np.ndarray,
    window_start_us: int,
    output_endpoint_us: int,
) -> np.ndarray:
    """Golden EV-TTC raster equation, independent of the production recurrence.

    This is the frozen filter-state subset of
    ``ev_processor.h@59c498b``: events in a 7 ms window contribute
    ``sign * alpha * (1-alpha)**(-j)`` at a 0.2 ms bin ``j``.  The snapshot
    then decays the complete state by ``(1-alpha)**35``.  An event at the
    output boundary is intentionally excluded: the release snapshots first
    and inserts that triggering event at bin zero of the next window.
    """

    alpha = np.asarray(EXP6_ALPHAS, dtype=np.float64)
    dt_us = int(round(EXP6_INTERNAL_DT_MS * 1_000.0))
    output_bins = int(EXP6_OUTPUT_TIME_BINS)
    assert output_endpoint_us - window_start_us == output_bins * dt_us
    reference = np.asarray(state, dtype=np.float64).copy()
    timestamps = np.asarray(events.t_us, dtype=np.int64)
    active = (timestamps >= window_start_us) & (timestamps < output_endpoint_us)
    for event_index in np.flatnonzero(active):
        bin_index = int((timestamps[event_index] - window_start_us) // dt_us)
        assert 0 <= bin_index < output_bins
        contribution = alpha * np.power(1.0 - alpha, -bin_index)
        reference[:, events.y[event_index], events.x[event_index]] += (
            float(events.polarity[event_index]) * contribution
        )
    reference *= np.power(1.0 - alpha, output_bins)[:, None, None]
    return reference.astype(np.float32)


def test_garl_timevolume_matches_frozen_helper_and_channel_order() -> None:
    events = _events(t_us=(900_000, 925_000, 950_000, 999_999, 1_000_000))
    endpoint = 1_000_000
    roi = torch.tensor([0, 0, 8, 8])
    output = GarlTimeVolumeRepresentation().encode(events, endpoint, roi)
    reference, counts = official_timevolume_roi_np(
        (0, 0, 8, 8),
        events.x,
        events.y,
        events.t_us,
        time_window_s=0.1,
        number_of_planes=20,
    )
    expected = official_resize_feature(reference, (128, 128))
    torch.testing.assert_close(output.tensor, expected, atol=1e-6, rtol=1e-5)
    assert output.tensor.shape == (20, 128, 128)
    assert output.event_count == int(counts.sum())
    assert output.source == "isolated_garl_timevolume_frontend_not_full_garl_parity"
    assert output.support_start_us == 900_000
    assert output.support_end_us == endpoint
    assert output.diagnostics["roi_xmax"] == 8.0


def test_garl_empty_window_is_finite_zero_and_future_events_are_rejected() -> None:
    representation = GarlTimeVolumeRepresentation(target_size=(8, 8))
    empty = EventBatch.empty(
        width=8,
        height=8,
        sequence_id="empty",
        t_start_us=0,
        t_end_us=0,
    )
    output = representation.encode(empty, 1_000, torch.tensor([0, 0, 8, 8]))
    assert output.event_count == 0
    assert output.finite
    assert torch.count_nonzero(output.tensor) == 0
    with pytest.raises(ValueError, match="after endpoint"):
        representation.encode(_events(t_us=(0, 1_001)), 1_000, torch.tensor([0, 0, 8, 8]))


def test_exp6_pins_the_official_evttc_sources_and_raster_contract() -> None:
    assert EXP6_OFFICIAL_COMMIT_SHA == "59c498b71ae526bc2d7e570c82a078306a996b93"
    assert EXP6_PROCESSOR_SOURCE_PATH == "ev_ttc/include/ev_ttc/ev_processor.h"
    assert EXP6_PROCESSOR_SOURCE_SHA256 == (
        "439384787969f36f72bdc72e3f6a058c33847f7f8a70454a44313ffc0e9d511e"
    )
    assert EXP6_CONFIG_SOURCE_PATH == "ev_ttc/include/ev_ttc/config.h"
    assert EXP6_CONFIG_SOURCE_SHA256 == (
        "d30bfe8b292cb8505b1e1841bb76ebbeb2e1f34b3dce13c85b383252d4a44fe7"
    )
    assert EXP6_ALPHAS == (0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035)
    assert EXP6_INTERNAL_DT_MS == 0.2
    assert EXP6_OUTPUT_INTERVAL_MS == 7.0
    assert EXP6_OUTPUT_TIME_BINS == 35
    assert EXP6_RASTER_CONTRACT["scheduling"] == (
        "EV-TTC triggers from its first event when elapsed time is >=7 ms and can overshoot; "
        "V8 instead fixes boundaries at EventBatch.t_start_us+n*7000 us"
    )


def test_exp6_matches_official_window_equation_across_snapshots_boundary_and_reset() -> None:
    events = _events(
        x=(1, 2, 3, 4, 5, 1, 2, 3, 4),
        y=(1, 2, 3, 4, 5, 1, 2, 3, 4),
        t_us=(0, 200, 1_200, 6_800, 7_000, 7_200, 9_000, 13_800, 14_000),
        polarity=(1, -1, 1, -1, 1, -1, 1, -1, 1),
        start_us=0,
    )
    roi = torch.tensor([0, 0, 8, 8])
    representation = CausalExponentialStateRepresentation(target_size=(8, 8))
    first_packet = _events(
        x=(1, 2, 3, 4, 5),
        y=(1, 2, 3, 4, 5),
        t_us=(0, 200, 1_200, 6_800, 7_000),
        polarity=(1, -1, 1, -1, 1),
        start_us=0,
    )
    second_packet = _events(
        x=(1, 2, 3, 4),
        y=(1, 2, 3, 4),
        t_us=(7_200, 9_000, 13_800, 14_000),
        polarity=(-1, 1, -1, 1),
        start_us=0,
    )

    expected_first = _official_exp6_snapshot_reference(
        events=events,
        state=np.zeros((6, 8, 8), dtype=np.float32),
        window_start_us=0,
        output_endpoint_us=7_000,
    )
    first = representation.encode(first_packet, 7_000, roi)
    torch.testing.assert_close(
        first.tensor,
        official_resize_feature(expected_first, (8, 8)),
        atol=1e-6,
        rtol=1e-6,
    )
    same_boundary = representation.snapshot(7_000, roi)
    alternate_roi = representation.snapshot(7_000, torch.tensor([0, 0, 7, 7]))
    torch.testing.assert_close(same_boundary.tensor, first.tensor, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        alternate_roi.tensor,
        official_resize_feature(expected_first[:, :7, :7], (8, 8)),
        atol=1e-6,
        rtol=1e-6,
    )
    with pytest.raises(RuntimeError, match="advance beyond the pending EXP6 boundary"):
        representation.update(
            EventBatch.empty(
                width=8,
                height=8,
                sequence_id="sequence-a",
                t_start_us=0,
                t_end_us=7_000,
            ),
            7_000,
        )

    expected_second = _official_exp6_snapshot_reference(
        events=events,
        state=expected_first,
        window_start_us=7_000,
        output_endpoint_us=14_000,
    )
    second = representation.encode(second_packet, 14_000, roi)
    torch.testing.assert_close(
        second.tensor,
        official_resize_feature(expected_second, (8, 8)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert second.diagnostics["state_event_count"] == 8.0

    representation.reset()
    replay = representation.encode(first_packet, 7_000, roi)
    torch.testing.assert_close(replay.tensor, first.tensor, atol=1e-6, rtol=1e-6)
    assert representation.last_reset_reason == "manual"
    assert replay.diagnostics["reset_reason"] == 1.0
    assert replay.diagnostics["internal_dt_ms"] == 0.2
    assert replay.diagnostics["alpha_5"] == EXP6_ALPHAS[5]
    assert replay.diagnostics["warmup_duration_us"] == 7_000.0
    assert replay.diagnostics["state_event_count"] == 4.0
    assert replay.tensor.dtype == torch.float32
    assert replay.tensor.shape == (6, 8, 8)


def test_exp6_prefix_chunks_match_single_update_including_split_timestamp() -> None:
    """Stage 30 died allocating a 373M-event window; prefixes must stay bit-identical."""

    roi = torch.tensor([0, 0, 8, 8])
    whole = _events(
        x=(1, 2, 3, 4, 5, 6, 7),
        y=(1, 2, 3, 4, 5, 6, 7),
        t_us=(0, 200, 200, 200, 1_200, 6_800, 7_000),
        polarity=(1, -1, 1, -1, 1, -1, 1),
        start_us=0,
    )
    single = CausalExponentialStateRepresentation(target_size=(8, 8))
    single.update(whole, 7_000)
    chunked = CausalExponentialStateRepresentation(target_size=(8, 8))
    prefixes = (
        _events(x=(1,), y=(1,), t_us=(0,), polarity=(1,), start_us=0),
        _events(x=(2, 3), y=(2, 3), t_us=(200, 200), polarity=(-1, 1), start_us=0),
        _events(x=(4,), y=(4,), t_us=(200,), polarity=(-1,), start_us=0),
        _events(x=(5, 6), y=(5, 6), t_us=(1_200, 6_800), polarity=(1, -1), start_us=0),
    )
    for prefix in prefixes:
        chunked.ingest_prefix(prefix)
    chunked.update(
        _events(x=(7,), y=(7,), t_us=(7_000,), polarity=(1,), start_us=0),
        7_000,
    )
    torch.testing.assert_close(
        chunked.snapshot(7_000, roi).tensor,
        single.snapshot(7_000, roi).tensor,
        atol=1e-6,
        rtol=1e-6,
    )
    assert chunked.snapshot(7_000, roi).diagnostics["state_event_count"] == 6.0


def test_exp6_vectorized_bins_match_scalar_loop_and_ingest_large_packet() -> None:
    """Stage 30 died on a Python listcomp over ingest timestamps; keep numpy bins."""

    representation = CausalExponentialStateRepresentation(target_size=(8, 8))
    roi = torch.tensor([0, 0, 8, 8])
    representation.update(_events(t_us=(0,), polarity=(1,)), 0)
    stamps = np.asarray([0, 200, 200, 400, 1_000, 7_000], dtype=np.int64)
    legacy = np.asarray([representation._bin_for_timestamp(int(value)) for value in stamps])
    vectorized = representation._bins_for_timestamps(stamps)
    np.testing.assert_array_equal(vectorized, legacy)
    with pytest.raises(ValueError, match="precedes the stable per-sequence origin"):
        representation._bins_for_timestamps(np.asarray([-1], dtype=np.int64))

    count = 1_048_576 + 64
    times = np.repeat(np.arange(32, dtype=np.int64) * 200 + 200, count // 32)
    large = EventBatch(
        x=np.ones(count, dtype=np.int32),
        y=np.ones(count, dtype=np.int32),
        t_us=times,
        polarity=np.ones(count, dtype=np.int8),
        width=8,
        height=8,
        sequence_id="sequence-a",
        t_start_us=0,
        t_end_us=int(times[-1]),
    )
    ingested = representation.update(large, int(times[-1]))
    assert ingested == count
    snapshot = representation.snapshot(int(times[-1]), roi, event_count=ingested)
    assert snapshot.finite
    assert torch.isfinite(snapshot.tensor).all()


def test_exp6_empty_packet_decays_existing_state_without_nan() -> None:
    representation = CausalExponentialStateRepresentation(target_size=(8, 8))
    roi = torch.tensor([0, 0, 8, 8])
    first = representation.encode(_events(t_us=(0,), polarity=(1,)), 0, roi)
    empty = EventBatch.empty(
        width=8,
        height=8,
        sequence_id="sequence-a",
        t_start_us=0,
        t_end_us=1_000,
    )
    later = representation.encode(empty, 1_000, roi)
    assert later.event_count == 0
    assert later.finite
    assert torch.isfinite(later.tensor).all()
    assert torch.linalg.vector_norm(later.tensor) < torch.linalg.vector_norm(first.tensor)


def test_exp6_prefix_is_causal_and_future_packets_cannot_change_prior_snapshot() -> None:
    roi = torch.tensor([0, 0, 8, 8])
    prefix = _events(t_us=(0, 200, 400), polarity=(1, 1, -1))
    baseline = CausalExponentialStateRepresentation(target_size=(8, 8)).encode(prefix, 400, roi)
    stateful = CausalExponentialStateRepresentation(target_size=(8, 8))
    replay = stateful.encode(prefix, 400, roi)
    future = _events(t_us=(600, 800), polarity=(1, -1))
    stateful.encode(future, 800, roi)
    torch.testing.assert_close(baseline.tensor, replay.tensor, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="after endpoint"):
        CausalExponentialStateRepresentation().encode(_events(t_us=(0, 600)), 400, roi)


def test_exp6_resets_on_sequence_change_and_endpoint_rollback() -> None:
    representation = CausalExponentialStateRepresentation(target_size=(8, 8))
    roi = torch.tensor([0, 0, 8, 8])
    representation.encode(_events(t_us=(0, 200)), 200, roi)
    changed = representation.encode(_events(sequence_id="sequence-b", t_us=(0,)), 0, roi)
    assert representation.last_reset_reason == "sequence_changed"
    assert changed.diagnostics["reset_reason"] == 2.0
    rollback = representation.encode(_events(sequence_id="sequence-b", t_us=(0,)), -0, roi)
    assert rollback.diagnostics["reset_count"] >= 1.0
    # Advance then roll back to make the rollback reset observable.
    representation.encode(_events(sequence_id="sequence-b", t_us=(200,)), 200, roi)
    reset_output = representation.encode(_events(sequence_id="sequence-b", t_us=(0,)), 0, roi)
    assert representation.last_reset_reason == "endpoint_rollback"
    assert reset_output.diagnostics["reset_reason"] == 3.0
    assert reset_output.support_end_us <= reset_output.endpoint_us


def test_v8_batch_allows_two_or_three_steps_and_preserves_v4_steps() -> None:
    for steps in (2, 3):
        batch = ScientificRecoveryV8Batch(
            representations=torch.zeros((2, steps, 6, 4, 4)),
            endpoint_us=torch.tensor([[0, 1] if steps == 2 else [0, 1, 2]] * 2),
            token_id=["token-a", "token-b"],
            sequence_id=["sequence-a", "sequence-b"],
            track_id=["track-a", "track-b"],
            target_ttc=torch.ones(2),
            sample_weight=torch.ones(2),
            metadata={"event_count": torch.ones(2)},
        )
        assert batch.representations.shape[1] == steps
    assert EVENT_V4_STEPS == 3
    with pytest.raises(ValueError, match="steps"):
        ScientificRecoveryV8Batch(
            representations=torch.zeros((1, 4, 6, 4, 4)),
            endpoint_us=torch.zeros((1, 4)),
            token_id=["token"],
            sequence_id=["sequence"],
            track_id=["track"],
            target_ttc=torch.ones(1),
            sample_weight=torch.ones(1),
            metadata={},
        )
    with pytest.raises(ValueError, match="metadata"):
        ScientificRecoveryV8Batch(
            representations=torch.zeros((2, 2, 6, 4, 4)),
            endpoint_us=torch.zeros((2, 2)),
            token_id=["a", "b"],
            sequence_id=["a", "b"],
            track_id=["a", "b"],
            target_ttc=torch.ones(2),
            sample_weight=torch.ones(2),
            metadata={"bad": ["only-one"]},
        )
    with pytest.raises(ValueError, match="monotonic"):
        ScientificRecoveryV8Batch(
            representations=torch.zeros((1, 2, 6, 4, 4)),
            endpoint_us=torch.tensor([[2, 1]]),
            token_id=["token"],
            sequence_id=["sequence"],
            track_id=["track"],
            target_ttc=torch.ones(1),
            sample_weight=torch.ones(1),
            metadata={},
        )
    with pytest.raises(ValueError, match="finite"):
        ScientificRecoveryV8Batch(
            representations=torch.full((1, 2, 6, 4, 4), float("nan")),
            endpoint_us=torch.zeros((1, 2)),
            token_id=["token"],
            sequence_id=["sequence"],
            track_id=["track"],
            target_ttc=torch.ones(1),
            sample_weight=torch.ones(1),
            metadata={},
        )


def test_protocol_runtime_check_and_rejects_non_integer_roi() -> None:
    representation = GarlTimeVolumeRepresentation()
    assert isinstance(representation, TemporalEndpointRepresentation)
    with pytest.raises(ValueError, match="integer"):
        representation.encode(_events(t_us=(0,)), 0, torch.tensor([0.5, 0, 8, 8]))


def test_c1_temporal_channel_gate_is_uniform_at_initialization_and_finite() -> None:
    """The preregistered C1 gate starts as the exact fixed EXP6 control."""

    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig

    config = CausalScaleTTCConfig(
        in_channels=6,
        hidden_dim=32,
        geometry_dim=64,
        residual_depth=1,
        foreground_temporal_smoothing=0.0,
        foreground_temporal_smoothing_mode="causal_left",
        temporal_channel_gate_enabled=True,
        temporal_channel_gate_patch_grid=4,
        temporal_channel_gate_hidden_dim=16,
    )
    model = CausalScaleTTC(config).eval()
    inputs = torch.rand(2, 3, 6, 32, 32)
    delta = torch.full((2, 2), 0.05)
    with torch.no_grad():
        output = model(inputs, delta)
    weights = output.diagnostics["temporal_channel_gate_weights"]
    assert weights.shape == (2, 3, 6, 4, 4)
    torch.testing.assert_close(weights.sum(dim=2), torch.ones(2, 3, 4, 4))
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 6.0), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(output.ttc_mean_seconds).all()
    assert torch.isfinite(output.diagnostics["temporal_channel_gate_entropy"]).all()


def test_c1_temporal_channel_gate_rejects_non_exp6_channel_count() -> None:
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig

    with pytest.raises(ValueError, match="EXP6 six-channel input"):
        CausalScaleTTCConfig(in_channels=20, temporal_channel_gate_enabled=True)
