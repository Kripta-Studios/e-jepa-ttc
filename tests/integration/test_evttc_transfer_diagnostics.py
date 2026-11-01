import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.evttc_competition.runtime import CompetitionRuntime, PreparedObservation
from operational.evttc_transfer_audit.diagnostics import expansion_reference, serialize, trace_query
from operational.final_refit.training import PhaseOnlyFinal
from operational.rgb_port import transfer as engine


def test_trace_large_tensor_records_statistics_without_dumping_feature_maps() -> None:
    value = torch.ones((2, 5000))
    result = serialize(value)
    assert isinstance(result, dict)
    assert result["shape"] == [2, 5000]
    assert result["finite_count"] == 10000
    assert result["minimum"] == result["maximum"] == 1.0
    assert len(result["sha256"]) == 64


@pytest.mark.parametrize("delta", [0.05, 0.08, 0.1, 0.2])
def test_expansion_distinguishes_endpoints_at_real_cadence(delta: float) -> None:
    result = expansion_reference(1.0, 1.0 + delta, delta)
    assert result["current_ttc"] == pytest.approx(1.0)
    assert result["previous_ttc"] == pytest.approx(1.0 + delta)


def test_approach_recession_and_static_scene() -> None:
    assert expansion_reference(1.1, 1.0, 0.1)["current_ttc"] == pytest.approx(-1.1)
    assert expansion_reference(1.0, 1.0, 0.1)["status"] == "no_expansion"
    with pytest.raises(ValueError):
        expansion_reference(1.0, 2.0, 0.0)


def test_rotation_only_projection_has_depth_dependent_parallax() -> None:
    from operational.evttc_competition.geometry import Calibration, project_event_box_to_rgb

    calibration = Calibration(
        event_intrinsics=np.array([100.0, 100.0, 50.0, 40.0]),
        event_distortion=np.zeros(4),
        event_resolution=(100, 80),
        rgb_intrinsics=np.array([100.0, 100.0, 50.0, 40.0]),
        rgb_distortion=np.zeros(4),
        rgb_resolution=(100, 80),
        rotation_event_to_rgb=np.eye(3),
        source_kind="test",
        transform_source="test",
    )
    projected = project_event_box_to_rgb((40.0, 30.0, 60.0, 50.0), calibration)
    # A real +0.1 m camera translation shifts pixels by f*t/Z: 10 px at 1 m,
    # 1 px at 10 m. The rotation-only adapter cannot express either translation.
    assert projected == pytest.approx((40.0, 30.0, 60.0, 50.0))
    near_full = np.array([50.0, 30.0, 70.0, 50.0])
    far_full = np.array([41.0, 30.0, 61.0, 50.0])
    assert np.abs(near_full - projected).max() == 10.0
    assert np.abs(far_full - projected).max() == 1.0


def test_tracing_preserves_prediction_clock_shift_and_future_invariance(monkeypatch) -> None:
    # Expensive producer replaced at its boundary; real contexts and final head execute.
    def expert(models, prefix, sensor, delta):
        base = sensor.mean() + delta.mean()
        return torch.ones((1, 17)) * base, torch.tensor([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(engine, "_expert_block", expert)
    torch.manual_seed(7)
    config = TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    runtime = CompetitionRuntime(
        {
            "E_CTX_MATCHED": TemporalRefiner(config).eval(),
            "R_CTX": TemporalRefiner(config).eval(),
            "PHASE_ONLY_EVENT_S7_TRAIN40": PhaseOnlyFinal().eval(),
        },
        {"event": (torch.zeros(17), torch.ones(17)), "rgb": (torch.zeros(17), torch.ones(17))},
        device="cpu",
    )

    def observation(modality, shift=0, value=0.2):
        channels = 12 if modality == "event" else 3
        return PreparedObservation(
            modality,
            np.full((2, channels, 128, 128), value, np.float32),
            (shift, shift + 100000),
            shift,
            shift + 100000,
            shift + 100000,
        )

    event, rgb = observation("event"), observation("rgb")
    baseline = runtime.predict([event], [rgb], 100000)
    traced = trace_query(runtime, [event], [rgb], 100000)
    assert traced["prediction"]["model_ttc"] == baseline.model_ttc
    assert traced["accepted"]["event"][0]["delta_t_seconds"] == pytest.approx([0.1])
    assert traced["module_calls"]["PHASE_ONLY_EVENT_S7_TRAIN40"]
    shifted = trace_query(
        runtime, [observation("event", 500000)], [observation("rgb", 500000)], 600000
    )
    assert shifted["prediction"]["model_ttc"] == baseline.model_ttc
    future = trace_query(runtime, [event, observation("event", 200000, 2.0)], [rgb], 100000)
    assert future["prediction"]["model_ttc"] == baseline.model_ttc
    assert not any(module._forward_hooks for module in runtime.models.values())
