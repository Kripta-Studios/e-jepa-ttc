"""Native V13 parity, real clocks, modality guards, and bounded input reuse."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.rgb_port.recipe import resolved_recipe
from operational.rgb_port.train_heads import _model
from operational.rgb_port.train_producers import build_producer
from operational.rgb_port_streaming.inputs import EventInputs, RGBInputs
from operational.rgb_port_streaming.runtime import (
    NativeExperts,
    NativeStream,
    Request,
    temporal_forward,
)
from operational.streaming_revision.packets import PacketRing


@pytest.mark.parametrize("modality", ["event", "rgb"])
@pytest.mark.parametrize("steps", [2, 3])
def test_native_vectorized_experts_match_reference(modality, steps):
    ids = ["E_A5_MATCHED", "E_C2F_MATCHED"] if modality == "event" else ["R_A5", "R_C2F"]
    models = [
        build_producer(
            resolved_recipe(
                Path("configs/rgb_port/producers.json"),
                fit_id=name,
                producer_population=32,
                role_manifest_sha256="a" * 64,
            )
        )
        for name in ids
    ]
    pair = _model("PAIR_E_MATCHED" if modality == "event" else "PAIR_R")
    inputs = torch.rand(3, steps, 12 if modality == "event" else 3, 128, 128)
    dt = torch.full((3, steps - 1), 0.1)
    reference = NativeExperts(
        *models, pair, modality=modality, parent_sha256="a" * 64, vectorized=False, batch_size=1
    )
    optimized = NativeExperts(*models, pair, modality=modality, parent_sha256="a" * 64)
    with torch.inference_mode():
        torch.testing.assert_close(
            optimized(inputs, dt), reference(inputs, dt), atol=2e-5, rtol=2e-5
        )
    for original, wrapped in zip(models, optimized.models, strict=True):
        for a, b in zip(original.parameters(), wrapped.parameters(), strict=True):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    with pytest.raises(ValueError, match="Student must bind"):
        NativeExperts(
            *models, pair, modality=modality, parent_sha256="a" * 64, student=nn.Identity()
        )


class StubExperts:
    modality = "event"
    schema = EVENT_PHASE17_SHA256
    parents = "a" * 64
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def __call__(self, sensor, delta):
        self.calls.append(tuple(sensor.shape))
        return sensor[:, 0, 0, 0, 0, None].expand(-1, 17).clone()


def stream():
    experts = StubExperts()
    normalizer = FrozenNormalizer(
        np.zeros(17, np.float32),
        np.ones(17, np.float32),
        "H",
        experts.schema,
        experts.parents,
        "b" * 64,
        "event",
    )
    return NativeStream(experts, normalizer), experts


def req(anchor, available=None):
    return Request(
        anchor,
        anchor if available is None else available,
        ((anchor - 200000, anchor - 100000), (anchor - 100000, anchor)),
        (0.0, 0.0, 20.0, 20.0),
        (0.1,),
        lambda: torch.full((2, 12, 4, 4), float(anchor)),
    )


def test_reuse_batches_only_new_observations_and_resets_identity():
    runtime, experts = stream()
    first = runtime.prepare("s", "o", 400000, [req(300000), req(400000)])
    second = runtime.prepare("s", "o", 500000, [req(300000), req(400000), req(500000)])
    assert first["computed"] == 2 and second["computed"] == 1 and second["reused"] == 2
    assert [c[0] for c in experts.calls] == [2, 1]
    assert second["anchors"] == [300000, 400000, 500000]
    changed = runtime.prepare("s", "different", 500000, [req(300000), req(400000), req(500000)])
    assert changed["reused"] == 0
    runtime.prepare("s", "different", 500000, [req(300000, 500000)])
    assert experts.calls[-1][0] == 1  # Different ROI-availability clock cannot be exact reuse.


def test_invalid_future_or_excess_history_rejected_before_reader():
    runtime, experts = stream()
    with pytest.raises(ValueError, match="unavailable"):
        runtime.prepare("s", "t", 400000, [req(400001)])
    with pytest.raises(ValueError, match="650"):
        runtime.prepare("s", "t", 1000000, [req(200000)])
    assert not experts.calls


def test_rgb_frame_cache_exact_native_crops_and_causal_mixed_lengths():
    calls = []
    frame = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)

    def reader(seq, t):
        calls.append((seq, t))
        return frame

    inputs = RGBInputs(reader, max_bytes=20000)
    a = inputs.request(
        "s",
        300000,
        timestamps=[100000, 200000, 300000],
        available=300000,
        roi=(0.0, 0.0, 30.0, 30.0),
    )
    b = inputs.request(
        "s",
        400000,
        timestamps=[200000, 300000, 400000],
        available=400000,
        roi=(0.0, 0.0, 30.0, 30.0),
    )
    x, y = a.load(), b.load()
    assert len(calls) == 4 and torch.equal(x[1:], y[:2])
    assert inputs.retained_bytes <= inputs.max_bytes
    with pytest.raises(ValueError):
        inputs.request(
            "s", 300000, timestamps=[200000, 400000], available=400000, roi=(0.0, 0.0, 30.0, 30.0)
        )


def test_event_packets_and_voxel_cache_preserve_native_inputs():
    ring = PacketRing()
    raw = {
        "x": np.array([10, 11, 12], np.int32),
        "y": np.array([10, 11, 12], np.int32),
        "t": np.array([150000, 250000, 350000], np.int64),
        "p": np.array([1, -1, 1], np.int8),
    }
    ring.push(raw, 0, 500000)
    inputs = EventInputs(ring)
    request = inputs.request(
        "s",
        "t",
        400000,
        anchor=400000,
        available=400000,
        windows=[(100000, 200000), (200000, 300000), (300000, 400000)],
        roi=(0.0, 0.0, 30.0, 30.0),
    )
    x = request.load()
    assert x.shape == (3, 12, 128, 128)
    assert torch.equal(x, request.load())
    assert inputs.preparer.last_diagnostics["voxel_misses"] == 0


def test_h2_slicing_keeps_native_clock_semantics():
    runtime, _ = stream()
    prepared = runtime.prepare("s", "t", 500000, [req(300000), req(400000), req(500000)])

    class Head(nn.Module):
        def forward(self, *args):
            return args

    actual = temporal_forward(Head(), prepared, history=2)
    assert torch.equal(actual[1], prepared["timing"][:, -2:])


def test_student_rejects_heldout_training_and_v12_weights(tmp_path):
    from operational.rgb_port_streaming.distill import fit, load

    with pytest.raises(ValueError, match="role H"):
        fit(
            np.ones((2, 17), np.float32),
            np.array(["s1", "s2"]),
            tmp_path / "out",
            modality="event",
            parent_sha256="a" * 64,
            fit_role="V",
            updates=1,
        )
    path = tmp_path / "legacy.pt"
    torch.save({"schema": "a5_feature_student_v1"}, path)
    with pytest.raises(ValueError, match="V13"):
        load(path)


def test_native_current_mask_adapter_preserves_scalar_mask():
    from operational.rgb_port_streaming.contracts import NativeObservation

    value = NativeObservation(
        torch.zeros(2, 128),
        torch.zeros(2, 3),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, 3),
        torch.tensor([True, False]),
    )
    value.validate()
    assert value.known.shape == (2,)
    from dataclasses import replace

    with pytest.raises(ValueError, match="known mask"):
        replace(value, known=value.known.float()).validate()


def test_private_transport_code_identity_is_not_shared_between_models():
    from operational.rgb_port_streaming.transport import install_transport

    class Model(nn.Module):
        def _forward_impl(self, x):
            return x

    a, b = Model(), Model()
    original = Model._forward_impl.__code__
    install_transport(a)
    install_transport(b)
    assert a._forward_impl.__func__.__code__ is not b._forward_impl.__func__.__code__
    assert a._forward_impl.__func__.__code__ is not original
    assert Model._forward_impl.__code__ is original


def test_native_student_roundtrip_with_disjoint_sequence_folds(tmp_path):
    import json

    from operational.rgb_port_streaming.distill import fit, load

    values = np.random.default_rng(7).normal(size=(8, 17)).astype(np.float32)
    state = torch.get_rng_state().clone()
    fit(
        values,
        np.array(["a"] * 4 + ["b"] * 4),
        tmp_path / "student",
        modality="event",
        parent_sha256="a" * 64,
        fit_role="H",
        updates=1,
    )
    assert torch.equal(state, torch.get_rng_state())
    model, binding = load(tmp_path / "student/student.pt")
    assert binding["fit_role"] == "H" and binding["schema_sha256"] == EVENT_PHASE17_SHA256
    assert torch.isfinite(model(torch.from_numpy(values))).all()
    report = json.loads((tmp_path / "student/RESULT.json").read_text())
    assert not set(report["training_sequences"]) & set(report["validation_sequences"])
