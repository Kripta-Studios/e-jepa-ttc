"""RGB anchor, missing-modality and equal-capacity event-ablation contracts."""

import copy
import io

import pytest
import torch

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, phase_to_ttc
from operational.joint_measurement.fusion_model import NativePhaseConversion, RGBAnchorComplement


def example():
    return (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([[4.0, 3.0, 2.0], [2.0, 3.0, 4.0], [5.0, 3.0, 1.0]]),
        torch.tensor([6.0, float("nan"), -4.0]),
        torch.tensor([-6.0, 2.987654, 4.0]),
        torch.tensor([True, False, True]),
    )


def test_initial_prediction_anchors_rgb_without_averaging_opposite_signs():
    model = RGBAnchorComplement(2, 3)
    values = example()
    output = model(*values)
    torch.testing.assert_close(output["ttc"][[0, 2]], values[2][[0, 2]])
    assert torch.equal(output["ttc"][1:2], values[3][1:2])
    assert torch.equal(output["used_rgb"], values[-1])


def test_all_missing_rgb_skips_network_and_copies_event_output_exactly():
    model = RGBAnchorComplement(2, 3)
    calls = []
    handle = model.network.register_forward_hook(lambda *args: calls.append(True))
    rgb, event, anchor, fallback, available = example()
    rgb.fill_(float("nan"))
    event.fill_(float("nan"))
    anchor.fill_(float("nan"))
    available.fill_(False)
    try:
        result = model(rgb, event, anchor, fallback, available)
    finally:
        handle.remove()
    assert not calls
    assert torch.equal(result["ttc"], fallback)


def test_zero_control_removes_event_features_after_nonzero_head_initialization():
    true = RGBAnchorComplement(2, 3)
    with torch.no_grad():
        true.network[-1].weight.fill_(0.02)
    zero = copy.deepcopy(true)
    zero.mode = "ZERO"
    assert sum(p.numel() for p in true.parameters()) == sum(p.numel() for p in zero.parameters())
    assert all(torch.equal(a, b) for a, b in zip(true.parameters(), zero.parameters(), strict=True))
    values = example()
    expected = zero(*values)
    values[1].fill_(float("nan"))
    actual = zero(*values)
    assert torch.equal(actual["ttc"], expected["ttc"])
    with pytest.raises(ValueError, match="features must be finite"):
        true(*values)


def test_available_rows_train_correction_and_missing_row_keeps_exact_fallback():
    model = RGBAnchorComplement(2, 3)
    values = example()
    result = model(*values)
    result["raw_phase"][[0, 2]].sum().backward()
    head = model.network[-1]
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0
    assert head.bias.grad.abs().sum() > 0
    assert torch.equal(result["ttc"][1:2], values[3][1:2])


@pytest.mark.parametrize("invalid", ["shape", "availability", "anchor", "fallback", "dtype"])
def test_invalid_fusion_inputs_rejected(invalid):
    values = list(example())
    if invalid == "shape":
        values[1] = values[1][:, :2]
    elif invalid == "availability":
        values[-1] = values[-1].float()
    elif invalid == "anchor":
        values[2][0] = 0
    elif invalid == "fallback":
        values[3][1] = float("nan")
    else:
        values[0] = values[0].double()
    with pytest.raises(ValueError):
        RGBAnchorComplement(2, 3)(*values)


@pytest.mark.parametrize("mode", ["TRUE", "ZERO"])
def test_script_reload_keeps_dynamic_availability_and_exact_fallback(mode):
    model = RGBAnchorComplement(2, 3, mode=mode).eval()
    with torch.no_grad():
        model.network[-1].weight.fill_(0.01)
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(model), buffer)
    buffer.seek(0)
    loaded = torch.jit.load(buffer, map_location="cpu").eval()
    for available in ([True, False, True], [False, False, False], [True, True, True]):
        values = list(example())
        values[-1] = torch.tensor(available)
        values[2] = torch.tensor([6.0, 3.0, -4.0])
        values[0][~values[-1]] = float("nan")
        values[2][~values[-1]] = float("nan")
        with torch.inference_mode():
            expected, actual = model(*values), loaded(*values)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        assert torch.equal(actual["ttc"][~values[-1]], values[3][~values[-1]])


def test_scriptable_converter_matches_native_values_and_derivatives():
    points = torch.cat(
        (
            torch.linspace(-1, 9, 1001),
            torch.tensor([-100.0, MIN_PHASE, -1e-8, 0.0, 1e-8, MAX_PHASE, 100.0]),
        )
    )
    reference = points.clone().requires_grad_(True)
    eager = points.clone().requires_grad_(True)
    scripted = points.clone().requires_grad_(True)
    expected = phase_to_ttc(reference)
    model = NativePhaseConversion()
    actual = model(eager)
    portable = torch.jit.script(model)(scripted)
    assert torch.equal(expected, actual)
    torch.testing.assert_close(portable, expected, rtol=1e-6, atol=1e-6)
    for output in (expected, actual, portable):
        output.sum().backward()
    assert torch.equal(reference.grad, eager.grad)
    torch.testing.assert_close(scripted.grad, reference.grad, rtol=1e-6, atol=1e-6)
