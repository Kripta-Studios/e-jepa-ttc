from __future__ import annotations

import io

import pytest
import torch

from operational.final_refit.training import PhaseOnlyFinal
from scripts.runpod_final_export_compat import ScriptablePhaseOnlyFinal


def _models() -> tuple[PhaseOnlyFinal, ScriptablePhaseOnlyFinal]:
    torch.manual_seed(17)
    source = PhaseOnlyFinal().eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.2, 0.2)
    compatible = ScriptablePhaseOnlyFinal().eval()
    compatible.load_state_dict(source.state_dict())
    return source, compatible


def test_scripted_state_and_outputs_are_exact() -> None:
    source, compatible = _models()
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(compatible), buffer)
    buffer.seek(0)
    restored = torch.jit.load(buffer).eval()
    assert set(source.state_dict()) == set(restored.state_dict())
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    for batch in (1, 7):
        features = torch.randn(batch, 340)
        widths = torch.randn(batch, 2)
        expected = source(features, widths)
        observed = restored(features, widths)
        assert set(expected) == set(observed)
        for name in expected:
            torch.testing.assert_close(observed[name], expected[name], atol=1e-6, rtol=0)


def test_phase_edges_preserve_original_branches() -> None:
    source, compatible = _models()
    restored = torch.jit.script(compatible)
    phases = torch.tensor([-100.0, -0.6931461805599453, -1e-8, 0.0, 1e-8, 8.0, 100.0])
    features = torch.zeros((len(phases), 340))
    features[:, 160] = phases
    widths = torch.zeros((len(phases), 2))
    expected = source(features, widths)
    observed = restored(features, widths)
    for name in expected:
        torch.testing.assert_close(observed[name], expected[name], atol=1e-6, rtol=0)


@pytest.mark.parametrize("shape", [(340,), (2, 339), (2, 341)])
def test_feature_shape_guard_is_preserved(shape: tuple[int, ...]) -> None:
    scripted = torch.jit.script(ScriptablePhaseOnlyFinal().eval())
    with pytest.raises(
        (torch.jit.Error, RuntimeError, ValueError), match="PHASE_ONLY expects frozen"
    ):
        scripted(torch.zeros(shape), torch.zeros((shape[0], 2)))


def test_nonfinite_phase_guard_is_preserved() -> None:
    scripted = torch.jit.script(ScriptablePhaseOnlyFinal().eval())
    features = torch.zeros((1, 340))
    features[0, 160] = float("nan")
    with pytest.raises((torch.jit.Error, RuntimeError, ValueError), match="nonfinite phase"):
        scripted(features, torch.zeros((1, 2)))
