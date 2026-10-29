from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from operational.garl_evttc_competition.model import OfficialGarlRuntime, _NativeModel
from operational.garl_evttc_competition.prepare import Variant


class FakeModel:
    dT = 0.1  # noqa: N815 - upstream public API

    def __init__(self, heights: tuple[float, float]) -> None:
        self.heights = heights

    def forward_test(self, sensor: torch.Tensor):
        values = sensor.new_tensor([self.heights]).expand(len(sensor), -1)
        return values, None

    def float(self) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self

    def requires_grad_(self, requires_grad: bool = True) -> FakeModel:
        return self

    def to(self, device: torch.device) -> FakeModel:
        return self


def _runtime(variant: Variant, heights: tuple[float, float]) -> OfficialGarlRuntime:
    runtime = OfficialGarlRuntime.__new__(OfficialGarlRuntime)
    runtime.device = torch.device("cpu")
    runtime.models = {variant: cast(_NativeModel, FakeModel(heights))}
    return runtime


@pytest.mark.parametrize("variant,channels", [("full", 46), ("event", 40), ("visual", 6)])
def test_actual_delta_converts_raw_heights_without_clipping(variant: str, channels: int) -> None:
    typed_variant = cast(Variant, variant)
    runtime = _runtime(typed_variant, (2.0, 4.0))
    result = runtime.predict(typed_variant, np.zeros((channels, 128, 128), np.float32), 0.08)
    assert result.status == "ok"
    assert result.model_ttc == pytest.approx(0.16)
    assert result.native_delta_t_s == 0.1
    assert result.conversion_delta_t_s == 0.08


def test_signed_output_is_preserved() -> None:
    result = _runtime("event", (4.0, 2.0)).predict(
        "event", np.zeros((40, 128, 128), np.float32), 0.1
    )
    assert result.model_ttc == pytest.approx(-0.1)


def test_zero_denominator_returns_explicit_null() -> None:
    result = _runtime("visual", (2.0, 2.0)).predict(
        "visual", np.zeros((6, 128, 128), np.float32), 0.1
    )
    assert result.status == "unavailable" and result.model_ttc is None
    assert result.reason == "nonfinite_height_ratio_ttc"


def test_near_equal_heights_preserve_release_fp32_rounding() -> None:
    heights = (1.0000001, 1.0)
    result = _runtime("event", heights).predict("event", np.zeros((40, 128, 128), np.float32), 0.08)
    raw = np.asarray([heights], np.float32)
    expected = np.asarray([0.08], np.float32) / (np.float32(1.0) - raw[:, 0] / raw[:, 1])
    assert result.model_ttc == float(expected[0])


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "GARL_INVENTORY" not in os.environ
    or "GARL_CODE_ROOT" not in os.environ,
    reason="requires pinned public Garl assets and CUDA",
)
def test_three_pinned_models_match_direct_native_cuda_forward() -> None:
    runtime = OfficialGarlRuntime(
        Path(os.environ["GARL_CODE_ROOT"]), Path(os.environ["GARL_INVENTORY"]), "cuda"
    )
    for variant, channels in (("full", 46), ("event", 40), ("visual", 6)):
        typed_variant = cast(Variant, variant)
        sensor = np.full((channels, 128, 128), 0.01, np.float32)
        tensor = torch.from_numpy(sensor[None]).cuda()
        with torch.inference_mode():
            heights, _ = runtime.models[typed_variant].forward_test(tensor)
        raw = heights.float().cpu().numpy()[0]
        actual_native = runtime.predict(typed_variant, sensor, 0.1)
        native_expected = 0.1 / (1.0 - raw.reshape(1, -1)[:, 0] / raw.reshape(1, -1)[:, 1])
        actual = runtime.predict(typed_variant, sensor, 0.08)
        expected = np.asarray([0.08], np.float32) / (
            np.float32(1.0) - raw.reshape(1, -1)[:, 0] / raw.reshape(1, -1)[:, 1]
        )
        assert actual.heights == pytest.approx(raw.tolist(), rel=1e-6, abs=1e-6)
        if np.isfinite(native_expected[0]):
            assert actual_native.model_ttc == float(native_expected[0])
        if np.isfinite(expected[0]):
            assert actual.model_ttc == float(expected[0])
        assert runtime.bindings[variant]["strict_state_load"] is True
        assert runtime.bindings[variant]["parameter_count"] > 0
