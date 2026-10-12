"""Fixed-P2 RGB portable exports, with real RGB pixels and native frame intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

from operational.range_transfer.common import freeze, read, sha, write

from .common import ROOT, RUN, require_resources
from .export import TTCOnly
from .rgb_data import RGBPairDataset
from .rgb_evaluate import freeze_endpoints, load_model


def trace_portable(
    directory: Path,
    model: CausalScaleTTC | PairMeasurement,
    inputs: list[tuple[torch.Tensor, torch.Tensor]],
    binding: dict,
) -> dict:
    """Trace/serialize/reload a fixed RGB tensor contract without any optimizer."""
    if len(inputs) < 2 or binding.get("pixel_input_key") != "rgb":
        raise ValueError("At least two RGB examples and an explicit RGB input contract required")
    for pixels, delta in inputs:
        if (
            pixels.shape != (1, 2, 3, 128, 128)
            or delta.shape != (1, 1)
            or pixels.dtype != torch.float32
            or delta.dtype != torch.float32
            or pixels.device.type != "cpu"
            or delta.device.type != "cpu"
            or not torch.isfinite(pixels).all()
            or (pixels < 0).any()
            or (pixels > 1).any()
            or not torch.isfinite(delta).all()
            or not (delta > 0).all()
        ):
            raise ValueError("Portable RGB input violates the native P2 contract")
    freeze(directory / "BINDING.json", binding)
    if (directory / "COMPLETE.json").exists():
        receipt = read(directory / "COMPLETE.json")
        if any(sha(directory / name) != value for name, value in receipt["files"].items()):
            raise ValueError("RGB portable artifact changed")
        return receipt
    wrapper = TTCOnly(model).cpu().eval().requires_grad_(False)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, inputs[0], check_inputs=inputs[1:], strict=True)
        path = directory / "model.torchscript.pt"
        with path.open("wb") as stream:
            torch.jit.save(traced, stream)
        with path.open("rb") as stream:
            replay = torch.jit.load(stream, map_location="cpu").eval()
        checks = []
        for ordinal, example in enumerate(inputs):
            expected, actual = wrapper(*example), replay(*example)
            torch.testing.assert_close(
                actual, expected, rtol=binding["rtol"], atol=binding["atol_seconds"]
            )
            if not torch.isfinite(actual).all():
                raise ValueError("Nonfinite RGB portable output")
            checks.append(
                {
                    "example": ordinal,
                    "maxabs_s": float((actual - expected).abs().max()),
                    "expected": expected.tolist(),
                    "actual": actual.tolist(),
                }
            )
        np.savez(
            directory / "example_input.npz",
            rgb=inputs[0][0].numpy(),
            delta_t_s=inputs[0][1].numpy(),
        )
        write(directory / "example_output.json", {"signed_ttc_seconds": checks[0]["actual"]})
    write(directory / "REPLAY.json", {"checks": checks, "device": "cpu", "optimizer_updates": 0})
    write(
        directory / "normalization.json",
        {
            "pixels": "native decoded RGB uint8 converted to float32 divided by 255",
            "crop": "historical common ROI; preserve native interpolation and temporal pairing",
            "delta_t_s": "measured native RGB sensor-frame interval; never event-window delta",
            "sensor_preprocessing_included": False,
        },
    )
    names = [
        "BINDING.json",
        "model.torchscript.pt",
        "example_input.npz",
        "example_output.json",
        "REPLAY.json",
        "normalization.json",
    ]
    receipt = {"status": "COMPLETE", "files": {name: sha(directory / name) for name in names}}
    freeze(directory / "COMPLETE.json", receipt)
    return receipt


def export(fit_id: str) -> dict:
    """Export only endpoints sealed after the registered RGB comparison completes."""
    endpoints = freeze_endpoints()["endpoints"]
    if fit_id not in endpoints:
        raise ValueError("Only frozen RGB endpoints can be exported")
    require_resources(starting=True)
    torch.set_num_threads(1)
    binding = {
        "fit_id": fit_id,
        "endpoint": endpoints[fit_id],
        "pixel_input_key": "rgb",
        "input_shape": [1, 2, 3, 128, 128],
        "delta_shape": [1, 1],
        "dtype": "float32",
        "domain": "signed native TTC seconds",
        "test_rows": [0, 17, 83, 159, 255],
        "role": "P",
        "rtol": 1e-5,
        "atol_seconds": 1e-5,
        "includes_sensor_preprocessing": False,
        "sources": {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/rgb_export.py",
                "operational/joint_measurement/export.py",
            )
        },
    }
    model = load_model(fit_id)
    dataset = RGBPairDataset("P")
    try:
        inputs = []
        for index in binding["test_rows"]:
            batch = dataset.batch([index])
            inputs.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        return trace_portable(RUN / "rgb/exports" / fit_id, model, inputs, binding)
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_id")
    print(export(parser.parse_args().fit_id))
