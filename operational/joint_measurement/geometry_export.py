"""Fixed-shape portable geometry adapter, preserving the registered trainer bytes.

The training forward's Python set-of-shapes check is incompatible with tracing.
This adapter spells out the same tensor operations with external shape admission;
it shares the trained modules and math helpers and is parity-tested before export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from e_jepa_ttc.joint_measurement.model import bidirectional_multiscale
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha, write

from .common import ROOT
from .data import SensorDataset
from .geometry_evaluate import freeze_endpoint, load_model
from .geometry_model import GeometricPairMeasurement, relative_scale_phase, weighted_affine
from .geometry_train import DIRECTORY, FIT


class PortableGeometry(nn.Module):
    """Same trained tensors and geometry, with a separately admitted batch-one input contract."""

    def __init__(self, model: GeometricPairMeasurement) -> None:
        super().__init__()
        self.model = model

    def forward(self, events: Tensor, delta: Tensor) -> Tensor:
        """Emit signed TTC for already-validated [1,2,12,128,128] FP32 inputs."""
        model = self.model
        dt = delta.reshape(1)
        dense = model.encoder(events.flatten(0, 1)).reshape(1, 2, 64, 32, 32)
        previous, current = dense[:, 0], dense[:, 1]
        correlations, support = bidirectional_multiscale(previous, current)
        axis = (torch.arange(32, dtype=events.dtype, device=events.device) + 0.5) / 16 - 1
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((xx, yy))[None]
        spatial = model.spatial(
            torch.cat(
                (
                    previous,
                    current,
                    current - previous,
                    current * previous,
                    correlations,
                    support.to(previous.dtype),
                    coordinates,
                    dt[:, None, None, None].expand(-1, 1, 32, 32),
                ),
                dim=1,
            )
        )
        descriptor = torch.cat(
            (spatial.mean((-2, -1)), spatial.amax((-2, -1)), dt[:, None], dt.log()[:, None]), dim=1
        )
        gates = model.gates(descriptor).sigmoid()
        weights = functional.softplus(model.spatial_weights(spatial)) + 1e-4
        offsets = torch.arange(-2, 3, dtype=events.dtype, device=events.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        scales = []
        for index, scale in enumerate((1, 1, 2, 2)):
            volume = correlations[:, index * 25 : (index + 1) * 25]
            valid = support[:, index * 25 : (index + 1) * 25]
            probability = (volume / model.temperature).masked_fill(~valid, -torch.inf).softmax(1)
            dx = (probability * ox.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            dy = (probability * oy.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            interior = torch.zeros_like(weights[:, index])
            margin = 2 * scale
            interior[:, margin:-margin, margin:-margin] = 1
            matrix, _ = weighted_affine(torch.stack((dx, dy), 1), weights[:, index] * interior)
            log_axes = 0.5 * matrix.square().sum(1).clamp_min(1e-12).log()
            scales.append(gates[:, 0] * log_axes[:, 0] + (1 - gates[:, 0]) * log_axes[:, 1])
        fine = 0.5 * (scales[0] - scales[1])
        coarse = 0.5 * (scales[2] - scales[3])
        ratio = (1 - gates[:, 1]) * fine + gates[:, 1] * coarse
        return phase_to_ttc(relative_scale_phase(ratio, dt))


def trace_portable(
    directory: Path,
    model: GeometricPairMeasurement,
    inputs: list[tuple[Tensor, Tensor]],
    binding: dict,
) -> dict:
    """Trace and reload, comparing every example to the ORIGINAL registered forward."""
    if len(inputs) < 2:
        raise ValueError("At least two independently varying input examples required")
    for events, delta in inputs:
        if (
            events.shape != (1, 2, 12, 128, 128)
            or delta.shape != (1, 1)
            or events.dtype != torch.float32
            or delta.dtype != torch.float32
            or events.device.type != "cpu"
            or delta.device.type != "cpu"
            or not torch.isfinite(events).all()
            or not torch.isfinite(delta).all()
            or not (delta > 0).all()
        ):
            raise ValueError("Portable geometry requires native finite CPU FP32 batch-one inputs")
    freeze(directory / "BINDING.json", binding)
    if (directory / "COMPLETE.json").exists():
        receipt = read(directory / "COMPLETE.json")
        if any(sha(directory / name) != expected for name, expected in receipt["files"].items()):
            raise ValueError("Geometry portable artifact changed")
        return receipt
    model = model.cpu().eval().requires_grad_(False)
    wrapper = PortableGeometry(model)
    with torch.inference_mode():
        expected = [phase_to_ttc(model(*example)["raw_phase"]) for example in inputs]
        for example, target in zip(inputs, expected, strict=True):
            # Eager adapter parity is bit-exact; traced arithmetic allows declared rounding.
            if not torch.equal(wrapper(*example), target):
                raise ValueError("Geometry tensor adapter changed outputs")
        traced = torch.jit.trace(wrapper, inputs[0], check_inputs=inputs[1:], strict=True)
        path = directory / "model.torchscript.pt"
        with path.open("wb") as stream:
            torch.jit.save(traced, stream)
        with path.open("rb") as stream:
            loaded = torch.jit.load(stream, map_location="cpu").eval()
        checks = []
        for index, (example, target) in enumerate(zip(inputs, expected, strict=True)):
            actual = loaded(*example)
            torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-5)
            checks.append(
                {
                    "example": index,
                    "maxabs_s": float((actual - target).abs().max()),
                    "expected": target.tolist(),
                    "actual": actual.tolist(),
                }
            )
        np.savez(
            directory / "example_input.npz",
            events=inputs[0][0].numpy(),
            delta_t_s=inputs[0][1].numpy(),
        )
        write(directory / "example_output.json", {"signed_ttc_seconds": checks[0]["actual"]})
    write(
        directory / "REPLAY.json",
        {
            "checks": checks,
            "device": "cpu",
            "optimizer_updates": 0,
            "eager_adapter_bit_exact": True,
            "reference": "original registered GeometricPairMeasurement.forward",
        },
    )
    write(
        directory / "normalization.json",
        {
            "input": "admitted native P2 common-ROI FP32 event tensors and native delta",
            "includes_sensor_preprocessing": False,
            "input_shape": [1, 2, 12, 128, 128],
            "delta_shape": [1, 1],
            "requires_external_input_validation": True,
        },
    )
    names = (
        "BINDING.json",
        "model.torchscript.pt",
        "example_input.npz",
        "example_output.json",
        "REPLAY.json",
        "normalization.json",
    )
    result = {"status": "COMPLETE", "files": {name: sha(directory / name) for name in names}}
    freeze(directory / "COMPLETE.json", result)
    return result


def export() -> dict:
    """Export only a frozen scientific endpoint and replay five declared P queries."""
    endpoint = freeze_endpoint()["endpoints"][FIT]
    torch.set_num_threads(1)
    binding = {
        "fit_id": FIT,
        "endpoint": endpoint,
        "P_rows": [0, 17, 83, 159, 255],
        "rtol": 1e-5,
        "atol_seconds": 1e-5,
        "sources": {
            name: sha(ROOT / "operational/joint_measurement" / name)
            for name in ("geometry_model.py", "geometry_export.py")
        },
    }
    dataset = SensorDataset("P")
    try:
        inputs = []
        for index in binding["P_rows"]:
            batch = dataset.batch([index])
            inputs.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        return trace_portable(DIRECTORY / "exports" / FIT, load_model(), inputs, binding)
    finally:
        dataset.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(export())
