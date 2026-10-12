"""Portable fixed-P2 TorchScript exports of frozen primary endpoints."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha, write

from .common import RUN
from .data import SensorDataset
from .evaluate import freeze_primary
from .train import initialize, setup


class TTCOnly(nn.Module):
    """Expose events and real delta only; emit signed seconds."""

    def __init__(self, model: CausalScaleTTC | PairMeasurement) -> None:
        super().__init__()
        self.model = model

    def forward(self, events: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the existing native signed conversion."""
        if isinstance(self.model, CausalScaleTTC):
            return self.model(events, delta).ttc_mean_seconds.reshape(-1)
        return phase_to_ttc(self.model(events, delta)["raw_phase"]).reshape(-1)


def export(fit_id: str) -> dict:
    """Trace a fixed batch1 contract and verify on distinct P inputs in CPU."""
    endpoints = freeze_primary()["endpoints"]
    if fit_id not in endpoints:
        raise ValueError("Only frozen endpoints can be exported")
    directory = RUN / "exports" / fit_id
    directory.mkdir(parents=True, exist_ok=True)
    binding = {
        "endpoint": endpoints[fit_id],
        "batch_size": 1,
        "input_shape": [1, 2, 12, 128, 128],
        "delta_shape": [1, 1],
        "dtype": "float32",
        "domain": "signed native TTC",
        "test_rows": [0, 17, 83, 159, 255],
        "role": "P",
        "rtol": 1e-5,
        "atol_seconds": 1e-5,
        "includes_sensor_preprocessing": False,
    }
    freeze(directory / "BINDING.json", binding)
    if (directory / "COMPLETE.json").exists():
        receipt = read(directory / "COMPLETE.json")
        if any(sha(directory / name) != value for name, value in receipt["files"].items()):
            raise ValueError("Portable export changed")
        return receipt
    setup(7)
    model = initialize(fit_id)
    if fit_id != "A5-P2-FROZEN":
        payload = torch.load(
            RUN / "fits" / fit_id / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
    wrapper = TTCOnly(model).cpu().eval().requires_grad_(False)
    dataset = SensorDataset("P")
    try:
        inputs = []
        for index in binding["test_rows"]:
            batch = dataset.batch([index])
            inputs.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, inputs[0], check_inputs=inputs[1:], strict=True)
            path = directory / "model.torchscript.pt"
            # Python opens Unicode Windows paths correctly; the JIT C++ path route may not.
            with path.open("wb") as stream:
                torch.jit.save(traced, stream)
            with path.open("rb") as stream:
                replay = torch.jit.load(stream, map_location="cpu").eval()
            checks = []
            for index, example in zip(binding["test_rows"], inputs, strict=True):
                expected = wrapper(*example)
                actual = replay(*example)
                torch.testing.assert_close(
                    actual, expected, rtol=binding["rtol"], atol=binding["atol_seconds"]
                )
                checks.append(
                    {
                        "P_row": index,
                        "maxabs_s": float((actual - expected).abs().max()),
                        "expected": expected.tolist(),
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
            {"checks": checks, "device": "cpu", "scope": "fixed-shape tensor inference"},
        )
        write(
            directory / "normalization.json",
            {
                "type": "native preprocessing required before tensor input",
                "description": "exact admitted P2 common-ROI native tensors; no new normalization",
            },
        )
        names = [
            "model.torchscript.pt",
            "example_input.npz",
            "example_output.json",
            "BINDING.json",
            "normalization.json",
            "REPLAY.json",
        ]
        receipt = {
            "status": "COMPLETE",
            "fit_id": fit_id,
            "files": {name: sha(directory / name) for name in names},
        }
        freeze(directory / "COMPLETE.json", receipt)
        return receipt
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_id")
    print(json.dumps(export(parser.parse_args().fit_id), indent=2))
