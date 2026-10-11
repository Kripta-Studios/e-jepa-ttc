"""Compare repaired transfer with native H/V validation on real sensor inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import torch

from operational.rgb_port import transfer
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port_revision.transfer_inputs import prepare_event, raw_rgb_observations
from operational.rgb_port_streaming.contracts import NativeObservation

from .runtime import observation_contract


def main() -> None:
    """Load final models, compare both modalities, and never update weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = read_json_shared(args.manifest)["rows"]
    models = transfer._models(
        args.run, Path("configs/rgb_port/producers.json"), args.run / "P_MANIFEST.json", "cpu"
    )
    row = rows[0]
    prepared = prepare_event(row)
    rgb, reason = raw_rgb_observations(row)
    if not rgb:
        raise ValueError(f"First query has no RGB for parity: {reason}")
    sensor, meta = rgb[-1]
    inputs = {
        "E": (torch.from_numpy(prepared["own_events"]), torch.full((4, 2), 0.1)),
        "R": (torch.from_numpy(sensor).unsqueeze(0), torch.tensor([meta["delta_t_s"]])),
    }
    results = {}
    with torch.inference_mode():
        for modality, (sensor, delta) in inputs.items():
            try:
                transfer._expert_block(models, modality, sensor, delta)
            except ValueError as exc:
                if "known mask differs" not in str(exc):
                    raise
            else:
                raise AssertionError("Expected the original mask failure")
            with patch.object(transfer, "ProducerObservation", NativeObservation):
                reference = transfer._expert_block(models, modality, sensor, delta)
            with observation_contract():
                corrected = transfer._expert_block(models, modality, sensor, delta)
            for before, after in zip(reference, corrected, strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            results[modality] = {
                "phase17_and_experts_exact": True,
                "observations": len(sensor),
                "steps": sensor.shape[1],
            }
    atomic_write_json(
        args.output,
        {
            "status": "PASSED",
            "optimizer_updates": 0,
            "query_id": row["query_id"],
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
