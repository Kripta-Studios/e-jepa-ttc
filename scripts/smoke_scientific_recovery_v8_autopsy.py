#!/usr/bin/env python
"""Run the V8-A checkpoint replay path on a deterministic CPU fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.scientific_recovery_v8 import canonical_json_sha256  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scientific_recovery_v8/smoke_autopsy"),
    )
    args = parser.parse_args()
    replay_path = ROOT / "scripts" / "replay_scientific_recovery_v8_mechanisms.py"
    spec = importlib.util.spec_from_file_location("v8_autopsy_replay", replay_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the V8-A replay runner")
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)

    torch.manual_seed(17)
    config = CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=8,
        geometry_dim=8,
        residual_depth=1,
        dropout=0.0,
        foreground_temporal_smoothing_mode="causal_left",
        transport_enabled=True,
        transport_radius=1,
    )
    model = CausalScaleTTC(config).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "fixture_checkpoint.pt"
    replay_input = args.output_dir / "fixture_rows.pt"
    torch.save(
        {"model_config": config.__dict__, "model_state_dict": model.state_dict()}, checkpoint
    )
    torch.save(
        {
            "events": torch.rand(2, 3, 2, 16, 16),
            "delta_t_s": torch.full((2, 2), 0.01),
            "target_ttc": torch.tensor([2.0, 4.0]),
            "sample_weight": torch.ones(2),
            "token_id": ["fixture-a", "fixture-b"],
            "sequence_id": ["fixture-sequence-a", "fixture-sequence-b"],
            "track_id": ["fixture-track-a", "fixture-track-b"],
            "outer_fold": [0, 1],
            "seed": [7, 7],
            "endpoint_us": torch.tensor([[100, 200, 300], [400, 500, 600]]),
        },
        replay_input,
    )
    result = replay_module.run_replay(
        checkpoint=checkpoint,
        replay_input=replay_input,
        output_dir=args.output_dir / "replay",
        model_name="a5",
        config_sha256=canonical_json_sha256(model.checkpoint_config()),
        device_name="cpu",
    )
    if len(result["factorial_cells"]) != 5:
        raise RuntimeError("V8-A smoke did not produce all five factorial cells")
    print(json.dumps({"status": "passed", "manifest": result["artifact_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
