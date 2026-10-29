"""Export a dense CausalScaleTTC V8 checkpoint as fixed batch-one ONNX CPU inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import (  # noqa: E402
    assert_v8_delivery_paths_safe,
    export_v8_dense_onnx,
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--example-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-adapter", choices=("none", "strip_module_prefix"), default="none")
    args = parser.parse_args()
    assert_v8_delivery_paths_safe((args.checkpoint, args.example_input, args.output_dir))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_payload = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain model_config and model_state_dict")
    if args.state_adapter == "strip_module_prefix":
        state = {str(name).removeprefix("module."): value for name, value in state.items()}
    model = CausalScaleTTC(CausalScaleTTCConfig(**config_payload))
    model.load_state_dict(state, strict=True)
    with np.load(args.example_input, allow_pickle=False) as archive:
        required = ("representations", "delta_t_s")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"example input archive lacks fields: {missing}")
        inputs = {name: torch.from_numpy(np.asarray(archive[name])) for name in required}
    metadata = export_v8_dense_onnx(
        model,
        inputs,
        output_dir=args.output_dir,
        state_adapter_disclosure={
            "adapter": args.state_adapter,
            "checkpoint": args.checkpoint.name,
        },
        normalization={
            "representation": "frozen_v8_temporal_dense_tensor",
            "delta_t_s": "positive_seconds_between_causal_endpoints",
        },
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
