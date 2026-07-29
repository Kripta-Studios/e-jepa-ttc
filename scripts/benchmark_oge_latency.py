"""Benchmark a frozen OGE checkpoint on pre-voxelized tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEConfig
from e_jepa_ttc.runtime.oge_benchmark import benchmark_oge_model_only
from e_jepa_ttc.utils.io import write_structured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    assert_no_sealed_benchmark_paths((args.checkpoint, args.cache_manifest, args.output))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObjectGeometryJEPATTC(OGEConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = EAPObjectCacheDataset(args.cache_manifest, splits=("validation",))
    sample = dataset[0]

    def tensor(name: str) -> torch.Tensor:
        value = sample[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor field {name!r}.")
        return value.unsqueeze(0)

    end_us = tensor("context_window_end_us").float()
    inputs = {
        "context_events": tensor("context_events"),
        "context_times_s": (end_us - end_us[:, :1]) * 1e-6,
        "context_boxes": tensor("context_boxes"),
        "context_object_mask": tensor("context_object_mask"),
        "context_ego_actions": tensor("context_ego_actions"),
        "context_ego_action_mask": tensor("context_ego_action_mask"),
    }
    result = benchmark_oge_model_only(
        model,
        inputs,
        device=device,
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
    )
    dataset.close()
    write_structured(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
