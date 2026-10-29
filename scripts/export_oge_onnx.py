"""Export a frozen OGE checkpoint with a real cache sample and verify ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEConfig
from e_jepa_ttc.runtime.oge_export import export_oge_onnx


def _inputs(
    dataset: EAPObjectCacheDataset,
) -> dict[str, torch.Tensor]:
    sample = dataset[0]

    def tensor(name: str) -> torch.Tensor:
        value = sample[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor field {name!r}.")
        return value.unsqueeze(0)

    end_us = tensor("context_window_end_us").float()
    return {
        "context_events": tensor("context_events").float(),
        "context_times_s": (end_us - end_us[:, :1]) * 1e-6,
        "context_boxes": tensor("context_boxes").float(),
        "context_object_mask": tensor("context_object_mask").bool(),
        "context_ego_actions": tensor("context_ego_actions").float(),
        "context_ego_action_mask": tensor("context_ego_action_mask").bool(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    assert_no_sealed_benchmark_paths((args.checkpoint, args.cache_manifest, args.output_dir))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObjectGeometryJEPATTC(OGEConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = EAPObjectCacheDataset(args.cache_manifest, splits=(args.split,))
    metadata = export_oge_onnx(model, _inputs(dataset), output_dir=args.output_dir)
    dataset.close()
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
