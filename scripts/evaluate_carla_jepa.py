"""Evaluate a validation-selected CARLA JEPA checkpoint without TTC labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.training.carla_jepa import evaluate_carla_jepa  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/CARLA_DVS_Looming_Dataset/random_spawn"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/carla_dvs_looming_v1.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
    )
    parser.add_argument("--role", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    summary = evaluate_carla_jepa(
        root=args.root,
        manifest_path=args.manifest,
        split_path=args.split,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        role=args.role,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
