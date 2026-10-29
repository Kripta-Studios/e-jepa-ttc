"""Audit CARLA DVS Looming and create portable manifest/splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.data.carla_looming import (  # noqa: E402
    scan_carla_looming_root,
    summarize_carla_sequences,
    write_carla_looming_manifest,
    write_carla_looming_splits,
)


def main() -> int:
    """Run the pickle-free audit and blocked split preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/CARLA_DVS_Looming_Dataset/random_spawn"),
        help="Directory containing example_<index> sequence folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/carla_dvs_looming_v1.json"),
        help="Output manifest; stores only paths relative to --root.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
        help="Output train/validation/test split.",
    )
    parser.add_argument("--context-ms", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-event-validation",
        action="store_true",
        help=(
            "Scan every event for monotonicity, bounds and polarity. The default "
            "audit validates structure and temporal endpoints without reading 7.7B events."
        ),
    )
    args = parser.parse_args()

    sequences = scan_carla_looming_root(
        args.root,
        context_ms=args.context_ms,
        group_size=args.group_size,
        full_event_validation=args.full_event_validation,
    )
    try:
        root_hint = args.root.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        root_hint = "${CARLA_LOOMING_ROOT}"
    manifest = write_carla_looming_manifest(
        args.manifest,
        sequences,
        root_hint=root_hint,
        context_ms=args.context_ms,
    )
    split = write_carla_looming_splits(
        args.split,
        manifest_path=args.manifest,
        sequences=sequences,
        seed=args.seed,
        folds=args.folds,
    )
    print(
        json.dumps(
            {
                "root": str(args.root),
                "manifest": str(args.manifest),
                "manifest_artifact_sha256": manifest["artifact_sha256"],
                "split": str(args.split),
                "split_artifact_sha256": split["artifact_sha256"],
                "full_event_validation": args.full_event_validation,
                "summary": summarize_carla_sequences(sequences),
                "split_statistics": split["statistics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
