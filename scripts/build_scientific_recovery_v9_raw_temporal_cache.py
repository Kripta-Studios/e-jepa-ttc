"""Build the shared Stage 64 raw cache and exact cross-fitted A5 state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.data.crossfitted_a5_state import build_crossfitted_a5_states  # noqa: E402
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess  # noqa: E402
from e_jepa_ttc.data.raw_temporal_cache import build_raw_temporal_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--raw-train-root", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.output_root.exists():
        args.output_root.mkdir(parents=True)
    elif not args.resume:
        raise FileExistsError("output root exists; use --resume only for identical inputs")
    binding = pd.read_csv(args.binding, dtype={"sample_token": str})
    access = ReadOnlyTrainAccess(args.raw_train_root, args.train_parquet)
    cache = build_raw_temporal_cache(
        binding=binding,
        raw_train_root=args.raw_train_root,
        access=access,
        output_dir=args.output_root / "raw_temporal_cache",
        resume=args.resume,
    )
    source = args.stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62"
    state_root = args.output_root / "crossfitted_a5_state"
    if (state_root / "manifest.json").is_file():
        if not args.resume:
            raise FileExistsError("cross-fitted state already exists")
        state = json.loads((state_root / "manifest.json").read_text(encoding="utf-8"))
    else:
        state = build_crossfitted_a5_states(
            feature_cache_root=source / "feature_cache",
            router_root=(
                args.reference_root / "artifacts" / "scientific_recovery_v8" / "results" / "router"
            ),
            output_root=state_root,
        )
    access.write_ledger(args.output_root / "RAW_CACHE_READ_ACCESS_LEDGER.csv")
    print(
        json.dumps(
            {
                "cache_status": cache["status"],
                "cache_identity_sha256": cache["identity_sha256"],
                "state_artifact_type": state["artifact_type"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
