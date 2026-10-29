"""Verify companion PHASE17 assembly against every pre-amendment finite block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.evaluation.stage61_nested_pair_router import build_router_features
from e_jepa_ttc.models.three_expert_router import BASE8_FEATURES
from e_jepa_ttc.simplex_t.expert_phase import build_expert_features


def main() -> None:
    """Read cached original predictions only; never score or open supervision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve extension proof")
    blocks, observations = 0, 0
    for receipt in sorted(args.cache.glob("family*_query*.json")):
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        path = receipt.with_suffix(".npz")
        if compute_file_hash(str(path)) != saved["sha256"]:
            raise ValueError("cached payload changed")
        with np.load(path, allow_pickle=False) as arrays:
            values, points = arrays["features145"], arrays["expert_ttc"]
            if not np.isfinite(points).all():
                raise ValueError("this proof requires the pre-extension finite cohort")
            frames = []
            for expert in range(2):
                frame = pd.DataFrame(
                    values[:, :8].astype(np.float64), columns=np.asarray(BASE8_FEATURES)
                )
                frame["token_id"] = np.arange(len(points)).astype(str)
                frame["prediction_ttc"] = points[:, expert]
                frames.append(frame)
            old = build_router_features(frames[0], frames[1], points[:, 2])[1].to_numpy(np.float32)
            new = build_expert_features(frames[0], frames[1], points[:, 2])[1].to_numpy(np.float32)
            if old.tobytes() != new.tobytes() or new.tobytes() != values[:, :17].tobytes():
                raise ValueError(f"finite feature bytes changed: {path.name}")
            observations += len(points)
        blocks += 1
    write_new_json(
        args.output,
        {
            "status": "ALL_EXISTING_FINITE_BLOCKS_EXACT_PHASE_EXTENSION_PARITY",
            "blocks": blocks,
            "observations": observations,
            "all_exact": True,
            "expert_phase_source_sha256": compute_file_hash(
                "src/e_jepa_ttc/simplex_t/expert_phase.py"
            ),
            "targets_read": False,
            "optimizer_updates": 0,
        },
    )
    print(json.dumps({"blocks": blocks, "observations": observations, "all_exact": True}))


if __name__ == "__main__":
    main()
