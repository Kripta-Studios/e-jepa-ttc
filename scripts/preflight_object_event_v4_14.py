#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

LOCKED = {
    "base_blend": 0.20,
    "override_blend": 0.51,
    "negative_override_probability": 0.985,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v410-summary", type=Path, required=True)
    parser.add_argument("--v413-summary", type=Path, required=True)
    parser.add_argument("--v48-root", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.config, args.v410_summary, args.v413_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fusion = raw.get("fusion", {})
    if fusion != LOCKED:
        raise RuntimeError(f"v4.14 fusion is not locked to v4.13: {fusion}")
    aggregate = raw.get("aggregate", {})
    seeds = [int(seed) for seed in aggregate.get("seeds", [])]
    if seeds != [7, 13, 23]:
        raise RuntimeError(f"Unexpected v4.14 seeds: {seeds}")

    v410 = json.loads(args.v410_summary.read_text(encoding="utf-8"))
    if v410.get("artifact_type") != "object_event_v4_10_true_seed_fixed_fusion_robustness":
        raise RuntimeError("Unexpected v4.10 artifact")
    v413 = json.loads(args.v413_summary.read_text(encoding="utf-8"))
    if v413.get("artifact_type") != "object_event_v4_13_conservative_dual_head_fusion":
        raise RuntimeError("Unexpected v4.13 artifact")
    if not v413.get("passed"):
        raise RuntimeError("Seed-7 v4.13 development screen did not pass")
    if v413.get("config") != LOCKED:
        raise RuntimeError("Seed-7 v4.13 parameters differ from the locked parameters")

    checkpoints: dict[str, dict[str, object]] = {}
    for seed in seeds:
        path = args.v48_root / f"screen-seed-{seed}" / "best_gate_passing.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[str(seed)] = {
            "path": path.resolve().as_posix(),
            "artifact_type": payload.get("artifact_type"),
            "epoch": payload.get("epoch"),
        }

    result = {
        "status": "passed",
        "locked_fusion": LOCKED,
        "seeds": seeds,
        "v410_status": v410.get("status"),
        "v413_seed7_status": v413.get("status"),
        "v48_checkpoints": checkpoints,
        "scientific_contract": {
            "v413_parameters_are_not_retuned": True,
            "true_seed_probe_training_required": True,
            "median_probability_consensus_preregistered": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
