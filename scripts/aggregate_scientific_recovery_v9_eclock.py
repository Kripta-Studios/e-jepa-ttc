#!/usr/bin/env python
"""Aggregate only complete, identity-verified 8,192-row E-Clock OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.collision_clock_protocol import production_sequence_macro_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path, required=True)
    parser.add_argument("--required-sequences", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    hashes = json.loads(args.expected_hashes.read_text(encoding="utf-8"))
    sequences = json.loads(args.required_sequences.read_text(encoding="utf-8"))
    metrics = production_sequence_macro_metrics(
        frame,
        expected_hashes=hashes,
        required_sequences=sequences,
    )
    payload = sign_artifact(
        {
            "artifact_type": "eclock_x0_aggregate_v1",
            "arm_id": args.arm_id,
            "evidence_class": "scientific_oof",
            "scientific_result": True,
            "upstream_roi_is_box_conditioned": True,
            "explicit_foreground_height_interface_bypassed": args.arm_id
            in {"X0-BASE-U", "X0-DYN-U"},
            "loss_reduction": "mean_smooth_l1_benchmark_phase_error",
            "metrics": metrics,
        }
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
