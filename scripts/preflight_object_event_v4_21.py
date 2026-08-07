#!/usr/bin/env python3
"""Preflight for v4.21 box-pseudoflow oracle target audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--v420-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.cache_manifest, args.v419_summary, args.v420_summary, args.ensemble_train, args.ensemble_validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    v419 = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    if v419.get("decision", {}).get("recommendation") != "dense_correspondence_supported_train_box_pseudoflow_decoder":
        raise RuntimeError("v4.19 dense correspondence support is missing")
    v420 = json.loads(args.v420_summary.read_text(encoding="utf-8"))
    recommendation = v420.get("decision", {}).get("recommendation")
    if recommendation != "frozen_refiner_insufficient_move_pseudoflow_divergence_supervision_into_encoder":
        raise RuntimeError(f"Unexpected v4.20 recommendation: {recommendation}")
    print(json.dumps({
        "status": "passed",
        "v420_recommendation": recommendation,
        "scientific_contract": {
            "audit_target_before_encoder_unfreeze": True,
            "no_model_training": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
