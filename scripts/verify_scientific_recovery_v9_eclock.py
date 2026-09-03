#!/usr/bin/env python
"""Verify E-Clock protocol and signed JSON artifacts without trusting summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_gates import validate_x0_claim_scope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    schema_by_type = {
        "scientific_recovery_v9_eclock_protocol_v2": "protocol",
        "eclock_x0_reference_v2": "reference",
        "eclock_x0_dyn_w_not_executed_v2": "dyn_w_not_executed",
        "eclock_x0_checkpoint_manifest_v2": "checkpoint_manifest",
        "eclock_x0_resume_decision_v2": "resume_decision",
        "eclock_x0_fold_summary_v2": "fold_summary",
        "eclock_x0_bootstrap_v2": "bootstrap_artifact",
        "eclock_x0_aggregate_v2": "aggregate",
    }
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(payload):
            raise ValueError(f"artifact signature mismatch: {path}")
        schema_name = schema_by_type.get(payload.get("artifact_type"))
        if schema_name is not None:
            schema_path = (
                repo / f"schemas/scientific_recovery_v9_eclock_{schema_name}_v2.schema.json"
            )
            jsonschema.Draft202012Validator(
                json.loads(schema_path.read_text(encoding="utf-8"))
            ).validate(payload)
        if "upstream_roi_is_box_conditioned" in payload and "arm_id" in payload:
            validate_x0_claim_scope(payload, arm_id=str(payload["arm_id"]))
        print(f"VERIFIED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
