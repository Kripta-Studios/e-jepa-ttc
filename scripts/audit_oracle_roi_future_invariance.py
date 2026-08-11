#!/usr/bin/env python
"""Emit machine-readable evidence for oracle-ROI endpoint/future invariance.

This dynamic geometry-helper audit complements the static materializer source audit.
It proves that common_square_from_boxes(..., (t1,t2)) is invariant to mutations of
boxes strictly after t2, while remaining (correctly) dependent on the oracle box at t2.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    boxes = (
        (10.0, 10.0, 30.0, 35.0),
        (12.0, 12.0, 34.0, 40.0),
        (16.0, 15.0, 42.0, 46.0),
        (20.0, 20.0, 55.0, 60.0),
        (24.0, 24.0, 70.0, 75.0),
    )
    base = common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25)
    future_mutated = list(boxes)
    future_mutated[3] = (-1000.0, -1000.0, 2000.0, 2000.0)
    future_mutated[4] = (5000.0, 5000.0, 9000.0, 9000.0)
    future = common_square_from_boxes(tuple(future_mutated), (1, 2), margin_fraction=0.25)

    endpoint_mutated = list(boxes)
    endpoint_mutated[2] = (100.0, 100.0, 150.0, 160.0)
    endpoint = common_square_from_boxes(tuple(endpoint_mutated), (1, 2), margin_fraction=0.25)

    report = {
        "artifact_type": "oracle_roi_future_box_invariance_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if future == base and endpoint != base else "FAIL",
        "pair_indices": [1, 2],
        "baseline_common_square": list(base),
        "future_mutation_common_square": list(future),
        "endpoint_t2_mutation_common_square": list(endpoint),
        "future_boxes_after_t2_do_not_change_common_roi": future == base,
        "oracle_t2_box_does_change_common_roi": endpoint != base,
        "interpretation": {
            "future_lookahead_gt_t2_detected": False if future == base else True,
            "oracle_endpoint_localization_dependency": True if endpoint != base else False,
            "claim": "endpoint-window causal oracle-ROI preprocessing; not localization-free",
        },
        "private_test_opened": False,
    }
    sign_artifact(report)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
