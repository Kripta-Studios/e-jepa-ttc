"""Verify the fixed H16 reference cohort without acquiring GPU or reading targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.h16_qa_plan import bind_h16_reference_receipts
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative outstanding reservations required")
    if args.report is not None and (
        args.report.exists()
        or not args.report.resolve().is_relative_to(work / "artifacts/simplex_t")
    ):
        raise ValueError("new companion-local reference report required")

    def resources() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + 1_048_576,
        )

    if not resources():
        print(json.dumps({"status": "PAUSED_RESOURCE", "gpu_replay_performed": False}))
        return 3
    result = bind_h16_reference_receipts(work)
    if not resources():
        print(json.dumps({"status": "PAUSED_RESOURCE", "gpu_replay_performed": False}))
        return 3
    if args.report is not None:
        write_new_json(args.report, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verified_references": len(result["references"]),
                "missing": [
                    {"family": row["family"], "query": row["query"]} for row in result["missing"]
                ],
                "report_sha256": sha256(args.report) if args.report is not None else None,
                "gpu_replay_performed": False,
                "optimizer_updates": 0,
                "scientific_admission": False,
            }
        )
    )
    return 10 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
