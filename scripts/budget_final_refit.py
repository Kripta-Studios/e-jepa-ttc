"""Regenerate scoped training estimates from actual historical runtime receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_bound(path: Path, bindings: dict[str, str]) -> dict[str, Any]:
    """Read a receipt and preserve the exact bytes used in the calculation."""
    raw = path.read_bytes()
    bindings[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def budget(workspace: Path, rotation_plan: Path) -> dict[str, Any]:
    """Estimate shared final producers without treating historical rates as guarantees."""
    bindings: dict[str, str] = {}
    rotations = read_bound(rotation_plan, bindings)
    original = workspace / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008/fits"
    result: dict[str, Any] = {}
    for fit in ("R_A5", "R_C2F"):
        overlap = read_bound(original / fit / "RGB_OVERLAP_RUNTIME.json", bindings)
        provenance = read_bound(original / fit / "RUN_PROVENANCE.json", bindings)
        measured_updates = overlap["completed_updates"] - overlap["origin"]["completed_updates"]
        if (
            measured_updates <= 0
            or overlap["prefetch"]["consumed_batches"] != measured_updates
            or overlap["checkpoint_sha256"] != provenance["checkpoint_sha256"]
            or overlap["objective_changed"] is not False
        ):
            raise ValueError(f"Inconsistent historical runtime receipt: {fit}")
        overlap_seconds = float(overlap["elapsed_s_since_restore"])
        interval_seconds = (
            datetime.fromisoformat(provenance["end_time"])
            - datetime.fromisoformat(provenance["start_time"])
        ).total_seconds()
        final_updates = 18 * ((88_744 + 31) // 32)
        auxiliary_updates = sum(item["updates"][fit] for item in rotations["rotations"][1:])
        optimized_rate = overlap_seconds / measured_updates
        interval_rate = interval_seconds / provenance["completed_updates"]
        result[fit] = {
            "final_updates": final_updates,
            "auxiliary_updates_if_rotation0_reused": auxiliary_updates,
            "historical_optimized_scope": {
                "updates": measured_updates,
                "seconds": overlap_seconds,
                "seconds_per_update": optimized_rate,
                "scope": "saved overlap-runtime interval since checkpoint restore",
            },
            "historical_provenance_scope": {
                "updates": provenance["completed_updates"],
                "seconds": interval_seconds,
                "seconds_per_update": interval_rate,
                "scope": "whole historical start/end interval; may include pauses",
            },
            "final_training_hours_from_optimized_interval": final_updates * optimized_rate / 3600,
            "final_training_hours_from_provenance_interval": final_updates * interval_rate / 3600,
            "all_new_rgb_hours_from_optimized_interval": (
                (auxiliary_updates + final_updates) * optimized_rate / 3600
            ),
            "all_new_rgb_hours_from_provenance_interval": (
                (auxiliary_updates + final_updates) * interval_rate / 3600
            ),
        }
    return {
        "schema": "final_refit_scoped_budget_v1",
        "status": "SHARED_PRODUCER_ESTIMATE_FULL_OOF_EXTRACTION_UNPRICED",
        "scientific_updates_executed_by_budget_script": 0,
        "authorized_immediate_training_scope": "two final TRAIN40 RGB producers only",
        "producer_updates": sum(item["final_updates"] for item in result.values()),
        "producers": result,
        "final_rgb_training_hours_from_optimized_interval": sum(
            item["final_training_hours_from_optimized_interval"] for item in result.values()
        ),
        "final_rgb_training_hours_from_provenance_interval": sum(
            item["final_training_hours_from_provenance_interval"] for item in result.values()
        ),
        "not_a_guaranteed_completion_time": True,
        "not_a_full_refit_budget": True,
        "excludes": [
            "H/V RGB cache preparation",
            "feature and history extraction",
            "all event producer fits for new OOF rotations",
            "PAIR/context/final readout fitting",
            "technical admission and recovery",
            "export and final Test12 inference",
        ],
        "shared_cache_policy": "one lossless input cache; no duplication per fold",
        "scientific_endpoint_policy": "fixed 18-epoch endpoints, no score-based selection",
        "source_bindings": bindings,
    }


def main() -> None:
    """Write a reproducible planning artifact; never launch jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--rotation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = budget(args.workspace.resolve(), args.rotation_plan.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "producer_updates",
                    "final_rgb_training_hours_from_optimized_interval",
                    "final_rgb_training_hours_from_provenance_interval",
                    "not_a_full_refit_budget",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
