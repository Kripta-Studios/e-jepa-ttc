"""Audit real RAW clock support and build a new, label-free causal pilot manifest.

This command reads timestamp metadata, calibration and causal bounding boxes.
It does not read RGB pixels, event coordinates, TTC targets or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import h5py  # noqa: E402

from operational.evttc_transfer.inputs import _canonical_sha256  # noqa: E402
from operational.train40_runtime.replay_inputs import generate_sensor_clock_replay  # noqa: E402


def sha256(path: Path) -> str:
    """Hash a small evidence or source file, never the complete RAW HDF5."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Create one evidence file without replacing existing bytes."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def stat_identity(path: Path) -> dict[str, int]:
    """Return the RAW identity already used by the preserved source manifest."""
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def audit_inventory(base: dict[str, Any]) -> dict[str, Any]:
    """Read the actual first/last timestamps of every registered physical source."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in base["rows"]:
        groups.setdefault(row["physical_sequence_id"], []).append(row)
    inventory = []
    for sequence, population in sorted(groups.items()):
        population = sorted(population, key=lambda row: int(row["anchor_us"]))
        first = population[0]
        path = Path(first["raw_path"]).resolve(strict=True)
        before = stat_identity(path)
        if any(
            Path(row["raw_path"]).resolve() != path or row["raw_stat"] != before
            for row in population
        ):
            raise ValueError(f"{sequence}: RAW identity differs from the frozen manifest")
        with h5py.File(path, "r") as handle:
            events = handle["prophesee/event_cam_left/t"]
            rgb = handle["blackflys/left/ts"]
            if not isinstance(events, h5py.Dataset) or not isinstance(rgb, h5py.Dataset):
                raise ValueError(f"{sequence}: timestamps must be datasets")
            if not len(events) or not len(rgb):
                raise ValueError(f"{sequence}: sensor timestamps must be nonempty")
            event_range = [int(events[0]), int(events[-1])]
            rgb_range = [int(rgb[0]), int(rgb[-1])]
            event_count, rgb_count = len(events), len(rgb)
        if stat_identity(path) != before:
            raise ValueError(f"{sequence}: RAW changed while reading metadata")
        inventory.append(
            {
                "physical_sequence_id": sequence,
                "raw_path": str(path),
                "raw_stat": before,
                "event_timestamp_count": event_count,
                "event_sensor_range_us": event_range,
                "event_duration_s": (event_range[1] - event_range[0]) / 1e6,
                "rgb_sensor_range_us": rgb_range,
                "rgb_frames": rgb_count,
                "base_query_count": len(population),
                "base_anchor_range_us": [
                    int(population[0]["anchor_us"]),
                    int(population[-1]["anchor_us"]),
                ],
                "base_anchor_duration_s": (
                    int(population[-1]["anchor_us"]) - int(population[0]["anchor_us"])
                )
                / 1e6,
                "raw_stat_preserved": True,
            }
        )
    durations = [row["event_duration_s"] for row in inventory]
    return {
        "schema": "train40_raw_clock_support_audit_v1",
        "scope": "HDF5_TIMESTAMP_ENDPOINTS_NOT_A_LATENCY_OR_COVERAGE_MEASUREMENT",
        "physical_sequence_count": len(inventory),
        "base_query_count": len(base["rows"]),
        "minimum_event_duration_s": min(durations),
        "maximum_event_duration_s": max(durations),
        "any_raw_sequence_at_least_60s": any(value >= 60 for value in durations),
        "sixty_second_natural_replay_demonstrated": False,
        "endpoint_only_cannot_establish_annotation_or_event_continuity": True,
        "raw_integrity_scope": "registered size/mtime checked before and after metadata read",
        "full_raw_sha256_computed": False,
        "rgb_pixels_read": False,
        "event_coordinates_or_polarities_read": False,
        "ttc_targets_read": False,
        "inventory": inventory,
    }


def main() -> None:
    """Write reproducible metadata evidence and a canonical causal pilot cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    parser.add_argument("--sequence", default="CPLA-low")
    parser.add_argument("--updates", type=int, default=250)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    base_path = args.base_manifest.resolve(strict=True)
    output = args.output.resolve()
    if output == base_path.parent or base_path.parent in output.parents:
        parser.error("--output must be outside the preserved base-manifest directory")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    base_hash = sha256(base_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    audit = audit_inventory(base)
    write_json(output / "raw_sequence_inventory.json", audit)
    selected = [row for row in base["rows"] if row["physical_sequence_id"] == args.sequence]
    if not selected:
        raise ValueError("Selected sequence is absent from the frozen manifest")
    first = min(int(row["anchor_us"]) for row in selected)
    last = first + (args.updates - 1) * 20_000
    configuration = {
        "sequence_ids": [args.sequence],
        "bounds_us": {args.sequence: [first, last]},
        "max_updates_per_sequence": args.updates,
        "step_us": 20_000,
        "annotation_delay_us": 0,
    }
    manifest = generate_sensor_clock_replay(
        base,
        sequence_ids=[args.sequence],
        bounds_us={args.sequence: (first, last)},
        max_updates_per_sequence=args.updates,
        step_us=20_000,
        annotation_delay_us=0,
    )
    checks = {
        "scheduled_count_exact": manifest["scheduled_updates"] == args.updates,
        "accepted_count_exact": manifest["accepted_updates"] == args.updates,
        "no_excluded_slots": manifest["excluded"] == [],
        "unchanged_base_bytes": sha256(base_path) == base_hash,
        "base_canonical_binding_exact": manifest["provenance"]["base_manifest_canonical_sha256"]
        == _canonical_sha256(base),
        "fixed_20ms_grid": [row["anchor_us"] for row in manifest["rows"]]
        == list(range(first, last + 1, 20_000)),
        "full_620ms_initial_sensor_support": all(
            first - 620_000 >= row["event_sensor_range_us"][0] for row in manifest["inventory"]
        ),
        "ttc_targets_not_read": manifest["targets_read"] is False,
        "no_optimizer_updates": manifest["optimizer_updates"] == 0,
    }
    manifest_path = output / "causal_pilot_manifest.json"
    write_json(manifest_path, manifest)
    sources = [
        Path(__file__).resolve(),
        ROOT / "operational/train40_runtime/replay_inputs.py",
        ROOT / "operational/rgb_port_revision/transfer_inputs.py",
        ROOT / "operational/evttc_transfer/inputs.py",
    ]
    receipt = {
        "schema": "train40_causal_pilot_generation_receipt_v1",
        "status": "COMPLETE" if all(checks.values()) else "FAILED",
        "started_utc": started,
        "ended_utc": datetime.now(UTC).isoformat(),
        "command_argv": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "host": platform.node(),
        "base_manifest": str(base_path),
        "base_manifest_sha256": base_hash,
        "base_manifest_canonical_sha256": _canonical_sha256(base),
        "configuration": configuration,
        "configuration_hash": _canonical_sha256(configuration),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "output_sha256": {
            "raw_sequence_inventory.json": sha256(output / "raw_sequence_inventory.json"),
            "causal_pilot_manifest.json": sha256(manifest_path),
        },
        "checks": checks,
        "offered_duration_s_if_replayed_at_50hz": args.updates / 50,
        "sensor_anchor_span_s": (last - first) / 1e6,
        "latency_measured": False,
        "numerical_ttc_parity_measured": False,
        "gpu_used": False,
        "weights_or_normalizers_loaded": False,
        "targets_read": False,
        "optimizer_updates": 0,
    }
    write_json(output / "manifest_generation_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "checks": checks}))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
