"""Archive selected receipts and bounded log tails; generate an evidence-only timing report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FINAL_STAGES = (
    "PAIR_E_MATCHED",
    "PAIR_R",
    "E_CTX_MATCHED",
    "R_CTX",
    "PHASE_ONLY_EVENT_S7_TRAIN40",
)
FIT_FILES = (
    "RUN_PROVENANCE.json",
    "CHECKPOINT_RECEIPT.json",
    "UPDATE_JOURNAL.json",
    "WORKER_BINDING.json",
    "RESOURCE_WAIT.json",
)


def source_paths() -> list[Path]:
    """Enumerate named evidence, never scan datasets, checkpoint trees or storage recursively."""
    paths = [
        Path("/workspace/ejepa_final_runs/final") / name
        for name in (
            "TRAINING_EXPORT_STATUS.json",
            "FINAL_RECEIPT.json",
            "MODEL_SEAL.json",
            "OOF_PREPARATION.json",
        )
    ]
    paths += [
        Path("/workspace/ejepa_final_operations") / name
        for name in (
            "FINAL_QUEUE_STATUS.json",
            "FINAL_QUEUE_WORKERS.json",
            "EXTERNAL_JOBS.json",
            "export_test12.log",
            "queue.log",
        )
    ]
    paths += [
        Path("/workspace/ejepa_test12_probe") / name
        for name in (
            "STATUS.json",
            "PLAN.json",
            "DOWNLOAD_RESERVATIONS.json",
        )
    ]
    paths += [
        Path("/workspace/ejepa_event_runs") / name
        for name in (
            "EVENT_VOLUME_SNAPSHOT.json",
            "EVENT_DOWNSTREAM_RESULT_R2.json",
            "EVENT_DOWNSTREAM_RESULT_R3.json",
            "EVENT_DOWNSTREAM_RESULT_R4.json",
        )
    ]
    fits = [Path("/workspace/ejepa_final_runs/final/fits") / stage for stage in FINAL_STAGES]
    fits += [
        Path(f"/workspace/ejepa_event_runs/rotations/{rotation}/fits") / stage
        for rotation in (2, 3, 4)
        for stage in ("E_A5", "E_C2F", "PAIR_E", "E_CTX")
    ]
    fits += [
        Path(
            f"/workspace/ejepa_rgb_handoff/extracted_v10/roots/r{rotation}_{kind}"
            f"/rotations/{rotation}/fits/{stage}"
        )
        for rotation in (2, 3, 4)
        for stage, kind in (
            ("R_A5", "a5"),
            ("R_C2F", "c2f"),
            ("PAIR_R", "downstream"),
            ("R_CTX", "downstream"),
        )
    ]
    for fit in fits:
        paths.extend(fit / name for name in FIT_FILES)
    paths.extend(
        Path("/workspace/ejepa_final_operations") / f"{stage}.log" for stage in FINAL_STAGES
    )
    return paths


def phase_row(source: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Keep active-call throughput distinct from the complete run's calendar span."""
    start, end = value.get("start_time"), value.get("end_time")
    span = None
    if start and end:
        try:
            span = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        except (TypeError, ValueError):
            pass
    return {
        "source": str(source),
        "run_name": value.get("run_name"),
        "status": value.get("status"),
        "started_utc_reported": start,
        "ended_utc_reported": end,
        "calendar_span_seconds_including_pauses": span,
        "last_call_updates": value.get("updates_last_call"),
        "last_call_wall_seconds": value.get("elapsed_seconds_last_call"),
        "last_call_updates_per_second": value.get("updates_per_second_last_call"),
        "python_version": value.get("python_version"),
        "torch_version": value.get("torch_version"),
        "cuda_version": value.get("cuda_version"),
        "gpu_name": value.get("gpu_name"),
        "host": value.get("host"),
        "identity_sha256": value.get("identity_sha256"),
    }


def collect(output: Path, paths: list[Path]) -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = output / f"history_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    phases = []
    for ordinal, source in enumerate(paths):
        entry: dict[str, Any] = {"source": str(source)}
        try:
            before = source.stat()
            entry.update(
                source_bytes=before.st_size,
                mtime_utc=datetime.fromtimestamp(before.st_mtime, UTC).isoformat(),
            )
            tail = source.suffix == ".log"
            if not tail and before.st_size > 2 * 1024**2:
                entry["status"] = "SKIPPED_EXCEEDS_2_MIB"
            else:
                with source.open("rb") as stream:
                    if tail:
                        stream.seek(max(0, before.st_size - 64 * 1024))
                    data = stream.read(2 * 1024**2 + 1)
                after = source.stat()
                archived = f"{ordinal:03d}_{source.name}"
                (destination / archived).write_bytes(data)
                entry.update(
                    status="ARCHIVED",
                    archived=archived,
                    archived_bytes=len(data),
                    archived_sha256=hashlib.sha256(data).hexdigest(),
                    coverage="LAST_64_KIB" if tail else "COMPLETE_FILE",
                    changed_during_read=(before.st_mtime_ns, before.st_size)
                    != (after.st_mtime_ns, after.st_size),
                )
                if source.name == "RUN_PROVENANCE.json":
                    phases.append(phase_row(source, json.loads(data)))
        except (OSError, ValueError) as error:
            entry.update(status="UNAVAILABLE", error=type(error).__name__)
        entries.append(entry)
    summary = {
        "schema": "ejepa_final_diagnostic_evidence_v1",
        "captured_utc": datetime.now(UTC).isoformat(),
        "capture_platform": platform.platform(),
        "archive_root": str(destination),
        "entries": entries,
        "phase_timings": phases,
        "collection_optimizer_updates": 0,
        "collection_inferences": 0,
        "evidence_limits": {
            "official_score": "Not established by these local/cloud training receipts.",
            "sota": "Not established; requires comparable official evaluation.",
            "kernel_latency": "Not measured by passive sampling or wall time.",
            "total_active_training_hours": "Not reconstructible from provenance spans alone.",
            "cost_usd": "Not measured; provider billing required, not inferred from GPU occupancy.",
            "transfer_time": "Requires matched start/end transfer receipts.",
            "data_loading_time": "Not separable from wall time without existing profiler evidence.",
            "oof": "Lineage checks are structural; TRAIN40 refit is not blind validation.",
        },
    }
    (destination / "EVIDENCE_MANIFEST.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Final refit diagnostic evidence",
        "",
        f"Captured UTC: {summary['captured_utc']}. No training or inference was repeated.",
        "",
        "The tables retain reported timestamps and last-call measurements. Calendar spans include "
        "pauses and host downtime; last-call rates cover only that invocation. "
        "Neither is GPU kernel latency.",
        "",
        "| Phase | Status | Last call updates | Last call wall seconds | Calendar span seconds |",
        "|---|---|---:|---:|---:|",
    ]
    for phase in phases:
        lines.append(
            f"| {phase['run_name']} | {phase['status']} | {phase['last_call_updates']} | "
            f"{phase['last_call_wall_seconds']} | "
            f"{phase['calendar_span_seconds_including_pauses']} |"
        )
    lines.extend(["", "## Claims requiring separate evidence", ""])
    lines.extend(f"- **{key}**: {value}" for key, value in summary["evidence_limits"].items())
    lines.extend(
        [
            "",
            "Raw receipts and bounded log tails are archived with SHA-256. Missing files and "
            "files changing during capture remain explicitly marked in EVIDENCE_MANIFEST.json. "
            "Receipt mtimes are inventory metadata, never assumed to be phase start/end times.",
            "",
        ]
    )
    (destination / "DIAGNOSTIC_EVIDENCE_REPORT.md").write_text("\n".join(lines))
    return {
        "archive_root": str(destination),
        "captured_utc": summary["captured_utc"],
        "archived_files": sum(entry["status"] == "ARCHIVED" for entry in entries),
        "archived_bytes": sum(entry.get("archived_bytes", 0) for entry in entries),
        "phase_rows": len(phases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.output, source_paths())))


if __name__ == "__main__":
    main()
