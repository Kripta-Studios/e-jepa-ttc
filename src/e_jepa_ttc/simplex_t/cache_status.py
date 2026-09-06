"""Read-only operational state; receipts and live processes are distinct evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psutil


def context_cache_status(worktree: Path) -> dict[str, Any]:
    """Report actual extraction activity without scoring or asserting full QA."""
    root = worktree / "artifacts/simplex_t/T1"
    cache = root / "context_features_fp32"
    receipts = list(cache.glob("family*_query*.json"))
    observations = 0
    for path in receipts:
        if path.stat().st_size > 65536:
            raise ValueError("cache receipt exceeds metadata bound")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not path.with_suffix(".npz").is_file() or not 1 <= receipt["rows"] <= 16:
            raise ValueError("receipt lacks a valid-sized payload")
        observations += receipt["rows"]
    live = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            command = process.info["cmdline"] or []
            if (
                "python" in str(process.info["name"]).lower()
                and any(Path(arg).name == "build_simplex_t_context_features.py" for arg in command)
                and Path(process.info["cwd"]).resolve() == worktree.resolve()
            ):
                live.append(process.info["pid"])
        except (psutil.Error, TypeError, OSError):
            continue
    audits = []
    for path in root.glob("CONTEXT_CACHE_AUDIT_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "PARTIAL_TEMPORAL_CACHE_CONTENT_VERIFIED":
            audits.append((record["completed_query_blocks"], path.name, record["observations"]))
    latest = max(audits) if audits else None
    return {
        "status": (
            "LIVE_CONTEXT_EXTRACTION"
            if live
            else "RECEIPTS_COMPLETE_AWAITING_FINAL_QA"
            if len(receipts) == 24576
            else "RESUMABLE_PARTIAL_EXTRACTION"
            if receipts
            else "NOT_STARTED"
        ),
        "live_python_pids": sorted(live),
        "pid_note": "May include Python launcher and worker; not a count of separate jobs",
        "declared_completed_blocks": len(receipts),
        "declared_observations": observations,
        "required_D0_query_family_blocks": 24576,
        "payload_hashes_rechecked_by_status": False,
        "last_saved_content_audit": (
            {"blocks": latest[0], "file": latest[1], "observations": latest[2]} if latest else None
        ),
        "scientific_completion_proven": False,
    }
