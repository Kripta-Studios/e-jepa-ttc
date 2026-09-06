"""Bounded SIMPLEX-T T0 evidence; this module never trains or decodes media."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq

BASE_COMMIT = "57b39cb2b9a5ec8378755f8f350634c007aa822e"


def resource_headroom(
    *, available_ram: int, process_tree_rss: int, written_volume_free: list[int]
) -> dict[str, Any]:
    """Apply absolute byte floors; volume capacity never affects eligibility."""
    if min([available_ram, process_tree_rss, *written_volume_free]) < 0:
        raise ValueError("resource counters must be nonnegative")
    reasons = []
    if available_ram < 8 * 1024**3:
        reasons.append("HOST_AVAILABLE_BELOW_8_GIB")
    if process_tree_rss > 4 * 1024**3:
        reasons.append("PROCESS_TREE_RSS_ABOVE_4_GIB")
    if any(free < 60 * 1024**3 for free in written_volume_free):
        reasons.append("WRITTEN_VOLUME_FREE_BELOW_60_GIB")
    return {"has_headroom": not reasons, "reasons": reasons}


def sha256(path: Path) -> str:
    """Hash a bounded metadata/source file, never an implicit raw source."""
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"metadata hash exceeds 32 MiB bound: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_handoff(root: Path) -> dict[str, str]:
    """Check every listed payload and reject paths outside the handoff."""
    root = root.resolve()
    hashes: dict[str, str] = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or relative in hashes:
            raise ValueError(f"invalid handoff entry: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"handoff checksum mismatch: {relative}")
        hashes[relative] = actual
    if not hashes:
        raise ValueError("empty handoff")
    return hashes


def parquet_header(path: Path, *, identity_projection: bool) -> dict[str, Any]:
    """Read a TRAIN footer and optionally sequence IDs; never target columns."""
    if path.name != "train.parquet":
        raise ValueError("only explicit public TRAIN metadata is allowed at T0")
    if not path.is_file():
        return {"path": str(path), "status": "MISSING"}
    started = time.monotonic()
    with pq.ParquetFile(path) as source:
        result: dict[str, Any] = {
            "path": str(path),
            "status": "FOOTER_ONLY",
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "rows": source.metadata.num_rows,
            "schema": str(source.schema_arrow.remove_metadata()),
            "projected_columns": [],
            "target_values_read": False,
        }
        if identity_projection:
            counts: dict[str, int] = {}
            for batch in source.iter_batches(
                batch_size=4096, columns=["sequence_id"], use_threads=False
            ):
                for sequence in batch.column(0).to_pylist():
                    if not isinstance(sequence, str) or not sequence:
                        raise ValueError("invalid sequence identity")
                    counts[sequence] = counts.get(sequence, 0) + 1
            result.update(
                status="INPUT_IDENTITY_METADATA_ONLY",
                projected_columns=["sequence_id"],
                sequence_row_counts=dict(sorted(counts.items())),
            )
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def interface_status(paths: dict[str, Any]) -> dict[str, Any]:
    """Detect unresolved prerequisites without adopting guessed owner schemas."""
    shared = Path(paths["shared_coordination"])
    ack = shared / "SIMPLEX_T_STAGE70_ACK.json"
    stage_root = paths.get("stage70_worktree_read_only")
    discovered = {}
    if stage_root:
        campaign = Path(stage_root) / "artifacts/stage70_76_architecture"
        for name, relative in (
            ("role_manifest", "data_roles/DATA_ROLES.json"),
            ("time_charter", "time_charter/LABEL_TIME_CHARTER.json"),
        ):
            candidate = campaign / relative
            if candidate.is_file():
                discovered[name] = {"path": str(candidate), "sha256": sha256(candidate)}
    missing = []
    for name in ("role_manifest", "time_charter"):
        if not paths.get(name) and name not in discovered:
            missing.append(name)
    if not ack.is_file():
        missing.append("stage70_owner_acknowledgement")
    return {
        "execution_status": (
            "WAITING_SHARED_ROLE_MANIFEST" if missing else "INTERFACE_REVIEW_REQUIRED"
        ),
        "missing": missing,
        "discovered_owner_interfaces_not_yet_acknowledged": discovered,
        "acknowledgement_path": str(ack),
        "acknowledgement_exists": ack.is_file(),
        "acknowledgement_sha256": sha256(ack) if ack.is_file() else None,
        "authoritative_roles_adopted": False,
        "scientific_run_enabled": False,
        "reason": "Owner-authored roles/time/producer contracts require semantic integration.",
    }


def resources() -> dict[str, Any]:
    """Record host state without allocating a GPU or changing another process."""
    processes = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        if "python" in (process.info["name"] or "").lower():
            processes.append(process.info)
    drives = {}
    for drive in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(drive.mountpoint)
        except OSError:
            continue
        drives[drive.mountpoint] = {"total": usage.total, "free": usage.free}
    return {
        "host": socket.gethostname(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "processes": processes,
        "drives": drives,
        "gpu_allocation": "BUSY_UNACKNOWLEDGED",
        "heavy_io_allocation": "BUSY_UNACKNOWLEDGED",
        "leases_acquired": [],
    }


def audit(local_paths: Path) -> dict[str, Any]:
    """Collect bounded T0 evidence. No role-dependent rows or media are opened."""
    started = time.monotonic()
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    worktree = Path(paths["worktree"])
    handoff = Path(paths["handoff_root"])
    hashes = verify_handoff(handoff)
    eap = Path(paths["eap_root"])
    garl_input = Path(paths["garl_annotations_candidate"])
    garl = garl_input.parent.parent
    code = Path(paths["garl_code_candidate"])
    files = [eap / "README.md", garl / "README.md", code / "garl_ttc/datasets/ttc_dataset.py"]
    headers = {
        "FRAME": parquet_header(eap / "data/train.parquet", identity_projection=True),
        "TTC_PAIR_INPUT": parquet_header(garl_input, identity_projection=True),
        "TTC_PAIR_SUPERVISION": parquet_header(
            garl / "annotations/train.parquet", identity_projection=False
        ),
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    return {
        "artifact_type": "simplex_t_t0_audit_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": commit,
        "base_commit": BASE_COMMIT,
        "local_paths_sha256": sha256(local_paths),
        "resource_amendment_sha256": sha256(
            worktree / "configs/experiment/simplex_t_resource_amendment.json"
        ),
        "handoff_payload_hashes": hashes,
        "interfaces": interface_status(paths),
        "resources": resources(),
        "local_metadata": headers,
        "local_source_hashes": {str(p): sha256(p) if p.is_file() else None for p in files},
        "history_source_status": "HISTORY_SOURCE_PRIVILEGED_OR_UNRESOLVED",
        "object_timeline": None,
        "role_intersection": None,
        "history_coverage": None,
        "producer_ancestry_verified": False,
        "production_replay_parity": "NOT_RUN_PREREQUISITE_BLOCK",
        "scientific_fits": 0,
        "scientific_optimizer_updates": 0,
        "raw_forwards": 0,
        "holdout_opened": False,
        "new_architecture_scores_read": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Create new evidence without overwriting prior audit snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
