"""Inspect P3 prerequisites without loading Torch, models, targets or raw events."""

# External JSON and process records are checked at the admission boundary.
# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def metadata(path: Path, expected: str | None = None) -> dict[str, Any]:
    """Read bounded JSON metadata, verifying a supplied immutable binding."""
    if path.stat().st_size > 1_048_576:
        raise ValueError(f"metadata exceeds 1 MiB: {path}")
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"metadata binding changed: {path}")
    return json.loads(data)


def file_presence(path: Path) -> dict[str, Any]:
    """Stat a referenced source; do not read or claim validation of its payload."""
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "payload_hash_reverified": False,
    }


def verify_small_checkpoints(checkpoints: list[dict[str, Any]]) -> None:
    """Hash only the nine referenced small weights, within a 32 MiB light-I/O cap."""
    if sum(row["bytes"] or 0 for row in checkpoints) > 32 * 1024**2:
        raise ValueError("weight verification exceeds light-I/O cap; exclusive slot required")
    for row in checkpoints:
        if not row["exists"]:
            continue
        actual = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        if actual != row["registered_sha256"]:
            raise ValueError("referenced frozen producer weights changed")
        row["payload_hash_reverified"] = True


def selected_queries(
    selected: list[dict[str, Any]],
    pools: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    """Resolve each fixed token to a unique fold0 TRAIN producer and supplied ROI."""
    if len(selected) != 64 or len({q["sample_token"] for q in selected}) != 64:
        raise ValueError("exactly 64 unique registered TRAIN tokens required")
    positions: dict[str, list[tuple[str, dict, dict, int]]] = {}
    for label, manifest, arrays in pools:
        for i, token in enumerate(arrays["tokens"]):
            positions.setdefault(str(token), []).append((label, manifest, arrays, i))
    rows = []
    for query in selected:
        matches = positions.get(query["sample_token"], [])
        if len(matches) != 1:
            raise ValueError("selected token absent or ambiguous in authorized indexes")
        label, manifest, arrays, i = matches[0]
        family_id = int(arrays["producer_family"][0, i])
        if family_id not in (0, 1, 2):
            raise ValueError("selected query does not belong to fold0 TRAIN")
        family = manifest["families"][family_id]
        if family_id not in (0, 1, 2) or family["role"] != f"inner{family_id}":
            raise ValueError("selected query does not belong to fold0 TRAIN")
        sequence = str(arrays["sequences"][i])
        if sequence != query["sequence_id"] or not arrays["valid"][i, -1]:
            raise ValueError("selected sequence or current availability differs")
        rows.append(
            {
                "sample_token": query["sample_token"],
                "sequence_id": sequence,
                "pool": label,
                "index_row": i,
                "family_id": family_id,
                "experts": family["experts"],
                "valid_history_slots": int(arrays["valid"][i].sum()),
                "availability_scope": "retrospective context conditioned on supplied current ROI",
            }
        )
    return rows


def live_gpu() -> dict[str, Any]:
    """Observe GPU/process metadata once; an observation does not acquire a lease."""
    commands = {
        "gpu_csv": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv",
        ],
        "process_csv": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv",
        ],
    }
    observed: dict[str, Any] = {}
    try:
        for key, command in commands.items():
            observed[key] = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
                check=True,
            ).stdout
        pids = {
            int(row["pid"].strip())
            for row in csv.DictReader(
                io.StringIO(observed["process_csv"]),
                skipinitialspace=True,
            )
        }
        jobs = []
        for pid in sorted(pids):
            try:
                process = psutil.Process(pid)
                if "python" not in process.name().lower():
                    continue
                command = process.cmdline()
                jobs.append(
                    {
                        "pid": pid,
                        "created_unix": process.create_time(),
                        "command": command,
                        "rss_bytes": process.memory_info().rss,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                jobs.append({"pid": pid, "identity_unresolved": True})
        observed.update(status="OBSERVED", live_python_gpu_processes=jobs)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        observed.update(status="UNKNOWN", error=str(error), live_python_gpu_processes=[])
    return observed


def blockers(slot: Any, gpu: dict, sources_present: bool) -> list[str]:
    """Keep missing authority, live contention and missing sources distinct."""
    reasons = []
    if not slot:
        reasons.append("MISSING_EXPLICIT_EXCLUSIVE_GPU_HEAVY_IO_SLOT")
    else:
        reasons.append("SLOT_REQUIRES_OWNER_SCOPE_EXPIRY_AND_LIVE_VALIDATION")
    if gpu.get("status") != "OBSERVED":
        reasons.append("GPU_OBSERVATION_UNAVAILABLE")
    elif gpu.get("live_python_gpu_processes"):
        reasons.append("OTHER_LIVE_PYTHON_GPU_JOB")
    if not sources_present:
        reasons.append("REFERENCED_SOURCE_FILE_MISSING")
    return reasons


def inspect(root: Path) -> dict[str, Any]:
    """Bind known manifests and inventory only sources needed by the fixed sample."""
    root = root.resolve(strict=True)
    protocol = metadata(root / "artifacts/simplex_t/h16_replication_20261003/PROTOCOL.json")
    launch = protocol["launch"]
    local = metadata(Path(launch["local_paths"]))
    ack_path = Path(local["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack = metadata(ack_path)
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = metadata(Path(ancestry_ref["path"]), ancestry_ref["sha256"])
    configuration = metadata(
        Path(launch["source_configuration"]),
        launch["source_configuration_sha256"],
    )
    selected_path = root / "artifacts/simplex_t/nocturnal_20261003/profiling/SELECTION.json"
    post = metadata(root / "artifacts/simplex_t/post_campaign_20261003/PROTOCOL.json")
    selection = metadata(selected_path, post["selection_sha256"])
    pools = []
    bindings = []
    for label, relative in (
        ("D0", configuration["original"]["index_root"]["relative_path"]),
        ("D1", configuration["expansion"]["0"]["index_manifest"]["relative_path"]),
    ):
        directory = root / relative
        if directory.suffix == ".json":
            directory = directory.parent
        manifest = metadata(directory / "INDEX_MANIFEST.json")
        path = directory / "query_context_index.npz"
        if path.stat().st_size > 8_388_608:
            raise ValueError("index exceeds light metadata inspection budget")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != manifest["index_sha256"]:
            raise ValueError("query index binding changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                key: archive[key] for key in ("tokens", "sequences", "producer_family", "valid")
            }
        pools.append((label, manifest, arrays))
        bindings.append({"pool": label, "path": str(path), "sha256": sha})
    queries = selected_queries(selection["queries"], pools)
    producers = {(r["outer_fold"], r["role"], r["expert"]): r for r in ancestry["producers"]}
    allowed_sequences = set(configuration["original_sequences"]) | set(
        configuration["expansion_sequences"],
    )
    for query in queries:
        if query["sequence_id"] not in allowed_sequences:
            raise ValueError("selected query outside registered D1 population")
        for expert, sha in query["experts"].items():
            if producers[0, f"inner{query['family_id']}", expert]["checkpoint_sha256"] != sha:
                raise ValueError("selected index changes the acknowledged producer family")
    source_paths = {r["sha256"]: Path(r["path"]) for r in ancestry["input_bindings"].values()}
    hashes = sorted({sha for q in queries for sha in q["experts"].values()})
    checkpoints = [{**file_presence(source_paths[sha]), "registered_sha256": sha} for sha in hashes]
    verify_small_checkpoints(checkpoints)
    raw_root = (Path(local["eap_root"]) / "data/train").resolve(strict=True)
    raw = []
    for sequence in sorted({q["sequence_id"] for q in queries}):
        path = (raw_root / sequence / "events.h5").resolve()
        if not path.is_relative_to(raw_root):
            raise ValueError("raw reference leaves authorized TRAIN root")
        raw.append(file_presence(path))
    gpu = live_gpu()
    slot = ack["resources"].get("exclusive_gpu_and_heavy_io_slot")
    reasons = blockers(slot, gpu, all(r["exists"] for r in checkpoints + raw))
    return {
        "schema": "simplex_t_remaining_route_readiness_v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "BLOCKED",
        "blockers": reasons,
        "execution_admitted": False,
        "ack_path": str(ack_path),
        "ack_sha256": hashlib.sha256(ack_path.read_bytes()).hexdigest(),
        "ack_slot": slot,
        "local_gpu_flag": local.get("gpu_job_authorized_now"),
        "source_metadata_bound": True,
        "query_indexes": bindings,
        "selection_sha256": post["selection_sha256"],
        "queries": queries,
        "checkpoints": checkpoints,
        "raw_references": raw,
        "gpu": gpu,
        "payload_integrity": (
            "nine small producer weights rehashed; raw presence only, no event payload read"
        ),
        "remaining_implementation": (
            "canonical full/reduced route adapter integration and actual parity"
        ),
        "remaining_measurements": (
            "prospectively frozen warm/application-cold whole-route repetitions"
        ),
        "new_optimizer_updates": 0,
        "models_loaded": 0,
        "raw_payload_reads": 0,
        "frozen_deliveries_modified": False,
        "other_processes_modified": False,
    }


def main() -> int:
    """Write one independent durable inspection receipt, with no execution option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="new receipt; refuses overwrite")
    args = parser.parse_args()
    result = inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "blockers": result["blockers"],
                "queries": len(result["queries"]),
                "weights": len(result["checkpoints"]),
                "raw_files": len(result["raw_references"]),
                "receipt": str(args.output),
            }
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
