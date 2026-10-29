"""Immutable authorization for concurrent A5/C2F execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    identity_is_live,
    read_bytes_shared,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port_pipeline_v2.contracts import validate_pipeline_freeze
from operational.rgb_port_revision.migration import source_matches

SCHEMA = "rgb_port_concurrent_freeze_v1"
FIT_IDS = ("E_A5_MATCHED", "E_C2F_MATCHED")
SOURCE_FILES = ("__init__.py", "contracts.py", "queue.py", "producer.py")


def _canonical(value: Mapping[str, Any]) -> str:
    payload = {str(key): item for key, item in value.items() if key != "identity_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fit_origin(run: Path, fit_id: str) -> dict[str, Any]:
    fit = run / "fits" / fit_id
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    version = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if (
        not version.is_file()
        or sha256_file(version) != pointer.get("checkpoint_sha256")
        or int(pointer.get("completed_updates", -1)) < 0
    ):
        raise RuntimeError(f"Concurrent origin is not a full checkpoint: {fit_id}")
    target = run / "concurrent/origins" / f"{fit_id}_{int(pointer['completed_updates']):06d}.pt"
    if not target.is_file():
        atomic_write_bytes(target, read_bytes_shared(version))
    if sha256_file(target) != pointer["checkpoint_sha256"]:
        raise RuntimeError(f"Concurrent immutable origin changed: {fit_id}")
    return {
        "fit_id": fit_id,
        "completed_updates": int(pointer["completed_updates"]),
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "identity_sha256": pointer["identity_sha256"],
        "checkpoint_path": str(target.resolve(strict=True)),
    }


def build_concurrent_freeze(
    run: Path,
    *,
    c2f_child: Mapping[str, Any],
    output: Path | None = None,
) -> dict[str, Any]:
    """Bind the current checkpoints and the already-running C2F child."""
    run = run.resolve(strict=True)
    pipeline_path = run / "PIPELINE_FREEZE.json"
    acceleration_path = run / "ACCELERATION_FREEZE.json"
    pipeline = validate_pipeline_freeze(pipeline_path)
    source_root = Path(__file__).resolve().parent
    sources = {
        str((source_root / name).resolve(strict=True)): sha256_file(source_root / name)
        for name in SOURCE_FILES
    }
    origins = {fit_id: _fit_origin(run, fit_id) for fit_id in FIT_IDS}
    child = {"pid": int(c2f_child["pid"]), "create_time": float(c2f_child["create_time"])}
    if not identity_is_live(child):
        raise RuntimeError("C2F adoption child is not live with the recorded PID creation time")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "allowed_fit_ids": list(FIT_IDS),
        "max_concurrent_heavy": 2,
        "all_other_heavy_policy": "WAIT_UNTIL_BOTH_ALLOWED_FITS_TERMINAL",
        "pipeline_freeze_path": str(pipeline_path.resolve(strict=True)),
        "pipeline_freeze_sha256": sha256_file(pipeline_path),
        "pipeline_freeze_identity_sha256": pipeline["identity_sha256"],
        "acceleration_freeze_path": str(acceleration_path.resolve(strict=True)),
        "acceleration_freeze_sha256": sha256_file(acceleration_path),
        "origins": origins,
        "adopted_c2f_child": child,
        "cache_policy": {
            "event_fit_ids": list(FIT_IDS),
            "per_fit_max_bytes": {
                "E_A5_MATCHED": 512 * 1024**2,
                "E_C2F_MATCHED": 1024**3,
            },
            "graphs": {"E_A5_MATCHED": "V1_ONLY", "E_C2F_MATCHED": "DISABLED"},
        },
        "source_sha256": sources,
        "scientific_arms_added": 0,
    }
    payload["identity_sha256"] = _canonical(payload)
    target = output.resolve() if output else run / "CONCURRENT_FREEZE.json"
    if target.exists() and read_json_shared(target) != payload:
        raise RuntimeError("Concurrent freeze changed")
    if not target.exists():
        atomic_write_json(target, payload)
    return payload


def validate_concurrent_freeze(path: Path) -> dict[str, Any]:
    """Validate the additive concurrency authorization and all frozen bytes."""
    value = read_json_shared(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("allowed_fit_ids") != list(FIT_IDS)
        or value.get("max_concurrent_heavy") != 2
        or value.get("scientific_arms_added") != 0
        or value.get("identity_sha256") != _canonical(value)
    ):
        raise ValueError("Concurrent freeze differs from its authorized scope")
    pipeline_path = Path(value["pipeline_freeze_path"])
    pipeline = validate_pipeline_freeze(pipeline_path)
    if (
        sha256_file(pipeline_path) != value["pipeline_freeze_sha256"]
        or pipeline["identity_sha256"] != value["pipeline_freeze_identity_sha256"]
        or sha256_file(Path(value["acceleration_freeze_path"]))
        != value["acceleration_freeze_sha256"]
    ):
        raise RuntimeError("Concurrent parent freeze changed")
    expected_names = set(SOURCE_FILES)
    sources = value.get("source_sha256", {})
    if {Path(item).name for item in sources} != expected_names:
        raise ValueError("Concurrent source inventory differs")
    for source, digest in sources.items():
        if not source_matches(Path(source), digest, path.parent):
            raise RuntimeError(f"Concurrent source changed: {source}")
    cache = value.get("cache_policy", {})
    if (
        cache.get("event_fit_ids") != list(FIT_IDS)
        or cache.get("per_fit_max_bytes")
        != {"E_A5_MATCHED": 512 * 1024**2, "E_C2F_MATCHED": 1024**3}
        or cache.get("graphs") != {"E_A5_MATCHED": "V1_ONLY", "E_C2F_MATCHED": "DISABLED"}
    ):
        raise ValueError("Concurrent cache/graph scope changed")
    for fit_id, origin in value.get("origins", {}).items():
        if fit_id not in FIT_IDS or sha256_file(Path(origin["checkpoint_path"])) != origin.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError(f"Concurrent immutable origin changed: {fit_id}")
    return value


def validate_concurrent_lineage(run: Path, freeze_path: Path, fit_id: str) -> None:
    """Require a typed concurrent receipt after the immutable native origin."""
    freeze = validate_concurrent_freeze(freeze_path)
    if fit_id not in FIT_IDS:
        raise ValueError(f"Concurrent lineage is out of scope: {fit_id}")
    fit = run / "fits" / fit_id
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    update = int(pointer.get("completed_updates", -1))
    origin = freeze["origins"][fit_id]
    if update < int(origin["completed_updates"]):
        raise RuntimeError(f"Concurrent checkpoint moved behind its origin: {fit_id}")
    if update == int(origin["completed_updates"]):
        if (
            pointer.get("checkpoint_sha256") != origin["checkpoint_sha256"]
            or pointer.get("identity_sha256") != origin["identity_sha256"]
        ):
            raise RuntimeError(f"Concurrent origin changed: {fit_id}")
        return
    root = fit / "concurrent_checkpoints"
    receipt_path = root / f"checkpoint_{update:06d}.json"
    common = {
        "fit_id": fit_id,
        "concurrent_freeze_sha256": freeze["identity_sha256"],
        "concurrent_freeze_file_sha256": sha256_file(freeze_path),
        "cache_limit_bytes": freeze["cache_policy"]["per_fit_max_bytes"][fit_id],
        "graph_mode": freeze["cache_policy"]["graphs"][fit_id],
        "source_sha256": freeze["source_sha256"],
    }
    if receipt_path.is_file():
        value = read_json_shared(receipt_path)
        invalid = (
            value.get("schema") != "rgb_port_concurrent_checkpoint_receipt_v1"
            or value.get("status")
            not in {
                "COMPLETE",
                "RECOVERED_FROM_PENDING",
                "ADOPTED_V1_LINEAGE",
                "ADOPTED_V2_LINEAGE",
            }
            or int(value.get("completed_updates", -1)) != update
            or value.get("checkpoint_version") != pointer.get("version")
            or value.get("checkpoint_sha256") != pointer.get("checkpoint_sha256")
            or value.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or any(value.get(key) != item for key, item in common.items())
        )
    elif (root / "PENDING_EXECUTION_RECEIPT.json").is_file():
        pending = read_json_shared(root / "PENDING_EXECUTION_RECEIPT.json")
        invalid = (
            pending.get("schema") != "rgb_port_concurrent_pending_v1"
            or pending.get("status") != "PENDING"
            or int(pending.get("end_update", -1)) != update
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or any(pending.get(key) != item for key, item in common.items())
        )
    elif fit_id == "E_C2F_MATCHED":
        from operational.rgb_port_acceleration.queue import _validate_runtime_receipts

        _validate_runtime_receipts(
            run,
            require_endpoint=False,
            fit_id=fit_id,
        )
        invalid = False
    else:
        invalid = True
    if invalid:
        raise RuntimeError(f"Concurrent checkpoint lacks matching supplemental proof: {fit_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--c2f-pid", type=int, required=True)
    build.add_argument("--c2f-create-time", type=float, required=True)
    build.add_argument("--output", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    value = (
        build_concurrent_freeze(
            args.run,
            c2f_child={"pid": args.c2f_pid, "create_time": args.c2f_create_time},
            output=args.output,
        )
        if args.action == "build"
        else validate_concurrent_freeze(args.freeze)
    )
    print(value["identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
