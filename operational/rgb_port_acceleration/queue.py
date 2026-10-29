"""Runtime-only supervisor wrapper for the admitted E_A5 CUDA-graph route."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import run as frozen_queue
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

from .contracts import FIT_ID, METRICS_FIT_IDS, ORIGIN_UPDATE, validate_acceleration_freeze

PRODUCER_MODULE = "operational.rgb_port.train_producers"
ACCELERATED_MODULE = "operational.rgb_port_acceleration.producer"
RUNTIME_SCHEMA = "rgb_port_acceleration_runtime_v1"
VERSION_SCHEMA = "rgb_port_acceleration_checkpoint_receipt_v1"
PENDING_SCHEMA = "rgb_port_acceleration_pending_v1"


def _runtime_members(run: Path) -> list[str]:
    members = [
        "ACCELERATION_FREEZE.json",
        "acceleration/E_A5_CUDAGRAPH_ADMISSION.json",
        "acceleration/E_A5_ORIGIN_006189.pt",
        "resume_audit_20261009/ASTRA_REVIEW.md",
        "resume_audit_20261009/QUEUE_AUDIT.json",
        "resume_audit_20261009/OPTIMIZATION_AUDIT.json",
        "resume_audit_20261009/CUDA_CHECKPOINT_RESUME_QA.json",
        "resume_audit_20261009/E_A5_CUDAGRAPH_ADMISSION.json",
        "resume_audit_20261009/verify_accelerated_restore.py",
        "optional:resume_audit_20261009/ACCELERATED_RESTORE_INTEGRATION_QA.json",
        "repo-tree:operational/rgb_port_acceleration",
        "repo:tests/test_rgb_port_acceleration_queue.py",
        "repo:tests/test_rgb_port_acceleration_producer.py",
    ]
    for fit_id in METRICS_FIT_IDS:
        fit = run / "fits" / fit_id
        if (fit / "ACCELERATION_RUNTIME.json").is_file():
            members.append(f"fits/{fit_id}/ACCELERATION_RUNTIME.json")
        if fit_id == FIT_ID and (fit / "PREWARM_QA.json").is_file():
            members.append(f"fits/{fit_id}/PREWARM_QA.json")
        receipt_root = fit / "acceleration_checkpoints"
        if receipt_root.is_dir():
            members.extend(
                str(path.relative_to(run)).replace("\\", "/")
                for path in sorted(receipt_root.glob("checkpoint_*.json"))
            )
            members.extend(
                str(path.relative_to(run)).replace("\\", "/")
                for path in sorted(receipt_root.glob("prewarm_*.json"))
            )
            members.extend(
                str(path.relative_to(run)).replace("\\", "/")
                for path in sorted(receipt_root.glob("runtime_*.json"))
            )
    return members


def _augmented_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for field in ("heavy_command_markers", "project_command_markers"):
        markers = value["resources"].setdefault(field, [])
        if ACCELERATED_MODULE not in markers:
            markers.append(ACCELERATED_MODULE)
    members = value["package"]["members"]
    for member in _runtime_members(Path(value["run_root"])):
        if member not in members:
            members.append(member)
    return value


def _route_command(
    original: Callable[[list[Any], Path], list[str]],
    freeze: Path,
    command: list[Any],
    repository: Path,
) -> list[str]:
    resolved = original(command, repository)
    try:
        module_index = resolved.index("-m") + 1
    except ValueError:
        return resolved
    if (
        module_index >= len(resolved)
        or resolved[module_index] != PRODUCER_MODULE
        or "--fit-id" not in resolved
    ):
        return resolved
    fit_index = resolved.index("--fit-id")
    if fit_index + 1 >= len(resolved) or resolved[fit_index + 1] not in METRICS_FIT_IDS:
        return resolved
    return [
        *resolved[:module_index],
        ACCELERATED_MODULE,
        "--acceleration-freeze",
        str(freeze.resolve(strict=True)),
        "--",
        *resolved[module_index + 1 :],
    ]


def _validate_runtime_receipts(
    run: Path, *, require_endpoint: bool, fit_id: str = FIT_ID
) -> dict[str, Any]:
    freeze_path = run / "ACCELERATION_FREEZE.json"
    freeze = validate_acceleration_freeze(freeze_path)
    freeze_file_sha256 = sha256_file(freeze_path)
    if fit_id not in METRICS_FIT_IDS:
        raise ValueError(f"Unscoped acceleration fit: {fit_id}")
    fit = run / "fits" / fit_id
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    completed = int(pointer.get("completed_updates", -1))
    origin_update = ORIGIN_UPDATE if fit_id == FIT_ID else 0
    if completed < origin_update:
        raise RuntimeError("E_A5 checkpoint moved behind the admitted native origin")
    version = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if not version.is_file() or pointer.get("checkpoint_sha256") != sha256_file(version):
        raise RuntimeError(f"{fit_id} checkpoint pointer does not bind durable bytes")
    if completed == ORIGIN_UPDATE and fit_id == FIT_ID:
        origin = freeze["origin"]
        if pointer.get("checkpoint_sha256") != origin.get("checkpoint_sha256") or pointer.get(
            "identity_sha256"
        ) != origin.get("identity_sha256"):
            raise RuntimeError("E_A5 native origin pointer differs from the admission")
        return freeze
    runtime_path = fit / "ACCELERATION_RUNTIME.json"
    pending_path = fit / "acceleration_checkpoints/PENDING_EXECUTION_RECEIPT.json"
    if (
        not runtime_path.is_file()
        or not (fit / "acceleration_checkpoints" / f"checkpoint_{completed:06d}.json").is_file()
    ):
        if not pending_path.is_file():
            raise FileNotFoundError(f"Accelerated {fit_id} progress lacks runtime or pending proof")
        pending = read_json_shared(pending_path)
        expected_mode = "graph_and_batched_metrics" if fit_id == FIT_ID else "batched_metrics_only"
        if (
            pending.get("schema") != PENDING_SCHEMA
            or pending.get("status") != "PENDING"
            or pending.get("fit_id") != fit_id
            or pending.get("mode") != expected_mode
            or int(pending.get("end_update", -1)) != completed
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or pending.get("acceleration_freeze_sha256") != freeze["identity_sha256"]
            or pending.get("acceleration_freeze_file_sha256") != freeze_file_sha256
            or pending.get("admission_sha256")
            != (freeze["admission_sha256"] if fit_id == FIT_ID else None)
            or pending.get("source_sha256") != freeze["source_sha256"]
            or pending.get("torch_backend_source_sha256") != freeze["torch_backend_source_sha256"]
        ):
            raise RuntimeError("Pending acceleration proof cannot repair the durable checkpoint")
        return freeze
    runtime = read_json_shared(runtime_path)
    expected_mode = "graph_and_batched_metrics" if fit_id == FIT_ID else "batched_metrics_only"
    runtime_update = int(runtime.get("last_saved_update", runtime.get("completed_updates", -1)))
    runtime_checkpoint = runtime.get("last_checkpoint_sha256", runtime.get("checkpoint_sha256"))
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("status") not in {"READY", "ACTIVE"}
        or runtime.get("fit_id") != fit_id
        or runtime.get("mode") != expected_mode
        or runtime.get("optimizer_updates") != 0
        or runtime.get("acceleration_freeze_sha256") != freeze["identity_sha256"]
        or runtime.get("acceleration_freeze_file_sha256") != freeze_file_sha256
        or runtime.get("backend") != ("cudagraphs" if fit_id == FIT_ID else None)
        or runtime.get("shape") != ({"batch_size": 32, "frames": 3} if fit_id == FIT_ID else None)
        or runtime.get("admission_sha256")
        != (freeze["admission_sha256"] if fit_id == FIT_ID else None)
        or runtime_update != completed
        or runtime_checkpoint != pointer.get("checkpoint_sha256")
        or runtime.get("source_sha256") != freeze["source_sha256"]
        or runtime.get("torch_backend_source_sha256") != freeze["torch_backend_source_sha256"]
    ):
        raise RuntimeError(f"Accelerated {fit_id} runtime receipt does not bind the checkpoint")
    versions = fit / "checkpoint_versions"
    receipts = fit / "acceleration_checkpoints"
    for version in versions.glob("checkpoint_*.pt"):
        update = int(version.stem.removeprefix("checkpoint_"))
        if fit_id == FIT_ID and update <= ORIGIN_UPDATE:
            continue
        receipt_path = receipts / f"checkpoint_{update:06d}.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Accelerated checkpoint lacks receipt: {version.name}")
        item = read_json_shared(receipt_path)
        if (
            item.get("schema") != VERSION_SCHEMA
            or item.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING"}
            or item.get("fit_id") != fit_id
            or int(item.get("completed_updates", -1)) != update
            or item.get("checkpoint_version") != version.name
            or item.get("checkpoint_sha256") != sha256_file(version)
            or item.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or item.get("runtime_snapshot") != f"runtime_{update:06d}.json"
            or item.get("acceleration_freeze_sha256") != freeze["identity_sha256"]
            or item.get("acceleration_freeze_file_sha256") != freeze_file_sha256
            or item.get("admission_sha256")
            != (freeze["admission_sha256"] if fit_id == FIT_ID else None)
            or item.get("source_sha256") != freeze["source_sha256"]
            or item.get("torch_backend_source_sha256") != freeze["torch_backend_source_sha256"]
            or item.get("mode") != expected_mode
            or item.get("backend") != ("cudagraphs" if fit_id == FIT_ID else None)
            or item.get("shape") != ({"batch_size": 32, "frames": 3} if fit_id == FIT_ID else None)
        ):
            raise RuntimeError(f"Acceleration checkpoint receipt changed: {version.name}")
    latest = read_json_shared(receipts / f"checkpoint_{completed:06d}.json")
    for receipt_path in receipts.glob("checkpoint_*.json"):
        item = read_json_shared(receipt_path)
        update = int(item.get("completed_updates", -1))
        runtime_archive = receipts / f"runtime_{update:06d}.json"
        expected_runtime_sha = item.get("runtime_receipt_sha256")
        if (
            item.get("runtime_snapshot") != runtime_archive.name
            or not runtime_archive.is_file()
            or sha256_file(runtime_archive) != expected_runtime_sha
        ):
            raise RuntimeError("Acceleration checkpoint lacks its immutable runtime receipt")
    latest_archive = receipts / f"runtime_{completed:06d}.json"
    if latest.get("runtime_receipt_sha256") != sha256_file(latest_archive):
        raise RuntimeError("Latest acceleration checkpoint does not bind its runtime archive")
    if fit_id == FIT_ID and (completed > ORIGIN_UPDATE or require_endpoint):
        prewarm_path = fit / "PREWARM_QA.json"
        if not prewarm_path.is_file():
            raise RuntimeError("Complete accelerated bundle requires PASSED prewarm QA")
        prewarm = read_json_shared(prewarm_path)
        exact = prewarm.get("post_warm_exact")
        prewarm_update = int(prewarm.get("completed_updates", -1))
        if prewarm_update == ORIGIN_UPDATE:
            prewarm_lineage_valid = (
                prewarm.get("checkpoint_sha256") == freeze["origin"]["checkpoint_sha256"]
            )
        else:
            prewarm_receipt_path = receipts / f"checkpoint_{prewarm_update:06d}.json"
            prewarm_lineage_valid = prewarm_receipt_path.is_file() and read_json_shared(
                prewarm_receipt_path
            ).get("checkpoint_sha256") == prewarm.get("checkpoint_sha256")
        if (
            prewarm.get("schema") != "rgb_port_acceleration_prewarm_qa_v1"
            or prewarm.get("status") != "PASSED"
            or prewarm.get("fit_id") != FIT_ID
            or prewarm.get("optimizer_updates") != 0
            or prewarm_update < ORIGIN_UPDATE
            or prewarm_update > completed
            or not prewarm_lineage_valid
            or prewarm.get("checkpoint_identity_sha256") != freeze["origin"]["identity_sha256"]
            or prewarm.get("acceleration_freeze_sha256") != freeze["identity_sha256"]
            or prewarm.get("acceleration_freeze_file_sha256") != freeze_file_sha256
            or not isinstance(exact, dict)
            or not exact
            or not all(value is True for value in exact.values())
        ):
            raise RuntimeError("E_A5 prewarm QA does not bind the acceleration freeze")
        archive = receipts / f"prewarm_{prewarm_update:06d}.json"
        if archive.exists() and read_json_shared(archive) != prewarm:
            raise RuntimeError("Archived E_A5 prewarm QA changed")
        if not archive.exists():
            atomic_write_json(archive, prewarm)
    return freeze


@contextmanager
def _patched_runtime(run: Path) -> Iterator[None]:
    freeze = run / "ACCELERATION_FREEZE.json"
    validate_acceleration_freeze(freeze)
    original_load = frozen_queue._load_config
    original_resolve = frozen_queue._resolve_command
    original_freeze = frozen_queue._freeze_inputs
    original_fit_complete = frozen_queue._fit_complete

    def load(path: Path) -> dict[str, Any]:
        return _augmented_config(original_load(path))

    def resolve(command: list[Any], repository: Path) -> list[str]:
        return _route_command(original_resolve, freeze, command, repository)

    def check_inputs(config_path: Path, config: dict[str, Any], run_root: Path) -> dict[str, Any]:
        result = original_freeze(config_path, config, run_root)
        validate_acceleration_freeze(freeze)
        for fit_id in METRICS_FIT_IDS:
            if (run_root / "fits" / fit_id / "CHECKPOINT_POINTER.json").is_file():
                _validate_runtime_receipts(run_root, require_endpoint=False, fit_id=fit_id)
        return result

    def fit_complete(task: Mapping[str, Any], run_root: Path) -> bool:
        complete = original_fit_complete(task, run_root)
        fit_id = task.get("fit_id")
        if complete and fit_id in METRICS_FIT_IDS:
            _validate_runtime_receipts(
                run_root, require_endpoint=fit_id == FIT_ID, fit_id=str(fit_id)
            )
        return complete

    with ExitStack() as stack:
        stack.enter_context(patch.object(frozen_queue, "_load_config", load))
        stack.enter_context(patch.object(frozen_queue, "_resolve_command", resolve))
        stack.enter_context(patch.object(frozen_queue, "_freeze_inputs", check_inputs))
        stack.enter_context(patch.object(frozen_queue, "_fit_complete", fit_complete))
        yield


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("execute", "resume"):
        item = sub.add_parser(action)
        item.add_argument("--run", type=Path, required=True)
        if action == "execute":
            item.add_argument("--config", type=Path, required=True)
        item.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    if args.action == "package":
        for fit_id in METRICS_FIT_IDS:
            if (run / "fits" / fit_id / "CHECKPOINT_POINTER.json").is_file():
                _validate_runtime_receipts(run, require_endpoint=fit_id == FIT_ID, fit_id=fit_id)
    forwarded = [args.action]
    if args.action == "execute":
        forwarded.extend(("--config", str(args.config)))
    else:
        forwarded.extend(("--run", str(run)))
    if getattr(args, "once", False):
        forwarded.append("--once")
    with _patched_runtime(run):
        return frozen_queue.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
