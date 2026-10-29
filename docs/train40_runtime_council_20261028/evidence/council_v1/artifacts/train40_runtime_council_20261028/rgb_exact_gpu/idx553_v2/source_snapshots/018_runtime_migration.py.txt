"""Fail-closed migration of a durable producer state to a new runtime binding.

The scientific checkpoint is never rewritten.  A migration copies its exact
bytes into a new run root and records an external old-to-new runtime binding.
The runner must validate that binding before it may restore the checkpoint
under changed executable sources; this module deliberately provides no
``ignore_identity`` escape hatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from operational.rgb_port.accounting import atomic_write_json, read_json_shared

from .recipe import canonical_sha256, sha256

ADMISSION_SCHEMA = "final_refit_runtime_migration_admission_v1"
MIGRATION_SCHEMA = "final_refit_runtime_migration_v1"
RUNTIME_BINDING_SCHEMA = "final_refit_runtime_resume_binding_v1"
RUNTIME_BINDING_V3_SCHEMA = "final_refit_runtime_resume_binding_v2"
OPTIMIZATION_ADMISSION_SCHEMA = "final_refit_rgb_optimization_admission_v1"
BATCH_MIGRATION_SCHEMA = "final_refit_runtime_migration_batch_v2"
FIT_MIGRATION_SCHEMA = "final_refit_runtime_fit_migration_v2"
CHECKPOINT_SCHEMA = "rgb_port_producer_checkpoint_v1"
CHECKPOINT_INTERVAL = 100
FAULT_PROBE_SCHEMA = "final_refit_checkpoint_fault_probe_v1"
THREAD_PROBE_SCHEMA = "final_refit_rgb_runtime_thread_probe_v1"
_FAULTS = {
    "commit_journal_before",
    "commit_journal_after",
    "save_version_after_write",
    "save_pointer_after",
    "save_alias_after",
    "save_journal_after",
    "save_receipt_after",
}

_STATE_FIELDS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_random_state",
    "python_random_state",
    "cursor",
)


def _canonical_plan_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific/data plan after removing only runtime bindings."""
    result = dict(value)
    result.pop("plan_sha256", None)
    result.pop("executed_source_sha256", None)
    inputs = dict(cast(Mapping[str, Any], result["input_sha256"]))
    inputs.pop("runtime_admission", None)
    inputs.pop("runtime_migration_admission", None)
    inputs.pop("runtime_optimization_admission", None)
    result["input_sha256"] = inputs
    return result


def _validate_optimization_admission(
    path: Path,
    *,
    old_plan: Mapping[str, Any],
    new_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the technical v3 admission without binding a moving checkpoint."""
    value = read_json_shared(path.resolve(strict=True))
    without_hash = dict(value)
    claimed = without_hash.pop("admission_sha256", None)
    required_checks = {"numerical_parity", "state_parity", "resume_parity", "rng_parity"}
    checks = value.get("checks")
    if (
        value.get("schema") != OPTIMIZATION_ADMISSION_SCHEMA
        or value.get("status") != "PASS"
        or claimed != canonical_sha256(without_hash)
        or value.get("scientific_optimizer_updates") != 0
        or not isinstance(value.get("technical_optimizer_updates"), int)
        or int(value["technical_optimizer_updates"]) < 0
        or not isinstance(value.get("technical_cases"), list)
        or len(value["technical_cases"]) == 0
        or not isinstance(value.get("admitted_options"), dict)
        or not value["admitted_options"]
        or not isinstance(checks, dict)
        or set(checks) != required_checks
        or any(checks[name] != "PASS" for name in required_checks)
        or value.get("old_plan_sha256") != old_plan.get("plan_sha256")
        or value.get("scientific_plan_sha256")
        != canonical_sha256(_canonical_plan_identity(old_plan))
    ):
        raise ValueError("Runtime optimization admission is invalid")
    if _canonical_plan_identity(old_plan) != _canonical_plan_identity(new_plan):
        raise ValueError("Runtime optimization changed the scientific/data producer plan")
    transitions = value.get("source_sha256_before_after")
    old_sources = cast(Mapping[str, str], old_plan["executed_source_sha256"])
    new_sources = cast(Mapping[str, str], new_plan["executed_source_sha256"])
    if not isinstance(transitions, dict) or set(transitions) != set(old_sources) | set(new_sources):
        raise ValueError("Optimization admission omits part of the source closure")
    for name in transitions:
        if transitions[name] != {
            "before_sha256": old_sources.get(name),
            "after_sha256": new_sources.get(name),
        }:
            raise ValueError("Optimization admission source transition differs")
    repository = Path(__file__).resolve().parents[2]
    for name, expected in new_sources.items():
        current = (repository / name).resolve(strict=True)
        if not current.is_relative_to(repository) or sha256(current) != expected:
            raise ValueError(f"New optimization runtime source differs: {name}")
    return value


def validate_optimization_admission(
    path: Path, *, old_plan_path: Path, new_plan_path: Path
) -> dict[str, Any]:
    """Public fail-closed validator used by plan and runner admission gates."""
    return _validate_optimization_admission(
        path.resolve(strict=True),
        old_plan=_validate_plan(old_plan_path.resolve(strict=True)),
        new_plan=_validate_plan(new_plan_path.resolve(strict=True)),
    )


def _validate_plan(path: Path) -> dict[str, Any]:
    value = read_json_shared(path)
    claimed = value.get("plan_sha256")
    without_hash = dict(value)
    without_hash.pop("plan_sha256", None)
    if claimed != canonical_sha256(without_hash):
        raise ValueError(f"Producer plan hash is invalid: {path}")
    if (
        value.get("schema") != "final_refit_rgb_producer_plan_v1"
        or value.get("status") != "FROZEN_AUTHORIZED_PRODUCER_SCOPE"
        or value.get("launch_authorized") is not True
    ):
        raise ValueError("Runtime migration requires an authorized producer plan")
    return value


class _Digest(Protocol):
    def update(self, value: bytes, /) -> object: ...


def _feed(digest: _Digest, kind: str, payload: bytes = b"") -> None:
    encoded_kind = kind.encode("utf-8")
    digest.update(len(encoded_kind).to_bytes(4, "big"))
    digest.update(encoded_kind)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _state_sha256(value: object) -> str:
    """Hash nested checkpoint state without depending on pickle byte layout."""
    import numpy as np
    import torch

    digest = hashlib.sha256(b"final_refit_checkpoint_state_v1\0")

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            _feed(
                digest,
                "tensor",
                json.dumps(
                    {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            _feed(digest, "tensor_bytes", raw)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            _feed(
                digest,
                "ndarray",
                json.dumps(
                    {"dtype": array.dtype.str, "shape": list(array.shape)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            _feed(digest, "ndarray_bytes", array.tobytes())
        elif isinstance(item, Mapping):
            _feed(digest, "mapping", str(len(item)).encode("ascii"))
            mapping = cast(Mapping[object, object], item)
            keyed = [(_state_sha256(key), key) for key in mapping]
            for key_hash, key in sorted(keyed, key=lambda pair: pair[0]):
                _feed(digest, "key_hash", key_hash.encode("ascii"))
                visit(mapping[key])
        elif isinstance(item, tuple):
            _feed(digest, "tuple", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, list):
            _feed(digest, "list", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif item is None:
            _feed(digest, "none")
        elif isinstance(item, bool):
            _feed(digest, "bool", b"1" if item else b"0")
        elif isinstance(item, int):
            _feed(digest, "int", str(item).encode("ascii"))
        elif isinstance(item, float):
            _feed(digest, "float", item.hex().encode("ascii"))
        elif isinstance(item, str):
            _feed(digest, "str", item.encode("utf-8"))
        elif isinstance(item, bytes):
            _feed(digest, "bytes", item)
        else:
            raise TypeError(f"Unsupported checkpoint state value: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported producer checkpoint")
    missing = [name for name in _STATE_FIELDS if name not in value]
    if missing:
        raise ValueError(f"Checkpoint omits resumable state: {missing}")
    if int(value.get("accumulation_index", -1)) != 0:
        raise ValueError("Migration is only valid at an optimizer boundary")
    return cast(dict[str, Any], value)


def _validate_source_evidence(
    source_directory: Path,
    checkpoint: Path,
    payload: Mapping[str, Any],
    curve_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    identity = str(payload["identity_sha256"])
    completed = int(payload["completed_updates"])
    checkpoint_hash = sha256(checkpoint)
    pointer = read_json_shared(source_directory / "CHECKPOINT_POINTER.json")
    receipt = read_json_shared(source_directory / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(source_directory / "UPDATE_JOURNAL.json")
    if (
        pointer.get("schema") != "rgb_port_checkpoint_pointer_v1"
        or pointer.get("identity_sha256") != identity
        or int(pointer.get("completed_updates", -1)) != completed
        or pointer.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("Checkpoint pointer does not bind the migration source")
    if (
        receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
        or receipt.get("identity_sha256") != identity
        or int(receipt.get("completed_updates", -1)) != completed
        or receipt.get("checkpoint_sha256") != checkpoint_hash
        or receipt.get("complete_state") is not True
        or int(receipt.get("accumulation_index", -1)) != 0
        or receipt.get("scientific_endpoint") is not False
    ):
        raise ValueError("Checkpoint receipt does not describe a paused complete state")
    if (
        journal.get("schema") != "rgb_port_update_journal_v1"
        or journal.get("identity_sha256") != identity
        or int(journal.get("completed_updates", -1)) != completed
        or int(journal.get("durable_updates", -1)) != completed
        or int(journal.get("pending_update_upper", -1)) != 0
        or int(journal.get("recovery_upper", -1)) != int(payload["recovery_upper"])
    ):
        raise ValueError("Update journal is not a zero-pending durable boundary")
    curves = _validate_curve_coverage(curve_directory, payload)
    return pointer, receipt, journal, curves


def _validate_curve_coverage(
    curve_directory: Path,
    payload: Mapping[str, Any],
    *,
    recoverable_upper: int | None = None,
) -> dict[str, str]:
    """Validate durable curve files plus the checkpoint's pending suffix."""
    completed = int(payload["completed_updates"])
    curves: dict[str, str] = {}
    curve_updates: list[int] = []
    for curve in sorted(curve_directory.glob("curve_*.json")):
        try:
            named_boundary = int(curve.stem.removeprefix("curve_"))
        except ValueError as error:
            raise ValueError(f"Malformed curve filename: {curve.name}") from error
        if named_boundary > completed and (
            recoverable_upper is None or named_boundary > recoverable_upper
        ):
            raise ValueError(f"Curve advances past the durable checkpoint: {curve.name}")
        curve_value = read_json_shared(curve)
        updates = curve_value.get("updates")
        if not isinstance(updates, list) or not updates:
            raise ValueError(f"Periodic curve has a discontinuous update range: {curve.name}")
        observed = [int(item.get("update", -1)) for item in updates]
        expected = list(range(observed[0], observed[-1] + 1))
        if observed != expected or observed[-1] != named_boundary:
            raise ValueError(f"Periodic curve has a discontinuous update range: {curve.name}")
        if named_boundary <= completed:
            curve_updates.extend(observed)
        curves[curve.name] = sha256(curve)
    for boundary in range(CHECKPOINT_INTERVAL, completed + 1, CHECKPOINT_INTERVAL):
        if f"curve_{boundary:06d}.json" not in curves:
            raise ValueError(f"Missing durable periodic curve: curve_{boundary:06d}.json")
    pending = cast(Mapping[str, Any], payload["cursor"]).get("pending_curve")
    if not isinstance(pending, list):
        raise ValueError("Checkpoint cursor omits pending_curve")
    pending_updates = [int(item.get("update", -1)) for item in pending]
    if sorted(curve_updates + pending_updates) != list(range(1, completed + 1)):
        raise ValueError("Saved and pending curves do not cover the durable update cursor exactly")
    return curves


def _validate_admission(
    path: Path,
    *,
    fit_id: str,
    old_identity: str,
    checkpoint_sha256: str,
    old_plan_sha256: str,
) -> dict[str, Any]:
    value = read_json_shared(path)
    without_hash = dict(value)
    claimed = without_hash.pop("admission_sha256", None)
    required = {
        "schema": ADMISSION_SCHEMA,
        "status": "ADMITTED_TECHNICAL_RUNTIME_MIGRATION",
        "fit_id": fit_id,
        "old_identity_sha256": old_identity,
        "parent_checkpoint_sha256": checkpoint_sha256,
        "old_plan_sha256": old_plan_sha256,
        "scientific_recipe_unchanged": True,
        "optimizer_mode_unchanged": True,
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
        "scientific_optimizer_updates": 0,
    }
    if claimed != canonical_sha256(without_hash) or any(
        value.get(name) != expected for name, expected in required.items()
    ):
        raise ValueError("Runtime migration admission is invalid")
    sources = value.get("source_sha256_before_after")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Admission must enumerate the runtime source closure")
    for name, pair in sources.items():
        if (
            not isinstance(name, str)
            or not isinstance(pair, dict)
            or set(pair) != {"before_sha256", "after_sha256"}
            or (
                pair["before_sha256"] is not None
                and (not isinstance(pair["before_sha256"], str) or len(pair["before_sha256"]) != 64)
            )
            or not isinstance(pair["after_sha256"], str)
            or len(pair["after_sha256"]) != 64
        ):
            raise ValueError("Malformed before/after runtime source binding")
    evidence = value.get("technical_evidence")
    required_evidence = {
        "state_parity": "PASS",
        "numerical_parity": "PASS",
        "runtime_benchmark": "PROFILED",
    }
    if not isinstance(evidence, dict) or set(evidence) != set(required_evidence):
        raise ValueError("Admission requires state, numerical, and benchmark evidence")
    for name, expected_status in required_evidence.items():
        reference = evidence[name]
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256", "scope"}
            or not isinstance(reference["scope"], str)
            or not reference["scope"].strip()
        ):
            raise ValueError(f"Malformed technical evidence reference: {name}")
        evidence_path = Path(str(reference["path"]))
        if not evidence_path.is_absolute():
            evidence_path = path.parent / evidence_path
        evidence_path = evidence_path.resolve(strict=True)
        if sha256(evidence_path) != reference["sha256"]:
            raise ValueError(f"Technical evidence changed after admission: {name}")
        evidence_value = read_json_shared(evidence_path)
        admitted = False
        if name == "state_parity":
            cases = evidence_value.get("cases")
            admitted = bool(
                evidence_value.get("schema") == FAULT_PROBE_SCHEMA
                and evidence_value.get("status") == expected_status
                and isinstance(cases, list)
                and len(cases) == len(_FAULTS)
                and {item.get("fault") for item in cases} == _FAULTS
                and all(
                    item.get("all_exact") is True
                    and item.get("bounded_by_save_interval") is True
                    and item.get("scientific_endpoint") is True
                    and isinstance(item.get("exact"), dict)
                    and item["exact"]
                    and all(item["exact"].values())
                    for item in cases
                )
            )
        elif name == "numerical_parity":
            from .eager_probe import validate_receipt

            try:
                validate_receipt(evidence_value)
                admitted = evidence_value.get("scientific_updates") == 0
            except ValueError:
                admitted = False
        else:
            parity = evidence_value.get("parity_vs_first_baseline")
            best = evidence_value.get("best_admitted_index")
            admitted = bool(
                evidence_value.get("schema") == THREAD_PROBE_SCHEMA
                and evidence_value.get("status") == expected_status
                and evidence_value.get("scientific_updates") == 0
                and isinstance(parity, list)
                and isinstance(best, int)
                and 0 <= best < len(parity)
                and parity[best].get("passed") is True
                and evidence_value.get("baseline_repeat_parity", {}).get("passed") is True
            )
        if not admitted:
            raise ValueError(f"Technical evidence is not admitted: {name}")
    return value


def migrate_runtime_state(
    *,
    source_directory: Path,
    source_checkpoint: Path,
    old_plan_path: Path,
    new_plan_path: Path,
    admission_path: Path,
    destination_run_root: Path,
    fit_id: str,
    curve_directory: Path | None = None,
) -> dict[str, Any]:
    """Copy one exact durable state into an empty runtime-v2 run root.

    This function performs no optimizer update and never mutates ``source_directory``.
    A caller must supply a separately reviewed admission whose hash is bound by the
    new producer plan.  The returned status is deliberately not a scientific
    endpoint and does not itself authorize resumption.
    """
    source_directory = source_directory.resolve(strict=True)
    source_checkpoint = source_checkpoint.resolve(strict=True)
    if source_checkpoint.parent != source_directory:
        raise ValueError("Source checkpoint must be an immutable file in source_directory")
    old_plan_path = old_plan_path.resolve(strict=True)
    new_plan_path = new_plan_path.resolve(strict=True)
    admission_path = admission_path.resolve(strict=True)
    curve_directory = (
        source_directory if curve_directory is None else curve_directory.resolve(strict=True)
    )
    destination_run_root = destination_run_root.resolve()
    if destination_run_root.exists() or destination_run_root.is_relative_to(source_directory):
        raise FileExistsError("Migration destination must be a new, separate run root")

    old_plan = _validate_plan(old_plan_path)
    new_plan = _validate_plan(new_plan_path)
    if _canonical_plan_identity(old_plan) != _canonical_plan_identity(new_plan):
        raise ValueError("Runtime migration changed the scientific/data producer plan")

    payload = _load_checkpoint(source_checkpoint)
    identity = str(payload["identity_sha256"])
    completed = int(payload["completed_updates"])
    if source_checkpoint.name != f"checkpoint_migration_origin_{completed:06d}.pt":
        raise ValueError("Migration parent must be a uniquely named immutable origin copy")
    if payload.get("identity", {}).get("fit_id") != fit_id:
        raise ValueError("Checkpoint fit identity differs")
    if payload.get("identity", {}).get("producer_contract_sha256") != sha256(old_plan_path):
        raise ValueError("Checkpoint identity does not bind the immutable old producer plan")
    checkpoint_hash = sha256(source_checkpoint)
    pointer, receipt, journal, curves = _validate_source_evidence(
        source_directory, source_checkpoint, payload, curve_directory
    )
    admission = _validate_admission(
        admission_path,
        fit_id=fit_id,
        old_identity=identity,
        checkpoint_sha256=checkpoint_hash,
        old_plan_sha256=str(old_plan["plan_sha256"]),
    )
    if new_plan["input_sha256"].get("runtime_migration_admission") != sha256(admission_path):
        raise ValueError("New plan does not bind the reviewed migration admission")
    old_sources = cast(Mapping[str, str], old_plan["executed_source_sha256"])
    new_sources = cast(Mapping[str, str], new_plan["executed_source_sha256"])
    source_transition = cast(
        Mapping[str, Mapping[str, str | None]], admission["source_sha256_before_after"]
    )
    if set(source_transition) != set(old_sources) | set(new_sources) or any(
        source_transition[name]
        != {
            "before_sha256": old_sources.get(name),
            "after_sha256": new_sources.get(name),
        }
        for name in source_transition
    ):
        raise ValueError("Admission does not bind the complete old-to-new source closure")
    repository = Path(__file__).resolve().parents[2]
    for name, expected in new_sources.items():
        current = (repository / name).resolve(strict=True)
        if not current.is_relative_to(repository) or sha256(current) != expected:
            raise ValueError(f"New runtime source differs after admission: {name}")

    state_hashes = {name: _state_sha256(payload[name]) for name in _STATE_FIELDS}
    external_identity = canonical_sha256(
        {
            "schema": MIGRATION_SCHEMA,
            "fit_id": fit_id,
            "old_identity_sha256": identity,
            "parent_checkpoint_sha256": checkpoint_hash,
            "old_plan_sha256": old_plan["plan_sha256"],
            "new_plan_sha256": new_plan["plan_sha256"],
            "runtime_migration_admission_sha256": sha256(admission_path),
            "state_sha256": state_hashes,
        }
    )

    fit_directory = destination_run_root / "fits" / fit_id
    versions = fit_directory / "checkpoint_versions"
    versions.mkdir(parents=True)
    version_name = f"checkpoint_{completed:06d}.pt"
    version = versions / version_name
    origin_directory = destination_run_root / "RUNTIME_MIGRATION_ORIGIN"
    origin_directory.mkdir()
    origin = origin_directory / version_name
    with tempfile.NamedTemporaryFile(dir=origin_directory, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source_checkpoint, temporary)
        if sha256(temporary) != checkpoint_hash:
            raise OSError("Copied checkpoint bytes differ")
        os.replace(temporary, origin)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        os.link(origin, version)
    except OSError:
        shutil.copyfile(origin, version)
    alias = fit_directory / "checkpoint_last.pt"
    try:
        os.link(version, alias)
    except OSError:
        shutil.copyfile(version, alias)
    for name in curves:
        shutil.copyfile(curve_directory / name, fit_directory / name)
        if sha256(fit_directory / name) != curves[name]:
            raise OSError(f"Copied curve bytes differ: {name}")

    new_pointer = {**pointer, "version": version_name}
    new_receipt = {**receipt, "checkpoint_path": str(alias.resolve())}
    atomic_write_json(fit_directory / "CHECKPOINT_POINTER.json", new_pointer)
    atomic_write_json(fit_directory / "CHECKPOINT_RECEIPT.json", new_receipt)
    atomic_write_json(fit_directory / "UPDATE_JOURNAL.json", journal)
    shutil.copyfile(new_plan_path, destination_run_root / "PRODUCER_PLAN.json")
    shutil.copyfile(admission_path, destination_run_root / "RUNTIME_MIGRATION_ADMISSION.json")

    copied = _load_checkpoint(origin)
    copied_state = {name: _state_sha256(copied[name]) for name in _STATE_FIELDS}
    if sha256(version) != checkpoint_hash or copied_state != state_hashes:
        raise OSError("Migrated resumable state differs from its immutable parent")
    result: dict[str, Any] = {
        "schema": MIGRATION_SCHEMA,
        "status": "MIGRATED_NOT_RESUMED",
        "fit_id": fit_id,
        "completed_updates": completed,
        "scientific_optimizer_updates": 0,
        "scientific_endpoint": False,
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
        "next_periodic_checkpoint_update": ((completed // CHECKPOINT_INTERVAL) + 1)
        * CHECKPOINT_INTERVAL,
        "old_identity_sha256": identity,
        "runtime_binding_sha256": external_identity,
        "parent_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": checkpoint_hash,
        },
        "migrated_checkpoint": {"path": str(origin.resolve()), "sha256": sha256(origin)},
        "initial_resume_checkpoint": {
            "path": str(version.resolve()),
            "sha256": sha256(version),
        },
        "old_plan_sha256": old_plan["plan_sha256"],
        "new_plan_sha256": new_plan["plan_sha256"],
        "runtime_migration_admission_sha256": sha256(admission_path),
        "source_evidence_sha256": {
            "pointer": sha256(source_directory / "CHECKPOINT_POINTER.json"),
            "receipt": sha256(source_directory / "CHECKPOINT_RECEIPT.json"),
            "journal": sha256(source_directory / "UPDATE_JOURNAL.json"),
            "curves": curves,
        },
        "state_sha256": state_hashes,
        "state_equality": {
            name: copied_state[name] == digest for name, digest in state_hashes.items()
        },
        "payload_identity_preserved": copied["identity_sha256"] == identity,
        "recovery_upper": int(copied["recovery_upper"]),
        "accumulation_index": int(copied["accumulation_index"]),
        "admission_status": admission["status"],
    }
    if (
        not all(result["state_equality"].values())
        or result["payload_identity_preserved"] is not True
    ):
        raise OSError("Runtime migration state equality check failed")
    result["migration_sha256"] = canonical_sha256(result)
    atomic_write_json(fit_directory / "RUNTIME_MIGRATION.json", result)
    return result


def validate_runtime_migration(path: Path) -> dict[str, Any]:
    """Validate a completed external binding before a runner consumes it."""
    value = read_json_shared(path.resolve(strict=True))
    without_hash = dict(value)
    claimed = without_hash.pop("migration_sha256", None)
    if (
        value.get("schema") != MIGRATION_SCHEMA
        or value.get("status") != "MIGRATED_NOT_RESUMED"
        or claimed != canonical_sha256(without_hash)
        or value.get("scientific_optimizer_updates") != 0
        or value.get("scientific_endpoint") is not False
        or value.get("checkpoint_interval_updates") != CHECKPOINT_INTERVAL
        or value.get("payload_identity_preserved") is not True
        or set(cast(Mapping[str, bool], value.get("state_equality", {}))) != set(_STATE_FIELDS)
        or not all(cast(Mapping[str, bool], value.get("state_equality", {})).values())
    ):
        raise ValueError("Invalid runtime migration receipt")
    parent = cast(Mapping[str, Any], value["parent_checkpoint"])
    migrated = cast(Mapping[str, Any], value["migrated_checkpoint"])
    if sha256(Path(str(parent["path"]))) != parent["sha256"]:
        raise ValueError("Migration parent checkpoint changed")
    if sha256(Path(str(migrated["path"]))) != migrated["sha256"]:
        raise ValueError("Migrated checkpoint changed")
    if parent["sha256"] != migrated["sha256"]:
        raise ValueError("Migration rewrote the scientific checkpoint")
    expected_binding = canonical_sha256(
        {
            "schema": MIGRATION_SCHEMA,
            "fit_id": value["fit_id"],
            "old_identity_sha256": value["old_identity_sha256"],
            "parent_checkpoint_sha256": parent["sha256"],
            "old_plan_sha256": value["old_plan_sha256"],
            "new_plan_sha256": value["new_plan_sha256"],
            "runtime_migration_admission_sha256": value["runtime_migration_admission_sha256"],
            "state_sha256": value["state_sha256"],
        }
    )
    if value.get("runtime_binding_sha256") != expected_binding:
        raise ValueError("Runtime binding identity differs")
    payload = _load_checkpoint(Path(str(migrated["path"])))
    recorded_states = cast(Mapping[str, str], value.get("state_sha256", {}))
    if set(recorded_states) != set(_STATE_FIELDS) or any(
        _state_sha256(payload[name]) != recorded_states[name] for name in _STATE_FIELDS
    ):
        raise ValueError("Migrated checkpoint state differs from the migration receipt")
    if (
        payload.get("identity_sha256") != value.get("old_identity_sha256")
        or int(payload.get("completed_updates", -1)) != int(value.get("completed_updates", -2))
        or int(payload.get("recovery_upper", -1)) != int(value.get("recovery_upper", -2))
        or int(payload.get("accumulation_index", -1)) != 0
    ):
        raise ValueError("Migrated checkpoint cursor identity differs")
    return value


def _execution_identity_without_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields replaced by the reviewed external runtime lineage."""
    result = dict(value)
    for name in (
        "identity_sha256",
        "producer_contract_path",
        "producer_contract_sha256",
        "executed_source_sha256",
    ):
        result.pop(name, None)
    return result


def _validate_execution_identity(
    value: Mapping[str, Any],
    *,
    new_plan_path: Path,
    fit_id: str,
) -> None:
    plan = _validate_plan(new_plan_path)
    claimed = value.get("identity_sha256")
    without_hash = dict(value)
    without_hash.pop("identity_sha256", None)
    if (
        claimed != canonical_sha256(without_hash)
        or value.get("schema") != "final_refit_rgb_producer_identity_v1"
        or value.get("fit_id") != fit_id
        or value.get("producer_contract_sha256") != sha256(new_plan_path)
        or Path(str(value.get("producer_contract_path"))).resolve() != new_plan_path
    ):
        raise ValueError("Current execution identity does not bind the new producer plan")
    sources = value.get("executed_source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Current execution identity has no executable source closure")
    expected_sources = dict(cast(Mapping[str, str], plan["executed_source_sha256"]))
    expected_sources.pop("operational/final_refit/producer_plan.py", None)
    expected_sources["configs/rgb_port/producers.json"] = str(
        plan["input_sha256"]["producer_config"]
    )
    if sources != expected_sources:
        raise ValueError("Current execution identity truncates the admitted source closure")
    repository = Path(__file__).resolve().parents[2]
    for name, expected in sources.items():
        path = (repository / str(name)).resolve(strict=True)
        if not path.is_relative_to(repository) or sha256(path) != expected:
            raise ValueError(f"Current execution source differs: {name}")


def bind_runtime_resume(
    *,
    migration_path: Path,
    new_plan_path: Path,
    execution_identity: Mapping[str, Any],
    binding_path: Path,
) -> dict[str, Any]:
    """Bind changed runtime provenance to an unchanged checkpoint identity.

    The returned ``checkpoint_identity`` is suitable for constructing
    ``ProducerCheckpoint``. It is not an override: it is returned only after
    validating the immutable migration origin, the admitted new plan, the
    current executable closure, and equality of every non-lineage identity
    field. Future checkpoints retain this scientific identity while this
    sidecar records which admitted runtime executed their continuation.
    """
    migration_path = migration_path.resolve(strict=True)
    new_plan_path = new_plan_path.resolve(strict=True)
    binding_path = binding_path.resolve()
    migration = validate_runtime_migration(migration_path)
    plan = _validate_plan(new_plan_path)
    if (
        plan["plan_sha256"] != migration["new_plan_sha256"]
        or sha256(new_plan_path) != sha256(Path(str(execution_identity["producer_contract_path"])))
        or plan["input_sha256"].get("runtime_migration_admission")
        != migration["runtime_migration_admission_sha256"]
    ):
        raise ValueError("Runtime migration and new producer plan differ")
    fit_id = str(migration["fit_id"])
    _validate_execution_identity(execution_identity, new_plan_path=new_plan_path, fit_id=fit_id)
    origin = Path(str(cast(Mapping[str, Any], migration["migrated_checkpoint"])["path"]))
    payload = _load_checkpoint(origin)
    checkpoint_identity = cast(Mapping[str, Any], payload["identity"])
    if payload["identity_sha256"] != migration[
        "old_identity_sha256"
    ] or _execution_identity_without_lineage(
        checkpoint_identity
    ) != _execution_identity_without_lineage(execution_identity):
        raise ValueError("New runtime changed the checkpoint's scientific execution identity")
    value: dict[str, Any] = {
        "schema": RUNTIME_BINDING_SCHEMA,
        "status": "BOUND_NOT_RESUMED",
        "fit_id": fit_id,
        "migration_path": str(migration_path),
        "migration_sha256": sha256(migration_path),
        "migration_identity_sha256": migration["migration_sha256"],
        "new_plan_path": str(new_plan_path),
        "new_plan_file_sha256": sha256(new_plan_path),
        "new_plan_sha256": plan["plan_sha256"],
        "runtime_migration_admission_sha256": migration["runtime_migration_admission_sha256"],
        "checkpoint_identity_sha256": payload["identity_sha256"],
        "execution_identity_sha256": execution_identity["identity_sha256"],
        "execution_source_sha256": execution_identity["executed_source_sha256"],
        "optimizer_mode_unchanged": True,
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
        "scientific_optimizer_updates_during_binding": 0,
        "identity_policy": "PRESERVED_CHECKPOINT_IDENTITY_EXTERNAL_RUNTIME_LINEAGE",
    }
    value["binding_sha256"] = canonical_sha256(value)
    if binding_path.exists():
        existing = read_json_shared(binding_path)
        if existing != value:
            raise ValueError("Fit is already bound to a different runtime execution identity")
    else:
        atomic_write_json(binding_path, value)
    return dict(checkpoint_identity)


def validate_runtime_binding(
    *,
    binding_path: Path,
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate an existing runtime sidecar and return checkpoint identity."""
    binding_path = binding_path.resolve(strict=True)
    binding = read_json_shared(binding_path)
    without_hash = dict(binding)
    claimed = without_hash.pop("binding_sha256", None)
    if (
        binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or binding.get("status") != "BOUND_NOT_RESUMED"
        or claimed != canonical_sha256(without_hash)
        or binding.get("scientific_optimizer_updates_during_binding") != 0
        or binding.get("optimizer_mode_unchanged") is not True
        or binding.get("checkpoint_interval_updates") != CHECKPOINT_INTERVAL
        or binding.get("execution_identity_sha256") != execution_identity.get("identity_sha256")
    ):
        raise ValueError("Runtime resume binding differs from current execution")
    migration_path = Path(str(binding["migration_path"]))
    new_plan_path = Path(str(binding["new_plan_path"]))
    if (
        sha256(migration_path) != binding["migration_sha256"]
        or sha256(new_plan_path) != binding["new_plan_file_sha256"]
    ):
        raise ValueError("Runtime resume binding parents changed")
    migration = validate_runtime_migration(migration_path)
    if migration["migration_sha256"] != binding["migration_identity_sha256"]:
        raise ValueError("Runtime migration identity changed")
    _validate_execution_identity(
        execution_identity,
        new_plan_path=new_plan_path,
        fit_id=str(binding["fit_id"]),
    )
    origin = Path(str(cast(Mapping[str, Any], migration["migrated_checkpoint"])["path"]))
    payload = _load_checkpoint(origin)
    checkpoint_identity = cast(Mapping[str, Any], payload["identity"])
    if payload["identity_sha256"] != binding[
        "checkpoint_identity_sha256"
    ] or _execution_identity_without_lineage(
        checkpoint_identity
    ) != _execution_identity_without_lineage(execution_identity):
        raise ValueError("Runtime binding no longer preserves checkpoint identity")
    return dict(checkpoint_identity)


def resolve_runtime_checkpoint_identity(
    *,
    run_root: Path,
    fit_id: str,
    new_identity: Mapping[str, Any],
    new_plan_path: Path,
) -> dict[str, Any]:
    """Resolve the preserved identity for initial and later runtime-v2 resumes.

    This is the runner integration entry point.  The immutable migration origin
    remains outside ``checkpoint_versions`` so normal version garbage collection
    cannot erase the proof.  The active pointer may advance, but never regress
    behind the migrated cursor or change checkpoint identity.
    """
    run_root = run_root.resolve(strict=True)
    fit_directory = run_root / "fits" / fit_id
    migration_path = fit_directory / "RUNTIME_MIGRATION.json"
    binding_path = fit_directory / "RUNTIME_RESUME_BINDING.json"
    if binding_path.is_file():
        checkpoint_identity = validate_runtime_binding(
            binding_path=binding_path,
            execution_identity=new_identity,
        )
    else:
        checkpoint_identity = bind_runtime_resume(
            migration_path=migration_path,
            new_plan_path=new_plan_path,
            execution_identity=new_identity,
            binding_path=binding_path,
        )
    migration = validate_runtime_migration(migration_path)
    pointer = read_json_shared(fit_directory / "CHECKPOINT_POINTER.json")
    version = fit_directory / "checkpoint_versions" / str(pointer.get("version"))
    if (
        pointer.get("schema") != "rgb_port_checkpoint_pointer_v1"
        or pointer.get("identity_sha256") != checkpoint_identity["identity_sha256"]
        or int(pointer.get("completed_updates", -1)) < int(migration["completed_updates"])
        or not version.is_file()
        or pointer.get("checkpoint_sha256") != sha256(version)
    ):
        raise ValueError("Active runtime checkpoint does not descend from the migration cursor")
    return checkpoint_identity


def runtime_checkpoint_binding_kwargs(*, run_root: Path, fit_id: str) -> dict[str, str]:
    """Return the two verified keyword arguments for ``ProducerCheckpoint``."""
    fit_directory = run_root.resolve(strict=True) / "fits" / fit_id
    binding_path = fit_directory / "RUNTIME_RESUME_BINDING.json"
    binding = read_json_shared(binding_path)
    without_hash = dict(binding)
    claimed = without_hash.pop("binding_sha256", None)
    if (
        binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or claimed != canonical_sha256(without_hash)
        or binding.get("fit_id") != fit_id
    ):
        raise ValueError("Runtime checkpoint binding keyword source is invalid")
    migration = validate_runtime_migration(fit_directory / "RUNTIME_MIGRATION.json")
    origin = cast(Mapping[str, Any], migration["migrated_checkpoint"])
    return {
        "execution_binding_sha256": str(claimed),
        "execution_binding_origin_checkpoint_sha256": str(origin["sha256"]),
    }


def _validate_prior_execution_lineage(
    *,
    fit_id: str,
    payload: Mapping[str, Any],
    pointer: Mapping[str, Any],
    receipt: Mapping[str, Any],
    journal: Mapping[str, Any],
    prior_binding_path: Path | None,
) -> dict[str, Any] | None:
    """Prove an unbound, v1-bound, or chained v2-bound parent runtime."""
    observed_values = [
        value.get("execution_binding_sha256") for value in (payload, pointer, receipt, journal)
    ]
    if all(value is None for value in observed_values):
        if prior_binding_path is not None:
            raise ValueError("A prior binding was supplied for an unbound checkpoint")
        return None
    if prior_binding_path is None:
        raise ValueError("Bound source state requires its prior binding")
    binding_path = prior_binding_path.resolve(strict=True)
    binding = read_json_shared(binding_path)
    without_hash = dict(binding)
    claimed = without_hash.pop("binding_sha256", None)
    if (
        binding.get("schema") not in {RUNTIME_BINDING_SCHEMA, RUNTIME_BINDING_V3_SCHEMA}
        or binding.get("fit_id") != fit_id
        or claimed != canonical_sha256(without_hash)
        or binding.get("checkpoint_identity_sha256") != payload.get("identity_sha256")
        or binding.get("scientific_optimizer_updates_during_binding") != 0
        or binding.get("optimizer_mode_unchanged") is not True
    ):
        raise ValueError("Prior runtime binding is invalid")
    prior_plan_path = Path(str(binding["new_plan_path"])).resolve(strict=True)
    if sha256(prior_plan_path) != binding.get("new_plan_file_sha256"):
        raise ValueError("Prior runtime plan changed")
    result: dict[str, Any] = {
        "path": str(binding_path),
        "sha256": sha256(binding_path),
        "binding_sha256": claimed,
        "new_plan_path": str(prior_plan_path),
        "new_plan_file_sha256": sha256(prior_plan_path),
    }
    if binding["schema"] == RUNTIME_BINDING_SCHEMA:
        if any(value is None for value in observed_values) or set(observed_values) != {claimed}:
            raise ValueError("V1-bound source state has inconsistent execution bindings")
        prior_migration_path = Path(str(binding["migration_path"])).resolve(strict=True)
        if sha256(prior_migration_path) != binding.get("migration_sha256"):
            raise ValueError("Prior migration receipt changed")
        prior_migration = validate_runtime_migration(prior_migration_path)
        if (
            prior_migration.get("fit_id") != fit_id
            or prior_migration.get("old_identity_sha256") != payload.get("identity_sha256")
            or prior_migration.get("migration_sha256") != binding.get("migration_identity_sha256")
        ):
            raise ValueError("Prior migration lineage differs from the checkpoint")
        result.update(
            migration_path=str(prior_migration_path),
            migration_file_sha256=sha256(prior_migration_path),
            migration_sha256=prior_migration["migration_sha256"],
            previous_execution_binding_sha256=None,
            prior_runtime_sidecar_binding_sha256=claimed,
            parent_embedded_execution_binding_sha256=claimed,
        )
        return result

    batch_path = Path(str(binding["batch_migration_path"])).resolve(strict=True)
    fit_migration_path = Path(str(binding["fit_migration_path"])).resolve(strict=True)
    if sha256(batch_path) != binding.get("batch_migration_file_sha256") or sha256(
        fit_migration_path
    ) != binding.get("fit_migration_file_sha256"):
        raise ValueError("Prior v2 runtime lineage files changed")
    batch = validate_runtime_batch_migration(
        batch_path,
        new_plan_path=prior_plan_path,
        optimization_admission_path=batch_path.parent / "RUNTIME_OPTIMIZATION_ADMISSION.json",
    )
    fit_migration = read_json_shared(fit_migration_path)
    if (
        batch.get("batch_migration_sha256") != binding.get("batch_migration_sha256")
        or batch["fits"][fit_id]["fit_migration_sha256"] != binding.get("fit_migration_sha256")
        or fit_migration.get("fit_migration_sha256") != binding.get("fit_migration_sha256")
    ):
        raise ValueError("Prior v2 batch lineage differs")
    previous = binding.get("previous_execution_binding_sha256")
    if len(set(observed_values)) != 1:
        raise ValueError("V2-bound durable parent has mixed embedded bindings")
    embedded = observed_values[0]
    allowed = {claimed}
    if pointer.get("checkpoint_sha256") == fit_migration["parent_checkpoint"]["sha256"]:
        allowed.add(previous)
    if embedded not in allowed:
        raise ValueError("V2-bound source state has an inadmissible execution binding")
    result.update(
        migration_path=str(fit_migration_path),
        migration_file_sha256=sha256(fit_migration_path),
        migration_sha256=fit_migration["fit_migration_sha256"],
        batch_migration_path=str(batch_path),
        batch_migration_sha256=batch["batch_migration_sha256"],
        previous_execution_binding_sha256=previous,
        prior_runtime_sidecar_binding_sha256=claimed,
        parent_embedded_execution_binding_sha256=embedded,
    )
    return result


def migrate_runtime_states_v3(
    *,
    source_run_root: Path,
    old_plan_path: Path,
    new_plan_path: Path,
    optimization_admission_path: Path,
    destination_run_root: Path,
    fits: Mapping[str, Mapping[str, Path | None]],
) -> dict[str, Any]:
    """Atomically migrate both trained RGB fits into the admitted v3 runtime."""
    expected_fits = {"R_A5", "R_C2F"}
    if set(fits) != expected_fits:
        raise ValueError("V3 runtime migration requires exactly R_A5 and R_C2F")
    source_run_root = source_run_root.resolve(strict=True)
    old_plan_path = old_plan_path.resolve(strict=True)
    new_plan_path = new_plan_path.resolve(strict=True)
    optimization_admission_path = optimization_admission_path.resolve(strict=True)
    destination_run_root = destination_run_root.resolve()
    if destination_run_root.exists() or destination_run_root.is_relative_to(source_run_root):
        raise FileExistsError("V3 migration destination must be new and separate")
    old_plan = _validate_plan(old_plan_path)
    new_plan = _validate_plan(new_plan_path)
    admission = _validate_optimization_admission(
        optimization_admission_path, old_plan=old_plan, new_plan=new_plan
    )
    admission_file_sha = sha256(optimization_admission_path)
    if new_plan["input_sha256"].get("runtime_optimization_admission") != admission_file_sha:
        raise ValueError("New plan does not bind the optimization admission")

    prepared: dict[str, dict[str, Any]] = {}
    required_spec = {
        "source_directory",
        "source_checkpoint",
        "curve_directory",
        "prior_runtime_binding",
    }
    for fit_id in sorted(expected_fits):
        spec = fits[fit_id]
        if set(spec) != required_spec:
            raise ValueError(f"Malformed migration specification for {fit_id}")
        source_directory = Path(str(spec["source_directory"])).resolve(strict=True)
        source_checkpoint = Path(str(spec["source_checkpoint"])).resolve(strict=True)
        curve_value = spec["curve_directory"]
        curve_directory = (
            source_directory if curve_value is None else Path(str(curve_value)).resolve(strict=True)
        )
        prior_value = spec["prior_runtime_binding"]
        prior_path = None if prior_value is None else Path(str(prior_value))
        if not source_directory.is_relative_to(source_run_root):
            raise ValueError("Fit source directory is outside the admitted source run")
        if source_checkpoint.parent != source_directory:
            raise ValueError("Fit parent must be an immutable checkpoint in its source directory")
        payload = _load_checkpoint(source_checkpoint)
        completed = int(payload["completed_updates"])
        if (
            completed <= 0
            or source_checkpoint.name != f"checkpoint_migration_origin_{completed:06d}.pt"
        ):
            raise ValueError("Each fit requires a trained, uniquely named immutable parent")
        if payload.get("identity", {}).get("fit_id") != fit_id:
            raise ValueError("Checkpoint fit identity differs")
        pointer, receipt, journal, curves = _validate_source_evidence(
            source_directory, source_checkpoint, payload, curve_directory
        )
        prior = _validate_prior_execution_lineage(
            fit_id=fit_id,
            payload=payload,
            pointer=pointer,
            receipt=receipt,
            journal=journal,
            prior_binding_path=prior_path,
        )
        if prior is None:
            if payload.get("identity", {}).get("producer_contract_sha256") != sha256(old_plan_path):
                raise ValueError("Unbound checkpoint does not bind the exact old plan")
        elif prior["new_plan_file_sha256"] != sha256(old_plan_path):
            raise ValueError("Prior runtime lineage does not terminate at the old plan")
        prepared[fit_id] = {
            "source_directory": source_directory,
            "source_checkpoint": source_checkpoint,
            "curve_directory": curve_directory,
            "payload": payload,
            "completed": completed,
            "checkpoint_sha256": sha256(source_checkpoint),
            "pointer": pointer,
            "receipt": receipt,
            "journal": journal,
            "curves": curves,
            "prior": prior,
            "state_sha256": {name: _state_sha256(payload[name]) for name in _STATE_FIELDS},
        }

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination_run_root.name}.", dir=destination_run_root.parent)
    )
    try:
        shutil.copyfile(new_plan_path, temporary_root / "PRODUCER_PLAN.json")
        shutil.copyfile(
            optimization_admission_path,
            temporary_root / "RUNTIME_OPTIMIZATION_ADMISSION.json",
        )
        fit_receipts: dict[str, dict[str, Any]] = {}
        for fit_id, item in prepared.items():
            fit_directory = temporary_root / "fits" / fit_id
            versions = fit_directory / "checkpoint_versions"
            versions.mkdir(parents=True)
            version_name = f"checkpoint_{item['completed']:06d}.pt"
            origin_directory = temporary_root / "RUNTIME_MIGRATION_ORIGIN" / fit_id
            origin_directory.mkdir(parents=True)
            origin = origin_directory / version_name
            shutil.copyfile(item["source_checkpoint"], origin)
            version = versions / version_name
            try:
                os.link(origin, version)
            except OSError:
                shutil.copyfile(origin, version)
            alias = fit_directory / "checkpoint_last.pt"
            try:
                os.link(version, alias)
            except OSError:
                shutil.copyfile(version, alias)
            if (
                sha256(origin) != item["checkpoint_sha256"]
                or sha256(alias) != item["checkpoint_sha256"]
            ):
                raise OSError("V3 migration rewrote checkpoint bytes")
            for name, expected_sha in item["curves"].items():
                target = fit_directory / name
                shutil.copyfile(item["curve_directory"] / name, target)
                if sha256(target) != expected_sha:
                    raise OSError(f"V3 migration rewrote curve bytes: {name}")
            atomic_write_json(
                fit_directory / "CHECKPOINT_POINTER.json",
                {**item["pointer"], "version": version_name},
            )
            atomic_write_json(
                fit_directory / "CHECKPOINT_RECEIPT.json",
                {
                    **item["receipt"],
                    "checkpoint_path": str(
                        (destination_run_root / "fits" / fit_id / "checkpoint_last.pt").resolve()
                    ),
                },
            )
            atomic_write_json(fit_directory / "UPDATE_JOURNAL.json", item["journal"])
            destination_evidence = {
                "pointer": sha256(fit_directory / "CHECKPOINT_POINTER.json"),
                "receipt": sha256(fit_directory / "CHECKPOINT_RECEIPT.json"),
                "journal": sha256(fit_directory / "UPDATE_JOURNAL.json"),
                "curves": {name: sha256(fit_directory / name) for name in item["curves"]},
                "version": sha256(version),
                "alias": sha256(alias),
            }
            fit_receipt: dict[str, Any] = {
                "schema": FIT_MIGRATION_SCHEMA,
                "status": "MIGRATED_NOT_RESUMED",
                "fit_id": fit_id,
                "completed_updates": item["completed"],
                "scientific_optimizer_updates": 0,
                "scientific_endpoint": False,
                "original_scientific_identity_sha256": item["payload"]["identity_sha256"],
                "parent_checkpoint": {
                    "path": str(item["source_checkpoint"]),
                    "sha256": item["checkpoint_sha256"],
                },
                "migrated_checkpoint": {
                    "path": str(
                        (
                            destination_run_root
                            / "RUNTIME_MIGRATION_ORIGIN"
                            / fit_id
                            / version_name
                        ).resolve()
                    ),
                    "sha256": sha256(origin),
                },
                "initial_resume_checkpoint": {
                    "path": str(
                        (
                            destination_run_root
                            / "fits"
                            / fit_id
                            / "checkpoint_versions"
                            / version_name
                        ).resolve()
                    ),
                    "sha256": sha256(version),
                },
                "state_sha256": item["state_sha256"],
                "prior_runtime_lineage": item["prior"],
                "source_evidence_sha256": {
                    "pointer": sha256(item["source_directory"] / "CHECKPOINT_POINTER.json"),
                    "receipt": sha256(item["source_directory"] / "CHECKPOINT_RECEIPT.json"),
                    "journal": sha256(item["source_directory"] / "UPDATE_JOURNAL.json"),
                    "curves": item["curves"],
                },
                "destination_evidence_sha256": destination_evidence,
            }
            fit_receipt["fit_migration_sha256"] = canonical_sha256(fit_receipt)
            atomic_write_json(fit_directory / "RUNTIME_MIGRATION_V3.json", fit_receipt)
            fit_receipts[fit_id] = fit_receipt
        batch: dict[str, Any] = {
            "schema": BATCH_MIGRATION_SCHEMA,
            "status": "MIGRATED_NOT_RESUMED",
            "scientific_optimizer_updates": 0,
            "scientific_endpoint": False,
            "old_plan_sha256": old_plan["plan_sha256"],
            "new_plan_sha256": new_plan["plan_sha256"],
            "optimization_admission_sha256": admission_file_sha,
            "optimization_admission_identity_sha256": admission["admission_sha256"],
            "fits": {
                fit_id: {
                    "fit_migration_sha256": receipt["fit_migration_sha256"],
                    "original_scientific_identity_sha256": receipt[
                        "original_scientific_identity_sha256"
                    ],
                    "completed_updates": receipt["completed_updates"],
                    "parent_checkpoint_sha256": receipt["parent_checkpoint"]["sha256"],
                    "prior_runtime_sidecar_binding_sha256": (
                        None
                        if receipt["prior_runtime_lineage"] is None
                        else receipt["prior_runtime_lineage"][
                            "prior_runtime_sidecar_binding_sha256"
                        ]
                    ),
                    "parent_embedded_execution_binding_sha256": (
                        None
                        if receipt["prior_runtime_lineage"] is None
                        else receipt["prior_runtime_lineage"][
                            "parent_embedded_execution_binding_sha256"
                        ]
                    ),
                }
                for fit_id, receipt in fit_receipts.items()
            },
        }
        batch["batch_migration_sha256"] = canonical_sha256(batch)
        atomic_write_json(temporary_root / "RUNTIME_MIGRATION_BATCH.json", batch)
        os.replace(temporary_root, destination_run_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return validate_runtime_batch_migration(
        destination_run_root / "RUNTIME_MIGRATION_BATCH.json",
        new_plan_path=destination_run_root / "PRODUCER_PLAN.json",
        optimization_admission_path=destination_run_root / "RUNTIME_OPTIMIZATION_ADMISSION.json",
    )


def validate_runtime_batch_migration(
    path: Path,
    *,
    new_plan_path: Path,
    optimization_admission_path: Path,
) -> dict[str, Any]:
    """Rehash the complete two-fit v3 migration before a runner consumes it."""
    path = path.resolve(strict=True)
    value = read_json_shared(path)
    without_hash = dict(value)
    claimed = without_hash.pop("batch_migration_sha256", None)
    fits = value.get("fits")
    if (
        value.get("schema") != BATCH_MIGRATION_SCHEMA
        or value.get("status") != "MIGRATED_NOT_RESUMED"
        or claimed != canonical_sha256(without_hash)
        or value.get("scientific_optimizer_updates") != 0
        or value.get("scientific_endpoint") is not False
        or not isinstance(fits, dict)
        or set(fits) != {"R_A5", "R_C2F"}
    ):
        raise ValueError("Invalid v3 batch migration receipt")
    plan = _validate_plan(new_plan_path.resolve(strict=True))
    admission_path = optimization_admission_path.resolve(strict=True)
    if (
        plan.get("plan_sha256") != value.get("new_plan_sha256")
        or sha256(admission_path) != value.get("optimization_admission_sha256")
        or plan["input_sha256"].get("runtime_optimization_admission")
        != value.get("optimization_admission_sha256")
    ):
        raise ValueError("V3 batch plan or admission binding differs")
    root = path.parent
    for fit_id, summary in fits.items():
        fit_path = root / "fits" / fit_id / "RUNTIME_MIGRATION_V3.json"
        receipt = read_json_shared(fit_path.resolve(strict=True))
        receipt_without_hash = dict(receipt)
        receipt_claimed = receipt_without_hash.pop("fit_migration_sha256", None)
        parent = cast(Mapping[str, Any], receipt.get("parent_checkpoint", {}))
        migrated = cast(Mapping[str, Any], receipt.get("migrated_checkpoint", {}))
        if (
            receipt.get("schema") != FIT_MIGRATION_SCHEMA
            or receipt.get("status") != "MIGRATED_NOT_RESUMED"
            or receipt.get("fit_id") != fit_id
            or receipt.get("scientific_optimizer_updates") != 0
            or receipt.get("scientific_endpoint") is not False
            or receipt_claimed != canonical_sha256(receipt_without_hash)
            or receipt_claimed != summary.get("fit_migration_sha256")
            or receipt.get("original_scientific_identity_sha256")
            != summary.get("original_scientific_identity_sha256")
            or parent.get("sha256") != summary.get("parent_checkpoint_sha256")
            or sha256(Path(str(parent.get("path"))).resolve(strict=True)) != parent.get("sha256")
            or sha256(Path(str(migrated.get("path"))).resolve(strict=True))
            != migrated.get("sha256")
            or migrated.get("sha256") != parent.get("sha256")
        ):
            raise ValueError(f"Invalid v3 fit migration receipt: {fit_id}")
        payload = _load_checkpoint(Path(str(migrated["path"])))
        states = cast(Mapping[str, str], receipt.get("state_sha256", {}))
        if (
            payload.get("identity_sha256") != receipt["original_scientific_identity_sha256"]
            or set(states) != set(_STATE_FIELDS)
            or any(_state_sha256(payload[name]) != states[name] for name in _STATE_FIELDS)
        ):
            raise ValueError(f"Migrated v3 state differs: {fit_id}")
        fit_directory = root / "fits" / fit_id
        initial = cast(Mapping[str, Any], receipt.get("initial_resume_checkpoint", {}))
        initial_path = Path(str(initial.get("path")))
        alias = fit_directory / "checkpoint_last.pt"
        active_pointer = read_json_shared(fit_directory / "CHECKPOINT_POINTER.json")
        active_version = (
            fit_directory / "checkpoint_versions" / str(active_pointer.get("version"))
        ).resolve(strict=True)
        active_payload = _load_checkpoint(active_version)
        active_completed = int(active_payload.get("completed_updates", -1))
        migrated_completed = int(receipt["completed_updates"])
        versions_directory = (fit_directory / "checkpoint_versions").resolve(strict=True)
        if (
            active_version.parent != versions_directory
            or active_version.name != f"checkpoint_{active_completed:06d}.pt"
        ):
            raise ValueError(f"V3 active checkpoint path is not canonical: {fit_id}")
        active_hash = sha256(active_version)
        if (
            active_pointer.get("schema") != "rgb_port_checkpoint_pointer_v1"
            or active_pointer.get("identity_sha256")
            != receipt["original_scientific_identity_sha256"]
            or int(active_pointer.get("completed_updates", -1)) != active_completed
            or active_pointer.get("checkpoint_sha256") != active_hash
        ):
            raise ValueError(f"V3 active checkpoint pointer differs: {fit_id}")
        observed_curve_hashes = {
            curve.name: sha256(curve) for curve in sorted(fit_directory.glob("curve_*.json"))
        }
        source_curves = cast(
            Mapping[str, str],
            cast(Mapping[str, Any], receipt["source_evidence_sha256"])["curves"],
        )
        destination_evidence = receipt.get("destination_evidence_sha256")
        prior = receipt.get("prior_runtime_lineage")
        prior_sidecar, parent_embedded = _prior_bindings_from_summary(summary)
        if (
            initial.get("sha256") != parent.get("sha256")
            or active_completed < migrated_completed
            or active_payload.get("identity_sha256")
            != receipt["original_scientific_identity_sha256"]
            or any(
                observed_curve_hashes.get(name) != digest for name, digest in source_curves.items()
            )
        ):
            raise ValueError(f"V3 destination evidence changed: {fit_id}")
        initial_exact = bool(
            active_completed == migrated_completed
            and initial_path.is_file()
            and initial_path.resolve().parent == versions_directory
            and initial_path.name == f"checkpoint_{migrated_completed:06d}.pt"
            and sha256(initial_path) == parent.get("sha256")
            and initial_path.resolve() == active_version
            and alias.is_file()
            and destination_evidence
            == {
                "pointer": sha256(fit_directory / "CHECKPOINT_POINTER.json"),
                "receipt": sha256(fit_directory / "CHECKPOINT_RECEIPT.json"),
                "journal": sha256(fit_directory / "UPDATE_JOURNAL.json"),
                "curves": observed_curve_hashes,
                "version": active_hash,
                "alias": sha256(alias),
            }
        )
        if initial_exact:
            _validate_curve_coverage(fit_directory, active_payload)
        else:
            binding_path = fit_directory / "RUNTIME_RESUME_BINDING.json"
            binding = read_json_shared(binding_path.resolve(strict=True))
            binding_without_hash = dict(binding)
            binding_sha = binding_without_hash.pop("binding_sha256", None)
            journal = read_json_shared(fit_directory / "UPDATE_JOURNAL.json")
            journal_completed = int(journal.get("completed_updates", -1))
            journal_durable = int(journal.get("durable_updates", -1))
            journal_pending = int(journal.get("pending_update_upper", -1))
            journal_recovery = int(journal.get("recovery_upper", -1))
            recovery_after_restore = (
                max(int(active_payload["recovery_upper"]), journal_recovery)
                + max(0, journal_completed - active_completed)
                + journal_pending
            )
            destination_curves = _validate_curve_coverage(
                fit_directory,
                active_payload,
                recoverable_upper=journal_completed + journal_pending,
            )
            allowed_bindings = {binding_sha}
            if active_hash == parent.get("sha256"):
                allowed_bindings.add(parent_embedded)
            if (
                binding.get("schema") != RUNTIME_BINDING_V3_SCHEMA
                or binding.get("fit_id") != fit_id
                or binding_sha != canonical_sha256(binding_without_hash)
                or binding.get("batch_migration_sha256") != value["batch_migration_sha256"]
                or binding.get("fit_migration_sha256") != receipt["fit_migration_sha256"]
                or active_payload.get("execution_binding_sha256") not in allowed_bindings
                or active_pointer.get("execution_binding_sha256") not in allowed_bindings
                or journal.get("schema") != "rgb_port_update_journal_v1"
                or journal.get("identity_sha256") != receipt["original_scientific_identity_sha256"]
                or journal.get("execution_binding_sha256") not in allowed_bindings
                or journal_completed < active_completed
                or journal_completed - active_completed + journal_pending > CHECKPOINT_INTERVAL
                or journal_durable < 0
                or journal_durable > active_completed
                or journal_pending not in {0, 1}
                or journal_recovery < 0
                or recovery_after_restore > 11_092
                or destination_curves != observed_curve_hashes
            ):
                raise ValueError(f"V3 recoverable descendant differs: {fit_id}")
        if (prior is None) != (prior_sidecar is None):
            raise ValueError(f"Prior binding summary differs: {fit_id}")
        if prior is not None and (
            sha256(Path(str(prior["path"])).resolve(strict=True)) != prior["sha256"]
            or prior.get("prior_runtime_sidecar_binding_sha256", prior.get("binding_sha256"))
            != prior_sidecar
            or prior.get("parent_embedded_execution_binding_sha256", prior.get("binding_sha256"))
            != parent_embedded
            or sha256(Path(str(prior["migration_path"])).resolve(strict=True))
            != prior["migration_file_sha256"]
        ):
            raise ValueError(f"Prior runtime lineage changed: {fit_id}")
    return value


def resolve_runtime_checkpoint_identity_v3(
    *,
    run_root: Path,
    fit_id: str,
    new_identity: Mapping[str, Any],
    new_plan_path: Path,
) -> dict[str, Any]:
    """Bind one fit from a validated two-fit batch to its v3 execution identity."""
    run_root = run_root.resolve(strict=True)
    new_plan_path = new_plan_path.resolve(strict=True)
    batch_path = run_root / "RUNTIME_MIGRATION_BATCH.json"
    admission_path = run_root / "RUNTIME_OPTIMIZATION_ADMISSION.json"
    batch = validate_runtime_batch_migration(
        batch_path,
        new_plan_path=new_plan_path,
        optimization_admission_path=admission_path,
    )
    if fit_id not in batch["fits"]:
        raise ValueError("Fit is outside the sealed v3 batch migration")
    _validate_execution_identity(new_identity, new_plan_path=new_plan_path, fit_id=fit_id)
    fit_directory = run_root / "fits" / fit_id
    migration_path = fit_directory / "RUNTIME_MIGRATION_V3.json"
    migration = read_json_shared(migration_path)
    origin = Path(str(migration["migrated_checkpoint"]["path"])).resolve(strict=True)
    payload = _load_checkpoint(origin)
    checkpoint_identity = cast(Mapping[str, Any], payload["identity"])
    if payload.get("identity_sha256") != migration.get(
        "original_scientific_identity_sha256"
    ) or _execution_identity_without_lineage(
        checkpoint_identity
    ) != _execution_identity_without_lineage(new_identity):
        raise ValueError("V3 execution changed the checkpoint scientific identity")
    binding_path = fit_directory / "RUNTIME_RESUME_BINDING.json"
    prior_sidecar, parent_embedded = _prior_bindings_from_summary(batch["fits"][fit_id])
    value: dict[str, Any] = {
        "schema": RUNTIME_BINDING_V3_SCHEMA,
        "status": "BOUND_NOT_RESUMED",
        "fit_id": fit_id,
        "batch_migration_path": str(batch_path),
        "batch_migration_file_sha256": sha256(batch_path),
        "batch_migration_sha256": batch["batch_migration_sha256"],
        "fit_migration_path": str(migration_path),
        "fit_migration_file_sha256": sha256(migration_path),
        "fit_migration_sha256": migration["fit_migration_sha256"],
        "new_plan_path": str(new_plan_path),
        "new_plan_file_sha256": sha256(new_plan_path),
        "new_plan_sha256": batch["new_plan_sha256"],
        "optimization_admission_sha256": batch["optimization_admission_sha256"],
        "checkpoint_identity_sha256": payload["identity_sha256"],
        "execution_identity_sha256": new_identity["identity_sha256"],
        "execution_source_sha256": new_identity["executed_source_sha256"],
        "previous_execution_binding_sha256": prior_sidecar,
        "origin_embedded_execution_binding_sha256": parent_embedded,
        "optimizer_mode_unchanged": True,
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
        "scientific_optimizer_updates_during_binding": 0,
        "identity_policy": "PRESERVED_ORIGINAL_SCIENTIFIC_IDENTITY_CHAINED_RUNTIME_LINEAGE",
    }
    value["binding_sha256"] = canonical_sha256(value)
    if binding_path.exists():
        existing = read_json_shared(binding_path)
        if existing != value:
            raise ValueError("Fit is already bound to a different v3 execution identity")
    else:
        atomic_write_json(binding_path, value)
    pointer = read_json_shared(fit_directory / "CHECKPOINT_POINTER.json")
    if (
        pointer.get("identity_sha256") != payload["identity_sha256"]
        or int(pointer.get("completed_updates", -1)) < int(migration["completed_updates"])
        or pointer.get("checkpoint_sha256")
        != sha256(fit_directory / "checkpoint_versions" / str(pointer.get("version")))
    ):
        raise ValueError("Active v3 checkpoint does not descend from its migration cursor")
    return dict(checkpoint_identity)


def runtime_checkpoint_binding_kwargs_v3(*, run_root: Path, fit_id: str) -> dict[str, str]:
    """Return verified ProducerCheckpoint binding arguments for a v3 fit."""
    fit_directory = run_root.resolve(strict=True) / "fits" / fit_id
    binding = read_json_shared(fit_directory / "RUNTIME_RESUME_BINDING.json")
    without_hash = dict(binding)
    claimed = without_hash.pop("binding_sha256", None)
    if (
        binding.get("schema") != RUNTIME_BINDING_V3_SCHEMA
        or binding.get("fit_id") != fit_id
        or claimed != canonical_sha256(without_hash)
    ):
        raise ValueError("V3 runtime checkpoint binding keyword source is invalid")
    batch_path = Path(str(binding["batch_migration_path"])).resolve(strict=True)
    plan_path = Path(str(binding["new_plan_path"])).resolve(strict=True)
    if sha256(batch_path) != binding.get("batch_migration_file_sha256") or sha256(
        plan_path
    ) != binding.get("new_plan_file_sha256"):
        raise ValueError("V3 runtime binding parents changed")
    batch = validate_runtime_batch_migration(
        batch_path,
        new_plan_path=plan_path,
        optimization_admission_path=(batch_path.parent / "RUNTIME_OPTIMIZATION_ADMISSION.json"),
    )
    migration_path = fit_directory / "RUNTIME_MIGRATION_V3.json"
    migration = read_json_shared(migration_path)
    batch_summary = batch["fits"][fit_id]
    batch_prior_sidecar, batch_parent_embedded = _prior_bindings_from_summary(batch_summary)
    binding_prior_sidecar, binding_parent_embedded = _prior_bindings_from_runtime_binding(binding)
    if (
        sha256(migration_path) != binding.get("fit_migration_file_sha256")
        or migration.get("fit_migration_sha256") != binding.get("fit_migration_sha256")
        or batch["fits"][fit_id]["fit_migration_sha256"] != binding.get("fit_migration_sha256")
        or batch_prior_sidecar != binding_prior_sidecar
        or batch_parent_embedded != binding_parent_embedded
    ):
        raise ValueError("V3 fit migration changed after binding")
    result = {
        "execution_binding_sha256": str(claimed),
        "execution_binding_origin_checkpoint_sha256": str(migration["parent_checkpoint"]["sha256"]),
    }
    if binding_parent_embedded is not None:
        result["execution_binding_origin_prior_sha256"] = binding_parent_embedded
    return result


def _prior_bindings_from_summary(
    summary: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Read one exact legacy or current prior-binding summary representation."""
    legacy = "prior_execution_binding_sha256" in summary
    current_sidecar = "prior_runtime_sidecar_binding_sha256" in summary
    current_embedded = "parent_embedded_execution_binding_sha256" in summary
    if legacy and not current_sidecar and not current_embedded:
        value = cast(str | None, summary["prior_execution_binding_sha256"])
        return value, value
    if not legacy and current_sidecar and current_embedded:
        return (
            cast(str | None, summary["prior_runtime_sidecar_binding_sha256"]),
            cast(str | None, summary["parent_embedded_execution_binding_sha256"]),
        )
    raise ValueError("Prior binding summary mixes legacy and current fields")


def _prior_bindings_from_runtime_binding(
    binding: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Read one exact legacy or current runtime-binding representation."""
    if "previous_execution_binding_sha256" not in binding:
        raise ValueError("Runtime binding omits its previous execution binding")
    previous = cast(str | None, binding["previous_execution_binding_sha256"])
    if "origin_embedded_execution_binding_sha256" not in binding:
        return previous, previous
    return (
        previous,
        cast(str | None, binding["origin_embedded_execution_binding_sha256"]),
    )
