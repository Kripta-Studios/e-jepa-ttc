"""Verified local staging and selective routing for immutable RGB inputs.

Scientific outputs remain on their durable volume.  Only manifest-bound shard
reads may be redirected to a byte-identical local mirror; source identities,
sampler state, RNG state, checkpoints and journals are not rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json

from .bundle import safe_member, verify
from .inventory import digest, read

GIB = 1024**3
DEFAULT_RESERVE_BYTES = 12 * GIB
READINESS_SCHEMA = "ejepa_local_immutable_bundle_v1"
ROUTING_SCHEMA = "ejepa_local_rgb_routing_v1"


def _content_identity(manifest: Mapping[str, Any]) -> str:
    value = [
        {
            "relative": safe_member(str(item["relative"])),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in manifest["files"]
    ]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _copy_verified(source: Path, target: Path, size: int, sha256: str) -> None:
    before = source.stat()
    if not source.is_file() or source.is_symlink() or before.st_size != size:
        raise ValueError(f"Staging source differs from manifest: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staging", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        checksum = hashlib.sha256()
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            while block := incoming.read(8 * 1024**2):
                checksum.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Staging source changed during copy: {source}")
        if temporary.stat().st_size != size or checksum.hexdigest() != sha256:
            raise ValueError(f"Staged bytes differ from manifest: {source}")
        os.replace(temporary, target)
        target.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def stage_bundle(
    logical_root: Path,
    stage_root: Path,
    *,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict[str, Any]:
    """Copy a complete manifest-bound bundle to a new/resumable local mirror."""
    logical, stage = logical_root.resolve(strict=True), stage_root.resolve()
    if logical == stage or logical.is_relative_to(stage) or stage.is_relative_to(logical):
        raise ValueError("Logical bundle and stage must be separate non-nested directories")
    if reserve_bytes < 4 * GIB:
        raise ValueError("Local staging must retain at least 4 GiB free")
    manifest_path = logical / "BUNDLE_MANIFEST.json"
    manifest = read(manifest_path)
    files = list(manifest.get("files", []))
    if not files:
        raise ValueError("Bundle manifest contains no files")
    required = sum(int(item["bytes"]) for item in files)
    stage.parent.mkdir(parents=True, exist_ok=True)
    reusable = 0
    if stage.is_dir():
        for item in files:
            candidate = stage / safe_member(str(item["relative"]))
            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and candidate.stat().st_size == int(item["bytes"])
                and digest(candidate) == str(item["sha256"])
            ):
                reusable += int(item["bytes"])
    free = shutil.disk_usage(stage.parent).free
    if free - (required - reusable) < reserve_bytes:
        raise OSError(
            f"Insufficient local disk: free={free}, required_new={required - reusable}, "
            f"reserve={reserve_bytes}"
        )
    stage.mkdir(parents=True, exist_ok=True)
    lock = stage.parent / f".{stage.name}.STAGING.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"A staging writer is already registered: {lock}") from error
    copied = reused = 0
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        for item in files:
            relative = safe_member(str(item["relative"]))
            source, target = logical / relative, stage / relative
            size, sha256 = int(item["bytes"]), str(item["sha256"])
            if (
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == size
                and digest(target) == sha256
            ):
                reused += 1
            else:
                _copy_verified(source, target, size, sha256)
                copied += 1
        _copy_verified(
            manifest_path,
            stage / "BUNDLE_MANIFEST.json",
            manifest_path.stat().st_size,
            digest(manifest_path),
        )
        staged = verify(stage)
        receipt = {
            "schema": READINESS_SCHEMA,
            "status": "READY",
            "logical_bundle_root": str(logical),
            "stage_root": str(stage),
            "bundle_manifest_sha256": digest(stage / "BUNDLE_MANIFEST.json"),
            "content_identity_sha256": _content_identity(staged),
            "profile": staged.get("profile"),
            "files": len(files),
            "bytes": required,
            "copied_files": copied,
            "reused_files": reused,
            "reserve_bytes": reserve_bytes,
            "scientific_outputs_relocated": False,
            "checkpoint_or_journal_relocated": False,
            "sampler_or_rng_changed": False,
        }
        readiness = stage.parent / f"{stage.name}.READY.json"
        atomic_write_json(readiness, receipt)
        return {**receipt, "readiness_path": str(readiness)}
    finally:
        lock.unlink(missing_ok=True)


def readiness_barrier(
    logical_root: Path, stage_root: Path, readiness_path: Path
) -> dict[str, Any]:
    """Re-hash the complete stage and require its exact logical-root binding."""
    logical, stage = logical_root.resolve(strict=True), stage_root.resolve(strict=True)
    receipt = read(readiness_path.resolve(strict=True))
    manifest = verify(stage)
    expected = {
        "schema": READINESS_SCHEMA,
        "status": "READY",
        "logical_bundle_root": str(logical),
        "stage_root": str(stage),
        "bundle_manifest_sha256": digest(stage / "BUNDLE_MANIFEST.json"),
        "content_identity_sha256": _content_identity(manifest),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Local immutable-cache readiness binding differs")
    if digest(logical / "BUNDLE_MANIFEST.json") != expected["bundle_manifest_sha256"]:
        raise ValueError("Logical bundle manifest differs from staged mirror")
    return receipt


def _local_path(path: Path, logical: Path, stage: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(logical):
        raise ValueError(f"RGB shard is outside the admitted logical bundle: {resolved}")
    local = (stage / resolved.relative_to(logical)).resolve(strict=True)
    if not local.is_relative_to(stage) or not local.is_file() or local.is_symlink():
        raise ValueError(f"Local RGB shard is not a regular staged file: {local}")
    return local


def route_rgb_source(
    source: object,
    *,
    logical_root: Path,
    stage_root: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    """Redirect only known prepared-RGB and teacher shard records to local bytes."""
    logical, stage = logical_root.resolve(strict=True), stage_root.resolve(strict=True)
    ready = readiness_barrier(logical, stage, readiness_path)
    identity_before = json.dumps(source.identity, sort_keys=True, default=str)
    base = getattr(source, "base", None)
    datasets = getattr(base, "datasets", None)
    teacher = getattr(base, "teacher", None)
    if not isinstance(datasets, Mapping) or teacher is None:
        raise TypeError("Local RGB routing requires a constructed fold RGB source")
    routed: list[dict[str, str]] = []
    rgb_patches: list[tuple[dict[str, Any], str]] = []
    for role, dataset in sorted(datasets.items()):
        cache = getattr(dataset, "prepared_cache", None)
        lookup = getattr(cache, "_lookup", None)
        if not isinstance(lookup, dict) or cache is None:
            raise ValueError(f"Role {role} lacks an admitted prepared RGB cache")
        unique: dict[int, dict[str, Any]] = {id(value[0]): value[0] for value in lookup.values()}
        for shard in unique.values():
            original = Path(str(shard["path"]))
            local = _local_path(original, logical, stage)
            if digest(local) != str(shard["sha256"]):
                raise ValueError(f"Local RGB shard hash mismatch: {local.name}")
            rgb_patches.append((shard, str(local)))
            routed.append({"kind": f"rgb_{role}", "logical": str(original), "local": str(local)})
    shards = getattr(teacher, "shards", None)
    if not isinstance(shards, tuple):
        raise TypeError("Teacher cache does not expose immutable shard records")
    patched_teacher = []
    for item in shards:
        original = Path(str(item["path"]))
        local = _local_path(original, logical, stage)
        if digest(local) != str(item["sha256"]):
            raise ValueError(f"Local teacher shard hash mismatch: {local.name}")
        patched_teacher.append({**item, "path": str(local)})
        routed.append({"kind": "teacher", "logical": str(original), "local": str(local)})
    for shard, local in rgb_patches:
        shard["path"] = local
    teacher.shards = tuple(patched_teacher)
    identity_after = json.dumps(source.identity, sort_keys=True, default=str)
    if identity_after != identity_before:
        raise RuntimeError("Physical RGB routing changed scientific source identity")
    mapping_payload = [
        {"kind": item["kind"], "logical": item["logical"], "local": item["local"]}
        for item in routed
    ]
    return {
        "schema": ROUTING_SCHEMA,
        "status": "READY_FOR_NUMERIC_PROOF",
        "logical_bundle_root": str(logical),
        "stage_root": str(stage),
        "readiness_sha256": digest(readiness_path),
        "content_identity_sha256": ready["content_identity_sha256"],
        "routes": len(routed),
        "route_map_sha256": hashlib.sha256(
            json.dumps(mapping_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scientific_source_identity_unchanged": True,
        "sampler_or_rng_changed": False,
        "checkpoint_or_journal_relocated": False,
        "numeric_proof_required_before_scientific_use": True,
    }


def _clear_rgb_resident_state(source: object) -> None:
    """Force a proof read through the currently routed physical paths."""
    base = source.base
    for dataset in base.datasets.values():
        cache = dataset.prepared_cache
        cache._loaded.clear()
        cache._loaded_bytes = 0
    base.teacher._loaded.clear()
    base.teacher._sizes.clear()
    base.teacher.retained_bytes = 0


def prove_rgb_route(
    source: object,
    *,
    logical_root: Path,
    stage_root: Path,
    readiness_path: Path,
    identity_function: Callable[[object], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require a bit-exact CPU batch identity before admitting local reads."""
    import random

    import numpy as np
    import torch

    if identity_function is None:
        from .runner import batch_identity

        identity_function = batch_identity
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    _clear_rgb_resident_state(source)
    logical_identity = identity_function(source)
    routing = route_rgb_source(
        source,
        logical_root=logical_root,
        stage_root=stage_root,
        readiness_path=readiness_path,
    )
    _clear_rgb_resident_state(source)
    local_identity = identity_function(source)
    numpy_after = np.random.get_state()
    rng_unchanged = (
        random.getstate() == python_state
        and numpy_after[0] == numpy_state[0]
        and np.array_equal(numpy_after[1], numpy_state[1])
        and numpy_after[2:] == numpy_state[2:]
        and torch.equal(torch.get_rng_state(), torch_state)
    )
    if logical_identity != local_identity or not rng_unchanged:
        raise ValueError("Local RGB routing failed bit-exact input/RNG proof")
    proof = {
        "schema": "ejepa_local_rgb_numeric_proof_v1",
        "status": "PASS",
        "routing": routing,
        "batch_identity": local_identity,
        "logical_vs_local_bit_exact": True,
        "python_numpy_torch_rng_unchanged": True,
        "optimizer_updates": 0,
        "scientific_outputs_relocated": False,
    }
    proof["proof_sha256"] = hashlib.sha256(
        json.dumps(proof, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "check"):
        child = subparsers.add_parser(name)
        child.add_argument("--logical-bundle", type=Path, required=True)
        child.add_argument("--stage", type=Path, required=True)
        child.add_argument("--readiness", type=Path)
    subparsers.choices["stage"].add_argument(
        "--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES
    )
    args = parser.parse_args()
    readiness = args.readiness or args.stage.parent / f"{args.stage.name}.READY.json"
    result = (
        stage_bundle(args.logical_bundle, args.stage, reserve_bytes=args.reserve_bytes)
        if args.command == "stage"
        else readiness_barrier(args.logical_bundle, args.stage, readiness)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
