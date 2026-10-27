"""Plan and build byte-exact sparse RGB TARs for cloud downstream histories."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from PIL import Image

from e_jepa_ttc.rgb_port.data import as_list, history_observations
from operational.final_refit.scientific_oof_features import role_rows
from operational.rgb_port.accounting import atomic_write_json


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _safe_relative(value: object) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"RGB frame reference escapes its media root: {text}")
    return path.as_posix()


def _read_descriptor(descriptor_path: Path) -> tuple[Path, dict[str, Path]]:
    descriptor_path = descriptor_path.resolve(strict=True)
    value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if value.get("schema") != "final_refit_rgb_source_descriptor_v1":
        raise ValueError("RGB source descriptor schema differs")
    manifests: dict[str, Path] = {}
    media_roots: set[Path] = set()
    for role in ("P", "H", "V"):
        raw = Path(str(value["role_manifests"][role]))
        manifest_path = (raw if raw.is_absolute() else descriptor_path.parent / raw).resolve(
            strict=True
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("role") != role:
            raise ValueError(f"RGB role manifest differs: {role}")
        manifests[role] = Path(str(manifest["rows_path"])).resolve(strict=True)
        media_roots.add(Path(str(manifest["eap_root"])).resolve(strict=True))
    if len(media_roots) != 1:
        raise ValueError("RGB role manifests do not share one media root")
    return media_roots.pop(), manifests


def enumerate_references(
    descriptor_path: Path,
    fold_contract: Path,
    *,
    rotations: tuple[int, ...] = (2, 3, 4),
    roles: tuple[str, ...] = ("refiner", "generation"),
) -> tuple[Path, dict[tuple[str, str], dict[str, Any]]]:
    """Enumerate exactly the members reached through canonical RGB histories."""
    media_root, row_paths = _read_descriptor(descriptor_path)
    native_parts: list[pd.DataFrame] = []
    for historical_role, path in row_paths.items():
        frame = pd.read_parquet(path)
        frame["_historical_role"] = historical_role
        native_parts.append(frame)
    native = pd.concat(native_parts, ignore_index=True).sort_values(
        "source_ordinal", kind="stable"
    )
    ordinals = native.source_ordinal.to_numpy(np.int64)
    if len(native) != 88_744 or not np.array_equal(ordinals, np.arange(88_744)):
        raise ValueError("P/H/V rows are not the exact native TRAIN40 ordinal union")
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for rotation in rotations:
        for role in roles:
            selected = role_rows(fold_contract.resolve(strict=True), rotation, role)
            for ordinal in selected.source_ordinal.to_numpy(np.int64):
                row = native.iloc[int(ordinal)].to_dict()
                for observation in history_observations(row):
                    shards = list(map(_safe_relative, as_list(observation["producer_shards"])))
                    members = list(map(_safe_relative, as_list(observation["producer_members"])))
                    if len(shards) not in {2, 3} or len(shards) != len(members):
                        raise ValueError("Canonical RGB history has invalid frame references")
                    for shard, member in zip(shards, members, strict=True):
                        key = (shard, member)
                        item = references.setdefault(
                            key,
                            {
                                "tar": shard,
                                "member": member,
                                "rotations": set(),
                                "roles": set(),
                            },
                        )
                        item["rotations"].add(rotation)
                        item["roles"].add(role)
    return media_root, references


def plan_supplement(
    descriptor_path: Path,
    fold_contract: Path,
    output: Path,
    *,
    rotations: tuple[int, ...] = (2, 3, 4),
    roles: tuple[str, ...] = ("refiner", "generation"),
) -> dict[str, Any]:
    """Scan TAR headers and write an exact member/size plan without copying pixels."""
    if not rotations or set(rotations) - {2, 3, 4} or not roles or set(roles) - {
        "refiner",
        "generation",
    }:
        raise ValueError("RGB supplement scope must use registered rotations and roles")
    media_root, references = enumerate_references(
        descriptor_path, fold_contract, rotations=rotations, roles=roles
    )
    by_tar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in references.values():
        by_tar[str(item["tar"])].append(item)
    tar_records: list[dict[str, Any]] = []
    payload_bytes = 0
    for relative in sorted(by_tar):
        source = (media_root / relative).resolve(strict=True)
        if not source.is_relative_to(media_root):
            raise ValueError("RGB TAR escapes its media root")
        members: list[dict[str, Any]] = []
        with tarfile.open(source, mode="r") as archive:
            for item in sorted(by_tar[relative], key=lambda value: str(value["member"])):
                info = archive.getmember(str(item["member"]))
                if not info.isfile() or info.size <= 0:
                    raise ValueError(f"RGB member is not a nonempty regular file: {info.name}")
                payload_bytes += info.size
                members.append(
                    {
                        "member": info.name,
                        "payload_bytes": info.size,
                        "rotations": sorted(item["rotations"]),
                        "roles": sorted(item["roles"]),
                    }
                )
        tar_records.append(
            {
                "relative": relative,
                "original_tar_bytes": source.stat().st_size,
                "members": members,
            }
        )
    payload = {
        "schema": "ejepa_rgb_frame_supplement_plan_v1",
        "status": "PLANNED_HEADERS_VERIFIED",
        "descriptor_sha256": _sha256(descriptor_path),
        "fold_contract_sha256": _sha256(fold_contract),
        "rotations": list(rotations),
        "roles": list(roles),
        "logical_media_root": "drives/E/eAP_dataset",
        "source_media_root": str(media_root),
        "tar_count": len(tar_records),
        "member_count": len(references),
        "unique_payload_bytes": payload_bytes,
        "minimum_sparse_tar_bytes": payload_bytes
        + sum((len(record["members"]) + 2) * 512 for record in tar_records),
        "tars": tar_records,
        "pixels_read": False,
        "optimizer_updates": 0,
    }
    payload["receipt_sha256"] = _canonical(payload)
    atomic_write_json(output, payload)
    return payload


class _HashingReader(io.RawIOBase):
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block


def _decoded_identity(payload: bytes) -> tuple[list[int], str]:
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return list(pixels.shape), hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def build_supplement(plan_path: Path, destination_root: Path, output: Path) -> dict[str, Any]:
    """Copy only planned compressed frame bytes into path-compatible sparse TARs."""
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_payload = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if (
        plan.get("schema") != "ejepa_rgb_frame_supplement_plan_v1"
        or plan.get("status") != "PLANNED_HEADERS_VERIFIED"
        or plan.get("receipt_sha256") != _canonical(plan_payload)
    ):
        raise ValueError("RGB supplement plan is invalid or changed")
    source_root = Path(str(plan["source_media_root"])).resolve(strict=True)
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for tar_record in plan["tars"]:
        relative = _safe_relative(tar_record["relative"])
        source = (source_root / relative).resolve(strict=True)
        destination = (destination_root / relative).resolve()
        if not source.is_relative_to(source_root) or not destination.is_relative_to(
            destination_root
        ):
            raise ValueError("RGB sparse TAR path escapes an authorized root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = destination.with_suffix(destination.suffix + ".json")
        if destination.exists() and receipt_path.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            existing_payload = {
                key: value for key, value in existing.items() if key != "receipt_sha256"
            }
            if (
                existing.get("schema") != "ejepa_rgb_sparse_tar_receipt_v1"
                or existing.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
                or existing.get("relative") != relative
                or existing.get("sparse_tar_bytes") != destination.stat().st_size
                or existing.get("sparse_tar_sha256") != _sha256(destination)
                or existing.get("receipt_sha256") != _canonical(existing_payload)
            ):
                raise ValueError(f"Existing RGB sparse TAR receipt differs: {relative}")
            records.append(existing)
            continue
        if destination.exists():
            members = []
            with tarfile.open(source, mode="r") as original, tarfile.open(
                destination, mode="r"
            ) as sparse:
                expected_names = {str(item["member"]) for item in tar_record["members"]}
                if set(sparse.getnames()) != expected_names:
                    raise ValueError(f"Interrupted sparse RGB TAR membership differs: {relative}")
                for member_index, planned in enumerate(tar_record["members"]):
                    name = str(planned["member"])
                    original_file = original.extractfile(name)
                    sparse_file = sparse.extractfile(name)
                    if original_file is None or sparse_file is None:
                        raise FileNotFoundError(f"Interrupted sparse RGB member absent: {name}")
                    with original_file, sparse_file:
                        original_bytes, sparse_bytes = original_file.read(), sparse_file.read()
                    if original_bytes != sparse_bytes or len(original_bytes) != int(
                        planned["payload_bytes"]
                    ):
                        raise ValueError(f"Interrupted sparse RGB payload differs: {name}")
                    member = {
                        **planned,
                        "compressed_payload_sha256": hashlib.sha256(original_bytes).hexdigest(),
                    }
                    if member_index == 0:
                        source_shape, source_pixel_sha = _decoded_identity(original_bytes)
                        sparse_shape, sparse_pixel_sha = _decoded_identity(sparse_bytes)
                        if (source_shape, source_pixel_sha) != (
                            sparse_shape,
                            sparse_pixel_sha,
                        ):
                            raise ValueError(
                                f"Interrupted sparse RGB decoder parity differs: {name}"
                            )
                        member.update(
                            decoded_rgb_shape=source_shape,
                            decoded_rgb_sha256=source_pixel_sha,
                            decoded_pixel_parity_sample=True,
                        )
                    members.append(member)
            recovered = {
                "schema": "ejepa_rgb_sparse_tar_receipt_v1",
                "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
                "plan_sha256": _sha256(plan_path),
                "relative": relative,
                "original_tar_bytes": source.stat().st_size,
                "sparse_tar_bytes": destination.stat().st_size,
                "sparse_tar_sha256": _sha256(destination),
                "members": members,
                "payload_recompressed": False,
                "decoded_pixel_parity": True,
                "decoded_pixel_parity_basis": (
                    "all compressed member bytes identical plus one deterministic decode per TAR"
                ),
                "decoded_pixel_samples": 1,
                "recovered_after_interrupted_postwrite_verification": True,
            }
            recovered["receipt_sha256"] = _canonical(recovered)
            atomic_write_json(receipt_path, recovered)
            records.append(recovered)
            continue
        pending = destination.with_name(f".{destination.name}.{os.getpid()}.pending")
        if destination.exists() or pending.exists():
            raise FileExistsError(f"Refusing to replace RGB supplement TAR: {destination}")
        members: list[dict[str, Any]] = []
        with tarfile.open(source, mode="r") as original, tarfile.open(pending, mode="w") as sparse:
            for member_index, planned in enumerate(tar_record["members"]):
                info = original.getmember(str(planned["member"]))
                extracted = original.extractfile(info)
                if extracted is None:
                    raise FileNotFoundError(f"RGB member disappeared: {info.name}")
                with extracted:
                    encoded = extracted.read()
                if len(encoded) != int(planned["payload_bytes"]):
                    raise ValueError(f"RGB member size changed: {info.name}")
                encoded_sha = hashlib.sha256(encoded).hexdigest()
                sparse.addfile(info, io.BytesIO(encoded))
                member = {**planned, "compressed_payload_sha256": encoded_sha}
                if member_index == 0:
                    shape, pixel_sha = _decoded_identity(encoded)
                    member.update(
                        decoded_rgb_shape=shape,
                        decoded_rgb_sha256=pixel_sha,
                        decoded_pixel_parity_sample=True,
                    )
                members.append(member)
        with pending.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, destination)
        expected = {str(item["member"]): item for item in members}
        with tarfile.open(destination, mode="r") as sparse:
            if set(sparse.getnames()) != set(expected):
                raise ValueError(f"Sparse RGB TAR membership differs: {relative}")
            for name, item in expected.items():
                extracted = sparse.extractfile(name)
                if extracted is None:
                    raise FileNotFoundError(f"Sparse RGB member disappeared: {name}")
                with extracted:
                    encoded = extracted.read()
                if (
                    len(encoded) != item["payload_bytes"]
                    or hashlib.sha256(encoded).hexdigest()
                    != item["compressed_payload_sha256"]
                ):
                    raise ValueError(f"Sparse RGB compressed payload parity differs: {name}")
                if item.get("decoded_pixel_parity_sample"):
                    shape, pixel_sha = _decoded_identity(encoded)
                    if (
                        shape != item["decoded_rgb_shape"]
                        or pixel_sha != item["decoded_rgb_sha256"]
                    ):
                        raise ValueError(f"Sparse RGB decoder parity differs: {name}")
        record = {
            "schema": "ejepa_rgb_sparse_tar_receipt_v1",
            "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
            "plan_sha256": _sha256(plan_path),
            "relative": relative,
            "original_tar_bytes": source.stat().st_size,
            "sparse_tar_bytes": destination.stat().st_size,
            "sparse_tar_sha256": _sha256(destination),
            "members": members,
            "payload_recompressed": False,
            "decoded_pixel_parity": True,
            "decoded_pixel_parity_basis": (
                "all compressed member bytes identical plus one deterministic decode per TAR"
            ),
            "decoded_pixel_samples": 1,
        }
        record["receipt_sha256"] = _canonical(record)
        atomic_write_json(receipt_path, record)
        records.append(record)
    result = {
        "schema": "ejepa_rgb_frame_supplement_v1",
        "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
        "plan_sha256": _sha256(plan_path),
        "logical_media_root": plan["logical_media_root"],
        "physical_root": str(destination_root),
        "tar_count": len(records),
        "member_count": sum(len(record["members"]) for record in records),
        "sparse_tar_bytes": sum(record["sparse_tar_bytes"] for record in records),
        "compressed_payload_bytes": sum(
            member["payload_bytes"] for record in records for member in record["members"]
        ),
        "payload_recompressed": False,
        "decoded_pixel_parity": True,
        "tars": records,
        "optimizer_updates": 0,
    }
    result["receipt_sha256"] = _canonical(result)
    atomic_write_json(output, result)
    return result


__all__ = ["build_supplement", "enumerate_references", "plan_supplement"]
