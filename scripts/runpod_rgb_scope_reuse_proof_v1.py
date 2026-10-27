"""Prove one RGB media scope is byte-identical to an already materialized scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_receipt(path: Path, schema: str, status: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema") != schema
        or value.get("status") != status
        or value.get("receipt_sha256") != _canonical(payload)
    ):
        raise ValueError(f"RGB reuse input receipt changed: {path}")
    return value


def _tar_members(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for tar in plan["tars"]:
        relative = str(tar["relative"])
        if relative in result:
            raise ValueError("RGB plan has duplicate TAR paths")
        members = {str(item["member"]): int(item["payload_bytes"]) for item in tar["members"]}
        if len(members) != len(tar["members"]):
            raise ValueError("RGB plan has duplicate TAR members")
        result[relative] = members
    return result


def prove_reuse(
    target_plan_path: Path,
    physical_plan_path: Path,
    physical_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind target role labels to identical immutable encoded members without relabelling."""
    paths = [
        target_plan_path.resolve(strict=True),
        physical_plan_path.resolve(strict=True),
        physical_manifest_path.resolve(strict=True),
    ]
    target = _read_receipt(
        paths[0], "ejepa_rgb_frame_supplement_plan_v1", "PLANNED_HEADERS_VERIFIED"
    )
    physical = _read_receipt(
        paths[1], "ejepa_rgb_frame_supplement_plan_v1", "PLANNED_HEADERS_VERIFIED"
    )
    manifest = _read_receipt(
        paths[2], "ejepa_rgb_frame_supplement_v1", "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
    )
    if manifest.get("plan_sha256") != _sha256(paths[1]):
        raise ValueError("Physical RGB manifest is not bound to its physical plan")
    target_members, physical_members = _tar_members(target), _tar_members(physical)
    if target_members != physical_members:
        raise ValueError("RGB target and physical plans are not exact member/byte sets")
    records = {str(item["relative"]): item for item in manifest["tars"]}
    if records.keys() != physical_members.keys():
        raise ValueError("Physical RGB manifest TAR set differs from its plan")
    physical_evidence: list[dict[str, Any]] = []
    for relative, planned_members in physical_members.items():
        record = records[relative]
        actual = {str(item["member"]): item for item in record["members"]}
        if actual.keys() != planned_members.keys() or any(
            int(actual[name]["payload_bytes"]) != size
            or not isinstance(actual[name].get("compressed_payload_sha256"), str)
            or len(actual[name]["compressed_payload_sha256"]) != 64
            for name, size in planned_members.items()
        ):
            raise ValueError("Physical RGB member evidence differs from plan")
        physical_evidence.append(
            {
                "relative": relative,
                "sparse_tar_bytes": int(record["sparse_tar_bytes"]),
                "sparse_tar_sha256": str(record["sparse_tar_sha256"]),
                "sidecar_receipt_sha256": str(record["receipt_sha256"]),
                "member_count": len(actual),
                "encoded_member_identity_sha256": _canonical(
                    {
                        name: {
                            "bytes": int(item["payload_bytes"]),
                            "sha256": str(item["compressed_payload_sha256"]),
                        }
                        for name, item in sorted(actual.items())
                    }
                ),
            }
        )
    result = {
        "schema": "ejepa_rgb_media_scope_reuse_proof_v1",
        "status": "PASS",
        "target": {
            "rotation": int(target["rotations"][0]),
            "role": str(target["roles"][0]),
            "plan_sha256": _sha256(paths[0]),
            "plan_receipt_sha256": target["receipt_sha256"],
        },
        "physical": {
            "rotation": int(physical["rotations"][0]),
            "role": str(physical["roles"][0]),
            "plan_sha256": _sha256(paths[1]),
            "plan_receipt_sha256": physical["receipt_sha256"],
            "manifest_sha256": _sha256(paths[2]),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
        },
        "same_logical_tar_paths": True,
        "same_exact_member_names_and_payload_bytes": True,
        "physical_sidecar_roles_relabelled": False,
        "target_role_authority_source": "target_plan_only",
        "encoded_payload_authority_source": "physical_manifest_and_unchanged_sidecars",
        "tar_count": len(physical_evidence),
        "member_count": sum(item["member_count"] for item in physical_evidence),
        "physical_evidence": physical_evidence,
        "optimizer_updates": 0,
    }
    result["receipt_sha256"] = _canonical(result)
    atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--physical-plan", type=Path, required=True)
    parser.add_argument("--physical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prove_reuse(
        args.target_plan, args.physical_plan, args.physical_manifest, args.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
