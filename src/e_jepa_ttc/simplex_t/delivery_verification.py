"""Read-only delivery verification against freshly bound scientific evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory, verify_bundle
from .delivery_documents import render_delivery_documents


def verify_delivery(
    output: Path,
    *,
    work_root: Path,
    analysis_commit: str,
    bind_verified_members: Callable[[], dict[str, BundleMember]],
    resource_ok: Callable[[], bool],
) -> dict:
    """Verify existing files without creating, repairing or deleting anything.

    As with assembly, the provider must reconstruct the full scientific graph,
    accounting and provenance via postprocessing_bundle_members. A ZIP checksum
    or a self-declared DELIVERY.json is not scientific completion evidence.
    """
    destination = output.resolve(strict=True)
    if not destination.is_relative_to(work_root.resolve(strict=True) / "artifacts"):
        raise ValueError("companion artifact delivery directory required")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: delivery verification")

    def read(path: Path) -> bytes:
        boundary()
        with path.open("rb") as stream:
            payload = stream.read(16_777_217)
        if len(payload) > 16_777_216:
            raise ValueError("delivery metadata exceeds bounded read")
        return payload

    boundary()
    original = bind_verified_members()
    validate_bundle_inventory({name: m.sha256 for name, m in original.items()})
    if any(type(m.bytes) is not int or m.bytes < 0 for m in original.values()):
        raise ValueError("exact nonnegative source byte counts required")

    def document(name: str) -> dict:
        member = original[name]
        payload = read(member.path)
        if len(payload) != member.bytes or hashlib.sha256(payload).hexdigest() != member.sha256:
            raise ValueError("delivery source document differs from scientific inventory")
        return json.loads(payload)

    report, decision = render_delivery_documents(
        document("postprocessing/SCIENTIFIC_GRAPH_COVERAGE.json"),
        document("postprocessing/CAMPAIGN_ACCOUNTING.json"),
        analysis_commit=analysis_commit,
    )

    def encoded(value: dict) -> bytes:
        return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )

    members = dict(original)
    for name, payload in (
        ("CODEX_SIMPLEX_T_FINAL_REPORT.md", report.encode("utf-8")),
        ("NEXT_DECISION_SIMPLEX_T.json", encoded(decision)),
    ):
        if name in members or read(destination / name) != payload:
            raise ValueError("delivery document differs from regenerated scientific evidence")
        members[name] = BundleMember(
            destination / name, hashlib.sha256(payload).hexdigest(), len(payload)
        )
    manifest = encoded(
        {
            "schema": "simplex_t_delivery_content_manifest_v1",
            "scope": "ARCHIVED_PAYLOADS_EXCLUDING_THIS_MANIFEST",
            "analysis_commit": analysis_commit,
            "members": {
                name: {"sha256": m.sha256, "bytes": m.bytes} for name, m in members.items()
            },
        }
    )
    if (
        "CONTENT_MANIFEST.json" in members
        or read(destination / "CONTENT_MANIFEST.json") != manifest
    ):
        raise ValueError("delivery content inventory differs from scientific evidence")
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    members["CONTENT_MANIFEST.json"] = BundleMember(
        destination / "CONTENT_MANIFEST.json", manifest_sha256, len(manifest)
    )
    archive = destination / f"E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_{analysis_commit[:12]}.zip"
    verified = verify_bundle(
        archive, {name: m.sha256 for name, m in members.items()}, resource_ok=resource_ok
    )
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while True:
            boundary()
            block = stream.read(1_048_576)
            if not block:
                break
            digest.update(block)
    archive_hash = digest.hexdigest()
    sidecar = f"{archive_hash}  {archive.name}"
    if read(archive.with_suffix(".zip.sha256")) not in {
        (sidecar + "\n").encode("ascii"),
        (sidecar + "\r\n").encode("ascii"),
    }:
        raise ValueError("delivery ZIP checksum sidecar differs")
    receipt = json.loads(read(destination / "DELIVERY.json"))
    required = 49 * 1024**2 + sum(
        2 * m.bytes + 4096 + 2 * len(name.encode("utf-8")) for name, m in original.items()
    )
    expected_bundle = {
        "status": "PINNED_BUNDLE_CREATED_AND_VERIFIED_NOT_SCIENTIFIC_COMPLETION",
        "path": str(archive),
        "sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "members": verified["members"],
        "uncompressed_bytes": verified["uncompressed_bytes"],
        "optimizer_updates": 0,
    }
    if (
        receipt.get("schema") != "simplex_t_delivery_assembly_v1"
        or receipt.get("status") != "DECLARED_SCIENTIFIC_PAYLOAD_TRANSPORT_VERIFIED"
        or receipt.get("bundle") != expected_bundle
        or receipt.get("content_manifest_sha256") != manifest_sha256
        or receipt.get("optimizer_updates_executed") != 0
        or receipt.get("campaign_complete") is not False
        or receipt.get("required_reservation_bound_bytes") != required
        or type(receipt.get("reserved_output_bytes")) is not int
        or receipt["reserved_output_bytes"] < required
    ):
        raise ValueError("delivery receipt differs from verified payload and reservation")
    boundary()
    if bind_verified_members() != original:
        raise ValueError("scientific delivery inventory changed during verification")
    return {
        "status": "DELIVERY_REVERIFIED_AGAINST_BOUND_SCIENTIFIC_EVIDENCE",
        "sha256": archive_hash,
        "archive": str(archive),
        "members": verified["members"],
        "optimizer_updates_executed": 0,
        "files_written": 0,
    }
