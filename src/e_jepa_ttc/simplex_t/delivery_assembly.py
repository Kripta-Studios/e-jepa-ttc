"""Assemble canonical delivery documents and a byte-verified explicit payload ZIP."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember, create_verified_bundle
from .bundle_integrity import validate_bundle_inventory
from .delivery_documents import render_delivery_documents


def assemble_delivery(
    output: Path,
    *,
    work_root: Path,
    analysis_commit: str,
    bind_verified_members: Callable[[], dict[str, BundleMember]],
    resource_ok: Callable[[], bool],
    reserved_output_bytes: int,
) -> dict:
    """Require a full scientific member provider, rechecking it around transport.

    The provider must call postprocessing_bundle_members with real admission and
    completed-graph validation. This composition cannot substitute for that
    configured authority. Partial outputs remain; retries need a new directory.
    The caller reserves all pending outputs, including the archive, before entry.
    """
    work = work_root.resolve(strict=True)
    destination = output.resolve()
    if not destination.is_relative_to(work / "artifacts") or destination.exists():
        raise ValueError("new companion artifact delivery directory required")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: delivery assembly")

    boundary()
    original = bind_verified_members()
    if type(reserved_output_bytes) is not int or reserved_output_bytes < 1:
        raise ValueError("positive explicit delivery output reservation required")
    validate_bundle_inventory({name: member.sha256 for name, member in original.items()})
    if any(type(member.bytes) is not int or member.bytes < 0 for member in original.values()):
        raise ValueError("exact nonnegative source byte counts required")

    def read(name: str) -> dict:
        boundary()
        member = original[name]
        with member.path.open("rb") as stream:
            payload = stream.read(16_777_217)
        if (
            len(payload) > 16_777_216
            or len(payload) != member.bytes
            or hashlib.sha256(payload).hexdigest() != member.sha256
        ):
            raise ValueError("delivery document input differs from verified member")
        return json.loads(payload)

    report, decision = render_delivery_documents(
        read("postprocessing/SCIENTIFIC_GRAPH_COVERAGE.json"),
        read("postprocessing/CAMPAIGN_ACCOUNTING.json"),
        analysis_commit=analysis_commit,
    )
    names = {
        "CODEX_SIMPLEX_T_FINAL_REPORT.md",
        "NEXT_DECISION_SIMPLEX_T.json",
        "CONTENT_MANIFEST.json",
    }
    if names & original.keys():
        raise ValueError("delivery documents collide with source inventory")
    # Conservative byte bound: ZIP payload expansion, entry headers/names and
    # up to 16 MiB of reports/inventory plus their archived copies. No free-space
    # percentage and no assumption that compression must shrink every input.
    required_reservation = 49 * 1024**2 + sum(
        2 * member.bytes + 4096 + 2 * len(name.encode("utf-8")) for name, member in original.items()
    )
    if reserved_output_bytes < required_reservation:
        raise ValueError(f"delivery reservation needs at least {required_reservation} bytes")
    boundary()
    destination.mkdir(parents=True)
    members = dict(original)
    metadata_bytes = 0

    def write(name: str, payload: bytes) -> None:
        nonlocal metadata_bytes
        boundary()
        metadata_bytes += len(payload)
        if metadata_bytes > 16_777_216:
            raise ValueError("delivery document exceeds reserved metadata bound")
        path = destination / name
        with path.open("xb") as stream:
            stream.write(payload)
        members[name] = BundleMember(path, hashlib.sha256(payload).hexdigest(), len(payload))

    def encoded(value: dict) -> bytes:
        return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )

    write("CODEX_SIMPLEX_T_FINAL_REPORT.md", report.encode("utf-8"))
    write("NEXT_DECISION_SIMPLEX_T.json", encoded(decision))
    write(
        "CONTENT_MANIFEST.json",
        encoded(
            {
                "schema": "simplex_t_delivery_content_manifest_v1",
                "scope": "ARCHIVED_PAYLOADS_EXCLUDING_THIS_MANIFEST",
                "analysis_commit": analysis_commit,
                "members": {
                    name: {"sha256": m.sha256, "bytes": m.bytes} for name, m in members.items()
                },
            }
        ),
    )

    def authority() -> None:
        boundary()
        if bind_verified_members() != original:
            raise ValueError("scientific delivery inventory changed")

    archive = destination / f"E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_{analysis_commit[:12]}.zip"
    result = create_verified_bundle(
        archive,
        members,
        validate_inventory_authority=authority,
        resource_ok=resource_ok,
    )
    boundary()
    with archive.with_suffix(".zip.sha256").open("x", encoding="ascii") as stream:
        stream.write(f"{result['sha256']}  {archive.name}\n")
    receipt = {
        "schema": "simplex_t_delivery_assembly_v1",
        "status": "DECLARED_SCIENTIFIC_PAYLOAD_TRANSPORT_VERIFIED",
        "bundle": result,
        "content_manifest_sha256": members["CONTENT_MANIFEST.json"].sha256,
        "optimizer_updates_executed": 0,
        "reserved_output_bytes": reserved_output_bytes,
        "required_reservation_bound_bytes": required_reservation,
        "campaign_complete": False,
        "completion_scope": "Caller must establish configured full campaign admission",
    }
    with (destination / "DELIVERY.json").open("xb") as stream:
        stream.write(encoded(receipt))
    return receipt
