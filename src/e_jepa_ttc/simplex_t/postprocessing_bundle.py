"""Bind a sealed postprocessing publication to explicit transport members."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory
from .campaign_accounting import AccountingPins, verify_campaign_accounting
from .history_bundle import HistoryPoolPins, history_bundle_members
from .phase_bundle import phase_bundle_members
from .postprocessing_inventory import inventory_postprocessing
from .provenance_bundle import provenance_bundle_members
from .resource_bundle import ResourceAttempt, resource_bundle_members
from .stage_gate import CanonicalPublication
from .technical_bundle import technical_bundle_members


def postprocessing_bundle_members(
    manifest: Path,
    *,
    manifest_sha256: str,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    work_root: Path,
    phases: dict[str, CanonicalPublication],
    history_pools: dict[str, HistoryPoolPins],
    accounting_pins: AccountingPins,
    resource_attempts: list[ResourceAttempt],
    verify_completed_graph: Callable[[], dict],
    validate_scientific_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Rehash all owned outputs and include the pinned manifest in the archive.

    This supplies postprocessing and all registered phase publications. The
    caller must separately include final reports and repeat this validation in
    the ZIP authority callback. Resource receipts cover declared observed
    attempts, not an exhaustive wall-time history after hard terminations.
    No fitting, output writes or campaign-completion inference occurs here.
    """
    validate_scientific_authority()
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: postprocessing bundle binding")
    if manifest.name != "POSTPROCESSING.json" or manifest.is_symlink():
        raise ValueError("canonical postprocessing manifest required")
    work = work_root.resolve(strict=True)
    if roots["work"].resolve(strict=True) != work:
        raise ValueError("provenance and publication work roots differ")
    if not manifest.resolve(strict=True).is_relative_to(work):
        raise ValueError("postprocessing manifest outside companion worktree")
    for binding in phases.values():
        if any(
            not path.resolve(strict=True).is_relative_to(work)
            for path in (binding.publication, binding.endpoints, binding.checkpoint_root)
        ):
            raise ValueError("phase publication outside companion worktree")
    # Administrative JSON only; cap allocation independently of host RAM size.
    with manifest.open("rb") as stream:
        payload = stream.read(16_777_217)
    if len(payload) > 16_777_216 or hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError("postprocessing manifest size or SHA256 differs")
    document = json.loads(payload)
    if (
        document.get("schema") != "simplex_t_campaign_postprocessing_v1"
        or document.get("freeze_sha256") != freeze_sha256
        or document.get("status")
        != "SEALED_ANALYSES_WEIGHTS_INTERFACE_ASSEMBLED_NOT_TRANSPORT_DELIVERY"
    ):
        raise ValueError("postprocessing schema, freeze or status differs")
    actual = inventory_postprocessing(
        manifest.parent, resource_ok=resource_ok, sealed_manifest_present=True
    )
    if actual != document.get("output_inventory"):
        raise ValueError("postprocessing output inventory differs")
    provenance_members = provenance_bundle_members(
        freeze,
        freeze_sha256=freeze_sha256,
        roots=roots,
        validate_scientific_authority=validate_scientific_authority,
        resource_ok=resource_ok,
    )
    resource_members = resource_bundle_members(
        resource_attempts,
        work_root=work,
        freeze_sha256=freeze_sha256,
        completed_stages=set(phases),
        resource_ok=resource_ok,
    )
    accounting = verify_campaign_accounting(
        **asdict(accounting_pins),
        freeze_sha256=freeze_sha256,
        phases=phases,
        work_root=work,
        verify_completed_graph=verify_completed_graph,
        resource_ok=resource_ok,
    )
    accounting_name = "CAMPAIGN_ACCOUNTING.json"
    pin = actual["members"].get(accounting_name)
    if (
        pin is None
        or pin["sha256"] != document.get("campaign_accounting_sha256")
        or document.get("optimizer_work_accounting_verified") is not True
        or pin["bytes"] > 8_388_608
    ):
        raise ValueError("postprocessing requires the bound optimizer accounting receipt")
    if json.loads((manifest.parent / accounting_name).read_text(encoding="utf-8")) != accounting:
        raise ValueError("postprocessing accounting differs from actual work evidence")
    accounting_members = {}
    for name in ("journal", "reconciliation", "ledger"):
        path = getattr(accounting_pins, name).resolve(strict=True)
        digest = getattr(accounting_pins, name + "_sha256")
        if not path.is_relative_to(work):
            raise ValueError("accounting evidence outside companion worktree")
        # The real accounting reader has just verified these exact bytes. ZIP
        # construction also rehashes every member and repeats this authority check.
        accounting_members[f"accounting/{name}.json"] = BundleMember(
            path, digest, path.stat().st_size
        )
    technical_members = technical_bundle_members(
        accounting_pins.reconciliation,
        reconciliation_sha256=accounting_pins.reconciliation_sha256,
        ledger=accounting_pins.ledger,
        ledger_sha256=accounting_pins.ledger_sha256,
        work_root=work,
        resource_ok=resource_ok,
    )
    # Regenerate from scientific publications, never follow caller-entered paths
    # in the serialized inventory before establishing their exact correspondence.
    phase_members = phase_bundle_members(
        phases,
        freeze_sha256=freeze_sha256,
        verify_completed_graph=verify_completed_graph,
        resource_ok=resource_ok,
    )
    expected_phase_pins = {
        name: {
            "work_relative_path": member.path.resolve(strict=True).relative_to(work).as_posix(),
            "sha256": member.sha256,
            "bytes": member.bytes,
        }
        for name, member in phase_members.items()
    }
    if not phase_members or expected_phase_pins != document.get("phase_payload_inventory"):
        raise ValueError("postprocessing phase inventory differs from verified publications")
    history_members = history_bundle_members(
        history_pools,
        work_root=work,
        validate_authority=validate_scientific_authority,
        resource_ok=resource_ok,
    )
    expected_history_pins = {
        name: {
            "work_relative_path": member.path.relative_to(work).as_posix(),
            "sha256": member.sha256,
            "bytes": member.bytes,
        }
        for name, member in history_members.items()
    }
    if not history_members or expected_history_pins != document.get("history_payload_inventory"):
        raise ValueError("postprocessing history inventory differs from frozen sources")
    validate_scientific_authority()
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: postprocessing bundle binding")
    with manifest.open("rb") as stream:
        if stream.read(len(payload) + 1) != payload:
            raise ValueError("postprocessing manifest changed during binding")
    members = {
        "postprocessing/" + name: BundleMember(manifest.parent / name, pin["sha256"], pin["bytes"])
        for name, pin in actual["members"].items()
    }
    members["postprocessing/POSTPROCESSING.json"] = BundleMember(
        manifest, manifest_sha256, len(payload)
    )
    if members.keys() & phase_members.keys():
        raise ValueError("postprocessing and phase archive names collide")
    members.update(phase_members)
    if members.keys() & history_members.keys():
        raise ValueError("history and publication archive names collide")
    members.update(history_members)
    if members.keys() & accounting_members.keys():
        raise ValueError("accounting and publication archive names collide")
    members.update(accounting_members)
    if members.keys() & technical_members.keys():
        raise ValueError("technical evidence and publication archive names collide")
    members.update(technical_members)
    if members.keys() & provenance_members.keys():
        raise ValueError("provenance and publication archive names collide")
    members.update(provenance_members)
    if members.keys() & resource_members.keys():
        raise ValueError("resource and publication archive names collide")
    members.update(resource_members)
    validate_bundle_inventory({name: member.sha256 for name, member in members.items()})
    return members
