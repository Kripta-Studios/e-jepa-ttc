"""Bind a sealed postprocessing publication to explicit transport members."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory
from .phase_bundle import phase_bundle_members
from .postprocessing_inventory import inventory_postprocessing
from .stage_gate import CanonicalPublication


def postprocessing_bundle_members(
    manifest: Path,
    *,
    manifest_sha256: str,
    freeze_sha256: str,
    work_root: Path,
    phases: dict[str, CanonicalPublication],
    verify_completed_graph: Callable[[], dict],
    validate_scientific_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Rehash all owned outputs and include the pinned manifest in the archive.

    This supplies postprocessing and all registered phase publications. The
    caller must separately include indices, provenance, accounting
    and reports, and repeat this validation in the ZIP authority callback.
    No fitting, output writes or campaign-completion inference occurs here.
    """
    validate_scientific_authority()
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: postprocessing bundle binding")
    if manifest.name != "POSTPROCESSING.json" or manifest.is_symlink():
        raise ValueError("canonical postprocessing manifest required")
    work = work_root.resolve(strict=True)
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
    validate_bundle_inventory({name: member.sha256 for name, member in members.items()})
    return members
