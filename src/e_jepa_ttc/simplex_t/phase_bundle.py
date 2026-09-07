"""Explicit prediction-publication members for the essential-results archive."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory
from .phase_manifest import fit_key
from .registry import registered_graph
from .stage_gate import CanonicalPublication


def phase_bundle_members(
    phases: dict[str, CanonicalPublication],
    *,
    freeze_sha256: str,
    verify_completed_graph: Callable[[], dict],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Bind every registered prediction table and both phase seals, without refits.

    The callback must run the actual scientific graph verifier, including phase
    states, independent OLD targets and gates. This reader adds transport pins;
    it does not replace those scientific checks. Compact weights, history and
    accounting remain separate required archive sections.
    """
    coverage = verify_completed_graph()
    if coverage.get("freeze_sha256") != freeze_sha256 or set(
        coverage.get("phase_fit_counts", {})
    ) != set(phases):
        raise ValueError("phase transport coverage differs from verified graph")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: phase bundle inventory")

    def pin(path: Path, expected: str) -> BundleMember:
        boundary()
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                boundary()
                data = stream.read(1_048_576)
                if not data:
                    break
                size += len(data)
                if size > before.st_size:
                    raise ValueError("phase bundle source grew")
                digest.update(data)
        if size != before.st_size or digest.hexdigest() != expected:
            raise ValueError("phase bundle source differs from publication pin")
        return BundleMember(path, expected, size)

    members: dict[str, BundleMember] = {}
    for stage, binding in sorted(phases.items()):
        prefix = f"publications/{stage}/"
        members[prefix + "ENDPOINTS.json"] = pin(binding.endpoints, binding.endpoints_sha256)
        members[prefix + "PREDICTIONS.json"] = pin(binding.publication, binding.publication_sha256)
        with binding.publication.open("rb") as stream:
            payload = stream.read(8_388_609)
        if (
            len(payload) > 8_388_608
            or hashlib.sha256(payload).hexdigest() != binding.publication_sha256
        ):
            raise ValueError("phase publication metadata changed or exceeds bound")
        state = json.loads(payload)
        contract = state["contract"]
        expected_keys = {
            fit_key(spec)
            for spec in registered_graph(**binding.availability)
            if spec.stage == stage
        }
        if (
            state.get("schema") != "simplex_t_phase_predictions_v1"
            or state.get("status") != "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"
            or contract["stage"] != stage
            or contract["availability"] != binding.availability
            or contract["freeze_sha256"] != freeze_sha256
            or contract["manifest_sha256"] != binding.endpoints_sha256
            or set(state["fits"]) != expected_keys
            or len(expected_keys) != coverage["phase_fit_counts"][stage]
        ):
            raise ValueError("phase transport publication contract differs")
        root = binding.publication.parent.resolve(strict=True)
        for key in sorted(expected_keys):
            entry = state["fits"][key]
            name = f"{key}/predictions.parquet"
            if entry["path"] != name:
                raise ValueError("canonical prediction payload path required")
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("prediction payload escapes publication root")
            members[prefix + name] = pin(path, entry["sha256"])
    validate_bundle_inventory({name: entry.sha256 for name, entry in members.items()})
    if verify_completed_graph() != coverage:
        raise ValueError("scientific graph changed during transport binding")
    return members
