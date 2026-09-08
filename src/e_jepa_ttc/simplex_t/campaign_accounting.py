"""Reconcile complete scientific endpoints with physical and technical work evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .phase_inference import validated_phase
from .phase_manifest import fit_key
from .physical_accounting import audit_physical_work
from .registry import registered_graph
from .stage_gate import CanonicalPublication
from .technical_accounting import verify_technical_accounting
from .training import load_checkpoint, state_digest


@dataclass(frozen=True)
class AccountingPins:
    """Exact post-fit work evidence; reservations alone cannot certify execution."""

    journal: Path
    journal_sha256: str
    reconciliation: Path
    reconciliation_sha256: str
    ledger: Path
    ledger_sha256: str


def verify_campaign_accounting(
    journal: Path,
    *,
    journal_sha256: str,
    freeze_sha256: str,
    phases: dict[str, CanonicalPublication],
    reconciliation: Path,
    reconciliation_sha256: str,
    ledger: Path,
    ledger_sha256: str,
    work_root: Path,
    verify_completed_graph: Callable[[], dict],
    resource_ok: Callable[[], bool],
) -> dict:
    """Require exact required-fit coverage and settled checkpoints before final accounting.

    The graph callback must verify actual freeze, sources, predictions and gates.
    Full planned branches are reserved but not counted as execution. All actual
    technical receipts are checked independently; legacy limitations are retained.
    """
    coverage = verify_completed_graph()
    if coverage["freeze_sha256"] != freeze_sha256 or set(phases) != set(
        coverage["phase_fit_counts"]
    ):
        raise ValueError("accounting differs from completed scientific graph")
    work = work_root.resolve(strict=True)
    if (
        not journal.resolve(strict=True).is_relative_to(work)
        or journal.name != "PHYSICAL_WORK.json"
    ):
        raise ValueError("companion scientific physical-work journal required")
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: campaign accounting")
    with journal.open("rb") as stream:
        payload = stream.read(16_777_217)
    if len(payload) > 16_777_216 or hashlib.sha256(payload).hexdigest() != journal_sha256:
        raise ValueError("scientific journal hash or size differs")
    state = json.loads(payload)
    technical = verify_technical_accounting(
        reconciliation,
        reconciliation_sha256=reconciliation_sha256,
        ledger=ledger,
        ledger_sha256=ledger_sha256,
        work_root=work,
        resource_ok=resource_ok,
    )
    flags = coverage["resolved_availability"]
    possible = registered_graph(
        d1=flags["d1"],
        density=flags["density"],
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )
    physical = audit_physical_work(
        state,
        expected_graph={fit_key(spec): spec.updates for spec in possible},
        technical_reserved=technical["reserved_updates"],
        resource_ok=resource_ok,
    )
    required = {fit_key(spec) for spec in registered_graph(**flags)}
    if (
        set(state["fits"]) != required
        or any(
            fit["completed"] != 2500 or fit["pending"] is not None for fit in state["fits"].values()
        )
        or coverage["fits_completed"] != len(required)
        or coverage["scientific_updates_completed"] != physical["saved_updates"]
    ):
        raise ValueError("scientific journal has missing, extra or unfinished fits")
    endpoint_keys = set()
    for stage, binding in phases.items():
        if binding.checkpoint_root.resolve(strict=True) != journal.parent.resolve(strict=True):
            raise ValueError("scientific phases do not share the physical-work root")
        _, endpoints = validated_phase(
            binding.endpoints,
            binding.checkpoint_root,
            manifest_sha256=binding.endpoints_sha256,
            freeze_sha256=freeze_sha256,
            stage=stage,
            availability=binding.availability,
            resource_ok=resource_ok,
        )
        for key, endpoint in endpoints.items():
            if key in endpoint_keys or key not in required:
                raise ValueError("endpoint coverage differs from journal")
            endpoint_keys.add(key)
            if not resource_ok():
                raise InterruptedError("PAUSED_RESOURCE: accounting checkpoint identity")
            fit = state["fits"][key]
            if fit["checkpoint_sha256"] != endpoint["checkpoint_sha256"]:
                # A status-only final save may alter bytes after journal settlement.
                # Validate all training state, not just weights, in that case.
                checkpoint = load_checkpoint(endpoint["resolved_checkpoint"])
                checkpoint.pop("status", None)
                if state_digest(checkpoint) != fit.get("checkpoint_state_sha256"):
                    raise ValueError("journal and sealed endpoint training states differ")
    if endpoint_keys != required or verify_completed_graph() != coverage:
        raise ValueError("accounting endpoint set or scientific graph changed")
    with journal.open("rb") as stream:
        if stream.read(len(payload) + 1) != payload:
            raise ValueError("scientific journal changed during accounting")
    lower = physical["saved_updates"] + technical["recorded_executed_updates"]
    upper = lower + physical["possible_lost_updates_upper"] + technical["uncertain_updates_upper"]
    if upper > 250000:
        raise ValueError("combined recorded optimizer-work upper bound exceeds cap")
    return dict(
        schema="simplex_t_campaign_accounting_v1",
        status="COMPLETE_REQUIRED_FITS_AND_WORK_ACCOUNTING_VERIFIED_NOT_FINAL_DELIVERY",
        journal_sha256=journal_sha256,
        freeze_sha256=freeze_sha256,
        scientific_fits_completed=len(required),
        scientific_saved_updates=physical["saved_updates"],
        technical=technical,
        physical=physical,
        recorded_total_updates_lower=lower,
        recorded_total_updates_upper=upper,
        checkpoint_training_identity_bound=True,
        optimizer_updates_executed=0,
        campaign_complete=False,
    )
