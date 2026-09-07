"""Package the exact underlying evidence used to reconcile technical optimizer work."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory
from .technical_accounting import verify_technical_accounting


def technical_bundle_members(
    reconciliation: Path,
    *,
    reconciliation_sha256: str,
    ledger: Path,
    ledger_sha256: str,
    work_root: Path,
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Reverify all evidence classes, retaining historical failure and resume receipts.

    Only references actually consumed by the technical accounting reader are
    traversed, never paths to datasets or checkpoints mentioned inside receipts.
    Historical noninstrumented evidence remains so; transport cannot upgrade it.
    Call again in the ZIP authority callback to detect any intervening change.
    """

    def verify() -> dict:
        return verify_technical_accounting(
            reconciliation,
            reconciliation_sha256=reconciliation_sha256,
            ledger=ledger,
            ledger_sha256=ledger_sha256,
            work_root=work_root,
            resource_ok=resource_ok,
        )

    verified = verify()
    work = work_root.resolve(strict=True)
    members: dict[str, BundleMember] = {}

    def add(path: Path, digest: str) -> bytes:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: technical evidence transport")
        target = path.resolve(strict=True)
        if not target.is_relative_to(work) or not target.is_file():
            raise ValueError("technical transport evidence outside companion")
        with target.open("rb") as stream:
            payload = stream.read(8_388_609)
        if len(payload) > 8_388_608 or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("technical transport evidence hash or size differs")
        name = "technical/" + target.relative_to(work).as_posix()
        member = BundleMember(target, digest, len(payload))
        if name in members and members[name] != member:
            raise ValueError("conflicting technical evidence pins")
        members[name] = member
        return payload

    audit = json.loads(add(reconciliation, reconciliation_sha256))
    add(ledger, ledger_sha256)
    for operation in audit["operations"]:
        evidence = work / operation["evidence"]["path"]
        add(evidence, operation["evidence"]["sha256"])
        category = operation["evidence_class"]
        if category == "CONTEMPORANEOUS_RECONCILIATION_NOT_INSTRUMENTED_COUNTER":
            support = operation["supporting_junit"]
            add(evidence.parent / support["path"], support["sha256"])
        elif category == "INSTRUMENTED_SUITE_COUNTER_INCLUDING_FAILED_RUNS":
            add(evidence.parent / "UPDATE_PROGRESS.json", operation["counters_sha256"])
    if verify() != verified:
        raise ValueError("technical accounting changed while binding transport")
    validate_bundle_inventory({name: member.sha256 for name, member in members.items()})
    return members
