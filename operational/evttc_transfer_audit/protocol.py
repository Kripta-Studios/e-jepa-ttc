"""Fail-closed recording/acquisition role validation, without model scores."""

from __future__ import annotations

from typing import Any

from .common import canonical_sha

ROLES = {"excluded", "development_exposed", "fit", "selection", "holdout_exposed", "holdout_clean"}
UNKNOWN = (None, "", "unknown", "unresolved", "PENDING")


def validate_protocol(
    policy: dict[str, Any],
    inventory: dict[str, Any],
    identities: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Propagate exclusions through aliases, recordings and acquisitions before role checks."""
    records = inventory["records"]
    ids = [r.get("sequence_id") for r in records]
    if (
        not records
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Population requires nonempty unique sequence IDs")
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    excluded = set(identities.get("excluded_recordings", []))
    violations: list[str] = []
    unresolved: list[str] = []
    if identities.get("status") != "verified":
        unresolved.append("identity_audit_unverified")
    for sequence in identities.get("sequences", []):
        if sequence["status"] != "unique":
            unresolved.append(f"identity:{sequence['competition_sequence']}:{sequence['status']}")
        for match in sequence["matches"]:
            excluded.add(match["sequence_id"])
    for row in records:
        sid = row["sequence_id"]
        find("id:" + sid)
        for alias in row.get("aliases", []):
            union("id:" + sid, "id:" + alias)
        for field in ("recording_group", "acquisition_group"):
            group = row.get(field)
            if group in UNKNOWN:
                unresolved.append(f"{sid}:{field}")
            else:
                union("id:" + sid, field + ":" + str(group))
        if row.get("review_status") != "approved" or not row.get("relationship_evidence"):
            unresolved.append(f"{sid}:relationship_review")
        if row.get("role") not in ROLES:
            violations.append(f"{sid}:invalid_role")
        exposure = row.get("historical_exposure")
        if exposure is None or "unknown" in exposure:
            unresolved.append(f"{sid}:historical_exposure")
        if row.get("role") == "holdout_clean" and exposure != ["verified_unexposed"]:
            violations.append(f"{sid}:clean_holdout_has_exposure_or_unknown_history")
    excluded_roots = {find("id:" + sid) for sid in excluded}
    effective = sorted(sid for sid in ids if find("id:" + sid) in excluded_roots)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        sid = row["sequence_id"]
        groups.setdefault(find("id:" + sid), []).append(row)
        if sid in effective and row["role"] != "excluded":
            violations.append(f"{sid}:related_to_excluded_recording")
    for members in groups.values():
        if len({r["role"] for r in members}) > 1:
            violations.append(
                "related_records_cross_roles:" + ",".join(r["sequence_id"] for r in members)
            )
    if lineage.get("status") != "verified":
        unresolved.append("component_ancestry_unresolved")
    status = "rejected" if violations else "unresolved" if unresolved else "verified"
    return {
        "schema": "evttc_transfer_protocol_v1",
        "status": status,
        "policy_sha256": canonical_sha(policy),
        "population_sha256": canonical_sha(inventory),
        "identities_sha256": canonical_sha(identities),
        "lineage_sha256": canonical_sha(lineage),
        "violations": violations,
        "unresolved": sorted(set(unresolved)),
        "effective_exclusions": effective,
        "recordings": len(records),
        "independent_groups": len(groups) if not unresolved and not violations else None,
        "training_allowed": (
            status == "verified"
            and policy.get("training_allowed") is True
            and any(r["role"] == "fit" for r in records)
        ),
        "records": records,
        "claim": "No independent generalization claim while acquisition or ancestry is unresolved.",
    }
