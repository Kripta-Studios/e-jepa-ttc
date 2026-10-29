"""Explicit expanded disk authorization; preserve every original scientific and RAM check."""

from __future__ import annotations

from .common import Campaign, digest, read


def authorization(c: Campaign) -> dict:
    """Require the user's new disk amendment to bind to the original frozen campaign."""
    value = read(c.out / "DISK_RESOURCE_AUTHORIZATION.json")
    if (
        value["parent_protocol_sha256"] != digest(c.out / "PROTOCOL.json")
        or value["parent_policy_sha256"] != digest(c.policy_path)
        or value["max_owned_bytes"] != 40_000_000_000
        or value["max_native_cache_bytes"] != 32_000_000_000
        or value["atomic_checkpoint_reservation_bytes"] != 1_000_000_000
        or value["scientific_recipe_changed"]
    ):
        raise ValueError("expanded disk authorization differs from the frozen campaign binding")
    return value


def install(c: Campaign) -> None:
    """Run the original guard, admitting only its exact old disk-limit error under the new cap."""
    from . import budget

    value = authorization(c)
    if getattr(budget.require, "_expanded_disk_authorization", None) == digest(
        c.out / "DISK_RESOURCE_AUTHORIZATION.json"
    ):
        return
    original = budget.require

    def require(campaign: Campaign) -> None:
        if campaign.out.resolve() != c.out.resolve():
            return original(campaign)
        try:
            original(campaign)
        except InterruptedError as error:
            if str(error) != (
                "owned campaign artifacts exceed10GB; no historical deletion permitted"
            ):
                raise
            # Original require has already checked resources and every update/recovery cap.
            used = budget._artifact_scans[str(campaign.out)][1]
            if used > value["max_owned_bytes"]:
                raise InterruptedError("owned campaign artifacts exceed authorized40GB") from error

    require._expanded_disk_authorization = digest(c.out / "DISK_RESOURCE_AUTHORIZATION.json")  # type: ignore[attr-defined]
    budget.require = require
