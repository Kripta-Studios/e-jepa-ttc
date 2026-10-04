"""Global physical ceilings without altering any historical or WIDE journal."""

from .common import Campaign, read


def accounting(c: Campaign) -> dict:
    """Include unresolved chunks conservatively; no unobserved execution claims."""
    wide, recovery, unresolved = 0, 0, 0
    path = c.out / "PHYSICAL_WORK.json"
    if path.exists():
        work = read(path)
        wide = work["accounting"]["scientific_saved_updates"]
        recovery = work["accounting"]["scientific_uncertain_lost_upper"]
        unresolved = sum(
            max(0, v["pending"][1] - v["completed"]) if v["pending"] else 0
            for v in work["fits"].values()
        )
    producer, heads = 0, 0
    for namespace in ("garl", "garl_heads"):
        path = c.out / namespace / "PHYSICAL_WORK.json"
        if not path.exists():
            continue
        work = read(path)
        if namespace == "garl":
            producer = work["saved_updates"]
            recovery += work["uncertain_lost_upper"]
            unresolved += sum(
                max(0, v["pending"][1] - v["saved"]) if v["pending"] else 0
                for v in work["fits"].values()
            )
        else:
            heads = work["accounting"]["scientific_saved_updates"]
            recovery += work["accounting"]["scientific_uncertain_lost_upper"]
            unresolved += sum(
                max(0, v["pending"][1] - v["completed"]) if v["pending"] else 0
                for v in work["fits"].values()
            )
    unsaved_confirmed = 0
    progress = c.out / "UPDATE_PROGRESS.json"
    if progress.exists():
        last = read(progress)
        durable = last["durable"]
        wide_ledger = c.out / "PHYSICAL_WORK.json"
        if wide_ledger.exists():
            durable = read(wide_ledger)["fits"].get(last["fit"], {}).get("completed", durable)
        unsaved_confirmed = max(0, last["confirmed"] - durable)
    interrupted = c.out / "data_recovery/INTERRUPTION_ACCOUNTING.json"
    confirmed_recovery = (
        read(interrupted)["confirmed_lost_updates_lower"] if interrupted.exists() else 0
    )
    return {
        "wide_saved": wide,
        "garl_producer_saved": producer,
        "garl_head_saved": heads,
        "scientific_saved_updates": wide + producer + heads,
        "recovery_uncertain_upper": recovery,
        "recovery_confirmed_lower": confirmed_recovery,
        "unresolved_execution_upper": unresolved,
        "physical_execution_upper": wide + producer + heads + recovery + unresolved,
        "unsaved_updates_confirmed_by_progress": unsaved_confirmed,
        "physical_execution_lower": wide
        + producer
        + heads
        + unsaved_confirmed
        + confirmed_recovery,
        "technical_synthetic_updates": 0,
        "physical_cap": 240000,
        "historical_reexecuted_updates": 0,
    }


def require(c: Campaign) -> None:
    """Enforce family caps, joint recovery ceiling and the owned artifact quota."""
    c.require_resources()
    counts = accounting(c)
    if (
        counts["wide_saved"] > 22500
        or counts["garl_producer_saved"] > 200000
        or counts["garl_head_saved"] > 15000
        or counts["recovery_uncertain_upper"] + counts["unresolved_execution_upper"] > 2300
        or counts["physical_execution_upper"] > 240000
    ):
        raise ValueError("authorized campaign physical work ceiling exceeded")
    used = sum(p.stat().st_size for p in c.out.rglob("*") if p.is_file())
    if used > 10_000_000_000:
        raise InterruptedError(
            "owned campaign artifacts exceed10GB; no historical deletion permitted"
        )
