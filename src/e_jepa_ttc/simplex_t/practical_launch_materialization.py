"""Resolve practical stages only through read-only canonical worker admission."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path


def resolve_practical_launch(
    stage: str,
    *,
    base: dict,
    frozen_availability: dict[str, bool],
    campaign_root: Path,
    publication: Callable[[str], dict],
    check_gate: Callable[[dict], dict],
) -> tuple[dict | None, dict]:
    """Distinguish a canonical negative gate from missing/broken evidence.

    The production caller binds check_gate to execute_configured_phase with
    verify_only=True and never supplies scores or hand-picked candidates.
    Probe destinations must stay absent so completed outputs cannot alter gate
    resolution. No checkpoints or predictions are written by that read-only API.
    """
    if stage not in {"T3", "T5"}:
        raise ValueError("only practical T3/T5 stages can be resolved")
    result = deepcopy(base)
    result.update(
        stage=stage,
        execution=str(campaign_root / stage / "execution"),
        publication=str(campaign_root / stage / "publication"),
        publications={},
    )
    flags = result["availability"]
    if any(flags[key] != frozen_availability[key] for key in ("d1", "density", "latent")):
        raise ValueError("practical materialization changes technical availability")
    flags.update(t3=False, replicate_scalar=False, replicate_latent=False)
    records = []
    candidates = (
        [("t3", "TPR")]
        if stage == "T3"
        else [("replicate_scalar", "TPR"), ("replicate_latent", "LATENT")]
    )
    for flag, family in candidates:
        if not frozen_availability[flag]:
            records.append(
                dict(flag=flag, family=family, eligible=False, reason="NOT_FROZEN_AVAILABLE")
            )
            continue
        pin = publication(family)
        probe = deepcopy(result)
        probe["availability"].update(t3=False, replicate_scalar=False, replicate_latent=False)
        probe["availability"][flag] = True
        probe["publications"] = {family: pin}
        probe_root = campaign_root / "gate_probes" / f"{stage}_{family}"
        if probe_root.exists():
            raise ValueError("read-only gate probe destination must remain absent")
        probe.update(
            execution=str(probe_root / "execution"), publication=str(probe_root / "publication")
        )
        try:
            checked = check_gate(probe)
        except ValueError as error:
            if str(error) != f"PRACTICAL_GATE_NOT_PASSED: {family}":
                raise
            eligible = False
        else:
            if checked.get("status") == "PAUSED_RESOURCE":
                raise InterruptedError("PAUSED_RESOURCE: practical stage resolution")
            if checked.get("status") != "PHASE_INCOMPLETE":
                raise ValueError("unexpected read-only gate probe outcome")
            eligible = True
        if probe_root.exists():
            raise ValueError("gate resolution unexpectedly created output")
        records.append(
            dict(
                flag=flag,
                family=family,
                eligible=eligible,
                reason="CANONICAL_GATE_PASSED" if eligible else "CANONICAL_GATE_NOT_PASSED",
                publication=pin,
            )
        )
        flags[flag] = eligible
        if eligible:
            result["publications"][family] = pin
    enabled = any(row["eligible"] for row in records)
    decision = dict(
        schema="simplex_t_practical_launch_decision_v1",
        stage=stage,
        freeze_sha256=base["freeze_sha256"],
        eligible=enabled,
        families=records,
        optimizer_updates=0,
        model_inference=False,
        scientific_completion=False,
    )
    return (result if enabled else None), decision
