"""Bind existing scientific outputs to the established T6 delivery worker."""

from __future__ import annotations

from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def delivery_launch(
    *,
    base: dict,
    stage_templates: dict[str, dict],
    accounting: dict,
    resource_attempts: list[dict],
    analysis_commit: str,
    attempt_root: Path,
) -> dict:
    """Hash known output manifests; semantic graph validation remains mandatory.

    This reads manifests, not prediction values or TTC targets. It cannot infer
    completed fits from filenames: the T6 worker independently checks complete
    endpoints, checkpoints, canonical gates and physical/technical accounting.
    """
    if "T2" not in stage_templates or not set(stage_templates) <= {"T2", "T3", "T4", "T5"}:
        raise ValueError("registered publications including T2 required")
    if len(analysis_commit) != 40 or set(analysis_commit) - set("0123456789abcdef"):
        raise ValueError("exact analysis commit required")
    work = Path(base["roots"]["work"]).resolve(strict=True)
    root = attempt_root.resolve()
    if not root.is_relative_to(work / "artifacts") or root == work / "artifacts":
        raise ValueError("new companion artifact delivery root required")
    if not resource_attempts:
        raise ValueError("observed resource attempts required")
    common = (
        "local_paths",
        "source_configuration",
        "source_configuration_sha256",
        "evidence_profile",
        "evidence_profile_sha256",
        "freeze",
        "freeze_sha256",
        "roots",
    )
    phases = {}
    for stage, template in stage_templates.items():
        if template["stage"] != stage or any(template[key] != base[key] for key in common):
            raise ValueError("phase template changes frozen scientific identity")
        if template["execution"] != base["execution"]:
            raise ValueError("phases must share one physical-work journal")
        publication = Path(template["publication"]) / f"{stage}_PREDICTIONS.json"
        endpoints = Path(template["execution"]) / f"{stage}_ENDPOINTS.json"
        phases[stage] = dict(
            publication=str(publication),
            publication_sha256=sha256(publication),
            endpoints=str(endpoints),
            endpoints_sha256=sha256(endpoints),
            checkpoint_root=str(Path(template["execution"]) / "fits"),
            availability=template["availability"],
        )
    if set(accounting) != {
        "journal",
        "journal_sha256",
        "reconciliation",
        "reconciliation_sha256",
        "ledger",
        "ledger_sha256",
    }:
        raise ValueError("exact physical and technical accounting references required")
    if (
        Path(accounting["journal"]).resolve()
        != Path(base["execution"]).resolve() / "PHYSICAL_WORK.json"
    ):
        raise ValueError("delivery accounting uses another physical journal")
    for name in ("journal", "reconciliation", "ledger"):
        if sha256(Path(accounting[name])) != accounting[name + "_sha256"]:
            raise ValueError("accounting input changed")
    return dict(
        schema="simplex_t_postprocessing_launch_v3",
        **{key: base[key] for key in common},
        output=str(root / "analysis"),
        publications=phases,
        accounting=accounting,
        delivery=dict(
            output=str(root / "delivery"),
            analysis_commit=analysis_commit,
            resource_attempts=resource_attempts,
        ),
    )
