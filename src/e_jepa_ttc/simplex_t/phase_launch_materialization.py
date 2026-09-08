"""Deterministic launch metadata for technical-availability stages, never fits."""

from __future__ import annotations

from pathlib import Path

from .registry import registered_graph


def technical_phase_launch(
    *,
    stage: str,
    freeze_launch: dict,
    frozen: dict,
    freeze_sha256: str,
    campaign_root: Path,
) -> dict:
    """Bind T2/T4 to the exact freeze; practical stages require another resolver.

    This produces input metadata only. The existing worker still independently
    checks the freeze, all source/QA pins, producer lineage and stage admission.
    T4 depends on technical latent availability, never on scalar improvement.
    """
    if stage not in {"T2", "T4"}:
        raise ValueError("practical stages require canonical publication resolution")
    if (
        freeze_launch.get("schema") != "simplex_t_freeze_launch_v1"
        or frozen.get("schema") != "simplex_t_scientific_freeze_v1"
        or frozen.get("status") != "FROZEN_INPUTS_STAGE_GATES_REQUIRED"
        or frozen.get("code_commit") != freeze_launch.get("code_commit")
        or frozen.get("holdout_authorized") is not False
        or len(freeze_sha256) != 64
        or set(freeze_sha256) - set("0123456789abcdef")
    ):
        raise ValueError("recognized exact frozen campaign required")
    work = Path(freeze_launch["roots"]["work"]).resolve(strict=True)
    output = campaign_root.resolve()
    if output == work / "artifacts" or not output.is_relative_to(work / "artifacts"):
        raise ValueError("campaign output must be a companion artifact subdirectory")
    flags = dict(frozen["source_contract"]["availability"])
    if flags.get("d1") is not True or flags.get("density") is not True:
        raise ValueError("this campaign requires resolved D0/D1/DENSE, no reduced fallback")
    # These flags do not affect T2/T4 fits. Do not imply practical authorization.
    flags.update(t3=False, replicate_scalar=False, replicate_latent=False)
    if not any(spec.stage == stage for spec in registered_graph(**flags)):
        raise ValueError("TECHNICALLY_UNAVAILABLE_STAGE")
    for field, category in (("source_configuration", "config"), ("evidence_profile", "qa")):
        target = Path(freeze_launch[field]).resolve(strict=True)
        digest = freeze_launch[field + "_sha256"]
        matches = [
            pin
            for pin in frozen["files"]
            if (Path(freeze_launch["roots"][pin["root"]]) / pin["relative_path"]).resolve()
            == target
        ]
        if (
            len(matches) != 1
            or matches[0]["sha256"] != digest
            or matches[0]["category"] != category
        ):
            raise ValueError("launch configuration is not bound by the scientific freeze")
    return {
        "schema": "simplex_t_frozen_phase_launch_v2",
        **{
            key: freeze_launch[key]
            for key in (
                "local_paths",
                "source_configuration",
                "source_configuration_sha256",
                "evidence_profile",
                "evidence_profile_sha256",
                "roots",
            )
        },
        "freeze": freeze_launch["output"],
        "freeze_sha256": freeze_sha256,
        "execution": str(output / stage / "execution"),
        "publication": str(output / stage / "publication"),
        "stage": stage,
        "availability": flags,
        "publications": {},
        "resource_receipt": str(
            work / "artifacts/simplex_t/resource_observations" / f"{stage}_template_unused.json"
        ),
    }
