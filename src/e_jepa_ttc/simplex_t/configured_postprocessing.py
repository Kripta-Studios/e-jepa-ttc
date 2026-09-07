"""Configured post-fit assembly with independently loaded, frozen OLD inputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .campaign_postprocessing import postprocess_completed_campaign
from .configuration_preflight import open_acknowledged_source_configuration
from .frozen_history import frozen_history_pools
from .old_cohort import load_old_evaluation_cohort
from .risk17_inputs import load_acknowledged_risk17
from .scientific_admission import validate_scientific_admission
from .scientific_freeze import read_scientific_freeze
from .stage_gate import CanonicalPublication


def postprocess_configured_campaign(
    output: Path,
    *,
    local_paths: Path,
    source_configuration: Path,
    source_configuration_sha256: str,
    evidence_profile: Path,
    evidence_profile_sha256: str,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    phases: dict[str, CanonicalPublication],
    resource_ok: Callable[[], bool],
) -> dict:
    """Assemble T6 analysis inputs without accepting caller-supplied targets or gates.

    Real admission and committed freeze verification precede source construction.
    The complete OLD cohort and RISK17 comparator are reloaded from exact freeze
    pins. The downstream graph check requires every enabled phase before analyses.
    Source ownership is released on success, pause and failure. Partial analysis
    output remains for audit; retry uses a new output directory, not new fits.
    This does not replace final transport packaging and technical accounting.
    """
    work = Path(json.loads(local_paths.read_text(encoding="utf-8"))["worktree"]).resolve(
        strict=True
    )
    if output.exists() or output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("new companion-local postprocessing directory required")
    if freeze.stat().st_size > 8_388_608 or sha256(freeze) != freeze_sha256:
        raise ValueError("postprocessing freeze bytes changed")
    requested = json.loads(freeze.read_text(encoding="utf-8"))

    def admission() -> None:
        validate_scientific_admission(
            requested,
            roots=roots,
            local_paths=local_paths,
            source_configuration=source_configuration,
            source_configuration_sha256=source_configuration_sha256,
            evidence_profile=evidence_profile,
            evidence_profile_sha256=evidence_profile_sha256,
            resource_ok=resource_ok,
        )

    record = read_scientific_freeze(
        freeze, expected_sha256=freeze_sha256, roots=roots, validate_prerequisites=admission
    )
    if record != requested:
        raise ValueError("requested and verified postprocessing freeze differ")
    sources, _ = open_acknowledged_source_configuration(
        local_paths, source_configuration, source_configuration_sha256
    )
    try:

        def frozen_digest(path: Path) -> str:
            target = path.resolve(strict=True)
            matches = [
                pin["sha256"]
                for pin in record["files"]
                if (roots[pin["root"]] / pin["relative_path"]).resolve(strict=True) == target
            ]
            if len(matches) != 1:
                raise ValueError("postprocessing input requires one exact freeze pin")
            return matches[0]

        historical = sources.historical_root.resolve(strict=True)
        index = historical / "FROZEN_EXPERT_TABLE_INDEX.json"
        ancestry = historical / "NESTED_ANCESTRY_AUDIT.json"
        if frozen_digest(ancestry) != sources.ancestry_sha256:
            raise ValueError("postprocessing cohort changes historical ancestry")
        pins = {index: frozen_digest(index), ancestry: sources.ancestry_sha256}
        for outer in range(3):
            for suffix in ("csv", "npz"):
                path = historical / "tables" / f"outer{outer}_outer_dev.{suffix}"
                pins[path] = frozen_digest(path)
        paths = json.loads(local_paths.read_text(encoding="utf-8"))
        ack = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
        replay = work / "artifacts/simplex_t/T1/risk17_frozen_replay/REPLAY.json"
        ridge = historical / "frozen_audit/extracted_input/run/stage65/ALL_RIDGE_FITS_FROZEN.json"
        for path in (ack, replay, ridge):
            pins[path] = frozen_digest(path)

        def validate() -> None:
            current = read_scientific_freeze(
                freeze,
                expected_sha256=freeze_sha256,
                roots=roots,
                validate_prerequisites=admission,
            )
            if current != record:
                raise ValueError("postprocessing freeze changed")
            for path, digest in pins.items():
                if not resource_ok():
                    raise InterruptedError("PAUSED_RESOURCE: postprocessing input bindings")
                if sha256(path) != digest:
                    raise ValueError("postprocessing historical input differs from frozen bytes")

        validate()
        cohort, receipt = load_old_evaluation_cohort(
            historical,
            ancestry_sha256=sources.ancestry_sha256,
            table_index_sha256=pins[index],
            allowed_sequences=sources.allowed_sequences,
            resource_ok=resource_ok,
        )
        for row in receipt["folds"]:
            for suffix, field in (("csv", "metadata_sha256"), ("npz", "arrays_sha256")):
                path = historical / "tables" / f"outer{row['outer_fold']}_outer_dev.{suffix}"
                if pins[path] != row[field]:
                    raise ValueError("postprocessing OLD receipt differs from frozen inputs")

        def risk17() -> pd.DataFrame:
            validate()
            return load_acknowledged_risk17(
                replay,
                manifest_sha256=pins[replay],
                ack_path=ack,
                ack_sha256=pins[ack],
                table_index_sha256=pins[index],
                ridge_manifest_sha256=pins[ridge],
                expected_identity=cohort,
                resource_ok=resource_ok,
            )

        return postprocess_completed_campaign(
            output,
            freeze=freeze,
            freeze_sha256=freeze_sha256,
            roots=roots,
            sources=sources,
            history_pools=frozen_history_pools(sources, record, roots=roots),
            phases=phases,
            expected_queries=cohort,
            load_verified_risk17=risk17,
            validate_authority_and_qa=validate,
            resource_ok=resource_ok,
        )
    finally:
        sources.release()
