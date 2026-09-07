"""Compose actual input authority and mandatory real QA for a prospective freeze."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .component_verification import verify_component_profile
from .configuration_preflight import open_acknowledged_source_configuration
from .coordination import verified_ack
from .expansion_authority import verify_expansion_authority
from .registry import registered_graph


def validate_scientific_admission(
    record: dict,
    *,
    roots: dict[str, Path],
    local_paths: Path,
    source_configuration: Path,
    source_configuration_sha256: str,
    evidence_profile: Path,
    evidence_profile_sha256: str,
    resource_ok: Callable[[], bool],
) -> dict:
    """Validate authority/QA, not merely the existence of freeze metadata.

    Use as the prerequisite callback around scientific_freeze publication/read.
    That API must still verify the complete committed code inventory, source
    preparation, normalization, graph and all pinned files before execution.
    Practical gates remain separate. No model is fitted or expert replay run.

    Expanded configurations require the exact supplementary owner ACK and all
    its references in the freeze. This does not substitute for source or replay QA.
    """
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: scientific input admission")
    local_hash = sha256(local_paths)
    if (
        source_configuration.stat().st_size > 1_048_576
        or sha256(source_configuration) != source_configuration_sha256
        or evidence_profile.stat().st_size > 1_048_576
        or sha256(evidence_profile) != evidence_profile_sha256
    ):
        raise ValueError("admission configuration bytes changed")
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if roots["work"].resolve(strict=True) != work:
        raise ValueError("scientific admission worktree differs from frozen roots")
    configuration = json.loads(source_configuration.read_text(encoding="utf-8"))
    expanded = any(key in configuration for key in ("expansion", "dense", "matched"))
    temporal = (
        verify_expansion_authority(local_paths, resource_ok=resource_ok) if expanded else None
    )
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    time_charter = ack["interfaces"]["time_charter"]

    def require_pin(path: Path, digest: str, category: str | None = None) -> None:
        target = path.resolve(strict=True)
        matches = [
            pin
            for pin in record["files"]
            if (roots[pin["root"]] / pin["relative_path"]).resolve(strict=True) == target
        ]
        if (
            len(matches) != 1
            or matches[0]["sha256"] != digest
            or (category is not None and matches[0]["category"] != category)
        ):
            raise ValueError("scientific freeze omits or changes an admission input pin")

    for path, digest, category in (
        (local_paths, local_hash, "config"),
        (source_configuration, source_configuration_sha256, "config"),
        (evidence_profile, evidence_profile_sha256, "qa"),
        (ack_path, ack_hash, "roles"),
        (Path(ancestry["path"]), ancestry["sha256"], "producers"),
        (Path(time_charter["path"]), time_charter["sha256"], "time"),
    ):
        require_pin(path, digest, category)
    contract = record["source_contract"]
    if temporal is not None:
        require_pin(Path(temporal["path"]), temporal["sha256"], "time")
        for entry in temporal["evidence"]:
            require_pin(Path(entry["path"]), entry["sha256"])
    if (
        contract["configuration_sha256"] != source_configuration_sha256
        or contract["authority_sha256"] != ack_hash
        or contract["availability"]["d1"] is not ("expansion" in configuration)
        or contract["availability"]["density"] is not ("dense" in configuration)
    ):
        raise ValueError("freeze changes acknowledged source configuration or data scope")
    sources, inspection = open_acknowledged_source_configuration(
        local_paths, source_configuration, source_configuration_sha256
    )
    try:
        graph = registered_graph(**contract["availability"])
        if any(spec not in sources.graph for spec in graph):
            raise ValueError("source configuration does not support the frozen candidate graph")
        index_path = sources.index_root / "INDEX_MANIFEST.json"
        index_hash = "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
        require_pin(index_path, index_hash)
        if sha256(index_path) != index_hash:
            raise ValueError("OLD context index differs from audited input-only amendment")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["ancestry"] != ancestry or inspection["ack_sha256"] != ack_hash:
            raise ValueError("context index or source loader changes acknowledged ancestry")
        qa = verify_component_profile(
            local_paths,
            evidence_profile,
            evidence_profile_sha256,
            resource_ok=resource_ok,
            require_h16=True,
            require_unit_qa=True,
            require_static_qa=True,
            require_types=True,
            require_powershell=True,
        )
        if (
            qa["local_paths_sha256"] != local_hash
            or qa["ancestry"]["ancestry_sha256"] != ancestry["sha256"]
        ):
            raise ValueError("real QA refers to another input authority")
        if (
            temporal is not None
            and verify_expansion_authority(local_paths, resource_ok=resource_ok) != temporal
        ):
            raise ValueError("supplementary temporal authority changed during admission")
        if (
            sha256(local_paths) != local_hash
            or sha256(source_configuration) != source_configuration_sha256
            or sha256(evidence_profile) != evidence_profile_sha256
            or verified_ack(ack_path, ack_hash) != ack
        ):
            raise ValueError("scientific admission inputs changed during validation")
        return {
            "status": "INPUT_AUTHORITY_AND_QA_VERIFIED_FREEZE_AND_GATES_REQUIRED",
            "source_configuration_sha256": source_configuration_sha256,
            "evidence_profile_sha256": evidence_profile_sha256,
            "ack_sha256": ack_hash,
            "time_scope": (
                "OLD8192_AND_ACKNOWLEDGED_EXPANDED_CURRENT_ROI_RETROSPECTIVE_CONTEXT"
                if temporal is not None
                else "OLD8192_CURRENT_ROI_RETROSPECTIVE_CONTEXT"
            ),
            "supplementary_time_ack_sha256": temporal["sha256"] if temporal is not None else None,
            "training_pools": [
                "D0",
                *(["D1"] if "expansion" in configuration else []),
                *(["DENSE_OLD"] if "dense" in configuration else []),
            ],
            "evaluation_cohort": "UNCHANGED_OLD8192",
            "online_annotation_availability_accredited": False,
            "optimizer_updates_executed": 0,
            "holdout_authorized": False,
        }
    finally:
        sources.release()
