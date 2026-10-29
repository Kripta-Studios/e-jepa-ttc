"""CLI preflight of source wiring against the acknowledged role/producer interface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .campaign_sources import CampaignSources
from .coordination import verified_ack
from .registry import registered_graph
from .source_configuration import sources_from_configuration


def inspect_source_configuration(
    local_paths: Path, source_config: Path, source_config_sha256: str, output: Path
) -> dict:
    """Inspect configuration only: do not load cache/target arrays, freeze or fit.

    The maximum registered graph describes possible later gated fits, not enabled
    stages. Expanded temporal authority remains a separate unresolved prerequisite.
    """
    if output.exists():
        raise FileExistsError("preserve existing source preflight")
    sources, result = open_acknowledged_source_configuration(
        local_paths, source_config, source_config_sha256
    )
    sources.release()
    write_new_json(output, result)
    return result


def open_acknowledged_source_configuration(
    local_paths: Path, source_config: Path, source_config_sha256: str
) -> tuple[CampaignSources, dict]:
    """Return unloaded sources bound to actual ACK roles and historical ancestry.

    This is configuration admission, not expanded time authority or a freeze.
    The caller owns releasing the returned adapter, including on exceptions.
    """
    if source_config.stat().st_size > 1_048_576 or sha256(source_config) != source_config_sha256:
        raise ValueError("source configuration changed or exceeds metadata bound")
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    config = json.loads(source_config.read_text(encoding="utf-8"))
    ack = verified_ack(
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318",
    )
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    if config["original"]["ancestry_sha256"] != ancestry["sha256"]:
        raise ValueError("source configuration changes acknowledged producer ancestry")
    if ("dense" in config) != ("matched" in config):
        raise ValueError("both registered matched-control configurations must be supplied together")
    roles = ack["interfaces"]["role_manifest"]["roles"]
    d1, density = "expansion" in config, "dense" in config
    graph = registered_graph(
        d1=d1,
        density=density,
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )
    historical = Path(ancestry["path"]).parent.resolve(strict=True)
    roots = {
        "work": Path(paths["worktree"]),
        "historical": historical,
        "stage70": Path(paths["stage70_worktree_read_only"]),
        "garl": Path(paths["garl_annotations_candidate"]).parent.parent,
    }
    sources = sources_from_configuration(
        source_config,
        expected_sha256=source_config_sha256,
        roots=roots,
        graph=graph,
        allowed_original_sequences=set(roles["original"]),
        allowed_expansion_sequences=set(roles["expansion"]) if d1 else set(),
    )
    if sources.historical_root != historical:
        sources.release()
        raise ValueError("source configuration changes acknowledged historical producer root")
    result = {
        "status": "SOURCE_CONFIGURATION_INSPECTED_NOT_PAYLOAD_VERIFIED_OR_FROZEN",
        "source_configuration_sha256": source_config_sha256,
        "local_paths_sha256": sha256(local_paths),
        "ack_sha256": "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318",
        "producer_ancestry_sha256": ancestry["sha256"],
        "maximum_registered_graph": [asdict(spec) for spec in graph],
        "maximum_registered_updates": sum(spec.updates for spec in graph),
        "graph_enables_practical_or_technical_gates": False,
        "feature_or_target_payloads_loaded": False,
        "new_query_time_authority_validated": False,
        "optimizer_updates": 0,
    }
    return sources, result
