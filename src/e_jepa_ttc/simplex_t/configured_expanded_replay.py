"""Pinned launch configuration for the separately leased D1/DENSE cache queue."""

from __future__ import annotations

import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .ancestry_evidence import verify_acknowledged_producers
from .coordination import shared_write_admission, verified_ack
from .expanded_execution import run_expanded_context_cache
from .expansion_authority import verify_expansion_authority
from .lifecycle import admitted
from .reuse_catalog import D0ReuseCatalog

PARENT_ACK_SHA256 = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
D0_INDEX_SHA256 = "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
POOL_INDEX_SHA256 = {
    "D1": "46a0749cb4a8c8394b14141e1a78e9181755a4718a44116fdd86b8aa82502cfe",
    "DENSE_OLD": "cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850",
}


def run_configured_expanded_replay(
    local_paths: Path,
    config_path: Path,
    config_sha256: str,
    *,
    max_new_queries: int,
    other_reserved_bytes: int,
    inspect_only: bool,
) -> dict:
    """Resolve real authority and pins; inspection never opens models or raw streams.

    Paths in the configuration are relative to the worktree, except preprocessing
    which is relative to the handoff parent. DENSE requires three complete pinned
    D0 catalogs, never a missing-cache fallback. Execution uses the existing shared
    CURRENT_REPLAY lease. Neither inspection nor cache completion authorizes fits.
    """
    if (
        type(inspect_only) is not bool
        or type(max_new_queries) is not int
        or max_new_queries < 1
        or type(other_reserved_bytes) is not int
        or other_reserved_bytes < 0
    ):
        raise ValueError("explicit bounded execution and outstanding reservations required")
    local_hash = sha256(local_paths)
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if config_path.stat().st_size > 1_048_576 or sha256(config_path) != config_sha256:
        raise ValueError("expanded launch configuration changed or exceeds metadata bound")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pool = config.get("pool")
    if (
        config.get("schema") != "simplex_t_expanded_replay_launch_v1"
        or pool not in POOL_INDEX_SHA256
    ):
        raise ValueError("unrecognized expanded replay launch")
    own_reservation = config.get("reserved_output_bytes")
    if type(own_reservation) is not int or own_reservation <= 0:
        raise ValueError("explicit positive pending cache output reservation required")

    def resources() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], other_reserved_bytes + own_reservation
        )

    if not resources():
        return {"status": "PAUSED_RESOURCE", "new_blocks": 0, "optimizer_updates": 0}

    def resolve(value: str, root: Path = work) -> Path:
        relative = Path(value)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("launch path must stay within its declared root")
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError("launch path resolves outside its declared root")
        return result

    index_root, dedup_root = resolve(config["index"]), resolve(config["dedup"])
    output = resolve(config["output"])
    required_name = (
        "expansion_context_features_fp32" if pool == "D1" else "dense_context_features_fp32"
    )
    if output.parent != work / "artifacts/simplex_t/T1" or output.name != required_name:
        raise ValueError("expanded output must use its separate T1 cache and shared replay lease")
    index_path = index_root / "INDEX_MANIFEST.json"
    if sha256(index_path) != POOL_INDEX_SHA256[pool]:
        raise ValueError("expanded index differs from the acknowledged query population")
    index_manifest = json.loads(index_path.read_text(encoding="utf-8"))
    # Conservative uncompressed per-query allowance plus identity/receipt overhead.
    required_reservation = index_manifest["queries"] * 3 * 131_072 + 1_048_576
    if own_reservation < required_reservation:
        raise ValueError("pending output reservation is smaller than the full expanded cache bound")
    dedup_path = dedup_root / "DEDUP_MANIFEST.json"
    if sha256(dedup_path) != config["dedup_sha256"]:
        raise ValueError("expanded deduplication manifest changed")
    original_root = resolve(config["original_index"])
    original_path = original_root / "INDEX_MANIFEST.json"
    if sha256(original_path) != D0_INDEX_SHA256:
        raise ValueError("original producer family index changed")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    families = original["families"]
    if index_manifest["families"] != families:
        raise ValueError("expanded index changes historical producer families")
    preprocessing = resolve(config["preprocessing"], work.parent)
    prep_hash = "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
    if sha256(preprocessing) != prep_hash:
        raise ValueError("historical preprocessing changed")
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack = verified_ack(ack_path, PARENT_ACK_SHA256)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    if original["ancestry"] != ancestry:
        raise ValueError("original index differs from acknowledged ancestry")
    authority = verify_expansion_authority(local_paths, resource_ok=resources)
    role = "D1_EXPANSION" if pool == "D1" else "DENSE_OLD"
    allowed = set(authority["scope"][role]["sequences"])
    original_roles = ack["interfaces"]["role_manifest"]["roles"]
    if allowed != set(original_roles["expansion" if pool == "D1" else "original"]):
        raise ValueError("supplementary scope differs from authoritative roles")
    reuse_entries = config.get("d0_reuse")
    if (pool == "DENSE_OLD") != (reuse_entries is not None):
        raise ValueError("DENSE requires complete D0 reuse configuration; D1 cannot borrow it")
    catalogs: dict[int, D0ReuseCatalog] = {}
    identities = None
    if pool == "DENSE_OLD":
        if not isinstance(reuse_entries, dict) or set(reuse_entries) != {"0", "1", "2"}:
            raise ValueError("DENSE requires all three complete D0 fold catalogs")
        for outer in range(3):
            if not resources():
                raise InterruptedError("PAUSED_RESOURCE: D0 reuse catalogs")
            entry = reuse_entries[str(outer)]
            catalogs[outer] = D0ReuseCatalog(
                compiled=resolve(entry["compiled"]),
                compiled_sha256=entry["compiled_sha256"],
                cache=resolve(entry["cache"]),
                index_root=original_root,
                dedup=resolve(entry["dedup"]),
                outer=outer,
            )
        identities = {outer: catalog.identity for outer, catalog in catalogs.items()}

    def validate() -> None:
        if sha256(local_paths) != local_hash or sha256(config_path) != config_sha256:
            raise ValueError("expanded local/launch configuration changed")
        verified_ack(ack_path, PARENT_ACK_SHA256)
        verify_expansion_authority(local_paths, resource_ok=resources)
        if sha256(original_path) != D0_INDEX_SHA256:
            raise ValueError("original family index changed during expanded replay")

    inspected = {
        "status": "EXPANDED_LAUNCH_INSPECTED_NOT_REPLAY_OR_SCIENTIFIC_ADMISSION",
        "pool": pool,
        "config_sha256": config_sha256,
        "index_manifest_sha256": POOL_INDEX_SHA256[pool],
        "supplementary_ack_sha256": authority["sha256"],
        "allowed_sequences": sorted(allowed),
        "reserved_output_bytes": own_reservation,
        "other_reserved_bytes": other_reserved_bytes,
        "d0_reuse_identities": identities,
        "output": str(output),
        "optimizer_updates": 0,
        "models_loaded": False,
    }
    if inspect_only:
        validate()
        return inspected
    # Check every acknowledged checkpoint/ancestor binding once before launching;
    # individual families also rehash their checkpoints when they are loaded.
    verify_acknowledged_producers(ack_path, PARENT_ACK_SHA256, resource_ok=resources)
    return run_expanded_context_cache(
        output,
        index_root=index_root,
        index_manifest_sha256=POOL_INDEX_SHA256[pool],
        dedup_root=dedup_root,
        dedup_manifest_sha256=config["dedup_sha256"],
        ancestry=Path(ancestry["path"]),
        ancestry_sha256=ancestry["sha256"],
        preprocessing=preprocessing,
        raw_train_root=Path(paths["eap_root"]) / "data/train",
        allowed_sequences=allowed,
        validate_expanded_authority=validate,
        resource_ok=resources,
        max_new_queries=max_new_queries,
        pool=pool,
        authorized_families=families,
        reuse_catalog_loader=catalogs.__getitem__ if identities is not None else None,
        reuse_expected_identities=identities,
    )
