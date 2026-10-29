"""Pinned launch configuration for the separately leased D1/DENSE cache queue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .ancestry_evidence import verify_acknowledged_producers
from .coordination import shared_write_admission, verified_ack
from .density_selection import validate_selected_rows
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
    verify_only: bool = False,
) -> dict:
    """Resolve real authority and pins; inspection never opens models or raw streams.

    Paths in the configuration are relative to the worktree, except preprocessing
    which is relative to the handoff parent. DENSE requires three complete pinned
    D0 catalogs, never a missing-cache fallback. Execution uses the existing shared
    CURRENT_REPLAY lease. Neither inspection nor cache completion authorizes fits.
    """
    if (
        type(inspect_only) is not bool
        or type(verify_only) is not bool
        or (inspect_only and verify_only)
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

    reuse_qa_ref = config.get("query_major_qa")
    if reuse_qa_ref is not None:
        if pool != "D1" or set(reuse_qa_ref) != {"path", "sha256", "contract_sha256"}:
            raise ValueError("query-major execution requires a pinned D1 input reuse QA")
        qa_path = resolve(reuse_qa_ref["path"])
        contract_path = qa_path.parent / "CONTRACT.json"
        if (
            not qa_path.is_relative_to(work / "artifacts/simplex_t/T0")
            or sha256(qa_path) != reuse_qa_ref["sha256"]
            or sha256(contract_path) != reuse_qa_ref["contract_sha256"]
        ):
            raise ValueError("query-major QA binding changed")
        qa = json.loads(qa_path.read_text("utf-8"))
        contract = json.loads(contract_path.read_text("utf-8"))
        if (
            qa["status"] != "QUERY_MAJOR_INPUT_REUSE_EXACT_PASS"
            or qa["compared_blocks_per_pass"] != 18
            or qa["optimizer_updates"] != 0
            or qa["sampled_rss_max_bytes"] > 4 * 1024**3
            or contract["queries"] != [455, 1366, 2277, 3187, 4097, 5007]
        ):
            raise ValueError("query-major QA coverage or resource evidence differs")
        baseline, optimized = qa["results"]
        if (
            baseline["mode"] != "baseline"
            or optimized["mode"] != "query_major"
            or baseline["preparations"] != 18
            or optimized["preparations"] != 6
            or optimized["hits"] != 12
            or not all(
                math.isfinite(row["seconds"]) and row["seconds"] > 0 for row in qa["results"]
            )
            or optimized["seconds"] >= baseline["seconds"]
        ):
            raise ValueError("query-major QA does not demonstrate an input reuse time saving")
        for value, digest in contract["pins"].items():
            if sha256(Path(value)) != digest:
                raise ValueError("input reuse QA dependencies changed")

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
    selection_ref = config.get("query_selection")
    selected_queries = None
    selection_path = None
    if selection_ref is not None:
        if pool != "D1" or set(selection_ref) != {"path", "sha256"}:
            raise ValueError("only an explicitly pinned D1 density selection is accepted")
        selection_path = resolve(selection_ref["path"])
        if (
            selection_path.stat().st_size > 1_048_576
            or sha256(selection_path) != selection_ref["sha256"]
        ):
            raise ValueError("D1 density selection changed")
        selection = json.loads(selection_path.read_text("utf-8"))
        if (
            selection["schema"] != "simplex_t_d1_density_selection_v1"
            or selection["amendment"] != "SIMPLEX_T_THROUGHPUT_2026-09-08"
            or selection["parent_manifest"]["sha256"] != POOL_INDEX_SHA256[pool]
            or selection["parent_index_sha256"] != index_manifest["index_sha256"]
            or selection["per_sequence_cap"] != 512
        ):
            raise ValueError("D1 selection changes its authorized parent or amendment")
        array_path = index_root / "query_context_index.npz"
        if sha256(array_path) != index_manifest["index_sha256"]:
            raise ValueError("D1 parent index bytes changed")
        selected_queries = np.asarray(selection["selected_original_rows"], dtype=np.int64)
        with np.load(array_path, allow_pickle=False) as data:
            validate_selected_rows(
                selected_queries, data["sequences"], data["anchor_us"], data["tokens"]
            )
    # Conservative uncompressed per-query allowance plus identity/receipt overhead.
    query_count = index_manifest["queries"] if selected_queries is None else len(selected_queries)
    required_reservation = query_count * 3 * 131_072 + 1_048_576
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
        if selection_path is not None:
            assert selection_ref is not None
            if sha256(selection_path) != selection_ref["sha256"]:
                raise ValueError("D1 selection changed during replay")

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
        "selected_queries": query_count,
        "query_selection": selection_ref,
        "query_major_qa": reuse_qa_ref,
    }
    if inspect_only:
        validate()
        return inspected
    if verify_only and not (output / "IDENTITY.json").exists():
        validate()
        return {**inspected, "status": "EXPANDED_CACHE_INCOMPLETE", "files_written": 0}
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
        selected_queries=selected_queries,
        selection_binding=(
            {**selection_ref, "path": str(selection_path)} if selection_ref is not None else None
        ),
        query_major=reuse_qa_ref is not None,
        input_reuse_qa_binding=reuse_qa_ref,
        reuse_catalog_loader=catalogs.__getitem__ if identities is not None else None,
        reuse_expected_identities=identities,
        **({"verify_only": True} if verify_only else {}),
    )
