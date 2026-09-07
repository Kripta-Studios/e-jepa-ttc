"""Wire expanded metadata, frozen producer inference and D0 reuse to one queue."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .dense_replay_reuse import DenseReplayReuse
from .expanded_history_loader import ExpandedHistoryLoader
from .expanded_inference import expanded_inference_family
from .expanded_replay_plan import inspect_expanded_replay, verify_d1_index_ancestry
from .expanded_replay_queue import run_expanded_blocks
from .expanded_stream_support import verify_expanded_stream_support, verify_stream_receipts
from .reuse_catalog import D0ReuseCatalog


def run_expanded_context_cache(
    output: Path,
    *,
    index_root: Path,
    index_manifest_sha256: str,
    dedup_root: Path,
    dedup_manifest_sha256: str,
    ancestry: Path,
    ancestry_sha256: str,
    preprocessing: Path,
    raw_train_root: Path,
    allowed_sequences: set[str],
    validate_expanded_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
    max_new_queries: int,
    pool: str = "D1",
    reuse_catalog_loader: Callable[[int], D0ReuseCatalog] | None = None,
    reuse_expected_identities: dict[int, dict] | None = None,
    authorized_families: list[dict] | None = None,
) -> dict:
    """Run only under independently verified expanded authority and absolute limits.

    The caller must verify the owner's supplementary time acknowledgment, exact
    role map and transitive exclusions, and the cooperative resource permission.
    This API does not invent an ACK schema or turn the original ACK into D1 access.
    DENSE additionally requires independently authorized family descriptors and
    pinned complete D0 identities for every fold. No missing-reuse fallback exists.
    """
    if pool not in {"D1", "DENSE_OLD"}:
        raise ValueError("unknown expanded pool")
    if pool == "DENSE_OLD":
        if (
            reuse_catalog_loader is None
            or reuse_expected_identities is None
            or set(reuse_expected_identities) != {0, 1, 2}
            or authorized_families is None
        ):
            raise ValueError("DENSE requires all D0 fold identities and authorized families")
    elif reuse_catalog_loader is not None or reuse_expected_identities is not None:
        raise ValueError("D1 cannot borrow D0 query blocks")
    validate_expanded_authority()
    if not resource_ok():
        return {"status": "PAUSED_RESOURCE", "new_blocks": 0, "optimizer_updates": 0}
    inspect_expanded_replay(index_root, manifest_sha256=index_manifest_sha256, pool=pool)
    prep_hash = "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
    if sha256(preprocessing) != prep_hash or sha256(ancestry) != ancestry_sha256:
        raise ValueError("frozen preprocessing or historical ancestry changed")
    manifest = json.loads((index_root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    if pool == "D1":
        verify_d1_index_ancestry(manifest, ancestry, ancestry_sha256)
    if authorized_families is not None and manifest["families"] != authorized_families:
        raise ValueError("expanded producer descriptors differ from authorized families")
    with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    if set(map(str, index["sequences"])) != allowed_sequences:
        raise ValueError("expanded index sequence set differs from authorized TRAIN pool")
    histories = ExpandedHistoryLoader(
        dedup_root,
        manifest_sha256=dedup_manifest_sha256,
        index_manifest_sha256=index_manifest_sha256,
        pool=pool,
        producer_family=index["producer_family"],
        valid=index["valid"],
    )
    stream_receipts = verify_expanded_stream_support(
        index,
        manifest,
        pool=pool,
        raw_train_root=raw_train_root,
        allowed_sequences=allowed_sequences,
        resource_ok=resource_ok,
    )
    parent = json.loads(ancestry.read_text(encoding="utf-8"))
    checkpoints = {r["sha256"]: Path(r["path"]) for r in parent["input_bindings"].values()}
    prep = json.loads(preprocessing.read_text(encoding="utf-8"))["config"]
    source_root = Path(__file__).parent
    code = {
        name: sha256(source_root / name)
        for name in (
            "expert_features.py",
            "expert_phase.py",
            "query_context_voxel.py",
            "context_raw_union.py",
            "expanded_execution.py",
            "expanded_inference.py",
            "expanded_replay_queue.py",
            "expanded_history_loader.py",
            "expanded_replay_plan.py",
            "expanded_stream_support.py",
            "dense_replay_reuse.py",
            "reuse_catalog.py",
            "cache_reuse.py",
        )
    }
    identity = {
        "schema": "simplex_t_query_context_fp32_cache_v1",
        "pool": pool,
        "index_sha256": manifest["index_sha256"],
        "preprocessing_sha256": prep_hash,
        "extractor_sha256": code["expert_features.py"],
        "expert_phase_sha256": code["expert_phase.py"],
        "voxel_sha256": code["query_context_voxel.py"],
        "union_reader_sha256": code["context_raw_union.py"],
        "runner_sha256": code["expanded_execution.py"],
        "expanded_code_sha256": code,
        "dedup_manifest_sha256": dedup_manifest_sha256,
        "ancestry_sha256": ancestry_sha256,
        "raw_stream_support": stream_receipts,
        "torch": str(torch.__version__),
        "batch_size": 16,
        "layout": "one query, chronological H16; absent slots zero, never exported",
        "precision": "FP32",
        "tf32": False,
        "optimizer_updates": 0,
    }
    reuse = None
    if pool == "DENSE_OLD":
        assert reuse_catalog_loader is not None and reuse_expected_identities is not None
        # JSON round-trip also isolates the frozen values from caller mutation.
        identity["d0_reuse_identities"] = json.loads(json.dumps(reuse_expected_identities))
        reuse = DenseReplayReuse(
            catalog_loader=reuse_catalog_loader,
            expected_identities={int(k): v for k, v in identity["d0_reuse_identities"].items()},
            histories=histories,
            index=index,
            extraction_identity=identity,
        )

    def validate() -> None:
        validate_expanded_authority()
        verify_stream_receipts(stream_receipts)
        for path, expected in (
            (index_root / "INDEX_MANIFEST.json", index_manifest_sha256),
            (dedup_root / "DEDUP_MANIFEST.json", dedup_manifest_sha256),
            (preprocessing, prep_hash),
            (ancestry, ancestry_sha256),
        ):
            if sha256(path) != expected:
                raise ValueError("expanded replay input authority binding changed")
        if any(sha256(source_root / name) != expected for name, expected in code.items()):
            raise ValueError("expanded replay implementation changed")

    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return run_expanded_blocks(
        output,
        identity=identity,
        index=index,
        families=manifest["families"],
        history_loader=histories,
        inference_family=lambda family: expanded_inference_family(
            family,
            families=manifest["families"],
            checkpoint_paths=checkpoints,
            index=index,
            history=histories(family // 4),
            raw_train_root=raw_train_root,
            allowed_sequences=allowed_sequences,
            preprocessing=prep,
            validate_prerequisites=validate,
        ),
        validate_prerequisites=validate,
        resource_ok=resource_ok,
        max_new_queries=max_new_queries,
        reuse_block=reuse,
    )


# Preserve the original D1-default API for existing callers.
run_d1_context_cache = run_expanded_context_cache
