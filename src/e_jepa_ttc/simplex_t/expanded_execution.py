"""Wire D1 input metadata, frozen producer inference and the shared replay queue."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .expanded_history_loader import ExpandedHistoryLoader
from .expanded_inference import expanded_inference_family
from .expanded_replay_plan import inspect_expanded_replay
from .expanded_replay_queue import run_expanded_blocks


def run_d1_context_cache(
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
) -> dict:
    """Run only under independently verified expanded authority and absolute limits.

    The caller must verify the owner's supplementary time acknowledgment, exact
    role map and transitive exclusions, and the cooperative resource permission.
    This API does not invent an ACK schema or turn the original ACK into D1 access.
    DENSE's D0-reuse integration remains separate; no DENSE fallback is allowed.
    """
    validate_expanded_authority()
    if not resource_ok():
        return {"status": "PAUSED_RESOURCE", "new_blocks": 0, "optimizer_updates": 0}
    inspect_expanded_replay(index_root, manifest_sha256=index_manifest_sha256, pool="D1")
    prep_hash = "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
    if sha256(preprocessing) != prep_hash or sha256(ancestry) != ancestry_sha256:
        raise ValueError("frozen preprocessing or historical ancestry changed")
    manifest = json.loads((index_root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    if (
        Path(manifest["ancestry"]["path"]).resolve() != ancestry.resolve()
        or manifest["ancestry"]["sha256"] != ancestry_sha256
    ):
        raise ValueError("D1 index changes acknowledged historical ancestry")
    with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    if set(map(str, index["sequences"])) != allowed_sequences:
        raise ValueError("D1 index sequence set differs from independently authorized TRAIN pool")
    histories = ExpandedHistoryLoader(
        dedup_root,
        manifest_sha256=dedup_manifest_sha256,
        index_manifest_sha256=index_manifest_sha256,
        pool="D1",
        producer_family=index["producer_family"],
        valid=index["valid"],
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
        )
    }
    identity = {
        "schema": "simplex_t_query_context_fp32_cache_v1",
        "pool": "D1",
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
        "torch": str(torch.__version__),
        "batch_size": 16,
        "layout": "one query, chronological H16; absent slots zero, never exported",
        "precision": "FP32",
        "tf32": False,
        "optimizer_updates": 0,
    }

    def validate() -> None:
        validate_expanded_authority()
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
    )
