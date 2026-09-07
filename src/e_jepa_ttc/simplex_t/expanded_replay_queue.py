"""Shared-lease, resumable expanded inference queue with atomic per-query receipts."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .compiled_context import validate_block
from .expanded_replay_plan import expanded_family_queries
from .lifecycle import ExclusiveLease


def run_expanded_blocks(
    output: Path,
    *,
    identity: dict,
    index: dict[str, np.ndarray],
    families: list[dict],
    history_loader: Callable[[int], np.ndarray],
    inference_family: Callable[
        [int], AbstractContextManager[Callable[[int], dict[str, np.ndarray]]]
    ],
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
    max_new_queries: int,
    reuse_block: Callable[[int, int], dict[str, np.ndarray] | None] | None = None,
) -> dict:
    """Schedule validated blocks; callbacks own pinned input and expert loading.

    The producer context must load the correct historical family once and release
    it on exit. The mandatory validator must verify actual temporal authority,
    all source pins and producer exclusions. This queue has no permissive loader
    or authority defaults and does not implement the production adapter itself.
    """
    if (
        identity.get("pool") not in {"D1", "DENSE_OLD"}
        or type(max_new_queries) is not int
        or max_new_queries < 1
    ):
        raise ValueError("expanded pool and positive execution slice required")
    validate_prerequisites()
    if (identity["pool"] == "DENSE_OLD") != (reuse_block is not None):
        raise ValueError("DENSE requires explicit D0 reuse; D1 cannot borrow D0 query blocks")
    groups = expanded_family_queries(
        index["producer_family"], families, queries=len(index["tokens"])
    )
    if index["valid"].shape != (len(index["tokens"]), 16) or index["valid"].dtype != bool:
        raise ValueError("H16 boolean availability required")
    completed = 0
    reused = 0
    started = time.monotonic()

    def result(status: str) -> dict:
        return {
            "status": status,
            "new_blocks": completed,
            "reused_blocks": reused,
            "optimizer_updates": 0,
        }

    # Same lock as D0: do not launch a second companion GPU replay.
    with ExclusiveLease(output.parent / "CURRENT_REPLAY.lock"):
        validate_prerequisites()
        if not resource_ok():
            return result("PAUSED_RESOURCE")
        output.mkdir(parents=True, exist_ok=True)
        identity_path = output / "IDENTITY.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("expanded extraction identity changed; no silent resume")
        else:
            write_new_json(identity_path, identity)
        for family, queries in groups.items():
            history = history_loader(family // 4)
            if history.shape != index["valid"].shape or history.dtype != np.int64:
                raise ValueError("expanded history shape or identity dtype differs")
            pending = []

            def validate(
                arrays: dict[str, np.ndarray], qi: int, bound_history: np.ndarray = history
            ) -> None:
                mask = index["valid"][qi]
                validate_block(
                    arrays,
                    bound_history[qi, mask],
                    index["anchor_us"][qi] - index["lag_us"][mask],
                    int(index["roi_available_us"][qi]),
                )

            for value in queries:
                qi = int(value)
                stem = output / f"family{family:02d}_query{qi:05d}"
                receipt, payload = stem.with_suffix(".json"), stem.with_suffix(".npz")
                if receipt.exists():
                    saved = json.loads(receipt.read_text(encoding="utf-8"))
                    if (
                        saved["query"] != qi
                        or saved["family"] != family
                        or sha256(payload) != saved["sha256"]
                    ):
                        raise ValueError("expanded completed block changed")
                    with np.load(payload, allow_pickle=False) as archive:
                        validate({key: archive[key] for key in archive.files}, qi)
                elif payload.exists() or stem.with_suffix(".partial").exists():
                    raise FileExistsError("unreceipted expanded block retained for recovery audit")
                else:
                    if reuse_block is not None:
                        validate_prerequisites()
                        if not resource_ok():
                            return result("PAUSED_RESOURCE")
                        shared = reuse_block(family, qi)
                        if shared is not None:
                            validate(shared, qi)
                            reused += 1
                            continue
                    pending.append((qi, stem))
            if not pending:
                continue
            validate_prerequisites()
            if not resource_ok():
                return result("PAUSED_RESOURCE")
            with inference_family(family) as infer:
                for qi, stem in pending:
                    validate_prerequisites()
                    if not resource_ok():
                        return result("PAUSED_RESOURCE")
                    arrays = infer(qi)
                    validate(arrays, qi)
                    temporary, payload = stem.with_suffix(".partial"), stem.with_suffix(".npz")
                    with temporary.open("xb") as stream:
                        np.savez_compressed(
                            stream,
                            features145=arrays["features145"],
                            expert_ttc=arrays["expert_ttc"],
                            pair_features=arrays["pair_features"],
                            known=arrays["known"],
                            observation_ids=arrays["observation_ids"],
                            anchor_us=arrays["anchor_us"],
                            available_us=arrays["available_us"],
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.rename(payload)
                    write_new_json(
                        stem.with_suffix(".json"),
                        {
                            "query": qi,
                            "family": family,
                            "rows": int(index["valid"][qi].sum()),
                            "sha256": sha256(payload),
                            "seconds_since_launch": time.monotonic() - started,
                        },
                    )
                    completed += 1
                    if completed >= max_new_queries:
                        return result("SLICE_COMPLETE")
    return result(
        "ALL_EXPANDED_INPUTS_AVAILABLE_WITH_D0_REUSE_NOT_SCIENTIFIC_FREEZE"
        if reused
        else "ALL_EXPANDED_BLOCKS_COMPLETE_NOT_SCIENTIFIC_FREEZE"
    )
