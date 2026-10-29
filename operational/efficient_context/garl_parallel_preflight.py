"""Audit future TRAIN feature routing and completed checkpoints beside the sole trainer."""

from __future__ import annotations

import argparse
import gc
import os
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, npz, read, release, spec


def producer_for(sequence: str, fold: int, fits: list[dict]) -> str:
    """Require one INNER-OOF producer that excluded the query sequence from training."""
    candidates = [
        fit
        for fit in fits
        if fit["outer"] == fold and fit["inner"] is not None and sequence in fit["excluded_inner"]
    ]
    if len(candidates) != 1:
        raise ValueError("query requires exactly one excluded INNER-OOF producer")
    fit = candidates[0]
    if sequence in fit["train_sequences"]:
        raise ValueError("INNER-OOF query sequence occurs in producer training")
    return str(fit["key"])


def validate_history(history: object, anchors: object) -> tuple:
    """Reject out-of-range IDs, missing current slots and future observations."""
    import numpy as np

    history = np.asarray(history)
    anchors = np.asarray(anchors)
    if history.ndim != 2 or history.shape[1] != 8 or anchors.ndim != 1:
        raise ValueError("expected the existing eight slots and one-dimensional anchors")
    if not np.issubdtype(history.dtype, np.integer):
        raise ValueError("history IDs must be integers")
    valid = history >= 0
    if not np.all(valid[:, -1]) or np.any(history < -1) or np.any(history >= len(anchors)):
        raise ValueError("history has missing current slots or out-of-range IDs")
    safe = np.maximum(history, 0)
    lags = anchors[safe[:, -1], None] - anchors[safe]
    if np.any(lags[valid] < 0):
        raise ValueError("future observation in historical context")
    return valid, lags


def guard(c: Campaign) -> dict:
    """Charge this CPU-only auxiliary process and the active trainer tree together."""
    import psutil

    from .deadline import permits
    from .fast_owned_scan import owned_bytes

    current = psutil.Process()
    roots = [current]
    lease = c.out / "WRITER.lock"
    if lease.exists():
        owner = read(lease)
        try:
            trainer = psutil.Process(owner["pid"])
            if trainer.create_time() == owner["create_time"]:
                roots.append(trainer)
        except psutil.NoSuchProcess:
            pass
    processes = {}
    for root in roots:
        for process in [root, *root.children(recursive=True)]:
            processes[process.pid] = process
    rss = 0
    for process in processes.values():
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    state = {
        "combined_tree_rss_bytes": rss,
        "available_bytes": psutil.virtual_memory().available,
        "owned_bytes": owned_bytes(c.out),
        "free_disk_bytes": psutil.disk_usage(str(c.out)).free,
    }
    # Keep2GB RSS headroom for transient training batches and checkpoint serialization.
    if (
        rss > 14_000_000_000
        or state["available_bytes"] < 4 * 1024**3
        or state["owned_bytes"] > 39_000_000_000
        or state["free_disk_bytes"] < 21_000_000_000
        or not permits(c)
    ):
        raise InterruptedError("auxiliary preflight paused to preserve trainer resources")
    return state


def checkpoint_audit(c: Campaign, folder: Path) -> list[dict]:
    """Verify immutable completed CPU states; never open the trainer's changing checkpoint."""
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.training import state_digest

    progress = c.out / "garl/INDEPENDENT_PROGRESS.json"
    endpoints = read(progress)["completed_fits"] if progress.exists() else []
    results = []
    for endpoint in endpoints:
        guard(c)
        key = endpoint["key"]
        path = Path(endpoint["checkpoint"])
        if digest(path) != endpoint["sha256"]:
            raise ValueError("completed checkpoint differs from its sealed bytes")
        state = torch.load(path, map_location="cpu", weights_only=True)
        seal = state.pop("state_sha256")
        if (
            state_digest(state) != seal
            or state["protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
            or state["key"] != key
            or state["status"] != "COMPLETE"
            or state["completed_updates"] != endpoint["updates"]
            or state["epoch"] != 51
            or len(state["losses"]) != endpoint["updates"]
            or not np.isfinite(state["losses"]).all()
            or not state["optimizer"]["state"]
            or any(
                name not in state
                for name in (
                    "model",
                    "scheduler",
                    "torch_rng",
                    "cuda_rng",
                    "sampler_rng",
                    "python_rng",
                    "numpy_rng",
                    "order",
                    "cursor",
                )
            )
        ):
            raise ValueError("completed native checkpoint is not a full fixed50-epoch state")
        result = {
            "fit": key,
            "checkpoint_sha256": endpoint["sha256"],
            "complete_state_sha256": seal,
            "updates": endpoint["updates"],
            "full_state_verified_on_CPU": True,
            "optimizer_updates": 0,
            "GPU_models_constructed": 0,
        }
        atomic_json(folder / f"checkpoint_{key}.json", result)
        results.append(result)
        del state
        gc.collect()
    return results


def execute(c: Campaign) -> dict:
    """Materialize TRAIN-only query plans and leakage contracts without native model evaluation."""
    import numpy as np
    import pyarrow.parquet as pq

    c.freeze()
    folder = c.out / "garl/parallel_preflight"
    folder.mkdir(parents=True, exist_ok=True)
    freeze = read(folder / "FREEZE.json")
    if (
        digest(Path(__file__)) != freeze["implementation_sha256"]
        or freeze["parent_protocol_sha256"] != digest(c.out / "PROTOCOL.json")
        or read(c.out / "TEST_RESULTS/parallel_preflight/QA.json")["status"] != "PASSED"
    ):
        raise ValueError("parallel preflight changed after QA")
    guard(c)
    audited = checkpoint_audit(c, folder)
    admission = read(c.out / "garl/ADMISSION.json")
    sources = c.sources()
    fragments = []
    try:
        for fold in range(3):
            resources = guard(c)
            target = folder / f"fold{fold}_TRAIN_PLAN.npz"
            receipt = target.with_suffix(".json")
            if receipt.exists():
                old = read(receipt)
                if digest(target) != old["sha256"]:
                    raise ValueError("committed TRAIN plan differs from its receipt")
                fragments.append(old)
                continue
            parent = sources.source(spec(sources, fold, 8), "inner_oof")
            with np.load(c.out / f"garl/admission/outer{fold}_tokens.npz", allow_pickle=False) as z:
                tokens = z["tokens"].astype(str)
            if len(tokens) != parent.population or len(set(tokens)) != len(tokens):
                raise ValueError("TRAIN plan differs from paired source population")
            metadata = pq.read_table(
                Path(c.local["garl_annotations_candidate"]),
                columns=["sample_token", "sequence_id", "event_windows_us", "boxes_xyxy"],
                filters=[("sample_token", "in", tokens.tolist())],
                use_threads=False,
            ).to_pylist()
            rows = {row["sample_token"]: row for row in metadata}
            if set(rows) != set(tokens):
                raise ValueError("missing TRAIN plan queries; no query dropping")
            history = parent.history[:, -8:]
            valid, lags = validate_history(history, parent.anchor_us)
            keys = np.asarray(
                [
                    producer_for(rows[token]["sequence_id"], fold, admission["fits"])
                    for token in tokens
                ]
            )
            windows = np.asarray([rows[token]["event_windows_us"] for token in tokens], np.int64)
            roi = np.asarray([rows[token]["boxes_xyxy"] for token in tokens], np.float64)
            identities = {}
            for sequence in sorted({rows[token]["sequence_id"] for token in tokens}):
                restored = read(c.out / f"data_recovery/files/{sequence}.json")
                stat = (c.raw / sequence / "events.h5").stat()
                if restored["status"] != "VERIFIED" or (stat.st_size, stat.st_mtime_ns) != (
                    restored["bytes"],
                    restored["mtime_ns"],
                ):
                    raise ValueError("TRAIN plan raw identity changed")
                identities[sequence] = restored["sha256"]
            guard(c)
            npz(
                target,
                tokens=tokens,
                producer_keys=keys,
                history=history,
                valid=valid,
                lags_us=lags,
                current_windows_us=windows,
                current_boxes_xyxy=roi,
            )
            result = {
                "fold": fold,
                "role": "inner_oof",
                "sha256": digest(target),
                "parent_source_sha256": parent.identity_sha256,
                "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
                "metadata_sha256": digest(Path(c.local["garl_annotations_candidate"])),
                "query_count": len(tokens),
                "valid_native_observation_count": int(valid.sum()),
                "producer_sequence_exclusion_verified_for_every_query": True,
                "raw_sha256": identities,
                "resources_before_fold": resources,
                "native_features_generated": 0,
                "GPU_models_constructed": 0,
                "optimizer_updates": 0,
                "OLD_DEV_or_other_holdout_inputs_opened": 0,
                "plan_is_independent_of_producer_weights": True,
                "final_feature_generator_retains_its_original_checks": True,
            }
            atomic_json(receipt, result)
            fragments.append(result)
            print("TRAIN_PLAN_COMPLETE", fold, len(tokens), int(valid.sum()), flush=True)
            del parent, history, valid, lags, rows, metadata, windows, roi, keys
            gc.collect()
    finally:
        release(sources)
    result = {
        "status": "COMPLETE",
        "completed_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "folds": fragments,
        "completed_checkpoint_audits": audited,
        "GPU_models_constructed": 0,
        "optimizer_updates": 0,
        "scientific_protocol_changed": False,
        "producer_training_continued_independently": True,
    }
    atomic_json(folder / "RESULTS.json", result)
    return result


def main() -> int:
    """Run resumable auxiliary audits; resource pauses never stop the heavy trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    try:
        execute(c)
    except InterruptedError as error:
        atomic_json(
            c.out / "garl/parallel_preflight/PAUSE.json",
            {"status": "PAUSED_RESOURCE", "error": str(error), "optimizer_updates": 0},
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
