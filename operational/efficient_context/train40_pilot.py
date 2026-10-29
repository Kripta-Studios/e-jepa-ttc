"""Measure a target-free TRAIN proxy for reusing the frozen Simplex-T H8 system."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_json, digest, npz, read

PILOT = ROOT / "artifacts/train40_feasibility_20261005"
_pool = None


def prepare(job: dict) -> tuple[object, dict]:
    """Read only the supplied current ROI and causal event windows on a CPU worker."""
    global _pool
    import torch

    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    torch.set_num_threads(4)
    if _pool is None:
        _pool = ReaderPool()
    begin = time.perf_counter()
    component: dict = {}
    tensor = encode_union(
        _pool.get(job["raw_path"]),
        job["windows"],
        job["lags"],
        job["valid"],
        tuple(job["roi"]),
        sequence_id=job["sequence_id"],
        roi_size=job["roi_size"],
        event_pixel_diff=job["event_pixel_diff"],
        profile=component,
    )
    return tensor.numpy(), {"preparation_ms": (time.perf_counter() - begin) * 1000, **component}


def select_queries(index: dict, families: list[dict], excluded: set[str]) -> list[dict]:
    """Choose four hash-ranked fresh TRAIN queries per expansion sequence, without labels."""
    import hashlib

    import numpy as np

    queries = []
    for sequence in sorted(set(index["sequences"].tolist())):
        candidates = []
        for row in np.flatnonzero(index["sequences"] == sequence):
            token = str(index["tokens"][row])
            family_id = int(index["producer_family"][0, row])
            family = families[family_id]
            if (
                token not in excluded
                and family["outer_fold"] == 0
                and family["role"] in {"inner0", "inner1", "inner2"}
                and bool(index["valid"][row, -1])
            ):
                rank = hashlib.sha256(("TRAIN40_PILOT_V1:" + token).encode()).hexdigest()
                candidates.append((rank, int(row), family_id))
        for _, row, family_id in sorted(candidates)[:4]:
            queries.append(
                {
                    "sample_token": str(index["tokens"][row]),
                    "sequence_id": sequence,
                    "pool": "D1",
                    "index_row": row,
                    "family_id": family_id,
                    "experts": families[family_id]["experts"],
                }
            )
    if not queries or len({q["sample_token"] for q in queries}) != len(queries):
        raise ValueError("nonempty unique fresh TRAIN pilot queries required")
    return sorted(queries, key=lambda q: (q["family_id"], q["sequence_id"], q["sample_token"]))


def run(c: Campaign) -> None:
    """Run the fixed H8 FP32 batch16 route, serial then four-worker CPU preparation."""
    import numpy as np
    import torch

    from .frozen_runtime import FrozenRuntime
    from .profile_safe import verify_raw

    original_out = c.out
    pause = read(original_out / "garl/USER_PAUSE_RECEIPT.json")
    if pause["status"] != "PAUSED_USER_STRATEGY" or pause["confirmed_lost_updates"] != 0:
        raise ValueError("verified complete native pause required before pilot")
    c.freeze()
    route_path = c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json"
    route = read(route_path)
    directory = c.historical / route["index_dirs"]["D1"]
    manifest = read(directory / "INDEX_MANIFEST.json")
    index_path = directory / "query_context_index.npz"
    if digest(index_path) != manifest["index_sha256"]:
        raise ValueError("TRAIN proxy index changed")
    with np.load(index_path, allow_pickle=False) as z:
        index = {key: z[key] for key in z.files}
    queries = select_queries(
        index, manifest["families"], {q["sample_token"] for q in route["queries"]}
    )
    prep_path = Path(route["preprocessing_path"])
    if digest(prep_path) != route["preprocessing_sha256"]:
        raise ValueError("frozen input preprocessing changed")
    prep = read(prep_path)["config"]
    raw_plan = read(original_out / "data_recovery/DOWNLOAD_PLAN.json")
    raw_pins = {v["sequence_id"]: v for v in raw_plan["files"]}
    jobs = []
    for query in queries:
        row = query["index_row"]
        raw = c.raw / query["sequence_id"] / "events.h5"
        receipt = read(original_out / "data_recovery/files" / (query["sequence_id"] + ".json"))
        expected = {"bytes": receipt["bytes"], "mtime_ns": receipt["mtime_ns"]}
        verify_raw(c, raw, expected)
        if (
            receipt["status"] != "VERIFIED"
            or raw.stat().st_size != receipt["bytes"]
            or receipt["bytes"] != raw_pins[query["sequence_id"]]["bytes"]
            or receipt["sha256"] != raw_pins[query["sequence_id"]]["sha256"]
            or receipt["path"] != str(raw)
        ):
            raise ValueError("pilot raw restoration is not verified")
        valid = index["valid"][row].copy()
        valid[:-8] = False
        jobs.append(
            {
                "raw_path": str(raw),
                "sequence_id": query["sequence_id"],
                "windows": index["base_windows_us"][row],
                "lags": index["lag_us"],
                "valid": valid,
                "roi": index["square_xyxy"][row],
                "roi_size": prep["roi_size"],
                "event_pixel_diff": prep["event_pixel_diff"],
            }
        )
    freeze = {
        "schema": "train40_frozen_H8_feasibility_pilot_v1",
        "native_pause_receipt_sha256": digest(original_out / "garl/USER_PAUSE_RECEIPT.json"),
        "original_protocol_sha256": digest(original_out / "PROTOCOL.json"),
        "historical_route_sha256": digest(route_path),
        "index_sha256": digest(index_path),
        "preprocessing_sha256": digest(prep_path),
        "source_sha256": digest(Path(__file__)),
        "queries": queries,
        "query_count": len(queries),
        "sequence_count": len({q["sequence_id"] for q in queries}),
        "population": (
            "fresh target-free queries from existing D1 expansion TRAIN index; proxy, not full40"
        ),
        "mode_order": ["serial", "four_workers"],
        "producer_batch": 16,
        "history": "original H8 most recent8 fixed slots",
        "optimizer_updates": 0,
        "weights": route["checkpoints"],
        "ROI_source": "supplied current ROI",
        "no_labels_or_evaluation_scores_read": True,
        "known_limitations": [
            "nine TRAIN40 raw sequences absent",
            "serial runs first; order and OS cache confounded",
            "no full40 encoder fitting or external evaluation",
        ],
    }
    pin = PILOT / "PILOT_FREEZE.json"
    if pin.exists():
        raise ValueError("preserve existing pilot freeze and fragments; inspect before resume")
    atomic_json(pin, freeze)
    c.out = PILOT
    c.policy = dict(c.policy, max_tree_rss_gib=16_000_000_000 / 1024**3)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    runtime = FrozenRuntime(c, route)
    results = {}
    references = {}
    start = time.monotonic()
    try:
        for mode in ("serial", "four_workers"):
            c.out = PILOT / "modes" / mode
            c.out.mkdir(parents=True, exist_ok=True)
            rows = []
            executor = ProcessPoolExecutor(max_workers=4) if mode == "four_workers" else None
            mode_start = time.perf_counter()
            try:
                for family_id in sorted({q["family_id"] for q in queries}):
                    members = [i for i, q in enumerate(queries) if q["family_id"] == family_id]
                    runtime.load(queries[members[0]])
                    pending = {}
                    for position, i in enumerate(members):
                        c.require_resources()
                        if time.monotonic() - start > 900:
                            raise InterruptedError(
                                "pilot15-minute execution cap reached; retain fragments"
                            )
                        if executor:
                            for next_i in members[position : position + 4]:
                                if next_i not in pending:
                                    pending[next_i] = executor.submit(prepare, jobs[next_i])
                            wait_start = time.perf_counter()
                            array, components = pending.pop(i).result()
                            wait_ms = (time.perf_counter() - wait_start) * 1000
                        else:
                            array, components = prepare(jobs[i])
                            wait_ms = components["preparation_ms"]
                        tensor = torch.from_numpy(array)
                        raw_hash = digest_array(array)
                        output = runtime.prepared(
                            queries[i], index, "H8", tensor, jobs[i]["valid"], compact=False
                        )
                        xs = output["xs"]
                        parity = "REFERENCE"
                        if mode == "serial":
                            references[i] = (raw_hash, xs, output["phase"], output["ttc"])
                        else:
                            reference = references[i]
                            if (
                                raw_hash != reference[0]
                                or any(
                                    not np.array_equal(a, b)
                                    for a, b in zip(xs, reference[1], strict=True)
                                )
                                or not np.array_equal(output["phase"], reference[2])
                                or not np.array_equal(output["ttc"], reference[3])
                            ):
                                raise ValueError(
                                    "pilot parallel raw/features/phase/TTC parity failed"
                                )
                            parity = "EXACT"
                        fragment = PILOT / f"fragments/{mode}_{i:03d}.json"
                        arrays_path = fragment.with_suffix(".npz")
                        arrays_path.parent.mkdir(parents=True, exist_ok=True)
                        npz(
                            arrays_path,
                            features=xs[0],
                            times=xs[1],
                            valid=xs[2],
                            experts=xs[3],
                            phase=output["phase"],
                            ttc=output["ttc"],
                        )
                        row = {
                            "mode": mode,
                            "query": queries[i],
                            "raw_tensor_sha256": raw_hash,
                            "parity": parity,
                            "wait_ms": wait_ms,
                            **components,
                            **{k: v for k, v in output.items() if k.endswith("_ms")},
                            "arrays_sha256": digest(arrays_path),
                        }
                        atomic_json(fragment, row)
                        rows.append(row)
                        atomic_json(
                            PILOT / "PROGRESS.json",
                            {
                                "status": "RUNNING",
                                "mode": mode,
                                "completed": len(rows),
                                "queries_per_mode": len(queries),
                                "optimizer_updates": 0,
                            },
                        )
                        del array, tensor, output
            finally:
                if executor:
                    executor.shutdown(wait=True, cancel_futures=True)
            elapsed = time.perf_counter() - mode_start
            results[mode] = {
                "queries": len(rows),
                "wall_seconds_including_model_load_startup_and_fragment_writes": elapsed,
                "queries_per_second": len(rows) / elapsed,
                "naive_88744_query_hours": elapsed * 88744 / len(rows) / 3600,
                "timings": {
                    key: {
                        "mean_ms": float(np.mean([r[key] for r in rows])),
                        "p50_ms": float(np.median([r[key] for r in rows])),
                        "p95_ms": float(np.quantile([r[key] for r in rows], 0.95)),
                    }
                    for key in ("preparation_ms", "wait_ms", "producer_ms", "head_ms")
                },
            }
        atomic_json(
            PILOT / "RESULTS.json",
            {
                "status": "COMPLETE",
                "freeze_sha256": digest(pin),
                "modes": results,
                "input_output_parity": "EXACT",
                "optimizer_updates": 0,
                "external_comparison_completed": False,
                "projections_are_not_validated_full40_ETA": True,
                "mode_order_confounds_causal_speedup": True,
            },
        )
    finally:
        runtime.close()
        if _pool is not None:
            _pool.close()


def digest_array(array: object) -> str:
    """Hash a full numeric CPU tensor without lossy casting or compression."""
    import hashlib

    import numpy as np

    a = np.asarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    """Execute only the explicitly authorized zero-update feasibility pilot."""
    os.environ.setdefault("PYTHONUTF8", "1")
    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    with Lease(c.out):
        run(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
