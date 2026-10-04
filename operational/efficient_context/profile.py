"""Paired frozen raw/ROI preparation measurements on the64 admitted TRAIN IDs."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .common import Campaign, atomic_bytes, atomic_json, digest, read


def run_profile(c: Campaign) -> None:
    """Commit each exact tensor comparison and independent R0 preparation request."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union

    from .frozen_runtime import FrozenRuntime

    c.require_resources()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    p = read(c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json")
    prep = read(Path(p["preprocessing_path"]))["config"]
    indexes = {}
    for name, relative in p["index_dirs"].items():
        directory = c.historical / relative
        manifest = read(directory / "INDEX_MANIFEST.json")
        if digest(directory / "query_context_index.npz") != manifest["index_sha256"]:
            raise ValueError("profile index changed")
        with np.load(directory / "query_context_index.npz", allow_pickle=False) as z:
            indexes[name] = {
                k: z[k]
                for k in (
                    "tokens",
                    "base_windows_us",
                    "lag_us",
                    "valid",
                    "square_xyxy",
                    "anchor_us",
                    "roi_available_us",
                )
            }
    pool = ReaderPool()
    runtime = FrozenRuntime(c, p)
    implementation = {
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "historical_route_sha256": digest(
            c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json"
        ),
        "queries": 64,
        "blocks": 3,
        "arms": ["H1", "H8", "H16", "WIDE"],
        "regimes": ["R0", "R2_segments_inside_R0"],
        "R1": "NOT_IMPLEMENTED",
        "routes": ["reference_batch16", "mapped_valid_slots"],
        "parity": {
            "features": 1e-4,
            "phase": 1e-5,
            "ttc_s": 0.01,
            "times": 1e-7,
            "mask": 0,
            "preprocessing": 0,
        },
        "files": [
            {"path": str(v), "sha256": digest(v)}
            for v in (
                Path(__file__),
                Path(__file__).with_name("frozen_runtime.py"),
                Path(__file__).parents[2] / "src/e_jepa_ttc/efficient_context/mapped_union.py",
            )
        ],
    }
    freeze = c.out / "profile/PROTOCOL.json"
    if freeze.exists() and read(freeze) != implementation:
        raise ValueError("frozen engineering implementation changed")
    if not freeze.exists():
        atomic_json(freeze, implementation)
    rows = []
    parities = []
    try:
        for mode in ("application_cold", "warm_block1", "warm_block2"):
            for qi, query in enumerate(p["queries"]):
                idx = indexes[query["pool"]]
                i = query["index_row"]
                if str(idx["tokens"][i]) != query["sample_token"]:
                    raise ValueError("TRAIN profile query mapping changed")
                raw_path = c.raw / query["sequence_id"] / "events.h5"
                expected = next(r for r in p["raw"] if r["path"] == str(raw_path))
                stat = raw_path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
                    raise ValueError("raw TRAIN binding changed")
                for label, length in (("H1", 1), ("H8", 8), ("H16", 16), ("WIDE", 8)):
                    if label not in runtime.heads:
                        continue
                    c.require_resources()
                    receipt = c.out / f"profile/fragments/{mode}_{qi:02d}_{label}.json"
                    if receipt.exists():
                        saved = read(receipt)
                        rows.extend(saved["requests"])
                        parities.append(saved["parity"])
                        continue
                    valid = idx["valid"][i].copy()
                    if label == "WIDE":
                        valid &= np.isin(np.arange(16), WIDE_SLOTS)
                    else:
                        valid[:-length] = False
                    order = ("reference", "mapped")
                    if (
                        int(
                            hashlib.sha256(
                                (mode + query["sample_token"] + label).encode()
                            ).hexdigest(),
                            16,
                        )
                        % 2
                    ):
                        order = order[::-1]
                    outputs = {}
                    prepared = {}
                    requests = []
                    runtime.load(query)
                    for route in order:
                        c.require_resources()
                        if mode == "application_cold":
                            pool.close()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        reader = pool.get(raw_path)
                        options = dict(
                            sequence_id=query["sequence_id"],
                            roi_size=prep["roi_size"],
                            event_pixel_diff=prep["event_pixel_diff"],
                        )
                        component = {}
                        if route == "mapped":
                            tensor = encode_union(
                                reader,
                                idx["base_windows_us"][i],
                                idx["lag_us"],
                                valid,
                                tuple(idx["square_xyxy"][i]),
                                profile=component,
                                **options,
                            )
                        else:
                            tensor = encode_context_union(
                                reader,
                                idx["base_windows_us"][i],
                                idx["lag_us"],
                                valid,
                                tuple(idx["square_xyxy"][i]),
                                **options,
                            )
                        preparation_ms = (time.perf_counter() - start) * 1000
                        result = runtime.prepared(
                            query, idx, label, tensor, valid, compact=route == "mapped"
                        )
                        elapsed = (time.perf_counter() - start) * 1000
                        outputs[route] = tensor
                        prepared[route] = result
                        requests.append(
                            {
                                "mode": mode,
                                "query": query["sample_token"],
                                "label": label,
                                "route": route,
                                "regime": "R0",
                                "milliseconds": elapsed,
                                "preparation_ms": preparation_ms,
                                **{
                                    k: result[k]
                                    for k in (
                                        "prepared_ms",
                                        "transfer_ms",
                                        "producer_ms",
                                        "normalization_ms",
                                        "head_ms",
                                        "producer_observations",
                                    )
                                },
                                "order": "/".join(order),
                                **component,
                            }
                        )
                    exact = torch.equal(outputs["reference"], outputs["mapped"])
                    parity = {
                        "query": query["sample_token"],
                        "label": label,
                        "mode": mode,
                        "status": "EXACT" if exact else "FAILED_INTEGRITY",
                        "max_abs": float((outputs["reference"] - outputs["mapped"]).abs().max()),
                    }
                    errors = {}
                    for j, name in enumerate(("features", "times", "mask", "experts")):
                        a, b = prepared["reference"]["xs"][j], prepared["mapped"]["xs"][j]
                        errors[name] = float(np.max(np.abs(a.astype(float) - b.astype(float))))
                        limit = {"features": 1e-4, "times": 1e-7, "mask": 0, "experts": 1e-5}[name]
                        if errors[name] > limit:
                            parity["status"] = "FAILED_INTEGRITY"
                    for name, limit in (("phase", 1e-5), ("ttc", 0.01)):
                        errors[name] = float(
                            np.max(np.abs(prepared["reference"][name] - prepared["mapped"][name]))
                        )
                        if errors[name] > limit:
                            parity["status"] = "FAILED_INTEGRITY"
                    parity["output_errors"] = errors
                    if label != "WIDE":
                        cached = runtime.cached[label]
                        parity["historical_input_errors"] = {
                            name: float(
                                np.max(
                                    np.abs(
                                        actual.astype(float)
                                        - cached[name][qi : qi + 1].astype(float)
                                    )
                                )
                            )
                            for name, actual in zip(
                                ("features", "times", "valid", "experts"),
                                prepared["reference"]["xs"],
                                strict=True,
                            )
                        }
                        if any(
                            v > {"features": 1e-4, "times": 1e-7, "valid": 0, "experts": 1e-5}[k]
                            for k, v in parity["historical_input_errors"].items()
                        ):
                            parity["status"] = "FAILED_INTEGRITY"
                    if parity["status"] != "FAILED_INTEGRITY":
                        parity["status"] = (
                            "NUMERICALLY_VALIDATED" if any(errors.values()) else "EXACT"
                        )
                    atomic_json(
                        receipt,
                        {
                            "requests": requests,
                            "parity": parity,
                            "raw_path": str(raw_path),
                            "raw_bytes": stat.st_size,
                        },
                    )
                    rows.extend(requests)
                    parities.append(parity)
                    if parity["status"] == "FAILED_INTEGRITY":
                        raise ValueError("frozen raw/producer/output parity failed: " + str(parity))
                    print(f"E1_{mode}_{qi + 1}/64_{label}_{parity['status']}", flush=True)
                    del outputs, tensor
        atomic_bytes(
            c.out / "PROFILE_COMPONENTS.csv", pd.DataFrame(rows).to_csv(index=False).encode()
        )
        grouped = pd.DataFrame(rows).groupby(["regime", "mode", "label", "route"]).milliseconds
        table = grouped.agg(["count", "median", lambda x: x.quantile(0.95)]).reset_index()
        table.columns = ["regime", "mode", "label", "route", "requests", "p50_ms", "p95_ms"]
        atomic_bytes(c.out / "RUNTIME_COMPARISON.csv", table.to_csv(index=False).encode())
        atomic_json(
            c.out / "INPUT_OUTPUT_PARITY.json",
            {
                "status": "NUMERICALLY_VALIDATED_EXACT_PREPROCESSING",
                "contexts": len(parities),
                "cases": parities,
                "producer_dispatch_changed": True,
                "R0_end_to_end_measured": True,
                "R1": "NOT_IMPLEMENTED_NO_COST_CLAIM",
                "R2": "TRANSFER_PRODUCERS_NORMALIZATION_HEAD_SEGMENTS_OF_EACH_R0_REQUEST",
                "gpu_shared": True,
                "optimizer_updates": 0,
                "WIDE": "COMPLETE"
                if "WIDE" in runtime.heads
                else "DEFERRED_UNTIL_THREE_ENDPOINT_SEAL",
            },
        )
    finally:
        pool.close()
        runtime.close()
