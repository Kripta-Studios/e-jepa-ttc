"""Independent TRAIN-only prepared-head measurements while raw storage is absent."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_bytes, atomic_json, digest, read


def run(c: Campaign) -> None:
    """Measure head segments only; never claim full producer or HDF5 speed."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS
    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from e_jepa_ttc.simplex_t.training import load_checkpoint

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    c.require_resources()
    parent = c.freeze()
    route_root = c.historical / "artifacts/simplex_t/shared_gpu_route_20261004"
    original = read(route_root / "PROTOCOL.json")
    night = c.historical / "artifacts/simplex_t/nocturnal_20261003"
    manifest = read(night / "PROFILE_INPUT_EXPORT.json")
    if digest(night / "PROFILE_INPUT_EXPORT.json") != original["head_index_sha256"]:
        raise ValueError("frozen historical profile inventory changed")
    models, inputs, pins = {}, {}, {}
    for label in ("H1", "H8", "H16"):
        record = manifest["models"][label + "_SEED7"]
        head = TemporalRefiner(TemporalConfig(**record["constructor"]["config"]))
        for kind in ("weights", "inputs"):
            file = night / record[kind + "_path"]
            if digest(file) != record[kind + "_sha256"]:
                raise ValueError("prepared historical head source changed")
            pins[label + "_" + kind] = record[kind + "_sha256"]
        with np.load(night / record["weights_path"], allow_pickle=False) as z:
            head.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
        models[label] = head.float().eval()
        with np.load(night / record["inputs_path"], allow_pickle=False) as z:
            inputs[label] = {k: z[k].copy() for k in ("features", "times", "valid", "experts")}
    checkpoint = c.out / "fits/seed7/fold0/checkpoint_last.pt"
    state = load_checkpoint(checkpoint)
    if state["status"] != "COMPLETED" or state["completed_updates"] != 2500:
        raise ValueError("prepared TRAIN profile requires the completed fold0 WIDE head")
    models["WIDE"] = load_endpoint(
        checkpoint,
        TemporalConfig(**parent["sources"]["0"]["model"]),
        seed=7,
        freeze_sha256=digest(c.out / "PROTOCOL.json"),
        train_source_sha256=parent["sources"]["0"]["wide_sha256"],
        endpoint_sha256=digest(checkpoint),
    )
    pins["WIDE_checkpoint"] = digest(checkpoint)
    inputs["WIDE"] = {
        "features": inputs["H16"]["features"][:, WIDE_SLOTS].copy(),
        "times": inputs["H16"]["times"][:, WIDE_SLOTS].copy(),
        "valid": inputs["H16"]["valid"][:, WIDE_SLOTS].copy(),
        "experts": inputs["H16"]["experts"].copy(),
    }
    indices = {}
    for pool, relative in original["index_dirs"].items():
        directory = c.historical / relative
        inventory = read(directory / "INDEX_MANIFEST.json")
        if digest(directory / "query_context_index.npz") != inventory["index_sha256"]:
            raise ValueError("prepared TRAIN query index changed")
        with np.load(directory / "query_context_index.npz", allow_pickle=False) as z:
            indices[pool] = {k: z[k].copy() for k in ("tokens", "lag_us")}
    for i, query in enumerate(original["queries"]):
        index = indices[query["pool"]]
        if str(index["tokens"][query["index_row"]]) != query["sample_token"]:
            raise ValueError("prepared TRAIN query identity changed")
        gap = np.diff(-index["lag_us"][list(WIDE_SLOTS)]) / 1e6
        valid = inputs["WIDE"]["valid"][i]
        inputs["WIDE"]["times"][i, :, 2] = 0
        inputs["WIDE"]["times"][i, 1:, 2] = gap.astype(np.float32) * valid[:-1]
        inputs["WIDE"]["times"][i, ~valid] = 0
    protocol = {
        "source_pins": pins,
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "queries": 64,
        "blocks": 3,
        "heads": ["H1", "H8", "H16", "WIDE"],
        "warmups_per_head": 10,
        "scope": "R2_HEAD_ONLY_CPU_FP32; excludes producers, transfer, raw, ROI and ingestion",
        "WIDE_train_fold0_endpoint_only": True,
        "OLD_DEV_opened_by_this_profile": False,
        "files": [{"path": str(Path(__file__)), "sha256": digest(Path(__file__))}],
    }
    freeze = c.out / "prepared_heads/PROTOCOL.json"
    if freeze.exists() and read(freeze) != protocol:
        raise ValueError("prepared head engineering freeze changed")
    if not freeze.exists():
        atomic_json(freeze, protocol)
    records = []
    for label, head in models.items():
        for _ in range(10):
            with torch.inference_mode():
                head(
                    *(
                        torch.from_numpy(inputs[label][k][0:1].copy())
                        for k in ("features", "times", "valid", "experts")
                    )
                )
        for block in range(3):
            for i, query in enumerate(original["queries"]):
                c.require_resources()
                receipt = c.out / f"prepared_heads/fragments/{label}_{block}_{i:02d}.json"
                if receipt.exists():
                    records.append(read(receipt))
                    continue
                xs = tuple(
                    torch.from_numpy(inputs[label][k][i : i + 1].copy())
                    for k in ("features", "times", "valid", "experts")
                )
                begin = time.perf_counter_ns()
                with torch.inference_mode():
                    output = head(*xs)
                    ttc = phase_to_ttc(output["point_phase"].double())
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                if not torch.isfinite(ttc).all():
                    raise ValueError("nonfinite prepared-head output")
                row = {
                    "label": label,
                    "block": block,
                    "query": query["sample_token"],
                    "milliseconds": elapsed,
                    "ttc_s": float(ttc[0]),
                    "point_phase": float(output["point_phase"][0]),
                    "regime": "R2_HEAD_ONLY_CPU_FP32",
                    "optimizer_updates": 0,
                }
                atomic_json(receipt, row)
                records.append(row)
    table = pd.DataFrame(records)
    stream = io.BytesIO()
    table.to_csv(stream, index=False)
    atomic_bytes(c.out / "prepared_heads/MEASUREMENTS.csv", stream.getvalue())
    summary = (
        table.groupby(["regime", "label", "block"])
        .milliseconds.agg(requests="count", p50_ms="median", p95_ms=lambda v: v.quantile(0.95))
        .reset_index()
    )
    atomic_bytes(c.out / "prepared_heads/RUNTIME.csv", summary.to_csv(index=False).encode())
    atomic_json(
        c.out / "PREPARED_HEAD_RESULTS.json",
        {
            "status": "COMPLETE",
            "requests": len(records),
            "optimizer_updates": 0,
            "scope": protocol["scope"],
            "R0_or_producer_R2_claim": False,
        },
    )
    print("PREPARED_TRAIN_HEADS_COMPLETE", len(records), flush=True)


def main() -> int:
    """Execute the viable prepared TRAIN segment independently of unavailable E:."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    with Lease(c.out):
        run(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
