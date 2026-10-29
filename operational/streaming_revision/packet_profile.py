"""CPU-only packet storage comparison on preselected TRAIN40 sensor queries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.efficient_context.common import digest
from operational.sota_evidence.test_inputs import (
    EXPOSURE_COLUMNS,
    INPUT_COLUMNS,
    describe_input,
    supported_job,
)
from operational.train40_system.durable_io import atomic_json

from .packets import PacketRing


def run(args: argparse.Namespace) -> None:
    """Require exact raw/ROI/voxel parity before reporting paired CPU cost."""
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "operational.streaming_revision._reference_packets", args.reference
    )
    if spec is None or spec.loader is None:
        raise ValueError("reference packet module required")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    tokens = set(pilot["queries"])
    source = pd.read_parquet(args.campaign / "TRAIN40_ROWS.parquet", columns=list(INPUT_COLUMNS))
    selected = source.loc[source.sample_token.isin(tokens)].sort_values(
        ["sequence_id", "timestamp_us"]
    )
    media = pd.read_parquet(args.raw_root / "data/train.parquet", columns=list(EXPOSURE_COLUMNS))
    exposure = {(r["sequence_id"], r["rgb_member_path"]): r for r in media.to_dict("records")}
    pool = ReaderPool()
    factories = {
        "reference": reference.PacketRing,
        "compact_arithmetic": PacketRing,
        "blocks_50ms": lambda **kw: PacketRing(packet_span_us=50000, **kw),
    }
    measurements, parity = [], []
    sequence = ""
    rings: dict = {}
    spans: dict = {}
    begin = time.monotonic()
    try:
        for ordinal, row in enumerate(selected.to_dict("records")):
            if time.monotonic() - begin > 180:
                raise TimeoutError("bounded CPU profile exceeded 180 seconds")
            job = describe_input(row, exposure, args.raw_root, split="train")
            reader = pool.get(job["path"])
            job = supported_job(
                job, int(reader.datasets["events/t"][0]), int(reader.datasets["events/t"][-1])
            )
            if sequence != job["sequence"]:
                sequence = job["sequence"]
                rings = {
                    key: factory(max_bytes=512 * 1024**2) for key, factory in factories.items()
                }
                spans = {}
            first = max(
                int(reader.datasets["events/t"][0]), int(np.asarray(job["windows"]).min()) - 350000
            )
            end = job["anchor"]
            for ring in rings.values():
                if ring.end is not None and ring.end < first:
                    ring.reset()
            previous = rings["reference"].end
            start = first if previous is None else previous
            raw = reader.read_window(start, end)
            for iteration in range(args.repeats):
                order = np.random.default_rng(ordinal * 100 + iteration).permutation(
                    list(factories)
                )
                outputs, tails = {}, {}
                for name in order:
                    ring = rings[name]
                    if iteration == 0:
                        tick = time.perf_counter()
                        ring.push(raw, start, end)
                        spans[name] = (time.perf_counter() - tick) * 1000
                    tick = time.perf_counter()
                    output = ring.read_roi(first, end, tuple(job["square"]), 5.0)
                    roi_ms = (time.perf_counter() - tick) * 1000
                    outputs[name] = output
                    tick = time.perf_counter()
                    tails[name] = ring.read_roi(
                        int(job["windows"][-1][0]), end, tuple(job["square"]), 5.0
                    )
                    tail_ms = (time.perf_counter() - tick) * 1000
                    measurements.append(
                        {
                            "ordinal": ordinal,
                            "sample_token": row["sample_token"],
                            "variant": name,
                            "iteration": iteration,
                            "push_ms": spans[name] if iteration == 0 else None,
                            "roi_ms": roi_ms,
                            "tail_roi_ms": tail_ms,
                            "retained_bytes": ring.retained_bytes,
                            "packets": len(ring.packets),
                            "input_events": len(raw["t"]),
                            "output_events": len(output["t"]),
                            "cold": previous is None,
                        }
                    )
                for name, values in outputs.items():
                    if any(not np.array_equal(values[k], outputs["reference"][k]) for k in values):
                        raise ValueError(f"{name}: raw ROI mismatch")
                    if any(
                        not np.array_equal(tails[name][k], tails["reference"][k])
                        for k in tails[name]
                    ):
                        raise ValueError(f"{name}: tail ROI mismatch")
                    mapped = map_roi(values, tuple(job["square"]), 128, 5.0)
                    window = job["windows"][-1]
                    encoded = encode_mapped(mapped, int(window[0]), int(window[1]), 128, sequence)
                    if name == "reference":
                        continue
                    expected = encode_mapped(
                        map_roi(outputs["reference"], tuple(job["square"]), 128, 5.0),
                        int(window[0]),
                        int(window[1]),
                        128,
                        sequence,
                    )
                    if not torch.equal(encoded, expected):
                        raise ValueError(f"{name}: voxel mismatch")
                parity.append(
                    {"ordinal": ordinal, "iteration": iteration, "raw_and_voxel_exact": True}
                )
            pd.DataFrame(measurements).to_csv(args.output / "MEASUREMENTS.csv", index=False)
            print(f"packet query {ordinal + 1}/{len(selected)}", flush=True)
    finally:
        pool.close()
    summary = []
    frame = pd.DataFrame(measurements)
    for name, group in frame.groupby("variant"):
        warm = group.loc[~group.cold]
        summary.append(
            {
                "variant": name,
                "warm_push_median_ms": float(warm.push_ms.median()),
                "roi_median_ms": float(group.roi_ms.median()),
                "warm_roi_median_ms": float(warm.roi_ms.median()),
                "warm_tail_roi_median_ms": float(warm.tail_roi_ms.median()),
                "peak_retained_bytes": int(group.retained_bytes.max()),
            }
        )
    pd.DataFrame(summary).to_csv(args.output / "SUMMARY.csv", index=False)
    atomic_json(
        args.output / "RESULT.json",
        cast(
            dict,
            {
                "status": "COMPLETE_CPU_PROFILE",
                "gpu_seconds": 0,
                "scope": "TRAIN40 queries; repeated ROI reads, single push per query; "
                "concurrent frozen GPU inference",
                "no_labels_read": True,
                "reference_sha256": digest(args.reference),
                "source_sha256": digest(Path(__file__)),
                "packets_sha256": digest(Path(__file__).with_name("packets.py")),
                "parity": parity,
                "summary": summary,
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("campaign", "raw-root", "pilot", "reference", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
