"""Fixed-64 TRAIN causal replay, paired preparation and frozen-output profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, read
from .r1_reader import ResidentReplay

ARMS = {"H1": 1, "H8": 8, "H16": 16, "WIDE": 8}
BLOCKS = ("application_cold", "warm_block1", "warm_block2")


class RuntimeCampaign(Campaign):
    """Read historical contracts but place every new runtime receipt in R1."""

    def __init__(self, base: Campaign, output: Path) -> None:
        self.base = base
        self.out = base.out
        self.output = output
        self.historical = base.historical

    def freeze(self) -> dict:
        """Validate the existing immutable scientific contract."""
        return self.base.freeze()

    def require_resources(self) -> None:
        """Keep two GiB host headroom and ten GB disk without old-root writes."""
        import psutil

        if (self.output / "STOP_REQUEST").exists():
            raise InterruptedError("R1 stop requested; completed paired receipts preserved")
        if psutil.virtual_memory().available < 2 * 1024**3:
            raise InterruptedError("R1 needs two GiB host headroom")
        if shutil.disk_usage(self.output).free < 10_000_000_000:
            raise InterruptedError("R1 needs ten GB output disk headroom")


def load_inputs(base: Campaign) -> tuple[dict, dict, dict]:
    """Resolve only the same frozen TRAIN query indexes as E1 R0."""
    import numpy as np

    path = base.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json"
    protocol = read(path)
    if len(protocol["queries"]) != 64:
        raise ValueError("exactly the original 64 TRAIN queries required")
    indexes = {}
    keys = (
        "tokens",
        "base_windows_us",
        "lag_us",
        "valid",
        "square_xyxy",
        "anchor_us",
        "roi_available_us",
    )
    for name, relative in protocol["index_dirs"].items():
        directory = base.historical / relative
        manifest = read(directory / "INDEX_MANIFEST.json")
        if digest(directory / "query_context_index.npz") != manifest["index_sha256"]:
            raise ValueError("fixed query index changed")
        with np.load(directory / "query_context_index.npz", allow_pickle=False) as z:
            indexes[name] = {key: z[key] for key in keys}
    return protocol, indexes, read(Path(protocol["preprocessing_path"]))["config"]


def schedule(protocol: dict, indexes: dict) -> list[tuple[int, dict]]:
    """Reorder the exact query set chronologically, keeping original parity IDs."""
    return sorted(
        enumerate(protocol["queries"]),
        key=lambda pair: (
            pair[1]["sequence_id"],
            int(indexes[pair[1]["pool"]]["anchor_us"][pair[1]["index_row"]]),
            pair[1]["sample_token"],
        ),
    )


def parity(reference: Tensor, optimized: Tensor, first: dict, second: dict) -> dict:
    """Enforce historical tolerances; exact preprocessing is mandatory."""
    import numpy as np
    import torch

    if not torch.equal(reference, optimized):
        raise ValueError("R1 preprocessing is not bit exact")
    errors: dict[str, float] = {"preprocessing": 0.0}
    for key, limit in dict(
        features=1e-4, times=1e-7, mask=0, experts=1e-5, phase=1e-5, ttc=0.01
    ).items():
        if not first:
            break
        if key in ("phase", "ttc"):
            a, b = first[key], second[key]
        else:
            j = ("features", "times", "mask", "experts").index(key)
            a, b = first["xs"][j], second["xs"][j]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite R1 output")
        errors[key] = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        if errors[key] > limit:
            raise ValueError(f"R1 output parity failed: {key}={errors[key]}")
    return {
        "status": "NUMERICALLY_VALIDATED" if any(errors.values()) else "EXACT",
        "max_absolute_errors": errors,
    }


def run(args: argparse.Namespace) -> None:
    """Execute zero-update engineering work, preserving each complete pair."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS
    from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader
    from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union

    from .frozen_runtime import FrozenRuntime
    from .profile_safe import verify_raw

    base = Campaign(args.protocol)
    base.freeze()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, indexes, prep = load_inputs(base)
    ordered = schedule(protocol, indexes)
    binding = dict(
        source_files={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("r1_reader.py"),
                Path(__file__).with_name("frozen_runtime.py"),
                ROOT / "src/e_jepa_ttc/efficient_context/mapped_union.py",
                ROOT / "src/e_jepa_ttc/simplex_t/context_raw_union.py",
                ROOT / "src/e_jepa_ttc/simplex_t/cached_event_reader.py",
            )
        },
        parent_protocol_sha256=digest(base.out / "PROTOCOL.json"),
        query_sha256=hashlib.sha256(
            json.dumps(protocol["queries"], sort_keys=True).encode()
        ).hexdigest(),
        capacity_bytes=args.capacity_mib * 1024**2,
        query_count=64,
        arms=list(ARMS),
        blocks=list(BLOCKS),
        optimizer_updates=0,
        replay_start="first requested history start per sequence and arm",
        ingestion="all intervening events through each exclusive cutoff; gaps charged",
        outputs="CPU_PREPROCESSING_ONLY" if args.cpu_only else "FROZEN_FP32_BATCH16",
        no_target_cache=True,
        no_future_ingestion=True,
        comparison="paired R1 reference versus R1 mapped; not against historical R0",
        phase="CPU_PARITY" if args.cpu_only else "GPU_MEASUREMENT",
    )
    dest = output / ("cpu_parity" if args.cpu_only else "measurement")
    dest.mkdir(exist_ok=True)
    freeze = dest / "PROTOCOL.json"
    if freeze.exists() and read(freeze) != binding:
        raise ValueError("R1 frozen implementation or input contract changed")
    if not freeze.exists():
        atomic_json(freeze, binding)
    torch.set_num_threads(2 if args.cpu_only else 4)
    torch.set_num_interop_threads(1 if args.cpu_only else 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    scoped = RuntimeCampaign(base, output)
    runtime = None if args.cpu_only else FrozenRuntime(scoped, protocol)
    scoped.out = dest
    all_rows, parities = [], []
    recovery_ms = 0.0
    blocks = BLOCKS[:1] if args.cpu_only else BLOCKS
    begin_run = time.perf_counter()
    try:
        for block in blocks:
            for label, length in ARMS.items():
                sources: dict[str, CachedEventReader] = {}
                readers: dict[str, ResidentReplay] = {}
                current_sequence = None
                try:
                    for position, (qi, query) in enumerate(ordered):
                        scoped.require_resources()
                        idx, row = indexes[query["pool"]], query["index_row"]
                        if str(idx["tokens"][row]) != query["sample_token"]:
                            raise ValueError("TRAIN query mapping changed")
                        path = base.raw / query["sequence_id"] / "events.h5"
                        if current_sequence != query["sequence_id"]:
                            for source in sources.values():
                                source.close()
                            expected = next(r for r in protocol["raw"] if Path(r["path"]) == path)
                            raw_binding = verify_raw(base, path, expected)
                            sources = {
                                route: CachedEventReader(path) for route in ("reference", "mapped")
                            }
                            readers = {
                                route: ResidentReplay(source, args.capacity_mib * 1024**2)
                                for route, source in sources.items()
                            }
                            current_sequence = query["sequence_id"]
                        valid = idx["valid"][row].copy()
                        if label == "WIDE":
                            valid &= np.isin(np.arange(16), WIDE_SLOTS)
                        else:
                            valid[:-length] = False
                        intervals = idx["base_windows_us"][row][None] - idx["lag_us"][:, None, None]
                        start, cutoff = int(intervals[valid].min()), int(intervals[valid].max())
                        receipt = dest / "fragments" / f"{block}_{label}_{qi:02d}.json"
                        if receipt.exists():
                            saved = read(receipt)
                            all_rows.extend(saved["requests"])
                            parities.append(saved["parity"])
                            recovery_begin = time.perf_counter()
                            for reader in readers.values():
                                reader.advance(start, cutoff)
                            recovery_ms += (time.perf_counter() - recovery_begin) * 1000
                            continue
                        if runtime is not None:
                            runtime.load(query)
                        routes = ["reference", "mapped"]
                        if (
                            int(
                                hashlib.sha256(
                                    (block + label + query["sample_token"]).encode()
                                ).hexdigest(),
                                16,
                            )
                            % 2
                        ):
                            routes.reverse()
                        tensors, predictions, rows = {}, {}, []
                        for route in routes:
                            scoped.require_resources()
                            if runtime is not None:
                                torch.cuda.synchronize()
                            before = dict(readers[route].stats)
                            started = time.perf_counter()
                            readers[route].advance(start, cutoff)
                            ingested = time.perf_counter()
                            encoder = encode_context_union if route == "reference" else encode_union
                            tensor = encoder(
                                cast(
                                    EAPEventReader,
                                    sources[route]
                                    if args.cpu_only and route == "reference"
                                    else readers[route],
                                ),
                                idx["base_windows_us"][row],
                                idx["lag_us"],
                                valid,
                                tuple(idx["square_xyxy"][row]),
                                sequence_id=query["sequence_id"],
                                roi_size=prep["roi_size"],
                                event_pixel_diff=prep["event_pixel_diff"],
                            )
                            prepared = time.perf_counter()
                            result = (
                                {}
                                if runtime is None
                                else runtime.prepared(
                                    query, idx, label, tensor, valid, compact=False
                                )
                            )
                            ended = time.perf_counter()
                            tensors[route], predictions[route] = tensor, result
                            stats = {
                                key: value - before[key]
                                for key, value in readers[route].stats.items()
                                if key != "peak_retained_bytes"
                            }
                            rows.append(
                                dict(
                                    block=block,
                                    label=label,
                                    route=route,
                                    regime="R1_REPLAY",
                                    query=query["sample_token"],
                                    original_query=qi,
                                    chronological_position=position,
                                    milliseconds=(ended - started) * 1000,
                                    preparation_ms=(prepared - ingested) * 1000,
                                    resident_bytes=readers[route].retained_bytes,
                                    peak_retained_bytes=readers[route].stats["peak_retained_bytes"],
                                    **stats,
                                    **{
                                        k: result[k]
                                        for k in (
                                            "transfer_ms",
                                            "producer_ms",
                                            "head_ms",
                                            "normalization_ms",
                                        )
                                        if k in result
                                    },
                                )
                            )
                        evidence = parity(
                            tensors["reference"],
                            tensors["mapped"],
                            predictions["reference"],
                            predictions["mapped"],
                        )
                        if runtime is not None and label != "WIDE":
                            historical_errors = {}
                            for name, actual, limit in zip(
                                ("features", "times", "valid", "experts"),
                                predictions["reference"]["xs"],
                                (1e-4, 1e-7, 0, 1e-5),
                                strict=True,
                            ):
                                expected = runtime.cached[label][name][qi : qi + 1]
                                error = float(
                                    np.max(np.abs(actual.astype(float) - expected.astype(float)))
                                )
                                if error > limit:
                                    raise ValueError(
                                        f"historical fixed input parity failed {name}: {error}"
                                    )
                                historical_errors[name] = error
                            evidence["historical_input_errors"] = historical_errors
                        evidence.update(query=query["sample_token"], label=label, block=block)
                        atomic_json(
                            receipt, dict(requests=rows, parity=evidence, raw_binding=raw_binding)
                        )
                        all_rows.extend(rows)
                        parities.append(evidence)
                        atomic_json(
                            dest / "PROGRESS.json",
                            dict(
                                completed_pairs=len(parities),
                                total_pairs=len(blocks) * 64 * 4,
                                block=block,
                                label=label,
                                query=query["sample_token"],
                                elapsed_s=time.perf_counter() - begin_run,
                                optimizer_updates=0,
                            ),
                        )
                        print(
                            f"R1 {block} {label} {position + 1}/64 {evidence['status']}", flush=True
                        )
                        del tensors, predictions, tensor
                finally:
                    for source in sources.values():
                        source.close()
        frame = pd.DataFrame(all_rows)
        atomic_bytes(dest / "REQUESTS.csv", frame.to_csv(index=False).encode())
        summaries = []
        for keys, group in frame.groupby(["block", "label", "route"]):
            wall_s = float(group.milliseconds.sum() / 1000)
            stream_s = float(group.stream_seconds.sum())
            block, label, route = cast(tuple[str, str, str], keys)
            summaries.append(
                dict(block=block, label=label, route=route)
                | dict(
                    requests=len(group),
                    p50_ms=float(group.milliseconds.median()),
                    p95_ms=float(group.milliseconds.quantile(0.95)),
                    total_request_seconds=wall_s,
                    ingested_stream_seconds=stream_s,
                    seconds_compute_per_stream_second=wall_s / stream_s,
                    queries_per_second=len(group) / wall_s,
                    ingestion_seconds=float(group.ingestion_ms.sum() / 1000),
                    fallback_reads=int(group.capacity_fallback_reads.sum()),
                )
            )
        atomic_json(
            dest / "RESULT.json",
            dict(
                status="COMPLETE",
                pairs=len(parities),
                summaries=summaries,
                parity=parities,
                recovery_ingestion_ms=recovery_ms,
                optimizer_updates=0,
                gpu_measured=not args.cpu_only,
                limitations=[
                    "Retrospective replay with supplied ROI; not an online availability claim",
                    "Two sensor caches; HDF5 decoding may internally touch full compression chunks",
                    "First requested history starts replay; earlier recording duration is excluded",
                    "Sparse original queries, not dense streaming throughput",
                    "Storage coldness uncontrolled; application-cold opens fresh handles",
                ],
            ),
        )
    finally:
        if runtime is not None:
            runtime.close()


def main() -> int:
    """Run CPU parity separately while the GPU training slot is occupied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/efficient_context_20261004/r1_20261008"
    )
    parser.add_argument("--capacity-mib", type=int, default=64)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
