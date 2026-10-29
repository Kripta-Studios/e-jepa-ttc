"""Real TRAIN zero-update parity admission for packed and graph H8 extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from operational.efficient_context.common import Lease, digest
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def _tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _model_sha(models: dict[str, Any]) -> str:
    result = hashlib.sha256()
    for model_name, model in sorted(models.items()):
        for name, value in sorted(model.state_dict().items()):
            result.update(model_name.encode())
            result.update(name.encode())
            result.update(_tensor_sha(value).encode())
    return result.hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": _tensor_sha(torch.get_rng_state()),
        "cuda": [_tensor_sha(value) for value in torch.cuda.get_rng_state_all()],
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
    }


def _exact(left: np.ndarray, right: np.ndarray) -> bool:
    """Require equal values, dtypes, shapes and bytes, including signed zero."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
        and left.tobytes() == right.tobytes()
    )


def _unique_graphs(counters: Any) -> int:  # noqa: ANN401 - Torch counter mapping
    return int(counters.get("stats", {}).get("unique_graphs", 0))


def _selected_rows(output: Path, sequences: np.ndarray) -> list[int]:
    rows = {0, 31, min(1000, len(sequences) - 1)}
    changes = np.flatnonzero(sequences[1:] != sequences[:-1]) + 1
    rows.update(int(value) for value in changes[:4])
    fragments = output / "h8_feature_fragments"
    if fragments.is_dir():
        receipt_rows = []
        for receipt in fragments.glob("query_*.json"):
            try:
                row = int(receipt.stem.split("_")[-1])
            except ValueError:
                continue
            if (fragments / f"query_{row:05d}.npz").is_file():
                receipt_rows.append(row)
        rows.update(sorted(receipt_rows)[-2:])
    return sorted(row for row in rows if 0 <= row < len(sequences))[:12]


def _job(
    raw_root: Path,
    row: int,
    index: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    import h5py

    from e_jepa_ttc.data.eap import _require_hdf5plugin

    sequence = str(index["sequences"][row])
    path = raw_root / "data/train" / sequence / "events.h5"
    _require_hdf5plugin()
    with h5py.File(path, "r") as handle:
        timestamps = handle["events/t"]
        if not isinstance(timestamps, h5py.Dataset):
            raise ValueError("Raw timestamps must be a dataset")
        start = int(timestamps[0])
    windows = index["windows_us"][row]
    valid = windows.min() - np.arange(7, -1, -1) * 50_000 >= start
    return (
        {
            "path": str(path),
            "sequence": sequence,
            "windows": windows,
            "square": index["square_xyxy"][row],
            "valid": valid,
        },
        valid,
    )


def run(output: Path, raw_root: Path, mode: str, admission_directory: Path) -> None:
    """Require exact canonical parity before the fast extractor can be frozen."""
    from operational.simplex_t_shared_route.adapter import extract as canonical_extract
    from operational.train40_system import history_features
    from operational.train40_system.h8_fast_extract import H8Extractor

    admission_directory.mkdir(parents=True, exist_ok=True)
    report_path = admission_directory / f"H8_FAST_{mode.upper()}_ADMISSION.json"
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "train40_h8_fast_extract_admission_v1",
        "status": "RUNNING",
        "mode": mode,
        "optimizer_updates": 0,
        "targets_read": False,
        "source_sha256": digest(Path(__file__)),
        "extractor_source_sha256": digest(Path(__file__).with_name("h8_fast_extract.py")),
    }
    atomic_json(report_path, report)
    models: dict[str, Any] = {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    history_features.worker_init()
    try:
        models, parents = history_features.load_producers(output, pair=True)
        initial_model = _model_sha(models)
        initial_rng = _rng_state()
        with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
            index = {
                key: stored[key] for key in ("windows_us", "square_xyxy", "sequences", "delta_t_s")
            }
        rows = _selected_rows(output, index["sequences"])
        prepared: dict[int, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
        observations = []
        for row in rows:
            job, valid = _job(raw_root, row, index)
            raw = history_features.prepare(job)
            events = torch.from_numpy(raw).to("cuda")
            delta = torch.full(
                (16, 2), float(index["delta_t_s"][row]), dtype=torch.float32, device="cuda"
            )
            prepared[row] = events, delta, valid

        extractor = H8Extractor(models, mode)
        # Warm only the candidate before comparisons; it never updates weights.
        first_events, first_delta, _ = prepared[rows[0]]
        warmup_calls = 3 if mode == "graph" else 1
        for _ in range(warmup_calls):
            extractor("H8_SEED7", models, first_events, first_delta)

        for row in rows:
            events, delta, valid = prepared[row]
            expected = canonical_extract("H8_SEED7", models, events, delta)
            actual = extractor("H8_SEED7", models, events, delta)
            if not _exact(actual, expected):
                raise ValueError(f"H8 fast extraction parity failed at TRAIN row {row}")
            stored_exact = None
            fragment = output / "h8_feature_fragments" / f"query_{row:05d}.npz"
            receipt = fragment.with_suffix(".json")
            if fragment.is_file() and receipt.is_file():
                metadata = json.loads(receipt.read_text(encoding="utf-8"))
                if metadata.get("sha256") != digest(fragment):
                    raise ValueError("Existing H8 feature receipt changed")
                with np.load(fragment, allow_pickle=False) as saved:
                    existing = saved["features"]
                masked = actual[-8:].copy()
                masked[~valid] = 0
                stored_exact = _exact(masked, existing)
                if not stored_exact:
                    raise ValueError("Fast features differ from stored canonical fragment")
            observations.append(
                {
                    "TRAIN_ordinal": row,
                    "sequence": str(index["sequences"][row]),
                    "all_16_features_exact": True,
                    "stored_last_8_masked_exact": stored_exact,
                    "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                }
            )

        a_row, b_row = rows[0], rows[1]
        a_events, a_delta, _ = prepared[a_row]
        b_events, b_delta, _ = prepared[b_row]
        a_first = extractor("H8_SEED7", models, a_events, a_delta).copy()
        extractor("H8_SEED7", models, b_events, b_delta)
        a_last = extractor("H8_SEED7", models, a_events, a_delta).copy()
        if not _exact(a_first, a_last):
            raise ValueError("Repeated A-B-A extraction changed features")

        graph_launches: dict[str, int] = {}
        if mode == "graph":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                extractor("H8_SEED7", models, a_events, a_delta)
            graph_launches = {
                event.key: int(event.count)
                for event in profile.key_averages()
                if "graphlaunch" in event.key.lower()
            }
            if not graph_launches:
                raise ValueError("Graph mode did not expose a CUDA GraphLaunch event")

        from torch._dynamo.utils import counters as dynamo_counters

        steady_unique_graphs_before = _unique_graphs(dynamo_counters)
        canonical_seconds, candidate_seconds = [], []
        benchmark_rows = rows[: min(6, len(rows))]
        for row in benchmark_rows:
            events, delta, _ = prepared[row]
            torch.cuda.synchronize()
            before = time.perf_counter()
            canonical_extract("H8_SEED7", models, events, delta)
            torch.cuda.synchronize()
            canonical_seconds.append(time.perf_counter() - before)
            before = time.perf_counter()
            extractor("H8_SEED7", models, events, delta)
            torch.cuda.synchronize()
            candidate_seconds.append(time.perf_counter() - before)
        steady_unique_graphs_after = _unique_graphs(dynamo_counters)
        steady_recompiles = steady_unique_graphs_after - steady_unique_graphs_before
        if mode == "graph" and steady_recompiles != 0:
            raise ValueError("H8 graph extraction recompiled during the steady benchmark")

        final_model = _model_sha(models)
        final_rng = _rng_state()
        if initial_model != final_model or initial_rng != final_rng:
            raise ValueError("H8 admission changed model state or CPU/CUDA RNG")
        peak_reserved = int(torch.cuda.max_memory_reserved())
        if peak_reserved >= 12 * 1024**3:
            raise ValueError("H8 fast extraction exceeds the admitted VRAM ceiling")
        process = psutil.Process()
        host_tree_rss = int(process.memory_info().rss)
        host_available = int(psutil.virtual_memory().available)
        if host_tree_rss >= 16_000_000_000 or host_available < 2 * 1024**3:
            raise ValueError("H8 fast extraction exceeds the admitted host-RAM boundary")
        report.update(
            {
                "status": "PASSED",
                "compatible": True,
                "parents": parents,
                "rows": rows,
                "observations": observations,
                "all_16_features_exact": True,
                "stored_fragments_checked": sum(
                    item["stored_last_8_masked_exact"] is True for item in observations
                ),
                "repeated_A_B_A_exact": True,
                "exact_bytes_including_signed_zero": True,
                "model_state_unchanged": True,
                "CPU_CUDA_RNG_unchanged": True,
                "graph_launches": graph_launches,
                "canonical_seconds": canonical_seconds,
                "candidate_seconds": candidate_seconds,
                "warmup_calls": warmup_calls,
                "steady_unique_graphs_before": steady_unique_graphs_before,
                "steady_unique_graphs_after": steady_unique_graphs_after,
                "steady_recompiles": steady_recompiles,
                "extractor": extractor.snapshot(),
                "peak_reserved_vram_bytes": peak_reserved,
                "host_tree_rss_bytes": host_tree_rss,
                "host_available_bytes": host_available,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(report_path, report)
    except BaseException as error:
        report.update(
            {
                "status": "FAILED",
                "compatible": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(report_path, report)
        raise
    finally:
        pool = history_features._reader_pool
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("packed", "graph"), required=True)
    parser.add_argument("--admission-directory", type=Path)
    arguments = parser.parse_args()
    root = arguments.output.resolve()
    directory = (
        arguments.admission_directory.resolve()
        if arguments.admission_directory is not None
        else root / "h8_fast_admission"
    )
    with Lease(root):
        run(root, arguments.raw_root.resolve(), arguments.mode, directory)
