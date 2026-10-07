"""Real TRAIN admission for native-graph H8 transport without feature writes."""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.efficient_context.common import digest
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.h8_fast_admission import (
    _exact,
    _job,
    _model_sha,
    _rng_state,
    _selected_rows,
)

REPORT_NAME = "H8_TRANSPORT_ADMISSION.json"
GPU_RESERVE_LIMIT = 3 * 1024**3


def _timed(call: Any) -> tuple[np.ndarray, float]:  # noqa: ANN401 - callable protocol is dynamic
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


@torch.inference_mode()
def run(output: Path, raw_root: Path) -> None:
    """Compare canonical packed extraction with native transport replay on TRAIN."""
    from e_jepa_ttc.models import causal_scale_ttc as model_module
    from operational.simplex_t_shared_route import adapter as canonical_adapter
    from operational.train40_system import history_features
    from operational.train40_system.h8_fast_extract import H8Extractor
    from operational.train40_system.h8_transport_graph import H8TransportGraph

    directory = output / "h8_transport_admission"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / REPORT_NAME
    report: dict[str, Any] = {
        "schema": "train40_h8_transport_admission_v1",
        "status": "RUNNING",
        "compatible": False,
        "started_utc": datetime.now(UTC).isoformat(),
        "optimizer_updates": 0,
        "targets_read": False,
        "public_TRAIN40_only": True,
        "writes_feature_fragments": False,
        "production_interrupted": False,
        "limitation": "Concurrent real-TRAIN admission; timings are not a production ETA claim.",
        "source_sha256": digest(Path(__file__)),
        "graph_source_sha256": digest(Path(__file__).with_name("h8_transport_graph.py")),
        "extractor_source_sha256": digest(Path(__file__).with_name("h8_fast_extract.py")),
        "canonical_adapter_source_sha256": digest(Path(canonical_adapter.__file__)),
        "history_source_sha256": digest(Path(history_features.__file__)),
    }
    atomic_json(destination, report)
    started = time.perf_counter()
    graph: H8TransportGraph | None = None
    history_features.worker_init()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        fraction = min(1.0, (GPU_RESERVE_LIMIT - 64 * 1024**2) / total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction)

        models, parents = history_features.load_producers(output, pair=True)
        if any(model.training for model in models.values()) or any(
            parameter.requires_grad for model in models.values() for parameter in model.parameters()
        ):
            raise ValueError("Transport admission requires frozen eval producers")
        initial_model = _model_sha(models)
        with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
            index = {
                key: stored[key] for key in ("windows_us", "square_xyxy", "sequences", "delta_t_s")
            }
        rows = _selected_rows(output, index["sequences"])
        if len(rows) != 9:
            raise ValueError(f"Transport admission requires nine bounded TRAIN rows, got {rows}")

        prepared: dict[int, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
        for row in rows:
            job, valid = _job(raw_root, row, index)
            events = torch.from_numpy(history_features.prepare(job)).to("cuda")
            delta = torch.full(
                (16, 2), float(index["delta_t_s"][row]), dtype=torch.float32, device="cuda"
            )
            prepared[row] = events, delta, valid

        initial_rng = _rng_state()
        extractor = H8Extractor(models, "packed")
        expected: dict[int, np.ndarray] = {}
        for row in rows:
            events, delta, _ = prepared[row]
            expected[row] = extractor("H8_SEED7", models, events, delta).copy()

        original_match = model_module.local_correlation_match
        original_features = model_module.transport_physical_features
        graph = H8TransportGraph(max_entries=8, warmup_calls=3)
        observations = []
        with graph.install():
            for row in rows:
                events, delta, valid = prepared[row]
                value = extractor("H8_SEED7", models, events, delta).copy()
                if not _exact(value, expected[row]):
                    raise ValueError(f"Transport graph differs from packed extraction at row {row}")
                fragment = output / "h8_feature_fragments" / f"query_{row:05d}.npz"
                receipt = fragment.with_suffix(".json")
                stored_exact = None
                if fragment.is_file() and receipt.is_file():
                    metadata = history_features.read(receipt)
                    if metadata.get("sha256") != digest(fragment):
                        raise ValueError("Historical H8 fragment receipt changed")
                    with np.load(fragment, allow_pickle=False) as stored:
                        historical = stored["features"]
                    masked = value[-8:].copy()
                    masked[~valid] = 0
                    stored_exact = _exact(masked, historical)
                    if not stored_exact:
                        raise ValueError("Transport graph differs from historical masked H8")
                observations.append(
                    {
                        "TRAIN_ordinal": row,
                        "sequence": str(index["sequences"][row]),
                        "canonical_all_16_exact": True,
                        "historic_masked_last_8_exact": stored_exact,
                    }
                )

            first_row, second_row = rows[:2]
            retained = extractor(
                "H8_SEED7", models, prepared[first_row][0], prepared[first_row][1]
            ).copy()
            first = retained.copy()
            extractor("H8_SEED7", models, prepared[second_row][0], prepared[second_row][1])
            last = extractor(
                "H8_SEED7", models, prepared[first_row][0], prepared[first_row][1]
            ).copy()
            if not _exact(first, last) or not _exact(retained, first):
                raise ValueError("Transport graph failed exact retained A-B-A parity")

        if model_module.local_correlation_match is not original_match or (
            model_module.transport_physical_features is not original_features
        ):
            raise ValueError("Transport graph aliases were not restored")

        captures_before_timing = int(graph.snapshot()["captures"])
        packed_seconds: list[float] = []
        graph_seconds: list[float] = []
        timing_rows = rows[:6]
        timing_order = []
        for iteration, row in enumerate(timing_rows):
            events, delta, _ = prepared[row]

            def packed_call(
                call_events: torch.Tensor = events,
                call_delta: torch.Tensor = delta,
            ) -> np.ndarray:
                return extractor("H8_SEED7", models, call_events, call_delta)

            def graph_call(
                call_events: torch.Tensor = events,
                call_delta: torch.Tensor = delta,
            ) -> np.ndarray:
                with graph.install():
                    return extractor("H8_SEED7", models, call_events, call_delta)

            order = ("packed", "graph") if iteration % 2 == 0 else ("graph", "packed")
            timing_order.append({"TRAIN_ordinal": row, "order": list(order)})
            measured: dict[str, np.ndarray] = {}
            for candidate in order:
                value, elapsed = _timed(graph_call if candidate == "graph" else packed_call)
                measured[candidate] = value
                (graph_seconds if candidate == "graph" else packed_seconds).append(elapsed)
            if not _exact(measured["packed"], measured["graph"]):
                raise ValueError("Alternating packed/graph timing outputs differ")

        captures_after_timing = int(graph.snapshot()["captures"])
        if captures_after_timing != captures_before_timing:
            raise ValueError("Steady transport timing captured an unexpected new graph")
        if _model_sha(models) != initial_model or _rng_state() != initial_rng:
            raise ValueError("Transport admission changed producer state or RNG")
        peak_reserved = int(torch.cuda.max_memory_reserved())
        if peak_reserved >= GPU_RESERVE_LIMIT:
            raise ValueError("Transport admission exceeded the three-GiB GPU reserve")

        median_packed = statistics.median(packed_seconds)
        median_graph = statistics.median(graph_seconds)
        report.update(
            {
                "status": "PASSED",
                "compatible": True,
                "finished_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "parents": parents,
                "rows": rows,
                "observations": observations,
                "canonical_all_16_exact": True,
                "historic_masked_last_8_exact": True,
                "stored_fragments_checked": sum(
                    item["historic_masked_last_8_exact"] is True for item in observations
                ),
                "exact_bytes_A_B_A": True,
                "retained_features_unchanged": True,
                "model_state_unchanged": True,
                "CPU_CUDA_RNG_unchanged": True,
                "aliases_restored": True,
                "alternating_same_inputs": True,
                "timing_rows": timing_rows,
                "timing_order": timing_order,
                "packed_seconds": packed_seconds,
                "graph_seconds": graph_seconds,
                "median_packed_seconds": median_packed,
                "median_graph_seconds": median_graph,
                "median_end_to_end_speed_ratio": median_packed / median_graph,
                "steady_graph_cache_no_new_captures": True,
                "graph": graph.snapshot(),
                "peak_reserved_bytes": peak_reserved,
                "gpu_memory_fraction": fraction,
            }
        )
        atomic_json(destination, report)
    except BaseException as error:
        report.update(
            {
                "status": "FAILED",
                "compatible": False,
                "finished_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(destination, report)
        raise
    finally:
        if graph is not None:
            if graph.snapshot()["installed"]:
                raise RuntimeError("Transport graph remained installed after admission")
            graph.close()
        pool = history_features._reader_pool
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.raw_root.resolve())
