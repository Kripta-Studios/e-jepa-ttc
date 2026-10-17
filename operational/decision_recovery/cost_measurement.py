"""Prospective paired batch-one cost measurement for frozen V19 endpoints.

The measured input is the immutable decoded V18 P2 feature pack, not raw event
camera packets.  Cold application setup and steady-state warm latency remain
separate.  This module performs inference only and never constructs an optimizer.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import common, evaluation, training

if TYPE_CHECKING:
    import torch

ARMS = ("L0", "LW")
QUERY_SEED = 20261017
QUERY_COUNT = 128
PACK_ROWS = 256
WARMUPS = 20
GPU_RESERVE_SECONDS = 900
DIRECTORY = training.destination("cost/batch1_cached_P2_to_TTC_v2")


def protocol() -> dict[str, Any]:
    """Return the fixed timing design without opening models or CUDA."""
    rng = np.random.default_rng(QUERY_SEED)
    indices = rng.choice(PACK_ROWS, QUERY_COUNT, replace=False).tolist()
    return {
        "schema": "v19_batch1_cached_P2_to_TTC_cost_v1",
        "arms": list(ARMS),
        "population": "P",
        "query_seed": QUERY_SEED,
        "indices": indices,
        "paired_orders": [rng.permutation(ARMS).tolist() for _ in indices],
        "query_scope": (
            "128 fixed rows sampled without replacement from admitted P pack_00000; "
            "technical latency sample, not a population metric"
        ),
        "batch_size": 1,
        "warmups_per_endpoint": WARMUPS,
        "precision": "CUDA_FP32_no_TF32",
        "deployment_route": "frozen TorchScript export; batch1; TTC and emitted phase",
        "input_scope": (
            "immutable V18 cached native P2 FP32 tensor (2x12x128x128) and native clock; "
            "does not include raw event-stream extraction or ROI construction"
        ),
        "cold_scope": (
            "application-cold reader/model load and first call; OS file cache uncontrolled"
        ),
        "warm_scope": (
            "application-warm pack handle and model after 20 forwards; each row still incurs "
            "ZIP member decode, CPU collation, H2D and synchronized forward"
        ),
        "stages": [
            "cached_P2_decode",
            "host_to_device",
            "forward_to_TTC",
            "cached_P2_to_TTC",
        ],
        "memory": "CUDA peak extra allocated per endpoint route and process RSS snapshots",
        "cold_memory_limit": (
            "model resident deltas are incremental in fixed L0,LW load order and descriptive; "
            "they are not an isolated between-endpoint memory comparison"
        ),
        "gpu_reserve_seconds": GPU_RESERVE_SECONDS,
        "optimizer_updates": 0,
        "backward": False,
        "comparison_limits": (
            "Do not compare as equivalent to V18 prepared-input timings, historical batches, "
            "or raw-sensor end-to-end latency"
        ),
    }


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_cost_measurement.py",
        Path(common.__file__),
        Path(evaluation.__file__),
        Path(training.__file__),
        common.ROOT / "operational/joint_measurement/data.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def _export_binding(arm: str) -> dict[str, Any]:
    directory = training.destination(f"evaluation/exports/{arm}")
    receipt = common.read_json(directory / "COMPLETE.json")
    if receipt["status"] != "COMPLETE" or any(
        common.sha(directory / name) != expected for name, expected in receipt["files"].items()
    ):
        raise ValueError(f"Frozen {arm} deployment export changed")
    return {
        "complete_sha256": common.sha(directory / "COMPLETE.json"),
        "torchscript_sha256": common.sha(directory / "model.torchscript.pt"),
        "replay_sha256": common.sha(directory / "REPLAY.json"),
    }


def _prediction_binding(arm: str) -> dict[str, str]:
    directory = training.destination(f"evaluation/predictions/P/{arm}")
    receipt = common.read_json(directory / "COMPLETE.json")
    prediction = directory / "predictions.parquet"
    if (
        receipt["status"] != "COMPLETE"
        or receipt["rows"] != 53890
        or common.sha(prediction) != receipt["sha256"]
    ):
        raise ValueError(f"Complete P predictions required for {arm} cost parity")
    return {
        "complete_sha256": common.sha(directory / "COMPLETE.json"),
        "predictions_sha256": receipt["sha256"],
    }


def _pack_binding() -> dict[str, Any]:
    from operational.range_transfer.common import digest

    folder = training.cache_root("P")
    complete = common.read_json(folder / "COMPLETE.json")
    binding = common.read_json(folder / "BINDING.json")
    receipt = common.read_json(folder / "pack_00000.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["rows"] != 53890
        or complete["binding"] != digest(binding)
        or receipt["start"] != 0
        or receipt["stop"] != PACK_ROWS
    ):
        raise ValueError("Complete admitted P pack0 is required")
    actual_pack_sha = common.sha(folder / "pack_00000.zip")
    if actual_pack_sha != receipt["sha256"]:
        raise ValueError("Admitted P pack0 bytes changed")
    return {
        "complete_sha256": common.sha(folder / "COMPLETE.json"),
        "binding_sha256": common.sha(folder / "BINDING.json"),
        "pack_receipt_sha256": common.sha(folder / "pack_00000.json"),
        "pack_sha256": actual_pack_sha,
        "pack_receipts_hash": complete["pack_receipts_hash"],
        "encoding": "outer ZIP_DEFLATED level1; inner NPZ arrays; immutable FP32 P2 cache",
    }


def registration_payload(source_commit: str) -> dict[str, Any]:
    """Resolve endpoint, export, prediction, cache and source identities on CPU."""
    evaluation_record = common.read_json(training.destination("evaluation/REGISTRATION.json"))
    endpoints = {arm: evaluation._endpoint(arm, training.registered()) for arm in ARMS}
    if endpoints != evaluation_record["endpoints"]:
        raise ValueError("Evaluation endpoints changed before prospective cost registration")
    payload = protocol()
    payload.update(
        endpoints=endpoints,
        exports={arm: _export_binding(arm) for arm in ARMS},
        P_predictions={arm: _prediction_binding(arm) for arm in ARMS},
        P_manifest_sha256=common.sha(training.manifest_path("P")),
        P_cache=_pack_binding(),
        evaluation_registration_sha256=common.sha(
            training.destination("evaluation/REGISTRATION.json")
        ),
        training_registration_sha256=common.sha(training.destination("training/REGISTRATION.json")),
        sources=_sources(),
        source_commit=source_commit,
        registration_precedes_gpu_measurement=True,
        system_promotion=False,
        official_comparison=False,
    )
    return payload


def register() -> dict[str, Any]:
    """Freeze the complete prospective cost contract before opening CUDA."""
    path = DIRECTORY / "REGISTRATION.json"
    if path.exists():
        return registered()
    sources = _sources()
    value = registration_payload(training.committed_sources(sources))
    common.write_json(path, value)
    return value


def registered() -> dict[str, Any]:
    value = common.read_json(DIRECTORY / "REGISTRATION.json")
    if value != registration_payload(value["source_commit"]):
        raise ValueError("Registered cost sources, endpoints or cache changed")
    return value


def _expected_outputs(record: dict[str, Any]) -> dict[str, dict[int, tuple[float, float]]]:
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.phase import ttc_to_phase

    result = {}
    indices = record["indices"]
    for arm in ARMS:
        path = training.destination(f"evaluation/predictions/P/{arm}/predictions.parquet")
        frame = pd.read_parquet(path, columns=["prediction_ttc"])
        result[arm] = {}
        for index, value in zip(indices, frame.iloc[indices].prediction_ttc, strict=True):
            ttc = float(value)
            phase = float(ttc_to_phase(torch.tensor([ttc], dtype=torch.float32)).item())
            result[arm][index] = (ttc, phase)
    return result


def _load_export(arm: str) -> torch.jit.ScriptModule:
    import torch

    path = training.destination(f"evaluation/exports/{arm}/model.torchscript.pt")
    with path.open("rb") as stream:
        return torch.jit.load(stream, map_location="cpu").eval()


def _inference(
    model: torch.jit.ScriptModule, events: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(events, delta)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("Deployment export output contract changed")
    return output


def validate_row(row: dict[str, Any], record: dict[str, Any], ordinal: int) -> None:
    """Reject reordered, incomplete, nonfinite or non-batch1 timing evidence."""
    expected = {
        "ordinal",
        "P_index",
        "order",
        "cached_P2_decode_s",
        "timings",
        "registration",
        "attempt_id",
        "utc",
    }
    if set(row) != expected or row["ordinal"] != ordinal:
        raise ValueError("Cost row schema or ordinal changed")
    if row["P_index"] != record["indices"][ordinal]:
        raise ValueError("Cost query order changed")
    if not row["registration"] or not row["attempt_id"] or not row["utc"]:
        raise ValueError("Cost row lacks registration, attempt or timestamp")
    if row["order"] != record["paired_orders"][ordinal] or set(row["timings"]) != set(ARMS):
        raise ValueError("Paired endpoint order changed")
    numbers = [row["cached_P2_decode_s"]]
    fields_required = {
        "host_to_device_s",
        "forward_to_TTC_s",
        "cached_P2_to_TTC_s",
        "ttc_seconds",
        "emitted_phase",
        "peak_extra_allocated_bytes",
        "memory_allocated_before_bytes",
    }
    for value in row["timings"].values():
        if set(value) != fields_required:
            raise ValueError("Endpoint timing schema changed")
        numbers.extend(
            value[name] for name in fields_required if name != "peak_extra_allocated_bytes"
        )
        if value["peak_extra_allocated_bytes"] < 0 or value["memory_allocated_before_bytes"] < 0:
            raise ValueError("Negative CUDA allocation witness")
    if not np.isfinite(np.asarray(numbers, dtype=np.float64)).all() or any(
        row["timings"][arm][name] <= 0
        for arm in ARMS
        for name in ("host_to_device_s", "forward_to_TTC_s", "cached_P2_to_TTC_s")
    ):
        raise ValueError("Nonfinite or nonpositive timing evidence")


def validate_outputs(
    row: dict[str, Any], expected: dict[str, dict[int, tuple[float, float]]]
) -> None:
    """Recheck both deployment outputs whenever durable evidence is reopened."""
    index = row["P_index"]
    for arm in ARMS:
        ttc, phase = expected[arm][index]
        observed = row["timings"][arm]
        if not np.isclose(observed["ttc_seconds"], ttc, rtol=1e-5, atol=1e-5) or not np.isclose(
            observed["emitted_phase"], phase, rtol=1e-5, atol=1e-5
        ):
            raise ValueError("Durable deployment output differs from frozen P prediction")


def validate_cold(
    payload: dict[str, Any],
    record: dict[str, Any],
    expected: dict[str, dict[int, tuple[float, float]]],
    registration_sha: str,
    attempt_id: str,
) -> None:
    """Reject altered cold evidence before its hash can enter COMPLETE."""
    from operational.range_transfer.common import digest

    top_level = {
        "registration",
        "registration_sha256",
        "attempt_id",
        "cold",
        "cold_model_load_order",
        "cold_memory_scope",
        "warmups_per_endpoint",
        "optimizer_updates",
    }
    required = {
        "torchscript_load_cpu_s",
        "model_to_cuda_s",
        "reader_init_s",
        "cached_P2_decode_s",
        "host_to_device_s",
        "first_forward_to_TTC_s",
        "cached_P2_to_TTC_s",
        "ttc_seconds",
        "emitted_phase",
        "model_resident_delta_bytes",
        "first_route_peak_allocated_bytes",
        "process_rss_bytes",
        "meaning",
    }
    if (
        set(payload) != top_level
        or payload.get("registration") != digest(record)
        or payload.get("registration_sha256") != registration_sha
        or payload.get("attempt_id") != attempt_id
        or payload.get("warmups_per_endpoint") != WARMUPS
        or payload.get("optimizer_updates") != 0
        or payload.get("cold_model_load_order") != list(ARMS)
        or payload.get("cold_memory_scope") != record["cold_memory_limit"]
        or set(payload.get("cold", {})) != set(ARMS)
    ):
        raise ValueError("Cold cost identity or contract changed")
    index = record["indices"][0]
    for arm in ARMS:
        value = payload["cold"][arm]
        if set(value) != required:
            raise ValueError("Cold endpoint evidence schema changed")
        numeric = [number for name, number in value.items() if name != "meaning"]
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all() or any(
            value[name] <= 0
            for name in (
                "torchscript_load_cpu_s",
                "model_to_cuda_s",
                "reader_init_s",
                "cached_P2_decode_s",
                "host_to_device_s",
                "first_forward_to_TTC_s",
                "cached_P2_to_TTC_s",
                "first_route_peak_allocated_bytes",
                "process_rss_bytes",
            )
        ):
            raise ValueError("Cold endpoint evidence is nonfinite or nonpositive")
        if value["model_resident_delta_bytes"] < 0:
            raise ValueError("Cold model resident delta is negative")
        ttc, phase = expected[arm][index]
        if not np.isclose(value["ttc_seconds"], ttc, rtol=1e-5, atol=1e-5) or not np.isclose(
            value["emitted_phase"], phase, rtol=1e-5, atol=1e-5
        ):
            raise ValueError("Cold deployment output differs from frozen P prediction")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fixed rows without treating them as scientific observations."""
    if len(rows) != QUERY_COUNT:
        raise ValueError("All fixed cost rows are required")
    result: dict[str, Any] = {
        "cached_P2_decode": _quantiles(
            np.asarray([row["cached_P2_decode_s"] for row in rows], dtype=np.float64)
        )
    }
    for arm in ARMS:
        result[arm] = {}
        for key in ("host_to_device_s", "forward_to_TTC_s", "cached_P2_to_TTC_s"):
            result[arm][key.removesuffix("_s")] = _quantiles(
                np.asarray([row["timings"][arm][key] for row in rows], dtype=np.float64)
            )
        result[arm]["max_peak_extra_allocated_bytes"] = max(
            row["timings"][arm]["peak_extra_allocated_bytes"] for row in rows
        )
    paired = {}
    for key in ("host_to_device_s", "forward_to_TTC_s", "cached_P2_to_TTC_s"):
        left = np.asarray([row["timings"]["L0"][key] for row in rows], dtype=np.float64)
        right = np.asarray([row["timings"]["LW"][key] for row in rows], dtype=np.float64)
        paired[key.removesuffix("_s")] = {
            "LW_minus_L0_seconds": _quantiles(right - left),
            "LW_over_L0_ratio": _quantiles(right / left),
        }
    result["paired"] = paired
    result["order_counts"] = {arm: sum(row["order"][0] == arm for row in rows) for arm in ARMS}
    return result


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional timings required")
    return {
        "n": len(values),
        "mean_seconds": float(values.mean()),
        "p50_seconds": float(np.quantile(values, 0.50)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "p99_seconds": float(np.quantile(values, 0.99)),
    }


def _admit_complete(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    if (
        result["status"] != "COMPLETE"
        or result["registration_sha256"] != registration_sha
        or result["attempt_id"] != attempt_id
        or common.sha(attempt / "COLD.json") != result["cold_sha256"]
        or set(result["row_sha256"]) != {f"row_{i:03d}.json" for i in range(QUERY_COUNT)}
        or result.get("optimizer_updates") != 0
        or result.get("backward") is not False
        or result.get("official_comparison") is not False
        or result.get("system_promotion") is not False
    ):
        raise ValueError("Completed cost receipt changed or is incomplete")
    expected = _expected_outputs(record)
    validate_cold(
        common.read_json(attempt / "COLD.json"),
        record,
        expected,
        registration_sha,
        attempt_id,
    )
    rows = []
    for ordinal in range(QUERY_COUNT):
        name = f"row_{ordinal:03d}.json"
        path = attempt / name
        if common.sha(path) != result["row_sha256"][name]:
            raise ValueError("Completed cost row bytes changed")
        row = common.read_json(path)
        validate_row(row, record, ordinal)
        validate_outputs(row, expected)
        if row["registration"] != registration_sha or row["attempt_id"] != attempt_id:
            raise ValueError("Completed cost row belongs to another attempt")
        rows.append(row)
    if summarize(rows) != result["results"]:
        raise ValueError("Completed cost summary does not reproduce")
    return result


def measure() -> dict[str, Any]:
    """Measure cold setup once and 128 paired warm batch1 routes with zero updates."""
    import psutil
    import torch

    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.train import setup
    from operational.range_transfer.common import digest

    record = registered()
    key = digest(record)
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    attempt_record = attempt / "ATTEMPT.json"
    if attempt_record.exists():
        if common.read_json(attempt_record) != {
            "attempt_id": attempt_id,
            "registration_sha256": registration_sha,
            "optimizer_updates": 0,
        }:
            raise ValueError("Cost attempt identity changed")
    else:
        common.write_json(
            attempt_record,
            {
                "attempt_id": attempt_id,
                "registration_sha256": registration_sha,
                "optimizer_updates": 0,
            },
        )
    complete = DIRECTORY / "COMPLETE.json"
    if complete.exists():
        return _admit_complete(common.read_json(complete), record)
    expected = _expected_outputs(record)
    process = psutil.Process(os.getpid())
    cold: dict[str, Any] = {}
    models: dict[str, Any] = {}
    setup(7)
    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.runtime_context(),
        training.gpu_charge(
            "V19_BATCH1_CACHED_P2_TO_TTC_COST", reserve=GPU_RESERVE_SECONDS
        ) as deadline,
        torch.inference_mode(),
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
        cold_path = attempt / "COLD.json"
        record_cold = not cold_path.exists()
        for arm in ARMS:
            if time.perf_counter() >= deadline:
                raise RuntimeError("GPU budget ended before both frozen models loaded")
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated()
            tick = time.perf_counter()
            model = _load_export(arm)
            loaded = time.perf_counter()
            model = model.cuda()
            torch.cuda.synchronize()
            ready = time.perf_counter()
            resident_delta = torch.cuda.memory_allocated() - allocated
            models[arm] = model
            if not record_cold:
                continue
            reader_tick = time.perf_counter()
            dataset = SensorDataset("P")
            reader_ready = time.perf_counter()
            try:
                decode_tick = time.perf_counter()
                batch = dataset.batch([record["indices"][0]])
                decoded = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                transfer_tick = time.perf_counter()
                events = batch.events.cuda()
                delta = batch.delta_t_s.reshape(1, 1).cuda()
                torch.cuda.synchronize()
                transferred = time.perf_counter()
                prediction, phase = _inference(model, events, delta)
                torch.cuda.synchronize()
                forwarded = time.perf_counter()
                observed, observed_phase = float(prediction.item()), float(phase.item())
                expected_ttc, expected_phase = expected[arm][record["indices"][0]]
                if not np.isclose(observed, expected_ttc, rtol=1e-5, atol=1e-5) or not np.isclose(
                    observed_phase, expected_phase, rtol=1e-5, atol=1e-5
                ):
                    raise ValueError("Cold deployment output differs from frozen P prediction")
                cold[arm] = {
                    "torchscript_load_cpu_s": loaded - tick,
                    "model_to_cuda_s": ready - loaded,
                    "reader_init_s": reader_ready - reader_tick,
                    "cached_P2_decode_s": decoded - decode_tick,
                    "host_to_device_s": transferred - transfer_tick,
                    "first_forward_to_TTC_s": forwarded - transferred,
                    "cached_P2_to_TTC_s": forwarded - decode_tick,
                    "ttc_seconds": observed,
                    "emitted_phase": observed_phase,
                    "model_resident_delta_bytes": resident_delta,
                    "first_route_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "process_rss_bytes": process.memory_info().rss,
                    "meaning": "application-cold; OS cache uncontrolled; descriptive single call",
                }
                del events, delta, prediction, phase
            finally:
                dataset.close()

        dataset = SensorDataset("P")
        try:
            warm = dataset.batch([record["indices"][0]])
            warm_events = warm.events.cuda()
            warm_delta = warm.delta_t_s.reshape(1, 1).cuda()
            for arm in ARMS:
                for _ in range(WARMUPS):
                    _inference(models[arm], warm_events, warm_delta)
            torch.cuda.synchronize()
            if record_cold:
                common.write_json(
                    cold_path,
                    {
                        "registration": key,
                        "registration_sha256": registration_sha,
                        "attempt_id": attempt_id,
                        "cold": cold,
                        "cold_model_load_order": list(ARMS),
                        "cold_memory_scope": record["cold_memory_limit"],
                        "warmups_per_endpoint": WARMUPS,
                        "optimizer_updates": 0,
                    },
                )
            else:
                prior_cold = common.read_json(cold_path)
                if (
                    prior_cold["registration"] != key
                    or prior_cold["registration_sha256"] != registration_sha
                    or prior_cold["attempt_id"] != attempt_id
                    or set(prior_cold["cold"]) != set(ARMS)
                    or prior_cold["cold_model_load_order"] != list(ARMS)
                ):
                    raise ValueError("Prior cold evidence belongs to another attempt")
            for ordinal, (index, order) in enumerate(
                zip(record["indices"], record["paired_orders"], strict=True)
            ):
                path = attempt / f"row_{ordinal:03d}.json"
                if path.exists():
                    prior = common.read_json(path)
                    validate_row(prior, record, ordinal)
                    validate_outputs(prior, expected)
                    if (
                        prior["registration"] != registration_sha
                        or prior["attempt_id"] != attempt_id
                    ):
                        raise ValueError("Prior timing row belongs to another attempt")
                    continue
                if time.perf_counter() >= deadline:
                    partial = {
                        "status": "PARTIAL",
                        "attempt_id": attempt_id,
                        "registration_sha256": registration_sha,
                        "next_ordinal": ordinal,
                        "optimizer_updates": 0,
                    }
                    common.write_json(attempt / "PARTIAL.json", partial)
                    del warm_events, warm_delta
                    models.clear()
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    return partial
                decode_tick = time.perf_counter()
                batch = dataset.batch([index])
                decoded = time.perf_counter()
                timings = {}
                for arm in order:
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    transfer_tick = time.perf_counter()
                    events = batch.events.cuda()
                    delta = batch.delta_t_s.reshape(1, 1).cuda()
                    torch.cuda.synchronize()
                    transferred = time.perf_counter()
                    prediction, phase = _inference(models[arm], events, delta)
                    torch.cuda.synchronize()
                    forwarded = time.perf_counter()
                    observed, observed_phase = float(prediction.item()), float(phase.item())
                    expected_ttc, expected_phase = expected[arm][index]
                    if not np.isclose(
                        observed, expected_ttc, rtol=1e-5, atol=1e-5
                    ) or not np.isclose(observed_phase, expected_phase, rtol=1e-5, atol=1e-5):
                        raise ValueError("Warm deployment output differs from frozen P prediction")
                    timings[arm] = {
                        "host_to_device_s": transferred - transfer_tick,
                        "forward_to_TTC_s": forwarded - transferred,
                        "cached_P2_to_TTC_s": decoded - decode_tick + forwarded - transfer_tick,
                        "ttc_seconds": observed,
                        "emitted_phase": observed_phase,
                        "peak_extra_allocated_bytes": (torch.cuda.max_memory_allocated() - before),
                        "memory_allocated_before_bytes": before,
                    }
                    del events, delta, prediction, phase
                row = {
                    "ordinal": ordinal,
                    "P_index": index,
                    "order": order,
                    "cached_P2_decode_s": decoded - decode_tick,
                    "timings": timings,
                    "registration": registration_sha,
                    "attempt_id": attempt_id,
                    "utc": datetime.now(UTC).isoformat(),
                }
                validate_row(row, record, ordinal)
                validate_outputs(row, expected)
                common.write_json(path, row)
        finally:
            dataset.close()
        rows = [common.read_json(attempt / f"row_{i:03d}.json") for i in range(QUERY_COUNT)]
        for ordinal, row in enumerate(rows):
            validate_row(row, record, ordinal)
            validate_outputs(row, expected)
        validate_cold(common.read_json(cold_path), record, expected, registration_sha, attempt_id)
        result = {
            "status": "COMPLETE",
            "registration_sha256": registration_sha,
            "attempt_id": attempt_id,
            "cold_sha256": common.sha(cold_path),
            "row_sha256": {
                f"row_{i:03d}.json": common.sha(attempt / f"row_{i:03d}.json")
                for i in range(QUERY_COUNT)
            },
            "results": summarize(rows),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "host": platform.node(),
            "optimizer_updates": 0,
            "backward": False,
            "official_comparison": False,
            "system_promotion": False,
            "comparison_limits": record["comparison_limits"],
        }
        common.write_json(complete, result)
        _admit_complete(result, record)
        del warm_events, warm_delta
        models.clear()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "measure"))
    args = parser.parse_args()
    print({"register": register, "measure": measure}[args.action]())
