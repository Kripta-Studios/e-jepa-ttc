"""Prospective batch-native and descriptive batch-one cost measurement.

The primary route reconstructs the exact four-row batches used to produce the
frozen P predictions.  Batch-one is measured as a separate deployment-shaped
route and its numerical drift is reported explicitly; it is never described as
exact parity with the historical batch-four predictions.
"""

from __future__ import annotations

import argparse
import gc
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import common, evaluation, training
from . import cost_measurement as v3

if TYPE_CHECKING:
    import torch

    from operational.joint_measurement.data import ObjectEventV4Batch

ARMS = v3.ARMS
QUERY_SEED = v3.QUERY_SEED
QUERY_COUNT = v3.QUERY_COUNT
PACK_ROWS = v3.PACK_ROWS
BATCH = 4
WARMUPS = 20
GPU_RESERVE_SECONDS = 180
RAW_PHASE_ABS_ENVELOPE = 1e-6
TTC_ABS_ENVELOPE_SECONDS = 0.01
SATURATION_TTC_SECONDS = 59.999
DIRECTORY = training.destination("cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v4")


def _batch_indices(index: int) -> list[int]:
    start = index - index % BATCH
    return list(range(start, start + BATCH))


def _position(index: int) -> int:
    return index % BATCH


def _is_saturated(ttc: float) -> bool:
    """Use the emitted TTC clamp, not raw/emitted phase magnitude."""
    return abs(ttc) >= SATURATION_TTC_SECONDS


def protocol() -> dict[str, Any]:
    """Return the fixed design without opening the cache, models, or CUDA."""
    base = v3.protocol()
    return {
        "schema": "v19_batch4_native_batch1_descriptive_cost_v4",
        "arms": list(ARMS),
        "population": "P",
        "query_seed": QUERY_SEED,
        "indices": base["indices"],
        "paired_orders": base["paired_orders"],
        "query_scope": base["query_scope"],
        "primary_route": (
            "eager CUDA FP32 on the historical contiguous batch4 containing each fixed query; "
            "latencies reported per batch and amortized per query"
        ),
        "descriptive_route": (
            "eager CUDA FP32 batch1 for the fixed query; numerical drift is measured and this "
            "route is not asserted exact against historical batch4 predictions"
        ),
        "historical_batch_size": BATCH,
        "warmups_per_endpoint_and_route": WARMUPS,
        "precision": "CUDA_FP32_no_TF32",
        "input_scope": base["input_scope"],
        "cold_scope": (
            "application-cold primary batch4 route; OS file cache uncontrolled; fixed L0,LW "
            "model load order is descriptive"
        ),
        "warm_scope": (
            "application-warm cache reader and models; includes ZIP member decode, collation, "
            "H2D and synchronized eager forward"
        ),
        "stages": ["cached_P2_decode", "host_to_device", "forward_to_TTC", "cached_P2_to_TTC"],
        "batch1_drift_contract": {
            "provenance": (
                "prospectively fixed after directed index210 diagnostic: observed LW raw-phase "
                "drift 3.213062882423401e-08 and TTC drift 0.000415802001953125 seconds"
            ),
            "raw_phase_absolute_envelope": RAW_PHASE_ABS_ENVELOPE,
            "ttc_absolute_envelope_seconds": TTC_ABS_ENVELOPE_SECONDS,
            "required_same_ttc_sign": True,
            "required_same_saturation_state": True,
            "saturation_rule": "abs(emitted TTC)>=59.999 seconds",
            "interpretation": "engineering guard against route failure, not an exact-parity claim",
        },
        "gpu_reserve_seconds": GPU_RESERVE_SECONDS,
        "optimizer_updates": 0,
        "backward": False,
        "official_comparison": False,
        "system_promotion": False,
        "comparison_limits": (
            "technical cached-P2 latency only; not raw-sensor end-to-end and not equivalent to "
            "historical V18 timing protocols"
        ),
    }


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_cost_measurement_v4.py",
        Path(v3.__file__),
        Path(common.__file__),
        Path(evaluation.__file__),
        Path(training.__file__),
        common.ROOT / "operational/joint_measurement/data.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def registration_payload(source_commit: str) -> dict[str, Any]:
    record = protocol()
    evaluation_record = common.read_json(training.destination("evaluation/REGISTRATION.json"))
    endpoints = {arm: evaluation._endpoint(arm, training.registered()) for arm in ARMS}
    if endpoints != evaluation_record["endpoints"]:
        raise ValueError("Evaluation endpoints changed before v4 cost registration")
    probe = training.destination("cost/parity_probe_index210_v1")
    probe_result = common.read_json(probe / "RESULT.json")
    probe_complete = common.read_json(probe / "COMPLETE.json")
    if (
        probe_result.get("status") != "COMPLETE_DIAGNOSTIC"
        or probe_result.get("optimizer_updates") != 0
        or probe_result.get("backward") is not False
        or probe_complete.get("result_sha256") != common.sha(probe / "RESULT.json")
    ):
        raise ValueError("Complete zero-update parity diagnostic required")
    record.update(
        endpoints=endpoints,
        P_predictions={arm: v3._prediction_binding(arm) for arm in ARMS},
        P_manifest_sha256=common.sha(training.manifest_path("P")),
        P_cache=v3._pack_binding(),
        parity_diagnostic={
            "registration_sha256": common.sha(probe / "REGISTRATION.json"),
            "result_sha256": common.sha(probe / "RESULT.json"),
            "complete_sha256": common.sha(probe / "COMPLETE.json"),
        },
        evaluation_registration_sha256=common.sha(
            training.destination("evaluation/REGISTRATION.json")
        ),
        training_registration_sha256=common.sha(training.destination("training/REGISTRATION.json")),
        sources=_sources(),
        source_commit=source_commit,
        registration_precedes_gpu_measurement=True,
    )
    return record


def register() -> dict[str, Any]:
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
        raise ValueError("Registered v4 cost sources, endpoints, cache, or diagnostic changed")
    return value


def _expected(record: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.phase import ttc_to_phase

    result: dict[str, dict[int, dict[str, float]]] = {}
    for arm in ARMS:
        frame = pd.read_parquet(
            training.destination(f"evaluation/predictions/P/{arm}/predictions.parquet"),
            columns=["prediction_ttc", "raw_phase"],
        )
        result[arm] = {}
        for index in record["indices"]:
            ttc = float(frame.iloc[index].prediction_ttc)
            result[arm][index] = {
                "raw_phase": float(frame.iloc[index].raw_phase),
                "ttc_seconds": ttc,
                "emitted_phase": float(
                    ttc_to_phase(torch.tensor([ttc], dtype=torch.float32)).item()
                ),
            }
    return result


def drift(observed: dict[str, float], expected: dict[str, float]) -> dict[str, Any]:
    """Describe and prospectively guard batch-one numerical drift."""
    raw_error = abs(observed["raw_phase"] - expected["raw_phase"])
    ttc_error = abs(observed["ttc_seconds"] - expected["ttc_seconds"])
    same_sign = bool(np.signbit(observed["ttc_seconds"]) == np.signbit(expected["ttc_seconds"]))
    same_saturation = _is_saturated(observed["ttc_seconds"]) == _is_saturated(
        expected["ttc_seconds"]
    )
    admitted = (
        np.isfinite([*observed.values(), raw_error, ttc_error]).all()
        and raw_error <= RAW_PHASE_ABS_ENVELOPE
        and ttc_error <= TTC_ABS_ENVELOPE_SECONDS
        and same_sign
        and same_saturation
    )
    return {
        "raw_phase_absolute_error": raw_error,
        "ttc_absolute_error_seconds": ttc_error,
        "same_ttc_sign": same_sign,
        "same_saturation_state": same_saturation,
        "observed_saturated": _is_saturated(observed["ttc_seconds"]),
        "expected_saturated": _is_saturated(expected["ttc_seconds"]),
        "admitted": bool(admitted),
    }


def validate_output(
    primary: dict[str, float], descriptive: dict[str, float], expected: dict[str, float]
) -> dict[str, Any]:
    """Require exact native-batch output and guarded descriptive batch-one drift."""
    if any(primary[name] != expected[name] for name in expected):
        raise ValueError("Native batch4 output differs from frozen P prediction")
    evidence = drift(descriptive, expected)
    if not evidence["admitted"]:
        raise ValueError("Descriptive batch1 output exceeds prospective drift contract")
    return evidence


def _forward(model: torch.nn.Module, batch: ObjectEventV4Batch) -> dict[str, Any]:
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase

    events = batch.events.cuda()
    delta = batch.delta_t_s.reshape(-1).cuda()
    torch.cuda.synchronize()
    output = model(events, delta)
    raw = output["raw_phase"]
    ttc = phase_to_ttc(raw)
    emitted = ttc_to_phase(ttc)
    torch.cuda.synchronize()
    return {"events": events, "delta": delta, "raw": raw, "ttc": ttc, "emitted": emitted}


def _timed_forward(
    model: torch.nn.Module, batch: ObjectEventV4Batch
) -> tuple[dict[str, Any], float, float, int]:
    """Time synchronized H2D and forward separately for one immutable batch."""
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    tick = time.perf_counter()
    events = batch.events.cuda()
    delta = batch.delta_t_s.reshape(-1).cuda()
    torch.cuda.synchronize()
    transferred = time.perf_counter()
    raw = model(events, delta)["raw_phase"]
    ttc = phase_to_ttc(raw)
    emitted = ttc_to_phase(ttc)
    torch.cuda.synchronize()
    forwarded = time.perf_counter()
    return (
        {"events": events, "delta": delta, "raw": raw, "ttc": ttc, "emitted": emitted},
        transferred - tick,
        forwarded - transferred,
        torch.cuda.max_memory_allocated() - before,
    )


def _values(output: dict[str, Any], position: int) -> dict[str, float]:
    return {
        "raw_phase": float(output["raw"][position].item()),
        "ttc_seconds": float(output["ttc"][position].item()),
        "emitted_phase": float(output["emitted"][position].item()),
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Finite one-dimensional timings required")
    return {
        "n": len(array),
        "mean_seconds": float(array.mean()),
        "p50_seconds": float(np.quantile(array, 0.5)),
        "p95_seconds": float(np.quantile(array, 0.95)),
        "p99_seconds": float(np.quantile(array, 0.99)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != QUERY_COUNT:
        raise ValueError("All fixed v4 cost rows are required")
    result: dict[str, Any] = {}
    for route in ("batch4_native", "batch1_descriptive"):
        result[route] = {}
        for arm in ARMS:
            result[route][arm] = {
                name.removesuffix("_s"): _stats([row["routes"][route][arm][name] for row in rows])
                for name in (
                    "cached_P2_decode_s",
                    "host_to_device_s",
                    "forward_to_TTC_s",
                    "cached_P2_to_TTC_s",
                )
            }
            result[route][arm]["max_peak_extra_allocated_bytes"] = max(
                row["routes"][route][arm]["peak_extra_allocated_bytes"] for row in rows
            )
        if route == "batch4_native":
            for arm in ARMS:
                result[route][arm]["amortized_per_query"] = {
                    name.removesuffix("_s"): _stats(
                        [row["routes"][route][arm][name] / BATCH for row in rows]
                    )
                    for name in (
                        "cached_P2_decode_s",
                        "host_to_device_s",
                        "forward_to_TTC_s",
                        "cached_P2_to_TTC_s",
                    )
                }
    result["batch1_drift_max"] = {
        arm: {
            "raw_phase_absolute_error": max(
                row["routes"]["batch1_descriptive"][arm]["drift"]["raw_phase_absolute_error"]
                for row in rows
            ),
            "ttc_absolute_error_seconds": max(
                row["routes"]["batch1_descriptive"][arm]["drift"]["ttc_absolute_error_seconds"]
                for row in rows
            ),
        }
        for arm in ARMS
    }
    return result


def validate_row(row: dict[str, Any], record: dict[str, Any], ordinal: int) -> None:
    if (
        set(row)
        != {
            "ordinal",
            "P_index",
            "order",
            "batch4_indices",
            "routes",
            "registration",
            "attempt_id",
            "utc",
        }
        or row["ordinal"] != ordinal
        or row["P_index"] != record["indices"][ordinal]
        or row["order"] != record["paired_orders"][ordinal]
        or row["batch4_indices"] != _batch_indices(row["P_index"])
        or set(row["routes"]) != {"batch4_native", "batch1_descriptive"}
    ):
        raise ValueError("V4 cost row identity or schema changed")
    for route in row["routes"].values():
        if set(route) != set(ARMS):
            raise ValueError("V4 cost arm set changed")
        for value in route.values():
            timings = [
                value[name]
                for name in (
                    "cached_P2_decode_s",
                    "host_to_device_s",
                    "forward_to_TTC_s",
                    "cached_P2_to_TTC_s",
                )
            ]
            if not np.isfinite(timings).all() or any(item <= 0 for item in timings):
                raise ValueError("Invalid v4 timing")


def validate_row_outputs(
    row: dict[str, Any], expected: dict[str, dict[int, dict[str, float]]]
) -> None:
    """Recompute all output and drift guards when durable evidence is reopened."""
    index = row["P_index"]
    output_names = ("raw_phase", "ttc_seconds", "emitted_phase")
    for arm in ARMS:
        primary = {name: row["routes"]["batch4_native"][arm][name] for name in output_names}
        descriptive = {
            name: row["routes"]["batch1_descriptive"][arm][name] for name in output_names
        }
        evidence = validate_output(primary, descriptive, expected[arm][index])
        if row["routes"]["batch1_descriptive"][arm].get("drift") != evidence:
            raise ValueError("Durable batch1 drift evidence changed")


def validate_cold(
    cold: dict[str, Any],
    record: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
    registration_sha: str,
    attempt_id: str,
) -> None:
    """Validate cold primary-route identity, values, parity, and nonnegative memory."""
    if (
        cold.get("registration_sha256") != registration_sha
        or cold.get("attempt_id") != attempt_id
        or cold.get("model_load_order") != list(ARMS)
        or cold.get("optimizer_updates") != 0
        or cold.get("backward") is not False
        or set(cold.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("Completed v4 cold evidence changed")
    first = record["indices"][0]
    positive = (
        "checkpoint_load_eager_cpu_s",
        "model_to_cuda_s",
        "reader_init_s",
        "cached_P2_decode_s",
        "host_to_device_s",
        "first_forward_to_TTC_s",
        "cached_P2_to_TTC_s",
    )
    for arm in ARMS:
        value = cold["arms"][arm]
        numbers = [value[name] for name in positive]
        if (
            not np.isfinite(numbers).all()
            or any(item <= 0 for item in numbers)
            or value["model_resident_delta_bytes"] < 0
            or value["first_route_peak_extra_allocated_bytes"] < 0
        ):
            raise ValueError("Invalid v4 cold timing or memory evidence")
        if any(value[name] != expected[arm][first][name] for name in expected[arm][first]):
            raise ValueError("Cold native batch4 output differs from frozen P prediction")


def _admit_complete(
    receipt: dict[str, Any],
    record: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    """Replay hashes, zero-update flags, rows, outputs, cold evidence, and summary."""
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    result_path = attempt / "RESULT.json"
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("registration_sha256") != registration_sha
        or receipt.get("result_sha256") != common.sha(result_path)
        or receipt.get("cold_sha256") != common.sha(attempt / "COLD.json")
        or receipt.get("optimizer_updates") != 0
        or receipt.get("backward") is not False
    ):
        raise ValueError("Completed v4 cost receipt changed")
    result = common.read_json(result_path)
    if (
        result.get("status") != "COMPLETE"
        or result.get("registration_sha256") != registration_sha
        or result.get("attempt_id") != attempt_id
        or result.get("optimizer_updates") != 0
        or result.get("backward") is not False
        or result.get("official_comparison") is not False
        or result.get("system_promotion") is not False
        or set(result.get("row_sha256", {})) != {f"row_{i:03d}.json" for i in range(QUERY_COUNT)}
    ):
        raise ValueError("Completed v4 cost result changed")
    rows = []
    for ordinal in range(QUERY_COUNT):
        name = f"row_{ordinal:03d}.json"
        path = attempt / name
        if common.sha(path) != result["row_sha256"][name]:
            raise ValueError("Completed v4 row bytes changed")
        row = common.read_json(path)
        validate_row(row, record, ordinal)
        validate_row_outputs(row, expected)
        if row["registration"] != registration_sha or row["attempt_id"] != attempt_id:
            raise ValueError("Completed v4 row belongs to another attempt")
        rows.append(row)
    if summarize(rows) != result["results"]:
        raise ValueError("Completed v4 summary does not reproduce")
    cold = common.read_json(attempt / "COLD.json")
    validate_cold(cold, record, expected, registration_sha, attempt_id)
    if result.get("checkpoint_sha256_after") != {
        arm: record["endpoints"][arm]["checkpoint_sha256"] for arm in ARMS
    }:
        raise ValueError("Completed v4 checkpoint identity changed")
    return result


def measure() -> dict[str, Any]:
    """Measure both frozen routes; no optimizer, backward, or model mutation."""
    import torch

    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.train import setup
    from operational.range_transfer.common import digest

    record = registered()
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    complete = DIRECTORY / "COMPLETE.json"
    if complete.exists():
        return _admit_complete(common.read_json(complete), record, _expected(record))

    expected = _expected(record)
    models: dict[str, Any] = {}
    dataset = None
    setup(7)
    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.runtime_context(),
        training.gpu_charge(
            "V19_BATCH4_NATIVE_BATCH1_DESCRIPTIVE_COST", reserve=GPU_RESERVE_SECONDS
        ) as deadline,
        torch.inference_mode(),
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            checkpoint_before = {arm: record["endpoints"][arm]["checkpoint_sha256"] for arm in ARMS}
            cold: dict[str, Any] = {}
            load_timings: dict[str, tuple[float, float, int]] = {}
            for arm in ARMS:
                allocated = torch.cuda.memory_allocated()
                tick = time.perf_counter()
                model = v3._load_eager(arm, record)
                loaded = time.perf_counter()
                models[arm] = model.cuda()
                torch.cuda.synchronize()
                ready = time.perf_counter()
                load_timings[arm] = (
                    loaded - tick,
                    ready - loaded,
                    torch.cuda.memory_allocated() - allocated,
                )
            reader_tick = time.perf_counter()
            dataset = SensorDataset("P")
            reader_ready = time.perf_counter()
            first = record["indices"][0]
            for arm in ARMS:
                decode_tick = time.perf_counter()
                cold_batch = dataset.batch(_batch_indices(first))
                decoded = time.perf_counter()
                output, h2d_s, forward_s, peak_bytes = _timed_forward(models[arm], cold_batch)
                values = _values(output, _position(first))
                if any(values[name] != expected[arm][first][name] for name in values):
                    raise ValueError("Cold native batch4 output differs from frozen P prediction")
                cpu_load_s, cuda_load_s, resident_bytes = load_timings[arm]
                cold[arm] = {
                    "checkpoint_load_eager_cpu_s": cpu_load_s,
                    "model_to_cuda_s": cuda_load_s,
                    "model_resident_delta_bytes": resident_bytes,
                    "reader_init_s": reader_ready - reader_tick,
                    "cached_P2_decode_s": decoded - decode_tick,
                    "host_to_device_s": h2d_s,
                    "first_forward_to_TTC_s": forward_s,
                    "cached_P2_to_TTC_s": decoded - decode_tick + h2d_s + forward_s,
                    "first_route_peak_extra_allocated_bytes": peak_bytes,
                    **values,
                }
                del output
            cold_path = attempt / "COLD.json"
            common.write_json(
                cold_path,
                {
                    "registration_sha256": registration_sha,
                    "attempt_id": attempt_id,
                    "meaning": "application-cold batch4 primary route; OS cache uncontrolled",
                    "model_load_order": list(ARMS),
                    "arms": cold,
                    "optimizer_updates": 0,
                    "backward": False,
                },
            )
            validate_cold(
                common.read_json(cold_path), record, expected, registration_sha, attempt_id
            )
            warm4, warm1 = dataset.batch(_batch_indices(first)), dataset.batch([first])
            for arm in ARMS:
                for _ in range(WARMUPS):
                    out4 = _forward(models[arm], warm4)
                    out1 = _forward(models[arm], warm1)
                    del out4, out1
            rows = []
            for ordinal, (index, order) in enumerate(
                zip(record["indices"], record["paired_orders"], strict=True)
            ):
                if time.perf_counter() >= deadline:
                    raise RuntimeError("GPU budget ended before v4 cost rows completed")
                routes = {"batch4_native": {}, "batch1_descriptive": {}}
                for route, indices in (
                    ("batch4_native", _batch_indices(index)),
                    ("batch1_descriptive", [index]),
                ):
                    for arm in order:
                        decode_tick = time.perf_counter()
                        batch = dataset.batch(indices)
                        decoded = time.perf_counter()
                        measured, h2d_s, forward_s, peak_bytes = _timed_forward(models[arm], batch)
                        position = _position(index) if route == "batch4_native" else 0
                        values = _values(measured, position)
                        entry = {
                            "cached_P2_decode_s": decoded - decode_tick,
                            "host_to_device_s": h2d_s,
                            "forward_to_TTC_s": forward_s,
                            "cached_P2_to_TTC_s": decoded - decode_tick + h2d_s + forward_s,
                            "peak_extra_allocated_bytes": peak_bytes,
                            **values,
                        }
                        if route == "batch4_native":
                            if any(values[name] != expected[arm][index][name] for name in values):
                                raise ValueError(
                                    "Native batch4 output differs from frozen P prediction"
                                )
                        else:
                            evidence = drift(values, expected[arm][index])
                            if not evidence["admitted"]:
                                raise ValueError(
                                    "Descriptive batch1 output exceeds prospective drift contract"
                                )
                            entry["drift"] = evidence
                        routes[route][arm] = entry
                        del measured
                row = {
                    "ordinal": ordinal,
                    "P_index": index,
                    "order": order,
                    "batch4_indices": _batch_indices(index),
                    "routes": routes,
                    "registration": registration_sha,
                    "attempt_id": attempt_id,
                    "utc": datetime.now(UTC).isoformat(),
                }
                validate_row(row, record, ordinal)
                common.write_json(attempt / f"row_{ordinal:03d}.json", row)
                rows.append(row)
            checkpoint_after = {
                arm: common.sha(
                    training.destination(f"training/fits/FULL-{arm}-S7/checkpoint_last.pt")
                )
                for arm in ARMS
            }
            if checkpoint_after != checkpoint_before:
                raise ValueError("Frozen checkpoints changed during v4 measurement")
            result = {
                "status": "COMPLETE",
                "registration_sha256": registration_sha,
                "registration_digest": digest(record),
                "attempt_id": attempt_id,
                "row_sha256": {
                    f"row_{i:03d}.json": common.sha(attempt / f"row_{i:03d}.json")
                    for i in range(QUERY_COUNT)
                },
                "cold_sha256": common.sha(cold_path),
                "results": summarize(rows),
                "checkpoint_sha256_after": checkpoint_after,
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "host": platform.node(),
                "optimizer_updates": 0,
                "backward": False,
                "official_comparison": False,
                "system_promotion": False,
            }
            result_path = attempt / "RESULT.json"
            common.write_json(result_path, result)
            common.write_json(
                complete,
                {
                    "status": "COMPLETE",
                    "registration_sha256": registration_sha,
                    "result_sha256": common.sha(result_path),
                    "cold_sha256": common.sha(cold_path),
                    "optimizer_updates": 0,
                    "backward": False,
                },
            )
            return result
        finally:
            if dataset is not None:
                dataset.close()
            models.clear()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "measure"))
    args = parser.parse_args()
    print({"register": register, "measure": measure}[args.action]())
