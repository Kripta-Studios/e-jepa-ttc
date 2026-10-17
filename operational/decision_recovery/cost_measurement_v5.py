"""V19 cost V5 with backend-consistent derived-phase parity.

Raw phase and TTC retain exact parity with frozen P predictions.  Emitted phase
is derived and checked against a CUDA reference generated from the exact saved
TTC values outside every timed region.  Batch-one remains descriptive under the
unchanged prospective V4 drift guard.
"""

from __future__ import annotations

import argparse
import gc
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    common,
    evaluation,
    training,
)
from . import (
    cost_measurement as v3,
)
from . import (
    cost_measurement_v4 as v4,
)
from . import (
    cost_v4_failure_probe as failure_probe,
)

ARMS = v4.ARMS
QUERY_COUNT = v4.QUERY_COUNT
WARMUPS = v4.WARMUPS
GPU_RESERVE_SECONDS = 180
DIRECTORY = training.destination("cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v5")
V4_DIRECTORY = v4.DIRECTORY
PROBE_DIRECTORY = failure_probe.DIRECTORY


def protocol() -> dict[str, Any]:
    value = v4.protocol()
    value.update(
        schema="v19_batch4_native_batch1_descriptive_cost_v5",
        emitted_phase_contract=(
            "derived CUDA phase is compared exactly with a CUDA reference computed from each "
            "exact frozen TTC outside timed regions; CPU-derived phase remains descriptive"
        ),
        v4_amendment=(
            "V4 stopped after 42 rows because CPU and CUDA ttc_to_phase differed by one float32 "
            "ULP at ordinal42 LW; raw phase and TTC were exact"
        ),
    )
    return value


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_cost_measurement_v5.py",
        Path(v4.__file__),
        Path(v3.__file__),
        Path(failure_probe.__file__),
        Path(common.__file__),
        Path(evaluation.__file__),
        Path(training.__file__),
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        common.ROOT / "operational/joint_measurement/data.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def registration_payload(source_commit: str) -> dict[str, Any]:
    value = protocol()
    v4_record = v4.registered()
    probe_result = common.read_json(PROBE_DIRECTORY / "RESULT.json")
    probe_complete = common.read_json(PROBE_DIRECTORY / "COMPLETE.json")
    lw = probe_result["observations"]["LW"]["comparisons"]
    if (
        probe_result.get("status") != "COMPLETE_DIAGNOSTIC"
        or probe_result.get("optimizer_updates") != 0
        or probe_result.get("backward") is not False
        or probe_complete.get("result_sha256") != common.sha(PROBE_DIRECTORY / "RESULT.json")
        or not lw["raw_vs_saved"]["exact"]
        or not lw["ttc_vs_saved"]["exact"]
        or lw["emitted_cuda_vs_cpu_from_saved_ttc"]["exact"]
        or not lw["emitted_cuda_vs_cuda_from_saved_ttc"]["exact"]
    ):
        raise ValueError("Complete directed V4 backend diagnosis required")
    attempt_id = common.sha(V4_DIRECTORY / "REGISTRATION.json")[:16]
    v4_attempt = V4_DIRECTORY / "attempts" / attempt_id
    partial = sorted(v4_attempt.glob("row_*.json"))
    if len(partial) != 42:
        raise ValueError("Preserved 42-row V4 attempt required")
    failed_receipt_path = V4_DIRECTORY / "FAILED_RECEIPT.json"
    failed_receipt = common.read_json(failed_receipt_path)
    if (
        failed_receipt.get("status") != "FAILED_DIAGNOSTICALLY_RESOLVED"
        or failed_receipt.get("completed_rows") != 42
        or failed_receipt.get("failure_ordinal") != 42
        or failed_receipt.get("failure_P_index") != 91
        or failed_receipt.get("optimizer_updates") != 0
        or failed_receipt.get("backward") is not False
        or failed_receipt.get("diagnostic_result_sha256")
        != common.sha(PROBE_DIRECTORY / "RESULT.json")
    ):
        raise ValueError("Resolved V4 failure receipt required")
    value.update(
        endpoints=v4_record["endpoints"],
        P_predictions=v4_record["P_predictions"],
        P_manifest_sha256=v4_record["P_manifest_sha256"],
        P_cache=v4_record["P_cache"],
        V4_failed_attempt={
            "registration_sha256": common.sha(V4_DIRECTORY / "REGISTRATION.json"),
            "cold_sha256": common.sha(v4_attempt / "COLD.json"),
            "partial_row_sha256": {path.name: common.sha(path) for path in partial},
            "failed_receipt_sha256": common.sha(failed_receipt_path),
        },
        directed_backend_probe={
            "registration_sha256": common.sha(PROBE_DIRECTORY / "REGISTRATION.json"),
            "result_sha256": common.sha(PROBE_DIRECTORY / "RESULT.json"),
            "complete_sha256": common.sha(PROBE_DIRECTORY / "COMPLETE.json"),
        },
        training_registration_sha256=v4_record["training_registration_sha256"],
        evaluation_registration_sha256=v4_record["evaluation_registration_sha256"],
        sources=_sources(),
        source_commit=source_commit,
        registration_precedes_gpu_measurement=True,
    )
    return value


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
        raise ValueError("Registered V5 cost identity changed")
    return value


def validate_primary(
    observed: dict[str, float], expected: dict[str, float], emitted_cuda: float
) -> None:
    """Keep exact raw/TTC parity and compare derived phase on the same backend."""
    if observed["raw_phase"] != expected["raw_phase"]:
        raise ValueError("Native batch4 raw phase differs from frozen P prediction")
    if observed["ttc_seconds"] != expected["ttc_seconds"]:
        raise ValueError("Native batch4 TTC differs from frozen P prediction")
    if observed["emitted_phase"] != emitted_cuda:
        raise ValueError("Native batch4 CUDA emitted phase differs from CUDA reference")


def _cuda_reference(
    expected: dict[str, dict[int, dict[str, float]]], record: dict[str, Any]
) -> dict[str, dict[int, float]]:
    """Generate the derived phase reference on CUDA before any timed row."""
    import torch

    from e_jepa_ttc.simplex_t.phase import ttc_to_phase

    result: dict[str, dict[int, float]] = {}
    for arm in ARMS:
        tensor = torch.tensor(
            [expected[arm][index]["ttc_seconds"] for index in record["indices"]],
            dtype=torch.float32,
            device="cuda",
        )
        phases = ttc_to_phase(tensor)
        torch.cuda.synchronize()
        result[arm] = {
            index: float(phase)
            for index, phase in zip(record["indices"], phases.cpu().tolist(), strict=True)
        }
    return result


def _validate_cuda_reference(
    payload: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
    record: dict[str, Any],
    registration_sha: str,
) -> dict[str, dict[int, float]]:
    if (
        payload.get("registration_sha256") != registration_sha
        or payload.get("backend") != "CUDA"
        or payload.get("optimizer_updates") != 0
        or payload.get("backward") is not False
        or set(payload.get("values", {})) != set(ARMS)
    ):
        raise ValueError("V5 CUDA phase reference identity changed")
    result: dict[str, dict[int, float]] = {}
    expected_keys = {str(index) for index in record["indices"]}
    for arm in ARMS:
        if set(payload["values"][arm]) != expected_keys:
            raise ValueError("V5 CUDA phase reference rows changed")
        result[arm] = {}
        for index in record["indices"]:
            item = payload["values"][arm][str(index)]
            if (
                set(item) != {"input_ttc_seconds", "emitted_phase_cuda"}
                or not np.isfinite([item["input_ttc_seconds"], item["emitted_phase_cuda"]]).all()
                or item["input_ttc_seconds"] != expected[arm][index]["ttc_seconds"]
            ):
                raise ValueError("V5 CUDA phase reference TTC changed")
            result[arm][index] = item["emitted_phase_cuda"]
    return result


def _validate_row_outputs(
    row: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
    cuda_reference: dict[str, dict[int, float]],
) -> None:
    index = row["P_index"]
    names = ("raw_phase", "ttc_seconds", "emitted_phase")
    for arm in ARMS:
        primary = {name: row["routes"]["batch4_native"][arm][name] for name in names}
        validate_primary(primary, expected[arm][index], cuda_reference[arm][index])
        descriptive = {name: row["routes"]["batch1_descriptive"][arm][name] for name in names}
        evidence = v4.drift(descriptive, expected[arm][index])
        if row["routes"]["batch1_descriptive"][arm].get("drift") != evidence:
            raise ValueError("Durable V5 batch1 drift evidence changed")


def _validate_cold(
    cold: dict[str, Any],
    record: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
    cuda_reference: dict[str, dict[int, float]],
    registration_sha: str,
    attempt_id: str,
) -> None:
    if (
        cold.get("registration_sha256") != registration_sha
        or cold.get("attempt_id") != attempt_id
        or cold.get("model_load_order") != list(ARMS)
        or cold.get("optimizer_updates") != 0
        or cold.get("backward") is not False
        or set(cold.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("V5 cold identity changed")
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
            raise ValueError("Invalid V5 cold timing or memory")
        validate_primary(value, expected[arm][first], cuda_reference[arm][first])


def _admit_complete(
    receipt: dict[str, Any],
    record: dict[str, Any],
    expected: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    result_path = attempt / "RESULT.json"
    reference_path = attempt / "CUDA_REFERENCE.json"
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("registration_sha256") != registration_sha
        or receipt.get("result_sha256") != common.sha(result_path)
        or receipt.get("cuda_reference_sha256") != common.sha(reference_path)
        or receipt.get("cold_sha256") != common.sha(attempt / "COLD.json")
        or receipt.get("optimizer_updates") != 0
        or receipt.get("backward") is not False
    ):
        raise ValueError("Completed V5 receipt changed")
    cuda_reference = _validate_cuda_reference(
        common.read_json(reference_path), expected, record, registration_sha
    )
    result = common.read_json(result_path)
    if (
        result.get("status") != "COMPLETE"
        or result.get("registration_sha256") != registration_sha
        or result.get("attempt_id") != attempt_id
        or result.get("cuda_reference_sha256") != common.sha(reference_path)
        or result.get("cold_sha256") != common.sha(attempt / "COLD.json")
        or result.get("optimizer_updates") != 0
        or result.get("backward") is not False
        or result.get("official_comparison") is not False
        or result.get("system_promotion") is not False
        or set(result.get("row_sha256", {})) != {f"row_{i:03d}.json" for i in range(QUERY_COUNT)}
    ):
        raise ValueError("Completed V5 result changed")
    rows = []
    for ordinal in range(QUERY_COUNT):
        name = f"row_{ordinal:03d}.json"
        path = attempt / name
        if common.sha(path) != result["row_sha256"][name]:
            raise ValueError("Completed V5 row bytes changed")
        row = common.read_json(path)
        v4.validate_row(row, record, ordinal)
        _validate_row_outputs(row, expected, cuda_reference)
        if row["registration"] != registration_sha or row["attempt_id"] != attempt_id:
            raise ValueError("Completed V5 row belongs to another attempt")
        rows.append(row)
    if v4.summarize(rows) != result["results"]:
        raise ValueError("Completed V5 summary does not reproduce")
    if result.get("checkpoint_sha256_after") != {
        arm: record["endpoints"][arm]["checkpoint_sha256"] for arm in ARMS
    }:
        raise ValueError("Completed V5 checkpoint identity changed")
    _validate_cold(
        common.read_json(attempt / "COLD.json"),
        record,
        expected,
        cuda_reference,
        registration_sha,
        attempt_id,
    )
    return result


def measure() -> dict[str, Any]:
    """Measure V5 with an untimed CUDA derived-phase reference and zero updates."""
    import torch

    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.train import setup

    record = registered()
    expected = v4._expected(record)
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    attempt_id = registration_sha[:16]
    attempt = DIRECTORY / "attempts" / attempt_id
    complete = DIRECTORY / "COMPLETE.json"
    if complete.exists():
        return _admit_complete(common.read_json(complete), record, expected)
    dataset = None
    models: dict[str, Any] = {}
    setup(7)
    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.runtime_context(),
        training.gpu_charge("V19_COST_V5", reserve=GPU_RESERVE_SECONDS) as deadline,
        torch.inference_mode(),
    ):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            cuda_reference = _cuda_reference(expected, record)
            reference_path = attempt / "CUDA_REFERENCE.json"
            common.write_json(
                reference_path,
                {
                    "registration_sha256": registration_sha,
                    "backend": "CUDA",
                    "meaning": "untimed derived phase from exact saved TTC",
                    "values": {
                        arm: {
                            str(index): {
                                "input_ttc_seconds": expected[arm][index]["ttc_seconds"],
                                "emitted_phase_cuda": cuda_reference[arm][index],
                            }
                            for index in record["indices"]
                        }
                        for arm in ARMS
                    },
                    "optimizer_updates": 0,
                    "backward": False,
                },
            )
            cuda_reference = _validate_cuda_reference(
                common.read_json(reference_path), expected, record, registration_sha
            )
            load: dict[str, tuple[float, float, int]] = {}
            for arm in ARMS:
                allocated = torch.cuda.memory_allocated()
                tick = time.perf_counter()
                model = v3._load_eager(arm, record)
                loaded = time.perf_counter()
                models[arm] = model.cuda()
                torch.cuda.synchronize()
                ready = time.perf_counter()
                load[arm] = (
                    loaded - tick,
                    ready - loaded,
                    torch.cuda.memory_allocated() - allocated,
                )
            reader_tick = time.perf_counter()
            dataset = SensorDataset("P")
            reader_ready = time.perf_counter()
            first = record["indices"][0]
            cold = {}
            for arm in ARMS:
                tick = time.perf_counter()
                batch = dataset.batch(v4._batch_indices(first))
                decoded = time.perf_counter()
                output, h2d_s, forward_s, peak = v4._timed_forward(models[arm], batch)
                values = v4._values(output, v4._position(first))
                validate_primary(values, expected[arm][first], cuda_reference[arm][first])
                cpu_load_s, cuda_load_s, resident = load[arm]
                cold[arm] = {
                    "checkpoint_load_eager_cpu_s": cpu_load_s,
                    "model_to_cuda_s": cuda_load_s,
                    "model_resident_delta_bytes": resident,
                    "reader_init_s": reader_ready - reader_tick,
                    "cached_P2_decode_s": decoded - tick,
                    "host_to_device_s": h2d_s,
                    "first_forward_to_TTC_s": forward_s,
                    "cached_P2_to_TTC_s": decoded - tick + h2d_s + forward_s,
                    "first_route_peak_extra_allocated_bytes": peak,
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
            _validate_cold(
                common.read_json(cold_path),
                record,
                expected,
                cuda_reference,
                registration_sha,
                attempt_id,
            )
            warm4 = dataset.batch(v4._batch_indices(first))
            warm1 = dataset.batch([first])
            for arm in ARMS:
                for _ in range(WARMUPS):
                    out4 = v4._forward(models[arm], warm4)
                    out1 = v4._forward(models[arm], warm1)
                    del out4, out1
            rows = []
            for ordinal, (index, order) in enumerate(
                zip(record["indices"], record["paired_orders"], strict=True)
            ):
                if time.perf_counter() >= deadline:
                    raise RuntimeError("GPU budget ended before V5 rows completed")
                routes = {"batch4_native": {}, "batch1_descriptive": {}}
                for route, indices in (
                    ("batch4_native", v4._batch_indices(index)),
                    ("batch1_descriptive", [index]),
                ):
                    for arm in order:
                        tick = time.perf_counter()
                        batch = dataset.batch(indices)
                        decoded = time.perf_counter()
                        output, h2d_s, forward_s, peak = v4._timed_forward(models[arm], batch)
                        position = v4._position(index) if route == "batch4_native" else 0
                        values = v4._values(output, position)
                        entry = {
                            "cached_P2_decode_s": decoded - tick,
                            "host_to_device_s": h2d_s,
                            "forward_to_TTC_s": forward_s,
                            "cached_P2_to_TTC_s": decoded - tick + h2d_s + forward_s,
                            "peak_extra_allocated_bytes": peak,
                            **values,
                        }
                        if route == "batch4_native":
                            validate_primary(
                                values, expected[arm][index], cuda_reference[arm][index]
                            )
                        else:
                            evidence = v4.drift(values, expected[arm][index])
                            if not evidence["admitted"]:
                                raise ValueError("Batch1 exceeds unchanged prospective drift guard")
                            entry["drift"] = evidence
                            entry["emitted_phase_difference_from_cuda_reference"] = abs(
                                values["emitted_phase"] - cuda_reference[arm][index]
                            )
                        routes[route][arm] = entry
                        del output
                row = {
                    "ordinal": ordinal,
                    "P_index": index,
                    "order": order,
                    "batch4_indices": v4._batch_indices(index),
                    "routes": routes,
                    "registration": registration_sha,
                    "attempt_id": attempt_id,
                    "utc": datetime.now(UTC).isoformat(),
                }
                v4.validate_row(row, record, ordinal)
                _validate_row_outputs(row, expected, cuda_reference)
                common.write_json(attempt / f"row_{ordinal:03d}.json", row)
                rows.append(row)
            checkpoint_after = {
                arm: common.sha(
                    training.destination(f"training/fits/FULL-{arm}-S7/checkpoint_last.pt")
                )
                for arm in ARMS
            }
            if checkpoint_after != {
                arm: record["endpoints"][arm]["checkpoint_sha256"] for arm in ARMS
            }:
                raise ValueError("Frozen checkpoint changed during V5")
            result = {
                "status": "COMPLETE",
                "registration_sha256": registration_sha,
                "attempt_id": attempt_id,
                "cuda_reference_sha256": common.sha(reference_path),
                "cold_sha256": common.sha(cold_path),
                "row_sha256": {
                    f"row_{i:03d}.json": common.sha(attempt / f"row_{i:03d}.json")
                    for i in range(QUERY_COUNT)
                },
                "results": v4.summarize(rows),
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
                    "cuda_reference_sha256": common.sha(reference_path),
                    "cold_sha256": common.sha(cold_path),
                    "optimizer_updates": 0,
                    "backward": False,
                },
            )
            return _admit_complete(common.read_json(complete), record, expected)
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
