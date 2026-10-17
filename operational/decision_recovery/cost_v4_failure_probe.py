"""Directed zero-update probe for the first missing V4 timing row."""

from __future__ import annotations

import argparse
import gc
import struct
from pathlib import Path
from typing import Any

from . import common, training
from . import cost_measurement as v3
from . import cost_measurement_v4 as v4

ORDINAL = 42
INDEX = 91
BATCH_INDICES = (88, 89, 90, 91)
POSITION = 3
ARMS = v4.ARMS
DIRECTORY = training.destination("cost/v4_failure_probe_ordinal42_v1")
V4_DIRECTORY = training.destination("cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v4")


def _bits(value: float) -> str:
    return struct.pack("!f", value).hex()


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_cost_v4_failure_probe.py",
        Path(v4.__file__),
        Path(v3.__file__),
        Path(training.__file__),
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        common.ROOT / "operational/joint_measurement/data.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def registration_payload(source_commit: str) -> dict[str, Any]:
    v4_registration = common.read_json(V4_DIRECTORY / "REGISTRATION.json")
    if v4_registration["indices"][ORDINAL] != INDEX or v4._batch_indices(INDEX) != list(
        BATCH_INDICES
    ):
        raise ValueError("Directed V4 failure location changed")
    attempt_id = common.sha(V4_DIRECTORY / "REGISTRATION.json")[:16]
    attempt = V4_DIRECTORY / "attempts" / attempt_id
    partial_rows = sorted(attempt.glob("row_*.json"))
    if [path.name for path in partial_rows] != [f"row_{i:03d}.json" for i in range(ORDINAL)]:
        raise ValueError("Expected exactly 42 preserved V4 rows before directed probe")
    return {
        "schema": "v19_cost_v4_failure_probe_ordinal42_v1",
        "v4_registration_sha256": common.sha(V4_DIRECTORY / "REGISTRATION.json"),
        "v4_cold_sha256": common.sha(attempt / "COLD.json"),
        "v4_partial_row_sha256": {path.name: common.sha(path) for path in partial_rows},
        "ordinal": ORDINAL,
        "P_index": INDEX,
        "batch_indices": list(BATCH_INDICES),
        "position": POSITION,
        "arms": list(ARMS),
        "comparisons": [
            "measured_CUDA_raw_vs_saved_raw",
            "measured_CUDA_TTC_vs_saved_TTC",
            "measured_CUDA_emitted_vs_CPU_emitted_from_saved_TTC",
            "measured_CUDA_emitted_vs_CUDA_emitted_from_saved_TTC",
        ],
        "float_witness": "decimal float32 value, absolute difference, exact equality, IEEE754 bits",
        "gpu_reserve_seconds": 60,
        "optimizer_updates": 0,
        "backward": False,
        "scope": "directed instrumentation only; V4 registration and parity criteria unchanged",
        "endpoints": v4_registration["endpoints"],
        "sources": _sources(),
        "source_commit": source_commit,
    }


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
        raise ValueError("Registered V4 failure probe identity changed")
    return value


def comparison(left: float, right: float) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "absolute_difference": abs(left - right),
        "exact": left == right,
        "left_float32_bits": _bits(left),
        "right_float32_bits": _bits(right),
    }


def run() -> dict[str, Any]:
    """Run one historical batch4 for both frozen endpoints and preserve all values."""
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.train import setup

    record = registered()
    registration_sha = common.sha(DIRECTORY / "REGISTRATION.json")
    result_path = DIRECTORY / "RESULT.json"
    complete_path = DIRECTORY / "COMPLETE.json"
    if complete_path.exists():
        complete = common.read_json(complete_path)
        if (
            complete
            != {
                "status": "COMPLETE",
                "registration_sha256": registration_sha,
                "result_sha256": common.sha(result_path),
                "optimizer_updates": 0,
                "backward": False,
            }
            or common.read_json(result_path).get("optimizer_updates") != 0
            or common.read_json(result_path).get("backward") is not False
        ):
            raise ValueError("Completed V4 failure probe changed")
        return common.read_json(result_path)

    v4_record = common.read_json(V4_DIRECTORY / "REGISTRATION.json")
    expected = v4._expected(v4_record)
    dataset = None
    models: dict[str, Any] = {}
    setup(7)
    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.runtime_context(),
        training.gpu_charge("V19_COST_V4_FAILURE_PROBE_ORDINAL42", reserve=60) as deadline,
        torch.inference_mode(),
    ):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            dataset = SensorDataset("P")
            batch = dataset.batch(list(BATCH_INDICES))
            observations = {}
            for arm in ARMS:
                if __import__("time").perf_counter() >= deadline:
                    raise RuntimeError("Probe GPU budget expired")
                models[arm] = v3._load_eager(arm, v4_record).cuda()
                events = batch.events.cuda()
                delta = batch.delta_t_s.reshape(-1).cuda()
                raw_tensor = models[arm](events, delta)["raw_phase"]
                ttc_tensor = phase_to_ttc(raw_tensor)
                emitted_tensor = ttc_to_phase(ttc_tensor)
                saved = expected[arm][INDEX]
                saved_ttc_cuda = torch.tensor(
                    [saved["ttc_seconds"]], dtype=torch.float32, device="cuda"
                )
                saved_emitted_cuda = ttc_to_phase(saved_ttc_cuda)
                torch.cuda.synchronize()
                measured = {
                    "raw_phase": float(raw_tensor[POSITION].item()),
                    "ttc_seconds": float(ttc_tensor[POSITION].item()),
                    "emitted_phase_cuda": float(emitted_tensor[POSITION].item()),
                }
                observations[arm] = {
                    "saved": {
                        **saved,
                        "emitted_phase_cuda_from_saved_ttc": float(saved_emitted_cuda.item()),
                    },
                    "measured": measured,
                    "comparisons": {
                        "raw_vs_saved": comparison(measured["raw_phase"], saved["raw_phase"]),
                        "ttc_vs_saved": comparison(measured["ttc_seconds"], saved["ttc_seconds"]),
                        "emitted_cuda_vs_cpu_from_saved_ttc": comparison(
                            measured["emitted_phase_cuda"], saved["emitted_phase"]
                        ),
                        "emitted_cuda_vs_cuda_from_saved_ttc": comparison(
                            measured["emitted_phase_cuda"], float(saved_emitted_cuda.item())
                        ),
                    },
                }
                del events, delta, raw_tensor, ttc_tensor, emitted_tensor, saved_ttc_cuda
            checkpoint_after = {
                arm: common.sha(
                    training.destination(f"training/fits/FULL-{arm}-S7/checkpoint_last.pt")
                )
                for arm in ARMS
            }
            if checkpoint_after != {
                arm: record["endpoints"][arm]["checkpoint_sha256"] for arm in ARMS
            }:
                raise ValueError("Endpoint checkpoint changed during directed probe")
            result = {
                "status": "COMPLETE_DIAGNOSTIC",
                "registration_sha256": registration_sha,
                "ordinal": ORDINAL,
                "P_index": INDEX,
                "batch_indices": list(BATCH_INDICES),
                "position": POSITION,
                "observations": observations,
                "checkpoint_sha256_after": checkpoint_after,
                "optimizer_updates": 0,
                "backward": False,
                "protocol_changed": False,
            }
            common.write_json(result_path, result)
            common.write_json(
                complete_path,
                {
                    "status": "COMPLETE",
                    "registration_sha256": registration_sha,
                    "result_sha256": common.sha(result_path),
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
    parser.add_argument("action", choices=("register", "run"))
    args = parser.parse_args()
    print({"register": register, "run": run}[args.action]())
