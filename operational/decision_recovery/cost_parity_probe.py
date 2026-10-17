"""Directed zero-update batch1/batch4 parity probe for V19 cost row 74."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import common, training

if TYPE_CHECKING:
    import torch

    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

ARMS = ("L0", "LW")
INDEX = 210
NEIGHBORS = (208, 209, 210, 211)
POSITION = 2
REPEATS = 5
ROOT = training.destination("cost/parity_probe_index210_v1")
V3 = training.destination("cost/batch1_cached_P2_to_TTC_v3_eager_cuda")


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_cost_parity_probe.py",
        Path(training.__file__),
        Path(common.__file__),
        common.ROOT / "operational/decision_recovery/cost_measurement.py",
        common.ROOT / "operational/joint_measurement/data.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def _endpoint(arm: str) -> dict[str, str]:
    directory = training.destination(f"training/fits/FULL-{arm}-S7")
    receipt = common.read_json(directory / "CHECKPOINT_RECEIPT.json")
    checkpoint = directory / "checkpoint_last.pt"
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 12000
        or common.sha(checkpoint) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Complete unchanged V19 endpoint required")
    return {
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "receipt_sha256": common.sha(directory / "CHECKPOINT_RECEIPT.json"),
    }


def payload(source_commit: str) -> dict[str, Any]:
    failure = common.read_json(V3 / "FAILURE.json")
    registration = common.read_json(V3 / "REGISTRATION.json")
    if (
        failure.get("status") != "PARITY_FAILURE"
        or failure.get("next_ordinal") != 74
        or failure.get("P_index") != INDEX
        or registration["indices"][74] != INDEX
    ):
        raise ValueError("Directed V3 failure identity changed")
    return {
        "schema": "v19_cost_batch_parity_probe_v1",
        "source_commit": source_commit,
        "sources": _sources(),
        "v3_registration_sha256": common.sha(V3 / "REGISTRATION.json"),
        "v3_failure_sha256": common.sha(V3 / "FAILURE.json"),
        "endpoints": {arm: _endpoint(arm) for arm in ARMS},
        "P_prediction_sha256": {
            arm: common.sha(
                training.destination(f"evaluation/predictions/P/{arm}/predictions.parquet")
            )
            for arm in ARMS
        },
        "index": INDEX,
        "neighbors": list(NEIGHBORS),
        "position": POSITION,
        "repeats": REPEATS,
        "device": "CUDA_FP32_no_TF32",
        "comparisons": ["batch1_vs_saved", "batch4_vs_saved", "batch1_vs_batch4"],
        "outputs": ["raw_phase", "ttc_seconds", "emitted_phase"],
        "diagnostics": ["absolute", "relative", "sign", "saturation"],
        "optimizer_updates": 0,
        "backward": False,
        "scope": "directed diagnostic only; no tolerance or protocol change",
    }


def register() -> dict[str, Any]:
    path = ROOT / "REGISTRATION.json"
    if path.exists():
        return registered()
    sources = _sources()
    value = payload(training.committed_sources(sources))
    validate_registration_identity(value)
    common.write_json(path, value)
    return value


def validate_registration_identity(value: dict[str, Any]) -> None:
    """Keep the directed failure query and zero-update scope immutable."""
    if (
        value.get("schema") != "v19_cost_batch_parity_probe_v1"
        or value.get("index") != INDEX
        or value.get("neighbors") != list(NEIGHBORS)
        or value.get("position") != POSITION
        or value.get("repeats") != REPEATS
        or value.get("optimizer_updates") != 0
        or value.get("backward") is not False
        or set(value.get("endpoints", {})) != set(ARMS)
    ):
        raise ValueError("Directed parity registration identity changed")


def registered() -> dict[str, Any]:
    value = common.read_json(ROOT / "REGISTRATION.json")
    validate_registration_identity(value)
    if value != payload(value["source_commit"]):
        raise ValueError("Parity probe registration or sources changed")
    return value


def compare(left: float, right: float, output: str = "ttc_seconds") -> dict[str, Any]:
    """Describe numerical drift without applying an acceptance tolerance."""
    if output not in {"raw_phase", "ttc_seconds", "emitted_phase"}:
        raise ValueError("Unknown parity output")
    absolute = abs(left - right)
    if output == "ttc_seconds":
        left_saturated, right_saturated = abs(left) >= 59.999, abs(right) >= 59.999
        saturation_rule = "abs(TTC)>=59.999"
    else:
        boundary = 1.0 / 60.0
        left_saturated, right_saturated = abs(left) <= boundary, abs(right) <= boundary
        saturation_rule = "abs(phase)<=1/60"
    return {
        "output": output,
        "left": left,
        "right": right,
        "absolute_difference": absolute,
        "relative_to_right": absolute / max(abs(right), 1e-12),
        "same_sign": bool(np.signbit(left) == np.signbit(right)),
        "left_saturated": bool(left_saturated),
        "right_saturated": bool(right_saturated),
        "saturation_rule": saturation_rule,
    }


def _saved() -> dict[str, dict[str, float]]:
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.phase import ttc_to_phase

    result = {}
    for arm in ARMS:
        frame = pd.read_parquet(
            training.destination(f"evaluation/predictions/P/{arm}/predictions.parquet"),
            columns=["prediction_ttc", "raw_phase"],
        )
        row = frame.iloc[INDEX]
        ttc = float(row.prediction_ttc)
        result[arm] = {
            "raw_phase": float(row.raw_phase),
            "ttc_seconds": ttc,
            "emitted_phase": float(ttc_to_phase(torch.tensor([ttc], dtype=torch.float32)).item()),
        }
    return result


def _load(arm: str, record: dict[str, Any]) -> torch.nn.Module:
    import torch

    path = training.destination(f"training/fits/FULL-{arm}-S7/checkpoint_last.pt")
    if common.sha(path) != record["endpoints"][arm]["checkpoint_sha256"]:
        raise ValueError("Endpoint changed after parity registration")
    payload_value = torch.load(path, map_location="cpu", weights_only=False)
    model = training.make_model()
    model.load_state_dict(payload_value["model_state_dict"], strict=True)
    return model.cuda().eval().requires_grad_(False)


def _forward(model: torch.nn.Module, batch: ObjectEventV4Batch) -> dict[str, list[float]]:
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase

    events = batch.events.cuda()
    delta = batch.delta_t_s.reshape(-1).cuda()
    raw = model(events, delta)["raw_phase"]
    ttc = phase_to_ttc(raw)
    emitted = ttc_to_phase(ttc)
    torch.cuda.synchronize()
    return {
        "raw_phase": raw.detach().cpu().double().tolist(),
        "ttc_seconds": ttc.detach().cpu().double().tolist(),
        "emitted_phase": emitted.detach().cpu().double().tolist(),
    }


def run() -> dict[str, Any]:
    """Run five repeated batch1 and batch4 forwards for both frozen endpoints."""
    import torch

    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.train import setup

    record = registered()
    result_path = ROOT / "RESULT.json"
    complete_path = ROOT / "COMPLETE.json"
    if result_path.exists():
        result = common.read_json(result_path)
        complete = common.read_json(complete_path)
        expected_complete = {
            "status": "COMPLETE",
            "registration_sha256": common.sha(ROOT / "REGISTRATION.json"),
            "result_sha256": common.sha(result_path),
            "optimizer_updates": 0,
            "backward": False,
        }
        if (
            complete != expected_complete
            or result.get("registration_sha256") != complete["registration_sha256"]
            or result.get("optimizer_updates") != 0
            or result.get("backward") is not False
            or result.get("protocol_changed") is not False
        ):
            raise ValueError("Parity result or completion receipt changed")
        return result
    setup(7)
    saved = _saved()
    dataset = None
    models = {}
    try:
        with (
            training.file_lock(training.destination("training/DRIVER.lock")),
            training.runtime_context(),
            training.gpu_charge("V19_COST_PARITY_INDEX210", reserve=180),
            torch.inference_mode(),
        ):
            dataset = SensorDataset("P")
            one, four = dataset.batch([INDEX]), dataset.batch(list(NEIGHBORS))
            observations = {}
            for arm in ARMS:
                model = _load(arm, record)
                models[arm] = model
                batch1 = [_forward(model, one) for _ in range(REPEATS)]
                batch4_full = [_forward(model, four) for _ in range(REPEATS)]
                batch4 = [
                    {name: values[name][POSITION] for name in record["outputs"]}
                    for values in batch4_full
                ]
                batch1_at_index = [
                    {name: values[name][0] for name in record["outputs"]} for values in batch1
                ]
                comparisons = {}
                for name in record["outputs"]:
                    comparisons[name] = {
                        "batch1_vs_saved": [
                            compare(values[name], saved[arm][name], name)
                            for values in batch1_at_index
                        ],
                        "batch4_vs_saved": [
                            compare(values[name], saved[arm][name], name) for values in batch4
                        ],
                        "batch1_vs_batch4": [
                            compare(left[name], right[name], name)
                            for left, right in zip(batch1_at_index, batch4, strict=True)
                        ],
                    }
                observations[arm] = {
                    "saved": saved[arm],
                    "batch1_repeats": batch1_at_index,
                    "batch4_repeats_at_position2": batch4,
                    "batch4_full_first_repeat": batch4_full[0],
                    "comparisons": comparisons,
                    "checkpoint_sha256_after": common.sha(
                        training.destination(f"training/fits/FULL-{arm}-S7/checkpoint_last.pt")
                    ),
                }
                if (
                    observations[arm]["checkpoint_sha256_after"]
                    != record["endpoints"][arm]["checkpoint_sha256"]
                ):
                    raise ValueError("Endpoint checkpoint changed during parity probe")
        result = {
            "status": "COMPLETE_DIAGNOSTIC",
            "registration_sha256": common.sha(ROOT / "REGISTRATION.json"),
            "observations": observations,
            "optimizer_updates": 0,
            "backward": False,
            "protocol_changed": False,
            "interpretation": "diagnose batch-size numerical drift before any cost redesign",
        }
        common.write_json(result_path, result)
        common.write_json(
            complete_path,
            {
                "status": "COMPLETE",
                "registration_sha256": result["registration_sha256"],
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
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    args = parser.parse_args()
    print({"register": register, "run": run}[args.action]())
