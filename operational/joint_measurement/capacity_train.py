"""Prospective family-D predictor-width contrast using the existing durable engine."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from . import geometry_train as engine
from .capacity_model import WidePairMeasurement, captured_backward
from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .data import cache_root, role_rows
from .evaluate import freeze_primary
from .fast_runtime import PersistentSensors
from .geometry_capture import contiguous_subset
from .geometry_capture_qa import STATE_KEYS
from .geometry_graph_probe import no_live_training_driver
from .qa import exact
from .resume_fast_proof import close
from .rgb_run import require_authorization
from .train import backward, committed_sources, sources

DIRECTORY = RUN / "adaptive/D1_predictor_width"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
FIT = "PAIR-CORR-WIDE-S7"
CONTROL = "PAIR-CORR-S7"
QA = RUN / "qa/capacity_D1"


def source_binding() -> dict:
    """Bind both the new predictor and the reused numerical/checkpoint implementation."""
    names = (
        "capacity_model",
        "capacity_train",
        "geometry_train",
        "geometry_capture",
        "geometry_capture_qa",
        "fast_runtime",
        "resume_fast_proof",
    )
    return {
        **sources(),
        **{
            f"operational/joint_measurement/{name}.py": sha(
                ROOT / f"operational/joint_measurement/{name}.py"
            )
            for name in names
        },
    }


def register() -> dict:
    """Choose width using P/H diagnostics only, before any D1 optimizer update."""
    require_authorization()
    committed_sources(source_binding())
    evidence_path = RUN / "diagnostics/conditional_families/P_H_FIT.json"
    evidence = read(evidence_path)
    p = evidence["models"][CONTROL]["populations"]["P"]
    if evidence["V_used"] or p["sign"] < 0.10:
        raise ValueError("D1 requires the documented substantial P sign error, without V selection")
    temporal = RUN / "diagnostics/real_temporal_P/RESULT.json"
    if not read(temporal)["grouped"][CONTROL]["geometry_followup_signal"]:
        raise ValueError(
            "Correspondence control must be verified before increasing predictor capacity"
        )
    control = RUN / "fits" / CONTROL
    identity = read(control / "IDENTITY.json")
    endpoints = freeze_primary()["endpoints"]
    endpoint_proof_path = RUN / "verification/PAIR_CORR_ENDPOINT.json"
    endpoint_proof = read(endpoint_proof_path)
    gradient_path = control / "ENCODER_GRADIENT.json"
    gradients = read(gradient_path)
    if not (
        endpoint_proof["status"] == "PASS"
        and endpoint_proof["checkpoint_sha256"] == endpoints[CONTROL]
        and endpoint_proof["encoder_tensors_changed"] > 0
        and all(row["finite"] for row in gradients.values())
        and any(row["gradient_l2"] > 0 for row in gradients.values())
    ):
        raise ValueError("D1 requires verified control gradients, weights and complete cursor")
    if (
        identity["sources"] != sources()
        or identity["config_hash"] != digest(config())
        or identity["seed"] != 7
        or identity["updates"] != 12000
    ):
        raise ValueError("Capacity control recipe changed")
    proof_path = DIRECTORY / "prototype/CPU.json"
    proof = read(proof_path)
    if proof["status"] != "PASS" or proof["model_source_sha256"] != sha(
        ROOT / "operational/joint_measurement/capacity_model.py"
    ):
        raise ValueError("A matching zero-update CPU arithmetic/gradient proof is required")
    rows = role_rows("P")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), role="P")
    if digest(masses.tolist()) != identity["mass_sha256"]:
        raise ValueError("P masses differ from the paired control")
    record = {
        "schema": "v18_adaptive_D1_predictor_width_v1",
        "experiment_id": "V18_ADAPTIVE_D1",
        "family": "D",
        "fits": [FIT],
        "seed": 7,
        "updates_per_fit": 12000,
        "maximum_new_scientific_updates": 12000,
        "maximum_synthetic_qa_updates": 30,
        "paired_control": {
            "fit_id": CONTROL,
            "checkpoint_sha256": endpoints[CONTROL],
            "identity_sha256": sha(control / "IDENTITY.json"),
            "updates": 12000,
            "reused_without_new_updates": True,
            "cost": "already charged; never count twice",
        },
        "hypothesis": (
            "Increasing spatial predictor capacity may reduce limited P fit without changing "
            "inputs, objective or native encoder initialization"
        ),
        "evidence": {
            "P_H_diagnostic_sha256": sha(evidence_path),
            "P_sign_error": p["sign"],
            "P_aggregate": p["aggregate"],
            "correspondence_sha256": sha(temporal),
            "control_gradient_proof_sha256": sha(gradient_path),
            "control_endpoint_proof_sha256": sha(endpoint_proof_path),
            "limit": (
                "Substantial P error enables a capacity test; it does not prove capacity is "
                "the cause. Frozen CORR has better P objective, so objective/optimization "
                "tradeoffs remain alternatives."
            ),
        },
        "only_scientific_factor": (
            "Spatial tower width96 to192 and readout hidden128 to256; native64 encoder "
            "remains fully trainable and identically initialized"
        ),
        "initialization": (
            "Same seed7 and constructor law; different tensor shapes imply different random "
            "tensors. Width-one constructor is bit-exact with the original. No widening "
            "of pretrained CNN or loss change."
        ),
        "capacity": proof["parameters"],
        "prototype_sha256": sha(proof_path),
        "parent_sha256": identity["parent"],
        "population": "P",
        "population_size": len(rows),
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_binding": identity["sensor_binding"],
        "mass_sha256": identity["mass_sha256"],
        "batch_size": 32,
        "microbatch_size": 4,
        "optimizer": config()["optimizer"],
        "objective": (
            "Unchanged emitted phase L1 plus0.1 raw phase L1, native signed decoder "
            "and global P masses"
        ),
        "runtime": (
            "FP32_no_TF32; full forward/backward capture after exact/numerical QA; "
            "owned gradients; pinned slices; lossless8GiB LRU; one prefetch"
        ),
        "mechanism_rule": (
            "At fixed12k, P aggregate improves>=3% versus free CORR with sign not worse; "
            "also report objective/bands. This is not H admission or promotion."
        ),
        "development_gates": config()["selection"],
        "evaluation": (
            "Freeze fixed12k before complete P/H. Same A5-P2 reference and original "
            "band/sign protections. Open V only after prospective H admission."
        ),
        "replication": (
            "Existing single selected family remains C1 fusion; D1 does not authorize "
            "another replication family"
        ),
        "official_claim": False,
        "sources": source_binding(),
    }
    registry = RUN / "EXPERIMENT_REGISTRY.jsonl"
    existing = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    others = {
        r["experiment_id"]
        for r in existing
        if r["experiment_id"].startswith("V18_ADAPTIVE_")
        and r["experiment_id"] != record["experiment_id"]
    }
    if len(others) >= 4:
        raise ValueError("Adaptive experiment count exhausted")
    freeze(REGISTRATION, record)
    prior = [r for r in existing if r["experiment_id"] == record["experiment_id"]]
    if prior:
        if any(r["registration_sha256"] != sha(REGISTRATION) for r in prior):
            raise ValueError("D1 registry changed")
    else:
        with registry.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "experiment_id": record["experiment_id"],
                        "registered_utc": utc(),
                        "status": "REGISTERED_BEFORE_UPDATES",
                        "registration_sha256": sha(REGISTRATION),
                        "maximum_scientific_updates": 12000,
                        "maximum_synthetic_qa_updates": 30,
                        "fits": [FIT],
                        "reused_control": CONTROL,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return record


def registered() -> dict:
    """Reject modifications to the registered numerical implementation."""
    record = read(REGISTRATION)
    if record["sources"] != source_binding():
        raise ValueError("D1 registered implementation changed")
    return record


@contextmanager
def engine_context(*, capture: bool) -> Iterator[None]:
    """Reuse the existing checkpoint engine in this process, under a separate D1 namespace."""
    with ExitStack() as stack:
        for name, value in {
            "FIT": FIT,
            "DIRECTORY": DIRECTORY,
            "REGISTRATION": REGISTRATION,
            "registered": registered,
            "source_binding": source_binding,
            "GeometricPairMeasurement": WidePairMeasurement,
            "backward": captured_backward if capture else backward,
            "subset": contiguous_subset,
        }.items():
            stack.enter_context(patch.object(engine, name, value))
        yield


def qa() -> dict:
    """Thirty counted CUDA updates: eager10, capture10 and capture5+restore5."""
    registered()
    if (QA / "RESULT.json").exists():
        result = read(QA / "RESULT.json")
        if result["registration_sha256"] != sha(REGISTRATION) or result["status"] != "PASS":
            raise ValueError("D1 QA failed or changed; retain evidence")
        return result
    if accounting()["technical"] + 30 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient D1 QA reserve")
    with gpu_charge("D1_width_CUDA_resume_QA", reserve=900) as deadline:
        for name, capture, limit in (
            ("eager", False, 10),
            ("continuous", True, 10),
            ("split", True, 5),
            ("split", True, 10),
        ):
            with engine_context(capture=capture):
                outcome = engine.segment(
                    QA / name,
                    engine.SyntheticSensors(),
                    np.full(8, 1 / 8),
                    limit=limit,
                    deadline=deadline,
                    device="cuda",
                    synthetic=True,
                )
            if outcome["completed_updates"] != limit:
                raise RuntimeError("D1 QA interrupted")
    eager, continuous, split = [
        torch.load(QA / name / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for name in ("eager", "continuous", "split")
    ]
    checks = {key: exact(continuous[key], split[key]) for key in STATE_KEYS}
    checks["curve"] = exact(continuous["cursor"]["curve"], split["cursor"]["curve"])
    numerical = {
        key: close(eager[key], continuous[key], rtol=1e-5, atol=1e-7) for key in STATE_KEYS
    }
    numerical["curve"] = close(
        eager["cursor"]["curve"], continuous["cursor"]["curve"], rtol=1e-5, atol=1e-7
    )
    result = {
        "status": "PASS" if all(checks.values()) and all(numerical.values()) else "FAIL",
        "exact_resume": checks,
        "eager_numerical": numerical,
        "technical_updates": 30,
        "registration_sha256": sha(REGISTRATION),
    }
    freeze(QA / "RESULT.json", result)
    if result["status"] != "PASS":
        raise ValueError("D1 full-state/numerical QA failed")
    return result


def run() -> None:
    """One D1 writer preserves full state through bounded GPU scopes."""
    require_authorization()
    no_live_training_driver()
    record = registered()
    with engine_context(capture=True), engine.exclusive_driver():
        qa()
        sensors = PersistentSensors()
        try:
            if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
                raise ValueError("D1 P sensor binding changed")
            masses = training_mass(
                sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
            )
            if digest(masses.tolist()) != record["mass_sha256"]:
                raise ValueError("D1 P mass changed")
            fit = DIRECTORY / "fits" / FIT
            provenance = fit / "RUN_PROVENANCE.json"
            if not provenance.exists():
                freeze(
                    provenance,
                    {
                        "experiment_id": record["experiment_id"],
                        "fit_id": FIT,
                        "git_commit": committed_sources(source_binding()),
                        "seed": 7,
                        "config_hash": digest(record),
                        "dataset_manifest_hash": record["P_manifest_sha256"],
                        "host": socket.gethostname(),
                        "python_version": platform.python_version(),
                        "torch_version": str(torch.__version__),
                        "cuda_version": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "start_time": utc(),
                        "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                    },
                )
            while True:
                path = fit / "CHECKPOINT_RECEIPT.json"
                if path.exists() and read(path)["status"] == "COMPLETE":
                    receipt = read(path)
                    if receipt["completed_updates"] != 12000 or receipt["checkpoint_sha256"] != sha(
                        fit / "checkpoint_last.pt"
                    ):
                        raise ValueError("D1 endpoint changed")
                    break
                require_resources(starting=True)
                write(
                    DIRECTORY / "RUNNER_STATE.json",
                    {"status": "TRAINING", "pid": os.getpid(), "utc": utc()},
                )
                with gpu_charge("D1_width_scientific_training", reserve=900) as deadline:
                    engine.segment(
                        fit, sensors, masses, limit=12000, deadline=deadline, device="cuda"
                    )
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "D1_ENDPOINT_COMPLETE",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "full_objective_complete": False,
                    "next": "Complete P/H evidence, cost, export and fixed development gates",
                },
            )
        except Exception as error:
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "ACTION_REQUIRED",
                    "error": str(error),
                    "utc": utc(),
                    "pid": os.getpid(),
                },
            )
            raise
        finally:
            sensors.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register()) if args.register_only else run()
