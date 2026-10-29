"""Fourth adaptive comparison: last residual block versus full encoder adaptation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.joint_measurement.loss import training_mass
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from . import correspondence_train as engine
from . import geometry_train
from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .compute_guard import guard_idle
from .data import cache_root, role_rows
from .fast_runtime import PersistentSensors
from .geometry_capture import contiguous_subset
from .geometry_capture_qa import STATE_KEYS
from .geometry_model import GeometricPairMeasurement
from .qa import exact, fixture
from .resume_fast_proof import close
from .rgb_run import require_authorization
from .train import committed_sources, initialize

DIRECTORY = RUN / "adaptive/A1_adaptation_depth"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
PARENT = RUN / "adaptive/B1_correspondence/fits/B1-CORRESPONDENCE-STAGE1-S7"
QA = RUN / "qa/adaptation_depth_A1"
MODES = ("LAST-BLOCK", "FULL")
UPDATES = 12000


def configure(model: nn.Module, mode: str) -> dict:
    """Change only gradient eligibility; keep identical stochastic forward policy."""
    if mode not in MODES or not isinstance(model.encoder, nn.Sequential):
        raise ValueError("Known adaptation mode and native sequential encoder required")
    children = list(model.encoder.children())
    if len(children) < 7 or not hasattr(children[-1], "network"):
        raise ValueError("Expected native stem and a final residual block")
    prefix = f"encoder.{len(children) - 1}."
    model.requires_grad_(True)
    if mode == "LAST-BLOCK":
        for name, parameter in model.named_parameters():
            if name.startswith("encoder.") and not name.startswith(prefix):
                parameter.requires_grad_(False)
    model.train()
    # Both arms retain the same dropout draws. GroupNorm has no running buffers.
    # This is partial adaptation, not an entirely frozen encoder operating in eval.
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.encoder.modules()
    ):
        raise ValueError("Running-stat normalization is outside this fixed recipe")
    return {
        "last_block_prefix": prefix,
        "trainable": [name for name, p in model.named_parameters() if p.requires_grad],
        "frozen": [name for name, p in model.named_parameters() if not p.requires_grad],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def folder(mode: str) -> Path:
    if mode not in MODES:
        raise ValueError("Unknown A1 arm")
    return DIRECTORY / "fits" / f"A1-{mode}-S7"


def source_binding() -> dict:
    return {
        **engine.source_binding(),
        **{
            f"operational/joint_measurement/{name}.py": sha(
                ROOT / f"operational/joint_measurement/{name}.py"
            )
            for name in ("adaptation_depth", "compute_guard", "correspondence_model")
        },
    }


def make_model(mode: str, stage: int = 2, *, synthetic: bool = False) -> tuple:
    """Both arms use the same complete P-only correspondence endpoint, never H weights."""
    if stage != 2:
        raise ValueError("A1 has one TTC stage only")
    receipt = read(PARENT / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 2000:
        raise ValueError("Complete B1 correspondence stage1 required")
    if sha(PARENT / "checkpoint_last.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("A1 shared parent changed")
    if (
        REGISTRATION.exists()
        and read(REGISTRATION)["parent_sha256"] != receipt["checkpoint_sha256"]
    ):
        raise ValueError("A1 parent differs from its prospective registration")
    native = initialize("A5-P2-FROZEN")
    if not isinstance(native, CausalScaleTTC):
        raise TypeError("Native private encoder required")
    model = GeometricPairMeasurement(native)
    payload = torch.load(PARENT / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    configure(model, mode)
    return model, receipt["checkpoint_sha256"]


def prototype() -> dict:
    """Prove identical native forwards and the specified gradient boundary without updates."""
    guard_idle()
    torch.set_num_threads(1)
    committed_sources(source_binding())
    batch = contiguous_subset(fixture(128), list(range(4)))
    checks, policies, initial_states, outputs = {}, {}, [], []
    parent_sha = sha(PARENT / "checkpoint_last.pt")
    for mode in MODES:
        torch.manual_seed(7)
        model, _ = make_model(mode)
        policies[mode] = configure(model, mode)
        before = copy.deepcopy(model.state_dict())
        initial_states.append(before)
        torch.manual_seed(17)
        outputs.append(model(batch.events, batch.delta_t_s)["raw_phase"].detach().clone())
        torch.manual_seed(17)
        engine.step_backward(
            model, [batch], torch.full((4,), 1 / 8), 8, correspondence=False, capture=False
        )
        checks[f"{mode}_finite_nonzero_encoder_gradient"] = all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.encoder.parameters()
            if p.requires_grad
        ) and any(bool(p.grad.norm() > 0) for p in model.encoder.parameters() if p.grad is not None)
        checks[f"{mode}_frozen_gradients_absent"] = all(
            p.grad is None for p in model.parameters() if not p.requires_grad
        )
        checks[f"{mode}_weights_unchanged"] = exact(before, model.state_dict())
    checks["identical_initial_weights"] = exact(*initial_states)
    checks["identical_forward_with_same_dropout_rng"] = exact(*outputs)
    checks["shared_parent_unchanged"] = sha(PARENT / "checkpoint_last.pt") == parent_sha
    checks["cuda_uninitialized"] = not torch.cuda.is_initialized()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "policies": policies,
        "sources": source_binding(),
        "parent_sha256": parent_sha,
        "optimizer_updates": 0,
    }
    freeze(DIRECTORY / "prototype/CPU.json", result)
    return result


def register() -> dict:
    """Freeze the fourth paired recipe before any synthetic or scientific optimizer update."""
    require_authorization()
    committed_sources(source_binding())
    proof = read(DIRECTORY / "prototype/CPU.json")
    if proof["status"] != "PASS" or proof["sources"] != source_binding():
        raise ValueError("A1 needs matching native zero-update proof")
    b1_path = RUN / "adaptive/B1_correspondence/evidence/COMPLETE.json"
    b1 = read(b1_path)
    if (
        b1["status"] != "B1_READY_FOR_INTERPRETATION"
        or not b1["selection"]["paired_mechanism"]["eligible"]
    ):
        raise ValueError("A1 requires complete positive paired B1 mechanism evidence")
    rows = role_rows("P")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), role="P")
    record = {
        "schema": "v18_adaptive_A1_v1",
        "experiment_id": "V18_ADAPTIVE_A1",
        "family": "A",
        "modes": list(MODES),
        "seed": 7,
        "updates_per_arm": UPDATES,
        "maximum_new_scientific_updates": 24000,
        "maximum_synthetic_qa_updates": 60,
        "hypothesis": (
            "Limit post-correspondence encoder adaptation to preserve learned matching "
            "while training the same TTC readout"
        ),
        "basis": (
            "B1 paired H mechanism signal and loss of part of known-scale matching after "
            "full TTC adaptation; not proof that frozen primary encoder cannot work"
        ),
        "B1_evidence_sha256": sha(b1_path),
        "prototype_sha256": sha(DIRECTORY / "prototype/CPU.json"),
        "parent_sha256": proof["parent_sha256"],
        "parent": (
            "Same P-only B1 correspondence stage1: historical C1 12000 + correspondence "
            "2000; reused, not recounted"
        ),
        "factor": (
            "Eligible encoder parameters: final residual block versus all blocks; "
            "all readout parameters trainable in both"
        ),
        "parameter_policy": proof["policies"],
        "training_mode": (
            "All modules train in both arms, including identical early-block dropout; "
            "frozen prefix has GroupNorm, no mutable running statistics. "
            "Encoder is partially, not wholly, frozen."
        ),
        "model": (
            "Unchanged GeometricPairMeasurement, native P2 input, same weights and temperature"
        ),
        "objective": (
            "Original emitted phase loss plus0.1 raw-phase term; "
            "original global P masses; no sign term"
        ),
        "optimizer": config()["optimizer"],
        "initial_optimizer": (
            "Fresh Adam/scheduler/sampler/RNG seed7 in both arms, 12000-step schedule; "
            "full-state restore within arm"
        ),
        "batch_size": 32,
        "microbatch_size": 4,
        "precision": "FP32_no_TF32",
        "sampling": (
            "Uniform P with replacement, identical torch.Generator seed7; keep all sampled IDs"
        ),
        "population_size": len(rows),
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_binding": sha(cache_root("P") / "COMPLETE.json"),
        "mass_sha256": digest(masses.tolist()),
        "freeze_rule": "Both12000 endpoints before any A1 H inference; no early stopping",
        "evaluation": (
            "Complete P/H, all bands/sign/MAE, paired2000 acquisition-group bootstrap, "
            "known-scale diagnostic, prepared128 P cost and portable examples"
        ),
        "mechanism_rule": (
            "LAST-BLOCK vs FULL: aggregate<=0.97 or crucial<=0.95 with aggregate<=1.02; "
            "every band<=1.10, sign<=control+0.005, identical full coverage"
        ),
        "development_gates": config()["selection"],
        "baseline": (
            "A5-P2 for development admission; FULL is the sole prospective paired control. "
            "Prior B1 10000-step endpoint is descriptive only"
        ),
        "V_rule": "No V rescue; any V evaluation requires original H admission and fixed endpoint",
        "replication": "No new replication family; C1/fusion already used",
        "runtime": (
            "Existing CUDA captured TTC backward and lossless8GiB sensor LRU/prefetch; "
            "fresh process per900s segment"
        ),
        "QA": "Each arm eager10, captured10, captured5+resume5:60 physical technical updates",
        "sources": source_binding(),
    }
    registry = RUN / "EXPERIMENT_REGISTRY.jsonl"
    entries = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    others = {
        r["experiment_id"]
        for r in entries
        if r["experiment_id"].startswith("V18_ADAPTIVE_")
        and r["experiment_id"] != record["experiment_id"]
    }
    if len(others) >= 4:
        raise ValueError("Four adaptive comparisons already registered")
    freeze(REGISTRATION, record)
    prior = [r for r in entries if r["experiment_id"] == record["experiment_id"]]
    if prior and any(r["registration_sha256"] != sha(REGISTRATION) for r in prior):
        raise ValueError("A1 registration changed")
    if not prior:
        with registry.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "experiment_id": record["experiment_id"],
                        "registered_utc": utc(),
                        "status": "REGISTERED_BEFORE_UPDATES",
                        "registration_sha256": sha(REGISTRATION),
                        "maximum_scientific_updates": 24000,
                        "maximum_synthetic_qa_updates": 60,
                        "fits": [folder(mode).name for mode in MODES],
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return record


def registered() -> dict:
    record = read(REGISTRATION)
    if record["sources"] != source_binding():
        raise ValueError("A1 registered training sources changed")
    return record


@contextmanager
def engine_context() -> Iterator[None]:
    """Reuse the exact admitted TTC transaction engine, with separate A1 identities."""
    with ExitStack() as stack:
        for key, value in {
            "REGISTRATION": REGISTRATION,
            "STEPS": {2: UPDATES},
            "registered": registered,
            "make_model": make_model,
        }.items():
            stack.enter_context(patch.object(engine, key, value))
        yield


def verify_change(target: Path, mode: str) -> None:
    record = registered()
    changes = read(target / "WEIGHT_CHANGE.json")
    if not changes["updates"]:
        return
    policy = record["parameter_policy"][mode]
    if any(changes["maxabs"][name] != 0 for name in policy["frozen"]):
        raise ValueError("A1 frozen prefix changed")
    if not any(
        value
        for name, value in changes["maxabs"].items()
        if name.startswith(policy["last_block_prefix"])
    ):
        raise ValueError("A1 last block did not learn")


def qa() -> dict:
    """Verify both gradient graphs and exact full-state resume before scientific updates."""
    guard_idle()
    registered()
    if (QA / "RESULT.json").exists():
        result = read(QA / "RESULT.json")
        if result["status"] != "PASS" or result["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("A1 QA is not admitted")
        return result
    if accounting()["technical"] + 60 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient technical update reserve")
    checks = {}
    with engine_context():
        for mode in MODES:
            target = QA / mode
            with gpu_charge(f"A1_QA_{mode}", reserve=900) as deadline:
                for name, capture, limit in (
                    ("eager", False, 10),
                    ("continuous", True, 10),
                    ("split", True, 5),
                    ("split", True, 10),
                ):
                    result = engine.segment(
                        target / name,
                        geometry_train.SyntheticSensors(),
                        np.full(8, 1 / 8),
                        mode,
                        2,
                        limit=limit,
                        deadline=deadline,
                        capture=capture,
                        synthetic=True,
                    )
                    if result["completed_updates"] != limit:
                        raise RuntimeError("A1 QA incomplete")
                    verify_change(target / name, mode)
            eager, continuous, split = [
                torch.load(
                    target / name / "checkpoint_last.pt", map_location="cpu", weights_only=False
                )
                for name in ("eager", "continuous", "split")
            ]
            resume = {key: exact(continuous[key], split[key]) for key in STATE_KEYS}
            resume["curve"] = exact(continuous["cursor"]["curve"], split["cursor"]["curve"])
            numerical = {
                key: close(eager[key], continuous[key], rtol=1e-5, atol=1e-7) for key in STATE_KEYS
            }
            numerical["curve"] = close(
                eager["cursor"]["curve"], continuous["cursor"]["curve"], rtol=1e-5, atol=1e-7
            )
            checks[mode] = {"exact_resume": resume, "eager_numerical": numerical}
    result = {
        "status": "PASS"
        if all(all(v.values()) for arm in checks.values() for v in arm.values())
        else "FAIL",
        "checks": checks,
        "registration_sha256": sha(REGISTRATION),
        "physical_technical_updates": 60,
    }
    freeze(QA / "RESULT.json", result)
    if result["status"] != "PASS":
        raise ValueError("A1 captured/resume QA failed")
    return result


def segment(mode: str) -> dict:
    """Restore complete state and advance one existing bounded CUDA segment."""
    guard_idle()
    record = registered()
    proof = read(QA / "RESULT.json")
    if proof["status"] != "PASS" or proof["registration_sha256"] != sha(REGISTRATION):
        raise ValueError("A1 CUDA QA required")
    require_resources(starting=True)
    sensors = PersistentSensors()
    target = folder(mode)
    try:
        if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
            raise ValueError("A1 sensory population changed")
        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        if digest(masses.tolist()) != record["mass_sha256"]:
            raise ValueError("A1 sampling masses changed")
        with gpu_charge(f"A1_{mode}", reserve=900) as deadline, engine_context():
            if not (target / "RUN_PROVENANCE.json").exists():
                freeze(
                    target / "RUN_PROVENANCE.json",
                    {
                        "experiment_id": record["experiment_id"],
                        "run_name": target.name,
                        "git_commit": committed_sources(source_binding()),
                        "config_hash": digest(record),
                        "seed": 7,
                        "dataset_manifest_hash": record["P_manifest_sha256"],
                        "split_version": "native P/H/V",
                        "host": socket.gethostname(),
                        "python_version": platform.python_version(),
                        "torch_version": str(torch.__version__),
                        "cuda_version": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "start_time": utc(),
                        "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                        "checkpoint_path": str(target / "checkpoint_last.pt"),
                        "metrics_path": str(target / "CURVE.csv"),
                        "parent_sha256": record["parent_sha256"],
                    },
                )
            result = engine.segment(
                target, sensors, masses, mode, 2, limit=UPDATES, deadline=deadline, capture=True
            )
            verify_change(target, mode)
        write(
            DIRECTORY / "segments" / f"{mode}_{result['completed_updates']:06d}.json",
            {"utc": utc(), "receipt": result, "accounting": accounting()},
        )
        return result
    finally:
        sensors.shutdown()


def run() -> None:
    """One coordinator and one fresh CUDA child; no silent retry after a failed child."""
    guard_idle()
    require_authorization()
    registered()
    proof = read(QA / "RESULT.json")
    if proof["status"] != "PASS" or proof["registration_sha256"] != sha(REGISTRATION):
        raise ValueError("Run A1 QA in its separate process first")
    python = ROOT.parent / "e-jepa-ttc/.venv/Scripts/python.exe"
    with patch.object(geometry_train, "DIRECTORY", DIRECTORY), geometry_train.exclusive_driver():
        try:
            for mode in MODES:
                target = folder(mode)
                path = target / "CHECKPOINT_RECEIPT.json"
                while True:
                    receipt = read(path) if path.exists() else None
                    if receipt and receipt["status"] == "COMPLETE":
                        if receipt["completed_updates"] != UPDATES or receipt[
                            "checkpoint_sha256"
                        ] != sha(target / "checkpoint_last.pt"):
                            raise ValueError("A1 endpoint changed")
                        verify_change(target, mode)
                        break
                    before = receipt["completed_updates"] if receipt else 0
                    with subprocess.Popen(
                        [
                            str(python),
                            "-u",
                            "-B",
                            "-m",
                            "operational.joint_measurement.adaptation_depth",
                            "segment",
                            "--arm",
                            mode,
                        ],
                        cwd=ROOT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    ) as process:
                        write(
                            DIRECTORY / "RUNNER_STATE.json",
                            {
                                "status": "TRAINING",
                                "mode": mode,
                                "pid": os.getpid(),
                                "child_launcher_pid": process.pid,
                                "utc": utc(),
                            },
                        )
                        code = process.wait()
                    if code != 0:
                        raise RuntimeError(f"A1 child failed with code {code}")
                    if read(path)["completed_updates"] <= before:
                        raise RuntimeError("A1 segment made no progress")
            freeze(
                DIRECTORY / "ENDPOINTS.json",
                {folder(mode).name: sha(folder(mode) / "checkpoint_last.pt") for mode in MODES},
            )
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "A1_ENDPOINTS_COMPLETE",
                    "utc": utc(),
                    "pid": os.getpid(),
                    "full_objective_complete": False,
                },
            )
        except Exception as error:
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "ACTION_REQUIRED",
                    "utc": utc(),
                    "pid": os.getpid(),
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prototype", "register", "qa", "segment", "run"))
    parser.add_argument("--arm", choices=MODES, default=MODES[0])
    args = parser.parse_args()
    result = (
        segment(args.arm)
        if args.action == "segment"
        else {"prototype": prototype, "register": register, "qa": qa, "run": run}[args.action]()
    )
    if result is not None:
        print(
            {key: result.get(key) for key in ("status", "completed_updates", "experiment_id")},
            flush=True,
        )
