"""B1 two-stage encoder correspondence versus equal-update supervised continuation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import geometry_train as engine
from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .correspondence_model import TRANSFORMS, step_backward
from .data import cache_root, role_rows
from .fast_runtime import PersistentSensors
from .geometry_capture import contiguous_subset
from .geometry_capture_qa import STATE_KEYS
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .qa import exact, fixture
from .resume_fast_proof import close
from .rgb_run import require_authorization
from .train import committed_sources, initialize, make_optimizer, setup, sources

DIRECTORY = RUN / "adaptive/B1_correspondence"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
PARENT = RUN / "adaptive/C1_geometry/fits/PAIR-CORR-GEOMETRY-S7"
MODES = ("CORRESPONDENCE", "TTC-CONTROL")
QA = RUN / "qa/correspondence_B1"
STEPS = {1: 2000, 2: 10000}


def folder(mode: str, stage: int) -> Path:
    """Separate every stage's optimizer, sampler, cursor and update journal."""
    if mode not in MODES or stage not in STEPS:
        raise ValueError("Unknown B1 arm or stage")
    return DIRECTORY / "fits" / f"B1-{mode}-STAGE{stage}-S7"


def source_binding() -> dict:
    """Bind all numerical, checkpoint and capture sources before any physical update."""
    names = (
        "correspondence_train",
        "correspondence_model",
        "geometry_train",
        "geometry_model",
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
    """Prospectively freeze one matched comparison, based only on P correspondence evidence."""
    require_authorization()
    committed_sources(source_binding())
    diagnostic = RUN / "diagnostics/known_scale_P/RESULT.json"
    prototype_path = DIRECTORY / "prototype/CPU.json"
    prototype = read(prototype_path)
    if prototype["status"] != "PASS" or prototype["sources"] != source_binding():
        raise ValueError("B1 requires its native zero-update CPU proof")
    receipt = read(PARENT / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 12000:
        raise ValueError("B1 requires the complete shared P-only C1 parent")
    if sha(PARENT / "checkpoint_last.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("C1 parent changed")
    rows = role_rows("P")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), role="P")
    record = {
        "schema": "v18_adaptive_B1_v1",
        "experiment_id": "V18_ADAPTIVE_B1",
        "family": "B",
        "modes": list(MODES),
        "seed": 7,
        "stages": {str(k): v for k, v in STEPS.items()},
        "maximum_new_scientific_updates": 24000,
        "maximum_synthetic_qa_updates": 90,
        "hypothesis": (
            "Explicit known-transform supervision can improve imperfect event correspondence "
            "and downstream TTC beyond the same additional updates of TTC supervision. "
            "P diagnostic has substantial signal, not collapse: approximately72% strict "
            "nonzero known-scale accuracy for C1 versus4% shuffled. Remaining errors and "
            "the worse TTC-adapted free CORR matching motivate this bounded contrast."
        ),
        "evidence_sha256": sha(diagnostic),
        "prototype_sha256": sha(prototype_path),
        "parent_sha256": receipt["checkpoint_sha256"],
        "parent": "Shared fixed C1 seed7, trained only on P;12000 historical updates reused",
        "stage1": (
            "2000 updates, encoder only in both arms; all readout parameters frozen. "
            "CORRESPONDENCE uses only latest P texture and known transforms, uniform rows; "
            "TTC-CONTROL uses real P2, original phase loss and global P masses. "
            "This isolates pretraining objective/input regime, not identical FLOPs."
        ),
        "stage2": (
            "10000 full-model TTC updates each, same original C1 objective, P2 and masses. "
            "Start from respective stage1 weights, reset Adam/scheduler and all RNG to seed7 "
            "in BOTH arms; same sampling sequence. No optimizer state reset within a stage."
        ),
        "correspondence": {
            "transforms_scale_dx_dy_pixels": [list(t) for t in TRANSFORMS],
            "assignment": "fixed four transforms in every microbatch, independent of labels",
            "warp": "bilinear zero-fill, align_cornersFalse, centred scale plus translation",
            "maps": "native32 and avgpool16, both directions, native5x5 cosine volume",
            "target": "bilinear probability on known displacement, no TTC targets",
            "support": "central half of each spatial axis, displacement within radius2",
            "temperature": 0.07,
            "reduction": "equal queries, positions, scales and directions",
            "limit": "Synthetic texture correspondence, NOT physical events or synthetic TTC",
        },
        "batch_size": 32,
        "microbatch_size": 4,
        "optimizer": config()["optimizer"],
        "precision": "FP32_no_TF32",
        "sampling": "uniform with replacement; seed7 per stage",
        "population": "P",
        "population_size": len(rows),
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_binding": sha(cache_root("P") / "COMPLETE.json"),
        "mass_sha256": digest(masses.tolist()),
        "freeze_rule": "Freeze both stage2 endpoints before any B1 H inference; no early stopping",
        "mechanism_rule": (
            "Full P/H report; paired H aggregate improvement>=3% or crucial>=5% with "
            "aggregate<=1.02 control, all bands<=1.10 control, sign<=control+0.005. "
            "Also report known-scale correspondence without treating it as TTC success."
        ),
        "development_gates": config()["selection"],
        "V_rule": "Only after fixed H admission; no V rescue or seed rescue",
        "replication": "Selected C1 fusion family already consumed; B1 is seed7 only",
        "runtime": "Existing lossless8GiB sensor LRU, prefetch, pinned microbatches, CUDA capture",
        "qa": "Each of3 stage/objective modes: eager10,capture10,capture5+restore5;90 total",
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
        raise ValueError("Adaptive comparison budget exhausted")
    freeze(REGISTRATION, record)
    prior = [r for r in existing if r["experiment_id"] == record["experiment_id"]]
    if prior:
        if any(r["registration_sha256"] != sha(REGISTRATION) for r in prior):
            raise ValueError("B1 registry changed")
    else:
        with registry.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "experiment_id": record["experiment_id"],
                        "registered_utc": utc(),
                        "status": "REGISTERED_BEFORE_UPDATES",
                        "registration_sha256": sha(REGISTRATION),
                        "maximum_scientific_updates": 24000,
                        "maximum_synthetic_qa_updates": 90,
                        "fits": [folder(mode, stage).name for stage in STEPS for mode in MODES],
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return record


def registered() -> dict:
    """Reject any post-registration change to scientific or training implementation."""
    record = read(REGISTRATION)
    if record["sources"] != source_binding():
        raise ValueError("B1 registered source changed")
    return record


def make_model(mode: str, stage: int, *, synthetic: bool) -> tuple:
    """Load identical C1 or the arm's durable stage1; freeze readout only during stage1."""
    source = PARENT if stage == 1 or synthetic else folder(mode, 1)
    receipt = read(source / "CHECKPOINT_RECEIPT.json")
    if (
        receipt["status"] != "COMPLETE"
        or sha(source / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
    ):
        raise ValueError("B1 source checkpoint is incomplete or changed")
    parent = initialize("A5-P2-FROZEN")
    if not isinstance(parent, CausalScaleTTC):
        raise TypeError("Native parent required")
    model = GeometricPairMeasurement(parent)
    payload = torch.load(source / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage == 2 or name.startswith("encoder."))
    return model, receipt["checkpoint_sha256"]


def prototype() -> dict:
    """Check the actual native parent/encoder and frozen heads on CPU, without optimizer steps."""
    torch.set_num_threads(1)
    committed_sources(source_binding())
    batch = contiguous_subset(fixture(128), list(range(4)))
    checks, initial_states, losses = {}, [], {}
    parent_sha = sha(PARENT / "checkpoint_last.pt")
    for mode in MODES:
        torch.manual_seed(7)
        model, source_sha = make_model(mode, 1, synthetic=True)
        initial = copy.deepcopy(model.state_dict())
        initial_states.append(initial)
        values = step_backward(
            model,
            [batch],
            torch.full((4,), 1 / 8),
            8,
            correspondence=mode == MODES[0],
            capture=False,
        )
        losses[mode] = values
        checks[f"{mode}_source"] = source_sha == parent_sha
        checks[f"{mode}_encoder_gradients"] = all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.encoder.parameters()
        ) and any(bool(p.grad.norm() > 0) for p in model.encoder.parameters() if p.grad is not None)
        checks[f"{mode}_readout_frozen"] = all(
            not p.requires_grad and p.grad is None
            for name, p in model.named_parameters()
            if not name.startswith("encoder.")
        )
        checks[f"{mode}_weights_unchanged"] = exact(initial, model.state_dict())
    checks["identical_initial_models"] = exact(*initial_states)
    checks["parent_unchanged"] = sha(PARENT / "checkpoint_last.pt") == parent_sha
    checks["cuda_uninitialized"] = not torch.cuda.is_initialized()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "losses": losses,
        "optimizer_updates": 0,
        "sources": source_binding(),
    }
    freeze(DIRECTORY / "prototype/CPU.json", result)
    return result


def segment(
    directory: Path,
    dataset: PersistentSensors,
    masses: np.ndarray,
    mode: str,
    stage: int,
    *,
    limit: int,
    deadline: float,
    capture: bool,
    synthetic: bool = False,
) -> dict:
    """Advance one full-state stage with the established transaction journal and sampler."""
    registered()
    setup(7)
    model, parent_sha = make_model(mode, stage, synthetic=synthetic)
    initial = copy.deepcopy(model.state_dict())
    model = model.to("cuda").train()
    endpoint = 10 if synthetic else STEPS[stage]
    identity = {
        "fit_id": directory.name,
        "mode": mode,
        "stage": stage,
        "registration_sha256": sha(REGISTRATION),
        "endpoint": endpoint,
        "population": "synthetic8" if synthetic else "P",
        "device": "cuda",
        "parent_sha256": parent_sha,
        "mass_sha256": digest(masses.tolist()),
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    optimizer, scheduler = make_optimizer(model, updates=STEPS[stage])
    sampler = torch.Generator().manual_seed(7)
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor = state.restore(model, optimizer, scheduler, sampler) if state.pointer.exists() else None
    curve, timings = (cursor["curve"], cursor["timings"]) if cursor else ([], [])
    if cursor is None:
        state.save(
            model,
            optimizer,
            scheduler,
            sampler,
            {"curve": curve, "timings": timings},
            status="RUNNING",
        )
    accounting()
    dataset.prepare(sampler.get_state(), pinned=True)
    try:
        while state.completed < min(limit, endpoint) and time.perf_counter() < deadline:
            if state.completed % 20 == 0:
                try:
                    require_resources()
                except RuntimeError as error:
                    if str(error).startswith("RESOURCE_WAIT"):
                        break
                    raise
            tick = time.perf_counter()
            ids = torch.randint(dataset.population, (32,), generator=sampler).tolist()
            batch = dataset.batch(ids)
            batches = [
                contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = step_backward(
                model,
                batches,
                torch.from_numpy(masses[ids]).float(),
                dataset.population,
                correspondence=stage == 1 and mode == "CORRESPONDENCE",
                capture=capture,
            )
            if state.completed == 0:
                gradients = {
                    name: float(p.grad.norm()) if p.grad is not None else 0.0
                    for name, p in model.encoder.named_parameters()
                }
                if not all(np.isfinite(list(gradients.values()))) or not any(gradients.values()):
                    raise ValueError("B1 encoder must receive finite nonzero gradients")
                write(directory / "ENCODER_GRADIENT.json", gradients)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            curve.append({"update": state.completed, "ids": ids, **losses, "norm": float(norm)})
            timings.append(time.perf_counter() - tick)
            if state.completed % 100 == 0:
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    sampler,
                    {"curve": curve, "timings": timings},
                    status="RUNNING",
                )
                accounting()
                print(
                    f"{directory.name} {state.completed}/{endpoint} loss={losses['loss']:.6g}",
                    flush=True,
                )
    finally:
        dataset.close()
    # As in the admitted engine, an exception never publishes a partial optimizer state.
    state.save(
        model,
        optimizer,
        scheduler,
        sampler,
        {"curve": curve, "timings": timings},
        status="COMPLETE" if state.completed == endpoint else "PAUSED",
    )
    changes = {
        key: float((value.detach().cpu() - initial[key]).abs().max())
        for key, value in model.state_dict().items()
    }
    if state.completed and not any(
        value for key, value in changes.items() if key.startswith("encoder.")
    ):
        raise ValueError("B1 encoder did not change")
    if stage == 1 and any(
        value for key, value in changes.items() if not key.startswith("encoder.")
    ):
        raise ValueError("B1 frozen readout changed")
    write(directory / "WEIGHT_CHANGE.json", {"updates": state.completed, "maxabs": changes})
    if curve:
        rows = [
            {**{k: v for k, v in row.items() if k != "ids"}, "seconds": seconds}
            for row, seconds in zip(curve, timings, strict=True)
        ]
        with (directory / "CURVE.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    accounting()
    return read(directory / "CHECKPOINT_RECEIPT.json")


def qa() -> dict:
    """Prove eager/captured equivalence and exact full-state resume for all three regimes."""
    registered()
    if (QA / "RESULT.json").exists():
        result = read(QA / "RESULT.json")
        if result["status"] != "PASS" or result["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Retained B1 QA failed or changed")
        return result
    if accounting()["technical"] + 90 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient B1 technical reserve")
    checks = {}
    for mode, stage in ((MODES[0], 1), (MODES[1], 1), (MODES[1], 2)):
        target = QA / f"{mode}-STAGE{stage}"
        with gpu_charge(f"B1_QA_{mode}_{stage}", reserve=900) as deadline:
            for name, capture, limit in (
                ("eager", False, 10),
                ("continuous", True, 10),
                ("split", True, 5),
                ("split", True, 10),
            ):
                result = segment(
                    target / name,
                    engine.SyntheticSensors(),
                    np.full(8, 1 / 8),
                    mode,
                    stage,
                    limit=limit,
                    deadline=deadline,
                    capture=capture,
                    synthetic=True,
                )
                if result["completed_updates"] != limit:
                    raise RuntimeError("B1 QA interrupted")
        eager, continuous, split = [
            torch.load(target / name / "checkpoint_last.pt", map_location="cpu", weights_only=False)
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
        checks[f"{mode}/{stage}"] = {"exact_resume": resume, "eager_numerical": numerical}
    result = {
        "status": "PASS"
        if all(all(values.values()) for modes in checks.values() for values in modes.values())
        else "FAIL",
        "checks": checks,
        "registration_sha256": sha(REGISTRATION),
        "physical_technical_updates": 90,
    }
    freeze(QA / "RESULT.json", result)
    if result["status"] != "PASS":
        raise ValueError("B1 CUDA numerical/resume QA failed")
    return result


def run() -> None:
    """Complete both stages, preserving every durable update and freezing both final endpoints."""
    require_authorization()
    no_live_training_driver()
    record = registered()
    with patch.object(engine, "DIRECTORY", DIRECTORY), engine.exclusive_driver():
        qa()
        sensors = PersistentSensors()
        try:
            if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
                raise ValueError("B1 P sensory cache changed")
            masses = training_mass(
                sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
            )
            if digest(masses.tolist()) != record["mass_sha256"]:
                raise ValueError("B1 P mass changed")
            for stage in STEPS:
                for mode in MODES:
                    target = folder(mode, stage)
                    provenance = target / "RUN_PROVENANCE.json"
                    if not provenance.exists():
                        freeze(
                            provenance,
                            {
                                "experiment_id": record["experiment_id"],
                                "run_name": target.name,
                                "git_commit": committed_sources(source_binding()),
                                "seed": 7,
                                "config_hash": digest(record),
                                "dataset_manifest_hash": record["P_manifest_sha256"],
                                "split_version": "native P/H/V",
                                "host": socket.gethostname(),
                                "python_version": platform.python_version(),
                                "torch_version": str(torch.__version__),
                                "cuda_version": torch.version.cuda,
                                "start_time": utc(),
                                "checkpoint_path": str(target / "checkpoint_last.pt"),
                                "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                            },
                        )
                    while True:
                        path = target / "CHECKPOINT_RECEIPT.json"
                        if path.exists() and read(path)["status"] == "COMPLETE":
                            receipt = read(path)
                            if receipt["completed_updates"] != STEPS[stage] or receipt[
                                "checkpoint_sha256"
                            ] != sha(target / "checkpoint_last.pt"):
                                raise ValueError("B1 completed endpoint changed")
                            break
                        require_resources(starting=True)
                        write(
                            DIRECTORY / "RUNNER_STATE.json",
                            {
                                "status": "TRAINING",
                                "pid": os.getpid(),
                                "mode": mode,
                                "stage": stage,
                                "utc": utc(),
                            },
                        )
                        with gpu_charge(f"B1_{mode}_stage{stage}", reserve=900) as deadline:
                            segment(
                                target,
                                sensors,
                                masses,
                                mode,
                                stage,
                                limit=STEPS[stage],
                                deadline=deadline,
                                capture=True,
                            )
            freeze(
                DIRECTORY / "ENDPOINTS.json",
                {
                    folder(mode, stage).name: sha(folder(mode, stage) / "checkpoint_last.pt")
                    for stage in STEPS
                    for mode in MODES
                },
            )
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "B1_ENDPOINTS_COMPLETE",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "full_objective_complete": False,
                },
            )
        except Exception as error:
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "ACTION_REQUIRED",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "error": str(error),
                },
            )
            raise
        finally:
            sensors.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    parser.add_argument("--prototype-cpu", action="store_true")
    args = parser.parse_args()
    if args.prototype_cpu:
        print(prototype())
    elif args.register_only:
        print(register())
    elif args.qa_only:
        no_live_training_driver()
        print(qa())
    else:
        run()
