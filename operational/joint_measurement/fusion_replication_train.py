"""Replicate the selected C1-to-RGB complement family, including its adapted encoder."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import platform
import socket
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, RUN, accounting, config, gpu_charge, require_resources
from .data import cache_root
from .fast_runtime import PersistentSensors
from .fusion_train import REGISTRATION as FUSION_REGISTRATION
from .fusion_train import freeze_endpoints
from .geometry_capture import contiguous_subset, owned_backward
from .geometry_capture_qa import OUT as CAPTURE_QA
from .geometry_capture_qa import STATE_KEYS, CapturedGeometry
from .geometry_capture_run import admission as capture_admission
from .geometry_graph_probe import no_live_training_driver
from .geometry_train import REGISTRATION as C1_REGISTRATION
from .geometry_train import SyntheticSensors
from .qa import exact
from .rgb_run import require_authorization
from .train import committed_sources, initialize, make_optimizer, setup

DIRECTORY = RUN / "replication/fusion_C1"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
SEEDS = (13, 23)


def register() -> dict:
    """Freeze one family, both adaptation seeds, paired heads and decisions before updates."""
    require_authorization()
    endpoints = freeze_endpoints()
    signal = read(RUN / "fusion/evaluation/RESULT.json")
    true = signal["results"]["V/FUSION-TRUE-S7"]["metrics"]
    zero = signal["results"]["V/FUSION-ZERO-S7"]["metrics"]
    checks = signal["gates"]["FUSION-TRUE-S7_vs_FUSION-ZERO-S7"]["checks"]
    if not (
        true["overall_MiD"] <= 0.97 * zero["overall_MiD"]
        and all(checks[k] for k in ("coverage", "small", "large", "negative", "sign"))
    ):
        raise ValueError("No practical controlled fusion signal to replicate")
    runtime = capture_admission()
    sources = {
        **runtime["sources"],
        "operational/joint_measurement/fusion_replication_train.py": sha(Path(__file__)),
    }
    committed_sources(sources)
    c1 = read(C1_REGISTRATION)
    heads = read(FUSION_REGISTRATION)
    record = {
        "schema": "v18_one_family_replication_v1",
        "family": "C1 adapted encoder plus RGB-anchored TRUE/ZERO complements",
        "seeds": list(SEEDS),
        "reused_seed7_endpoints": endpoints,
        "seed7_evidence_sha256": sha(RUN / "fusion/evaluation/RESULT.json"),
        "rationale": (
            "Controlled V aggregate improvement exceeds 3%, safeguards pass, paired "
            "group interval excludes zero; full promotion fails and F_TRUE interval "
            "includes zero"
        ),
        "event_recipe_sha256": sha(C1_REGISTRATION),
        "head_recipe_sha256": sha(FUSION_REGISTRATION),
        "event_updates_per_seed": 12000,
        "head_updates_per_mode_per_seed": 2500,
        "head_modes": ["TRUE", "ZERO"],
        "maximum_new_scientific_updates": 34000,
        "maximum_new_technical_updates": 20,
        "parent_sha256": c1["parent_sha256"],
        "P_manifest_sha256": c1["P_manifest_sha256"],
        "sensor_binding": c1["sensor_binding"],
        "mass_sha256": c1["mass_sha256"],
        "population_size": c1["population_size"],
        "optimizer": c1["optimizer"],
        "effective_batch": 32,
        "microbatch": 4,
        "head_recipes": heads["recipes"],
        "shared_initialization": (
            "Same historical P-only A5 and H-supervised R_CTX; replicate adaptation and"
            " every new module, not original pretraining"
        ),
        "fresh_features": (
            "Rebuild C1 H/V spatial features and H event normalizer for each seed; "
            "fixed unchanged RGB anchor inputs/normalizer"
        ),
        "head_seed_rule": (
            "Both TRUE/ZERO use the corresponding encoder seed13 or23; identical new "
            "head initialization"
        ),
        "fit_order": (
            "Both event producers, then their H features/normalizers and four heads; "
            "freeze all six endpoints before any replica V prediction"
        ),
        "evaluation": (
            "All seeds and controls retained; same complete V cohort, scorer, system "
            "gates and acquisition-group bootstrap; report per-seed and seed mean/std, "
            "no best-seed selection"
        ),
        "H_scope": (
            "Sequential supervised head fitting on H-trained R_CTX; not independent H validation"
        ),
        "scope_limit": (
            "Replicate multimodal complement mechanism only; does not replicate or "
            "promote standalone C1 versus free readout"
        ),
        "promotion": (
            "Requires original full-system gates and comparable full-pipeline cost; no "
            "automatic refit or official claim"
        ),
        "runtime": (
            "Admitted full forward/backward CUDA capture, FP32 no TF32, pinned "
            "contiguous microbatches, bounded 8GiB lossless sensor LRU, one-batch "
            "prefetch"
        ),
        "capture_proof_sha256": sha(CAPTURE_QA / "RESULT.json"),
        "sources": sources,
    }
    freeze(REGISTRATION, record)
    freeze(
        RUN / "replication/SELECTED_FAMILY.json",
        {
            "family": record["family"],
            "registration_sha256": sha(REGISTRATION),
            "maximum_families": 1,
        },
    )
    return record


def registered() -> dict:
    """Reject mutated numerical/runtime code without revisiting development outcomes."""
    record = read(REGISTRATION)
    for path, expected in record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Replication source changed: {path}")
    return record


def segment(
    directory: Path,
    sensors: PersistentSensors,
    masses: np.ndarray,
    seed: int,
    *,
    limit: int,
    deadline: float,
    synthetic: bool = False,
) -> dict:
    """Preserve C1 arithmetic with explicit initialization and sampler seeds."""
    record = registered()
    if seed not in SEEDS:
        raise ValueError("Unregistered replication seed")
    endpoint = 10 if synthetic else 12000
    identity = {
        "fit_id": f"C1-FUSION-REPLICA-S{seed}",
        "seed": seed,
        "registration_sha256": sha(REGISTRATION),
        "endpoint": endpoint,
        "population": "synthetic8" if synthetic else "P",
        "device": "cuda",
        "mass_sha256": digest(masses.tolist()),
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    setup(seed)
    parent = initialize("A5-P2-FROZEN")
    if not isinstance(parent, CausalScaleTTC):
        raise TypeError("Replication requires the native P-only initialization")
    initial = copy.deepcopy(parent.encoder.features.state_dict())
    model = CapturedGeometry(parent).to("cuda").train()
    del parent
    # Geometry exposes the same encoder/rest parameter groups as PairMeasurement.
    optimizer, scheduler = make_optimizer(
        cast(PairMeasurement, model), updates=record["event_updates_per_seed"]
    )
    sampler = torch.Generator().manual_seed(seed)
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor = state.restore(model, optimizer, scheduler, sampler) if state.pointer.exists() else None
    curve, timings = (cursor["curve"], cursor["timings"]) if cursor else ([], [])
    accounting()
    if cursor is None:
        state.save(
            model,
            optimizer,
            scheduler,
            sampler,
            {"curve": curve, "timings": timings},
            status="RUNNING",
        )
    sensors.prepare(sampler.get_state(), pinned=True)
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
            ids = torch.randint(sensors.population, (32,), generator=sampler).tolist()
            batch = sensors.batch(ids)
            batches = [
                contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = owned_backward(
                model, batches, torch.from_numpy(masses[ids]).float(), sensors.population
            )
            if state.completed == 0:
                gradients = {
                    name: float(p.grad.norm()) if p.grad is not None else 0.0
                    for name, p in model.encoder.named_parameters()
                }
                if not all(np.isfinite(list(gradients.values()))) or not any(gradients.values()):
                    raise ValueError("Replication encoder gradient contract failed")
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
                    f"C1-REPLICA-S{seed} {state.completed}/{endpoint} loss={losses['loss']:.6g}",
                    flush=True,
                )
    finally:
        sensors.close()
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
        for key, value in model.encoder.state_dict().items()
    }
    if state.completed and not any(changes.values()):
        raise ValueError("Replica encoder did not change")
    write(directory / "ENCODER_CHANGE.json", {"updates": state.completed, "maxabs": changes})
    if curve:
        rows = [
            {**{key: value for key, value in point.items() if key != "ids"}, "seconds": seconds}
            for point, seconds in zip(curve, timings, strict=True)
        ]
        with (directory / "CURVE.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    accounting()
    return read(directory / "CHECKPOINT_RECEIPT.json")


def qa() -> dict:
    """Twenty counted technical updates verify exact resume of the seeded captured engine."""
    root = RUN / "qa/fusion_replication_capture"
    if (root / "PASS.json").exists():
        proof = read(root / "PASS.json")
        if proof["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Replica QA binding changed")
        return proof
    if accounting()["technical"] + 20 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient QA reserve")
    with gpu_charge("fusion_replication_capture_QA", reserve=900) as deadline:
        for name, limit in (("continuous", 10), ("split", 5), ("split", 10)):
            outcome = segment(
                root / name,
                SyntheticSensors(),
                np.full(8, 1 / 8),
                13,
                limit=limit,
                deadline=deadline,
                synthetic=True,
            )
            if outcome["completed_updates"] != limit:
                raise RuntimeError("Replica QA interrupted; preserve states")
    payloads = [
        torch.load(root / name / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for name in ("continuous", "split")
    ]
    checks = {key: exact(payloads[0][key], payloads[1][key]) for key in STATE_KEYS}
    checks["curve"] = exact(payloads[0]["cursor"]["curve"], payloads[1]["cursor"]["curve"])
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "registration_sha256": sha(REGISTRATION),
        "technical_updates": 20,
    }
    freeze(root / f"{result['status']}.json", result)
    if result["status"] != "PASS":
        raise ValueError("Captured replica exact-resume proof failed")
    return result


def run() -> None:
    """Single writer finishes both fixed producer endpoints, retaining resources between scopes."""
    import msvcrt

    no_live_training_driver()
    record = register()
    with (DIRECTORY / "DRIVER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        sensors = None
        try:
            qa()
            sensors = PersistentSensors()
            if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
                raise ValueError("Replica P sensor identity changed")
            masses = training_mass(
                sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
            )
            if digest(masses.tolist()) != record["mass_sha256"]:
                raise ValueError("Replica P masses changed")
            for seed in SEEDS:
                folder = DIRECTORY / "fits" / f"C1-FUSION-REPLICA-S{seed}"
                if not (folder / "RUN_PROVENANCE.json").exists():
                    freeze(
                        folder / "RUN_PROVENANCE.json",
                        {
                            "experiment_id": "V18_FUSION_C1_REPLICATION",
                            "run_name": folder.name,
                            "seed": seed,
                            "git_commit": committed_sources(record["sources"]),
                            "config_hash": digest(record),
                            "dataset_manifest_hash": record["P_manifest_sha256"],
                            "split_version": "unchanged native P/H/V",
                            "host": socket.gethostname(),
                            "python_version": platform.python_version(),
                            "torch_version": str(torch.__version__),
                            "cuda_version": torch.version.cuda,
                            "gpu_name": torch.cuda.get_device_name(0),
                            "start_time": utc(),
                            "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                            "checkpoint_path": str(folder / "checkpoint_last.pt"),
                        },
                    )
                while True:
                    receipt = folder / "CHECKPOINT_RECEIPT.json"
                    if receipt.exists() and read(receipt)["status"] == "COMPLETE":
                        if (
                            read(receipt)["completed_updates"] != 12000
                            or sha(folder / "checkpoint_last.pt")
                            != read(receipt)["checkpoint_sha256"]
                        ):
                            raise ValueError("Replica endpoint changed")
                        break
                    state = {"status": "TRAINING", "seed": seed, "pid": os.getpid(), "utc": utc()}
                    write(DIRECTORY / "RUNNER_STATE.json", state)
                    require_resources(starting=True)
                    with gpu_charge(f"fusion_replication_C1_seed{seed}", reserve=900) as deadline:
                        result = segment(
                            folder, sensors, masses, seed, limit=12000, deadline=deadline
                        )
                    write(
                        DIRECTORY / "RUNNER_STATE.json",
                        {**state, "receipt": result, "accounting": accounting(), "utc": utc()},
                    )
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "REPLICA_PRODUCERS_COMPLETE",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "full_objective_complete": False,
                    "next": (
                        "Build independent H features/normalizers and four registered "
                        "heads; freeze all before V"
                    ),
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
            if sensors is not None:
                sensors.shutdown()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register()) if args.register_only else run()
