"""Family C: fixed geometry endpoint with a hash-bound, reused primary free control."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .data import cache_root, role_rows
from .evaluate import freeze_primary
from .fast_runtime import PersistentSensors
from .geometry_model import GeometricPairMeasurement
from .qa import exact, fixture, subset
from .train import backward, committed_sources, initialize, make_optimizer, setup, sources

FIT = "PAIR-CORR-GEOMETRY-S7"
CONTROL = "PAIR-CORR-S7"
DIRECTORY = RUN / "adaptive/C1_geometry"
REGISTRATION = DIRECTORY / "REGISTRATION.json"


def source_binding() -> dict:
    """Bind new geometry and reused scientific/runtime code before any update."""
    return {
        **sources(),
        **{
            f"operational/joint_measurement/{name}": sha(
                ROOT / "operational/joint_measurement" / name
            )
            for name in ("geometry_train.py", "geometry_model.py", "fast_runtime.py")
        },
    }


def register() -> dict:
    """Admit the signal and exact historical control; do not rerun identical 12k work."""
    authorization = read(RUN / "BUDGET_AUTHORIZATION_20261014.json")
    if authorization["status"] != "APPLIED" or not authorization["scope"].startswith(
        "All evidence-enabled V18 handoff branches:"
    ):
        raise ValueError("Family C requires full-program authority")
    diagnostic = RUN / "diagnostics/real_temporal_P/RESULT.json"
    if not read(diagnostic)["grouped"][CONTROL]["geometry_followup_signal"]:
        raise ValueError("No registered real-temporal correspondence signal")
    endpoints = freeze_primary()["endpoints"]
    control = RUN / "fits" / CONTROL
    identity = read(control / "IDENTITY.json")
    if (
        identity["sources"] != sources()
        or identity["config_hash"] != digest(config())
        or identity["seed"] != 7
        or identity["updates"] != 12000
        or identity["sensor_binding"] != sha(cache_root("P") / "COMPLETE.json")
    ):
        raise ValueError("Historical free control is not matched to the proposed recipe")
    proof = RUN / "verification/geometry_prototype_v1/CPU_INITIALIZATION_AND_GRADIENT.json"
    prototype = read(proof)
    if prototype["geometry_source_sha256"] != sha(
        ROOT / "operational/joint_measurement/geometry_model.py"
    ):
        raise ValueError("Repeat prototype verification after a geometry change")
    rows = role_rows("P")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), role="P")
    if identity["mass_sha256"] != digest(masses.tolist()):
        raise ValueError("Historical control used different population masses")
    record = {
        "schema": "v18_adaptive_C1_geometry_v1",
        "experiment_id": "V18_ADAPTIVE_C1",
        "family": "C",
        "fits": [FIT],
        "seed": 7,
        "updates_per_fit": 12000,
        "maximum_new_scientific_updates": 12000,
        "maximum_synthetic_qa_updates": 40,
        "paired_control": {
            "fit_id": CONTROL,
            "checkpoint_sha256": endpoints[CONTROL],
            "identity_sha256": sha(control / "IDENTITY.json"),
            "updates": 12000,
            "reused_without_new_updates": True,
            "cost": "already charged in primary; never count twice",
        },
        "hypothesis": (
            "Explicit affine scale/translation readout can exploit coherent nonzero correspondence "
            "that the free phase tower did not convert into acceptable TTC"
        ),
        "evidence_sha256": sha(diagnostic),
        "prototype_proof_sha256": sha(proof),
        "parent_sha256": identity["parent"],
        "population": "P",
        "population_size": len(rows),
        "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
        "sensor_binding": identity["sensor_binding"],
        "mass_sha256": identity["mass_sha256"],
        "batch_size": 32,
        "microbatch_size": 4,
        "optimizer": config()["optimizer"],
        "objective": "unchanged primary pair phase objective",
        "sampler": "uniform with replacement, independent CPU generator seed7",
        "geometry": {
            "temperature": 0.02,
            "radius": 2,
            "scales": [1, 2],
            "ridge": 1e-4,
            "weight_floor": 1e-4,
            "weights": "softplus of four spatial logits; initial logits zero",
            "gates": "sigmoid axis/resolution gates; initial logits zero",
            "clock": "current TTC = dt / expm1(log_size_ratio)",
            "phase_extension": "native inverse-TTC phase, tangent above q=9.9",
            "input": "same native event P2 common ROI, no bbox/GT in forward",
        },
        "capacity": {
            "free": prototype["free_parameters"],
            "geometry": prototype["geometry_parameters"],
            "claim": "readout architecture contrast; not exact parameter-count parity",
        },
        "runtime": {
            "precision": "FP32_no_TF32",
            "cache_limit_bytes": 8 * 1024**3,
            "persistent_archives": True,
            "prefetch_batches": 1,
            "pinned_cuda_input": True,
            "cuda_graph": False,
            "graph_status": "requires separate parity/resume admission for this new graph",
        },
        "development_gates": config()["selection"],
        "evaluation": (
            "fixed endpoint before H; compare A5-P2 and reused free control on complete P/H"
        ),
        "replication": "not yet selected",
        "official_claim": False,
        "sources": source_binding(),
    }
    registry = RUN / "EXPERIMENT_REGISTRY.jsonl"
    existing = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    others = {
        row["experiment_id"]
        for row in existing
        if row["experiment_id"].startswith("V18_ADAPTIVE_")
        and row["experiment_id"] != record["experiment_id"]
    }
    if len(others) >= 4:
        raise ValueError("Adaptive experiment count exhausted")
    freeze(REGISTRATION, record)
    prior = [row for row in existing if row["experiment_id"] == record["experiment_id"]]
    if prior:
        if any(row["registration_sha256"] != sha(REGISTRATION) for row in prior):
            raise ValueError("Adaptive registry binding changed")
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
                        "maximum_synthetic_qa_updates": 40,
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
    """Fail closed on any implementation change after registration."""
    record = read(REGISTRATION)
    if record["sources"] != source_binding():
        raise ValueError("Geometry implementation changed after registration")
    return record


class SyntheticSensors(PersistentSensors):
    """The actual prefetch protocol over a separate synthetic population."""

    def __init__(self) -> None:
        self.population = 8
        self.data = fixture(128)
        self.executor = self.future = self.shadow = None
        self.pinned = False

    def load(self, indices: list[int]) -> ObjectEventV4Batch:
        batch = subset(self.data, indices)
        if self.pinned:
            from dataclasses import fields

            for field in fields(batch):
                value = getattr(batch, field.name)
                if isinstance(value, torch.Tensor):
                    setattr(batch, field.name, value.pin_memory())
        return batch


def segment(
    directory: Path,
    dataset: PersistentSensors,
    masses: np.ndarray,
    *,
    limit: int,
    deadline: float,
    device: str,
    synthetic: bool = False,
) -> dict:
    """Advance the same sampler, optimizer and full checkpoint path for QA and science."""
    registered()
    endpoint = 10 if synthetic else 12000
    identity = {
        "fit_id": FIT,
        "registration_sha256": sha(REGISTRATION),
        "endpoint": endpoint,
        "population": "synthetic8" if synthetic else "P",
        "device": device,
        "mass_sha256": digest(masses.tolist()),
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    setup(7)
    parent = initialize("A5-P2-FROZEN")
    initial = copy.deepcopy(parent.encoder.features.state_dict())
    model = GeometricPairMeasurement(parent).to(device).train()
    del parent
    optimizer, scheduler = make_optimizer(model, updates=12000)
    sampler = torch.Generator().manual_seed(7)
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor = state.restore(model, optimizer, scheduler, sampler) if state.pointer.exists() else None
    accounting()
    curve = cursor["curve"] if cursor else []
    timings = cursor["timings"] if cursor else []
    if cursor is None:
        state.save(
            model,
            optimizer,
            scheduler,
            sampler,
            {"curve": curve, "timings": timings},
            status="RUNNING",
        )
    dataset.prepare(sampler.get_state(), pinned=device == "cuda")
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
                subset(batch, list(range(i, i + 4))).to(torch.device(device))
                for i in range(0, 32, 4)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = backward(
                model, batches, torch.from_numpy(masses[ids]).float(), dataset.population
            )
            if state.completed == 0:
                gradients = {
                    name: float(p.grad.norm()) if p.grad is not None else 0.0
                    for name, p in model.encoder.named_parameters()
                }
                if not all(np.isfinite(list(gradients.values()))) or not any(gradients.values()):
                    raise ValueError("Geometry encoder gradient contract failed")
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
                print(f"{FIT} {state.completed}/{endpoint} loss={losses['loss']:.6g}", flush=True)
    finally:
        dataset.close()
    # Never publish potentially partial optimizer state after an exception.
    # Pending/non-durable journal entries are recovered from the previous checkpoint.
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
        raise ValueError("Geometry encoder did not change")
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


def qa(device: str) -> dict:
    """Account exactly 20 synthetic updates per device, continuous10 vs 5+restore+5."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Explicit CPU/CUDA QA device required")
    registered()
    committed_sources(source_binding())
    root = RUN / "qa/geometry" / device
    if (root / "FAIL.json").exists():
        raise ValueError("Retained geometry QA failure requires investigation")
    if (root / "PASS.json").exists():
        result = read(root / "PASS.json")
        if result["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Geometry QA registration changed")
        return result
    if accounting()["technical"] + 20 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient synthetic reserve")
    require_resources(starting=True)
    scope = (
        gpu_charge("geometry_synthetic_qa", reserve=600)
        if device == "cuda"
        else nullcontext(float("inf"))
    )
    with scope as deadline:
        for mode, limit in (("continuous", 10), ("split", 5), ("split", 10)):
            outcome = segment(
                root / mode,
                SyntheticSensors(),
                np.full(8, 1 / 8),
                limit=limit,
                deadline=deadline,
                device=device,
                synthetic=True,
            )
            if outcome["completed_updates"] != limit:
                raise RuntimeError("Geometry QA interrupted; retain and resume checkpoints")
    payloads = [
        torch.load(root / mode / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for mode in ("continuous", "split")
    ]
    checks = {
        key: exact(payloads[0][key], payloads[1][key])
        for key in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "sampler_generator_state",
            "torch_rng_state",
            "cuda_rng_state_all",
            "numpy_random_state",
            "python_random_state",
        )
    }
    checks["sample_ids_losses_norms"] = exact(
        payloads[0]["cursor"]["curve"], payloads[1]["cursor"]["curve"]
    )
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "registration_sha256": sha(REGISTRATION),
        "physical_updates": 20,
        "cuda_initialized": torch.cuda.is_initialized(),
        "utc": utc(),
    }
    freeze(root / f"{result['status']}.json", result)
    if result["status"] != "PASS":
        raise AssertionError("Geometry full-state resume mismatch")
    return result


def fit(dataset: PersistentSensors) -> dict:
    """Run one owned GPU scope after both resume proofs, preserving a persistent CPU cache."""
    record = registered()
    commit = committed_sources(source_binding())
    for device in ("cpu", "cuda"):
        proof = read(RUN / "qa/geometry" / device / "PASS.json")
        if proof["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Geometry resume proof belongs to another recipe")
    directory = DIRECTORY / "fits" / FIT
    receipt = directory / "CHECKPOINT_RECEIPT.json"
    if receipt.exists() and read(receipt)["status"] == "COMPLETE":
        value = read(receipt)
        if (
            value["completed_updates"] != 12000
            or sha(directory / "checkpoint_last.pt") != value["checkpoint_sha256"]
        ):
            raise ValueError("Geometry final endpoint changed")
        return value
    require_resources(starting=True)
    if accounting()["physical_upper"] + 12000 > config()["caps"]["physical"]:
        raise RuntimeError("Insufficient geometry physical budget")
    if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
        raise ValueError("Geometry sensory cohort changed")
    masses = training_mass(
        dataset.rows.target_ttc.to_numpy(), dataset.rows.group_id.to_numpy(), role="P"
    )
    if digest(masses.tolist()) != record["mass_sha256"]:
        raise ValueError("Geometry population masses changed")
    provenance = directory / "RUN_PROVENANCE.json"
    if not provenance.exists():
        freeze(
            provenance,
            {
                "experiment_id": record["experiment_id"],
                "fit_id": FIT,
                "git_commit": commit,
                "registration_sha256": sha(REGISTRATION),
                "seed": 7,
                "config_hash": digest(record),
                "dataset_manifest_hash": record["P_manifest_sha256"],
                "parent_sha256": record["parent_sha256"],
                "start_time": utc(),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "host": socket.gethostname(),
                "status": "see CHECKPOINT_RECEIPT.json",
            },
        )
    with gpu_charge("geometry_scientific_training", reserve=900) as deadline:
        result = segment(directory, dataset, masses, limit=12000, deadline=deadline, device="cuda")
        write(
            directory / "STATUS.json",
            {
                **result,
                "observed_utc": utc(),
                "end_time": utc() if result["status"] == "COMPLETE" else None,
                "gpu_name": torch.cuda.get_device_name(0),
                "cache": dataset.counters(),
            },
        )
        return result


@contextmanager
def exclusive_driver() -> Iterator[None]:
    """Prevent two geometry writers independently of the shared GPU ownership lock."""
    import msvcrt

    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "DRIVER.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "qa", "fit"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    with exclusive_driver():
        if args.action == "register":
            print(register())
        elif args.action == "qa":
            print(qa(args.device))
        else:
            sensors = PersistentSensors()
            try:
                print(fit(sensors))
            finally:
                sensors.shutdown()
