"""Registered paired RGB adaptation, admitted only after primary P/H evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import socket
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .evaluate import freeze_primary
from .qa import exact, fixture, subset
from .rgb_data import RGBPairDataset
from .rgb_model import load_parent, pair_from_parent
from .train import backward, committed_sources, make_optimizer, setup, sources

REGISTRATION = RUN / "rgb/REGISTRATION.json"


def source_binding() -> dict:
    """Bind the reused objective plus this RGB implementation and input adapter."""
    return {
        **sources(),
        **{
            f"operational/joint_measurement/{name}": sha(
                ROOT / "operational/joint_measurement" / name
            )
            for name in ("rgb_train.py", "rgb_model.py", "rgb_data.py")
        },
        "src/e_jepa_ttc/rgb_port/data.py": sha(ROOT / "src/e_jepa_ttc/rgb_port/data.py"),
    }


def register(variant: str, reason: str) -> dict:
    """Freeze one RGB FT/frozen pair after the complete primary comparison exists."""
    if variant not in {"DIRECT", "CORR"} or not reason.strip():
        raise ValueError("Declare DIRECT/CORR and an evidence-based reason")
    freeze_primary()
    decision = RUN / "PRIMARY_SELECTION_FINAL.json"
    if not decision.is_file():
        raise ValueError("Complete primary P/H and cost interpretation before RGB registration")
    evidence = {str(decision.relative_to(RUN)): sha(decision)}
    for role in ("P", "H"):
        for fit_id in ["A5-P2-FROZEN", *config()["fits"]]:
            directory = RUN / "predictions" / role / fit_id
            receipt = read(directory / "COMPLETE.json")
            if (
                receipt["status"] != "COMPLETE"
                or sha(directory / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("RGB recipe requires intact complete primary P/H evidence")
            for name in ("COMPLETE.json", "METRICS.json"):
                path = directory / name
                evidence[str(path.relative_to(RUN))] = sha(path)
    _, parent = load_parent()
    dataset = RGBPairDataset("P")
    try:
        cache = dataset.source.prepared_cache
        assert cache is not None
        input_identity = {
            "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
            "P_rows_sha256": sha(dataset.source.rows_path),
            "cache_binding_sha256": cache.binding_sha256,
            "cache_manifest_sha256": cache.manifest_sha256,
            "population": dataset.population,
        }
    finally:
        dataset.close()
    result = {
        "experiment_id": "V18_RGB_PAIR",
        "variant": variant,
        "reason": reason,
        "evidence": evidence,
        "fits": [f"RGB-PAIR-{variant}-{mode}-S7" for mode in ("FT", "FROZEN")],
        "seed": 7,
        "updates_per_fit": 12000,
        "maximum_scientific_updates": 24000,
        "fusion_reserve_updates": 5000,
        "population": "P",
        "selection_population": "H_after_both_endpoints",
        "input": "last two real RGB frames, common ROI, native RGB sensor delta, query anchor",
        "input_identity": input_identity,
        "parent": parent,
        "batch_size": 32,
        "microbatch_size": 4,
        "precision": "FP32_no_TF32",
        "objective": "same V18 pair phase objective and P group/band masses",
        "optimizer": config()["optimizer"],
        "sampler": "uniform_with_replacement_shared_seed7",
        "control": "same new modules and objective; frozen encoder in eval mode",
        "development_gates": {
            **{
                k: v
                for k, v in config()["selection"].items()
                if k not in {"common_reference", "final_promotion"}
            },
            "common_reference": "R_A5-P2-FROZEN",
        },
        "reference": "R_A5-P2-FROZEN",
        "promotion": "requires full RGB/event/fusion comparison; no official claim",
        "sources": source_binding(),
    }
    freeze(REGISTRATION, result)
    registry = RUN / "EXPERIMENT_REGISTRY.jsonl"
    existing = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    records = [row for row in existing if row["experiment_id"] == result["experiment_id"]]
    if records:
        if any(row.get("registration_sha256") != sha(REGISTRATION) for row in records):
            raise ValueError("RGB registry disagrees with its frozen registration")
    else:
        with registry.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "experiment_id": result["experiment_id"],
                        "registered_utc": utc(),
                        "status": "REGISTERED_BEFORE_UPDATES",
                        "registration_sha256": sha(REGISTRATION),
                        "maximum_scientific_updates": 24000,
                        "maximum_synthetic_qa_updates": 40,
                        "fits": result["fits"],
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return result


def registered() -> dict:
    """Fail closed if the recipe or any implementation changed after registration."""
    record = read(REGISTRATION)
    if record["sources"] != source_binding():
        raise ValueError("RGB source differs from the registered implementation")
    return record


def segment(
    fit_id: str,
    directory: Path,
    dataset: RGBPairDataset | SyntheticRGB,
    masses: np.ndarray,
    *,
    limit: int,
    deadline: float,
    device: str,
    synthetic: bool = False,
) -> dict:
    """Advance a bounded durable trajectory using the same loop for QA and science."""
    record = registered()
    if fit_id not in record["fits"]:
        raise ValueError("Unregistered RGB fit")
    endpoint = 10 if synthetic else record["updates_per_fit"]
    identity = {
        "fit_id": fit_id,
        "registration_sha256": sha(REGISTRATION),
        "population": "synthetic8" if synthetic else "P",
        "endpoint": endpoint,
        "mass_sha256": digest(masses.tolist()),
        "device": device,
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    setup(record["seed"])
    parent, parent_identity = load_parent()
    if parent_identity != record["parent"]:
        raise ValueError("RGB parent changed after registration")
    model = (
        pair_from_parent(
            parent,
            correlation=record["variant"] == "CORR",
            frozen="FROZEN" in fit_id,
            seed=record["seed"],
        )
        .to(device)
        .train()
    )
    initial = copy.deepcopy(parent.encoder.features.state_dict())
    optimizer, scheduler = make_optimizer(model, updates=record["updates_per_fit"])
    sampler = torch.Generator().manual_seed(record["seed"])
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor = state.restore(model, optimizer, scheduler, sampler) if state.pointer.exists() else None
    accounting()
    curve = cursor["curve"] if cursor is not None else []
    timings = cursor["timings"] if cursor is not None else []
    if cursor is None:
        state.save(
            model,
            optimizer,
            scheduler,
            sampler,
            {"curve": curve, "timings": timings},
            status="RUNNING",
        )
    batch_size, micro = (4, 2) if synthetic else (32, 4)
    while state.completed < min(limit, endpoint) and time.perf_counter() < deadline:
        if state.completed % 20 == 0:
            try:
                require_resources()
            except RuntimeError:
                break
        tick = time.perf_counter()
        ids = torch.randint(dataset.population, (batch_size,), generator=sampler).tolist()
        batch = dataset.batch(ids)
        batches = [
            subset(batch, list(range(i, i + micro))).to(torch.device(device))
            for i in range(0, batch_size, micro)
        ]
        optimizer.zero_grad(set_to_none=True)
        losses = backward(model, batches, torch.from_numpy(masses[ids]).float(), dataset.population)
        if state.completed == 0:
            gradients = {
                name: float(p.grad.norm()) if p.grad is not None else 0.0
                for name, p in model.encoder.named_parameters()
            }
            if not all(np.isfinite(list(gradients.values()))) or (
                any(value > 0 for value in gradients.values()) == ("FROZEN" in fit_id)
            ):
                raise ValueError("RGB encoder gradient contract failed")
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
            print(f"{fit_id} {state.completed}/{endpoint} loss={losses['loss']:.6g}", flush=True)
    state.save(
        model,
        optimizer,
        scheduler,
        sampler,
        {"curve": curve, "timings": timings},
        status="COMPLETE" if state.completed == endpoint else "PAUSED",
    )
    changes = {
        k: float((v.detach().cpu() - initial[k]).abs().max())
        for k, v in model.encoder.state_dict().items()
    }
    if state.completed and (max(changes.values()) > 0) == ("FROZEN" in fit_id):
        raise ValueError("RGB encoder change contract failed")
    write(directory / "ENCODER_CHANGE.json", {"updates": state.completed, "maxabs": changes})
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
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    write(
        directory / "STATUS.json",
        {
            "fit_id": fit_id,
            "status": receipt["status"],
            "completed_updates": state.completed,
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "end_time": utc() if receipt["status"] == "COMPLETE" else None,
            "observed_utc": utc(),
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "precision": "FP32_no_TF32",
            "scope": "synthetic_qa" if synthetic else "RGB_P_scientific_training",
        },
    )
    return receipt


class SyntheticRGB:
    """Small RGB-shaped population for accounted optimizer/resume QA only."""

    population = 8

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20261012)
        self.data = replace(fixture(), events=torch.rand(8, 2, 3, 32, 32, generator=generator))

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        return subset(self.data, indices)


def qa() -> None:
    """Require exact CUDA 10 versus 5+restore+5 for both registered RGB arms."""
    record = registered()
    committed_sources(source_binding())
    require_resources(starting=True)
    if accounting()["technical"] + 40 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient synthetic update reserve for RGB QA")
    for fit_id in record["fits"]:
        root = RUN / "qa/rgb_cuda" / fit_id
        if (root / "FAIL.json").exists():
            raise AssertionError("Retained RGB CUDA QA failure requires review")
        if (root / "PASS.json").exists():
            if read(root / "PASS.json")["registration_sha256"] != sha(REGISTRATION):
                raise ValueError("RGB CUDA QA belongs to another registration")
            continue
        with gpu_charge("rgb_synthetic_qa/" + fit_id, reserve=300) as deadline:
            for mode, limit in (("continuous", 10), ("split", 5), ("split", 10)):
                result = segment(
                    fit_id,
                    root / mode,
                    SyntheticRGB(),
                    np.full(8, 1 / 8),
                    limit=limit,
                    deadline=deadline,
                    device="cuda",
                    synthetic=True,
                )
                if result["completed_updates"] != limit:
                    raise RuntimeError("RGB QA resource/deadline interruption; retain evidence")
            payloads = [
                torch.load(
                    root / mode / "checkpoint_last.pt", map_location="cpu", weights_only=False
                )
                for mode in ("continuous", "split")
            ]
            keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "sampler_generator_state",
                "torch_rng_state",
                "cuda_rng_state_all",
                "numpy_random_state",
                "python_random_state",
            ]
            checks = {key: exact(payloads[0][key], payloads[1][key]) for key in keys}
            checks["sample_ids_losses_norms"] = exact(
                payloads[0]["cursor"]["curve"], payloads[1]["cursor"]["curve"]
            )
        result = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "registration_sha256": sha(REGISTRATION),
            "physical_updates": 20,
            "utc": utc(),
        }
        freeze(root / f"{result['status']}.json", result)
        if result["status"] != "PASS":
            raise AssertionError("RGB exact CUDA resume failed")


def fit(fit_id: str) -> dict:
    """Train one registered RGB endpoint in one charged scope, with exact continuation."""
    record = registered()
    if fit_id not in record["fits"]:
        raise ValueError("Unregistered RGB fit")
    for name in record["fits"]:
        proof = read(RUN / "qa/rgb_cuda" / name / "PASS.json")
        if proof["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("RGB CUDA resume proof changed")
    commit = committed_sources(source_binding())
    directory = RUN / "rgb/fits" / fit_id
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    if receipt_path.exists() and read(receipt_path)["status"] == "COMPLETE":
        receipt = read(receipt_path)
        if (
            receipt["completed_updates"] != 12000
            or sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError("RGB endpoint differs")
        return receipt
    require_resources(starting=True)
    if accounting()["physical_upper"] + 12000 > config()["caps"]["physical"]:
        raise RuntimeError("Insufficient physical budget for RGB endpoint")
    dataset = RGBPairDataset("P")
    try:
        cache = dataset.source.prepared_cache
        assert cache is not None
        actual = {
            "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
            "P_rows_sha256": sha(dataset.source.rows_path),
            "cache_binding_sha256": cache.binding_sha256,
            "cache_manifest_sha256": cache.manifest_sha256,
            "population": dataset.population,
        }
        if actual != record["input_identity"]:
            raise ValueError("RGB sensory identity changed after registration")
        masses = training_mass(
            dataset.rows.target_ttc.to_numpy(), dataset.rows.group_id.to_numpy(), role="P"
        )
        provenance = directory / "RUN_PROVENANCE.json"
        if not provenance.exists():
            freeze(
                provenance,
                {
                    "git_commit": commit,
                    "registration_sha256": sha(REGISTRATION),
                    "parent": record["parent"],
                    "start_time": utc(),
                    "experiment_id": record["experiment_id"],
                    "fit_id": fit_id,
                    "seed": record["seed"],
                    "config_hash": digest(record),
                    "dataset_manifest_hash": record["input_identity"]["P_manifest_sha256"],
                    "python_version": platform.python_version(),
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "host": socket.gethostname(),
                    "status": "see CHECKPOINT_RECEIPT.json",
                },
            )
        with gpu_charge("rgb_scientific_training/" + fit_id, reserve=900) as deadline:
            return segment(
                fit_id, directory, dataset, masses, limit=12000, deadline=deadline, device="cuda"
            )
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "qa", "fit"])
    parser.add_argument("--variant", choices=["DIRECT", "CORR"])
    parser.add_argument("--reason")
    parser.add_argument("--fit-id")
    args = parser.parse_args()
    if args.action == "register":
        print(register(args.variant, args.reason or ""))
    elif args.action == "qa":
        qa()
    else:
        print(fit(args.fit_id))
