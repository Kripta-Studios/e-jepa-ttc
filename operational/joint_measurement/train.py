"""Fixed-endpoint V18 fits with a genuinely trainable encoder and durable resume."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import platform
import random
import socket
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.joint_measurement.loss import phase_components, training_mass
from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.simplex_t.phase import ttc_to_phase
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, RUN, accounting, config, gpu_charge, owner, require_resources
from .data import SensorDataset, cache_root
from .native_loss import effective_native_loss


def setup(seed: int) -> None:
    """Set the declared FP32 runtime and all optimization RNGs."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def native_config() -> tuple[CausalScaleTTCConfig, CausalScaleTTCLossConfig]:
    """Reuse the admitted native configuration without rebuilding defaults."""
    recipe = read(ROOT / "configs/rgb_port/producers.json")["producers"]["E_A5_MATCHED"]
    return CausalScaleTTCConfig(
        **recipe["model_config"], deterministic_transport_sampling=True
    ), CausalScaleTTCLossConfig(**recipe["loss_config"])


def initialize(fit_id: str, *, random_parent: bool = False) -> CausalScaleTTC | PairMeasurement:
    """Create a private native parent copy, then the registered intervention."""
    cfg, _ = native_config()
    native = CausalScaleTTC(cfg)
    if not random_parent:
        path = owner() / "fits/E_A5_MATCHED/checkpoint_last.pt"
        if sha(path) != read(RUN / "ADMISSION.json")["parent_sha256"]:
            raise ValueError("Parent changed before initialization")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        native.load_state_dict(payload["model_state_dict"], strict=True)
    if fit_id.startswith("NATIVE") or fit_id == "A5-P2-FROZEN":
        return native
    if not fit_id.startswith("PAIR-"):
        raise ValueError("Unregistered architecture")
    return PairMeasurement.from_native(
        native, use_correlation="DIRECT" not in fit_id, freeze_encoder="FROZEN" in fit_id
    )


def make_optimizer(model: CausalScaleTTC | PairMeasurement, *, updates: int) -> tuple:
    """Native and pair models receive the same declared backbone/rest LR groups."""
    recipe = config()["optimizer"]
    backbone = model.encoder.features if isinstance(model, CausalScaleTTC) else model.encoder
    ids = {id(p) for p in backbone.parameters()}
    groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad and id(p) in ids],
            "lr": recipe["backbone_lr"],
        },
        {
            "params": [p for p in model.parameters() if p.requires_grad and id(p) not in ids],
            "lr": recipe["readout_lr"],
        },
    ]
    optimizer = torch.optim.AdamW(
        [g for g in groups if g["params"]],
        weight_decay=recipe["weight_decay"],
        betas=tuple(recipe["betas"]),
        eps=recipe["eps"],
        foreach=False,
    )

    def schedule(step: int) -> float:
        warmup = min(recipe["warmup"], max(1, updates // 2))
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def backward(
    model: nn.Module,
    batches: list,
    weights: torch.Tensor | None,
    population: int,
    *,
    recompute: bool = True,
) -> dict:
    """Differentiate the registered effective batch exactly once per update."""
    if isinstance(model, CausalScaleTTC):
        _, loss_cfg = native_config()
        loss, components = effective_native_loss(model, batches, loss_cfg, recompute=recompute)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native objective")
        loss.backward()
        return {"loss": float(loss.detach()), **{k: float(v) for k, v in components.items()}}
    if weights is None:
        raise ValueError("Pair objective requires global P masses")
    size = sum(len(b.events) for b in batches)
    start = 0
    result = {"loss": 0.0, "emitted": 0.0, "raw": 0.0, "raw_emitted_gap": 0.0}
    for batch in batches:
        n = len(batch.events)
        raw = model(batch.events, batch.delta_t_s.reshape(n))["raw_phase"]
        terms = phase_components(raw, ttc_to_phase(batch.target_ttc_s))
        scaled = population * weights[start : start + n].to(raw.device)
        loss = ((terms["emitted"] + terms["raw"]) * scaled).sum() / size
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite pair objective")
        loss.backward()
        result["loss"] += float(loss.detach())
        for name, value in terms.items():
            result[name] += float((value.detach() * scaled).sum() / size)
        start += n
    return result


def sources() -> dict[str, str]:
    """Bind scientific inputs once per fit, not on the hot update path."""
    paths = list((ROOT / "src/e_jepa_ttc/joint_measurement").glob("*.py"))
    paths += [
        ROOT / "operational/joint_measurement" / name
        for name in ("train.py", "data.py", "native_loss.py", "common.py", "qa.py")
    ]
    paths += [
        ROOT / p
        for p in (
            "configs/joint_measurement/primary.json",
            "src/e_jepa_ttc/simplex_t/phase.py",
            "src/e_jepa_ttc/models/causal_scale_ttc.py",
            "src/e_jepa_ttc/models/local_transport.py",
            "src/e_jepa_ttc/losses/causal_scale_ttc.py",
            "operational/rgb_port/train_producers.py",
            "operational/rgb_port_revision/loss.py",
        )
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def committed_sources(binding: dict[str, str]) -> str:
    """Require an actual commit of the sources before scientific updates."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    for relative in binding:
        expected = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if expected.replace(b"\r\n", b"\n") != (ROOT / relative).read_bytes().replace(
            b"\r\n", b"\n"
        ):
            raise ValueError(f"Commit exact scientific source before training: {relative}")
    return commit


def fit(fit_id: str, *, max_updates: int | None = None) -> dict:
    """Train/resume one primary endpoint; never select a checkpoint on H/V."""
    recipe = config()
    if fit_id not in recipe["fits"]:
        raise ValueError("Fit is not in the registered primary matrix")
    directory = RUN / "fits" / fit_id
    endpoint = recipe["updates"]
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    if receipt_path.exists() and read(receipt_path)["status"] == "COMPLETE":
        receipt = read(receipt_path)
        if (
            receipt["completed_updates"] != endpoint
            or sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Completed endpoint changed")
        return receipt
    require_resources(starting=True)
    book = accounting()
    if book["physical_upper"] + endpoint > recipe["caps"]["physical"]:
        raise RuntimeError("Insufficient physical budget for registered endpoint")
    binding = sources()
    for tested_fit in recipe["fits"]:
        qa_receipt = read(RUN / "qa/cuda" / tested_fit / "PASS.json")
        if qa_receipt["source_hash"] != digest(binding):
            raise ValueError("CUDA resume checks must pass on the scientific source version")
    commit = committed_sources(binding)
    dataset = SensorDataset("P")
    masses = training_mass(
        dataset.rows.target_ttc.to_numpy(), dataset.rows.group_id.to_numpy(), role="P"
    )
    identity = {
        "fit_id": fit_id,
        "seed": recipe["seed"],
        "updates": endpoint,
        "sources": binding,
        "config_hash": digest(recipe),
        "sensor_binding": sha(cache_root("P") / "COMPLETE.json"),
        "parent": read(RUN / "ADMISSION.json")["parent_sha256"],
        "mass_sha256": digest(masses.tolist()),
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    started = utc()
    setup(recipe["seed"])
    model = initialize(fit_id)
    initial_backbone = copy.deepcopy(
        (
            model.encoder.features if isinstance(model, CausalScaleTTC) else model.encoder
        ).state_dict()
    )
    curves: list[dict] = []
    sampler = torch.Generator().manual_seed(recipe["seed"])
    state = ProducerCheckpoint(directory, identity, endpoint)
    try:
        with gpu_charge("scientific_training/" + fit_id, reserve=900) as deadline:
            model = model.to("cuda").train()
            optimizer, scheduler = make_optimizer(model, updates=endpoint)
            cursor = (
                state.restore(model, optimizer, scheduler, sampler)
                if state.pointer.exists()
                else None
            )
            # Restore may add non-durable/pending work to recovery_upper.
            # Enforce the V18 recovery cap before another optimizer boundary.
            accounting()
            if cursor is not None:
                curves = cursor["curves"]
            else:
                state.save(
                    model, optimizer, scheduler, sampler, {"curves": curves}, status="RUNNING"
                )
            provenance = directory / "RUN_PROVENANCE.json"
            if not provenance.exists():
                freeze(
                    provenance,
                    {
                        "experiment_id": "V18/" + fit_id,
                        "git_commit": commit,
                        "identity": identity["identity_sha256"],
                        "start_time": started,
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "host": socket.gethostname(),
                        "status": "see CHECKPOINT_RECEIPT.json",
                    },
                )
            limit = min(endpoint, state.completed + max_updates) if max_updates else endpoint
            while state.completed < limit and time.perf_counter() < deadline:
                if state.completed % 20 == 0:
                    try:
                        require_resources()
                    except RuntimeError:
                        break
                tick = time.perf_counter()
                ids = torch.randint(
                    dataset.population, (recipe["batch_size"],), generator=sampler
                ).tolist()
                batch = dataset.batch(ids)
                batches = []
                from dataclasses import fields

                from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

                for start in range(0, len(ids), recipe["microbatch_size"]):
                    stop = start + recipe["microbatch_size"]
                    values: dict[str, Any] = {
                        f.name: (
                            getattr(batch, f.name)[start:stop]
                            if getattr(batch, f.name) is not None
                            else None
                        )
                        for f in fields(batch)
                    }
                    batches.append(ObjectEventV4Batch(**values).to(torch.device("cuda")))
                optimizer.zero_grad(set_to_none=True)
                losses = backward(
                    model, batches, torch.from_numpy(masses[ids]).float(), dataset.population
                )
                if state.completed == 0:
                    backbone = (
                        model.encoder.features
                        if isinstance(model, CausalScaleTTC)
                        else model.encoder
                    )
                    witness = {
                        name: {
                            "requires_grad": value.requires_grad,
                            "has_gradient": value.grad is not None,
                            "gradient_l2": (
                                float(value.grad.norm()) if value.grad is not None else 0.0
                            ),
                            "finite": (
                                bool(torch.isfinite(value.grad).all())
                                if value.grad is not None
                                else True
                            ),
                        }
                        for name, value in backbone.named_parameters()
                    }
                    nonzero = any(v["gradient_l2"] > 0 for v in witness.values())
                    if not all(v["finite"] for v in witness.values()) or nonzero == (
                        "FROZEN" in fit_id
                    ):
                        raise ValueError("Encoder gradient contract failed before optimizer update")
                    write(directory / "ENCODER_GRADIENT.json", witness)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                state.begin()
                optimizer.step()
                scheduler.step()
                state.commit()
                curves.append(
                    {
                        "update": state.completed,
                        **losses,
                        "grad_norm": float(norm),
                        "clipped": float(norm) > 1,
                        "sample_ids": ids,
                        "seconds": time.perf_counter() - tick,
                    }
                )
                if state.completed % 100 == 0:
                    state.save(
                        model, optimizer, scheduler, sampler, {"curves": curves}, status="RUNNING"
                    )
                    accounting()
                    print(
                        f"{fit_id} {state.completed}/{endpoint} loss={losses['loss']:.6g}",
                        flush=True,
                    )
            state.save(
                model,
                optimizer,
                scheduler,
                sampler,
                {"curves": curves},
                status="COMPLETE" if state.completed == endpoint else "PAUSED",
            )
            encoder = model.encoder.features if isinstance(model, CausalScaleTTC) else model.encoder
            deltas = {
                k: float((v.detach().cpu() - initial_backbone[k]).abs().max())
                for k, v in encoder.state_dict().items()
            }
            changed = max(deltas.values()) > 0
            if state.completed and changed == ("FROZEN" in fit_id):
                raise ValueError("Encoder weight-change contract failed")
            write(
                directory / "ENCODER_CHANGE.json",
                {"changed": changed, "maxabs": deltas, "completed_updates": state.completed},
            )
    finally:
        dataset.close()
        accounting()
    if curves:
        keys = [k for k in curves[0] if k != "sample_ids"]
        with (directory / "CURVE.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row[k] for k in keys} for row in curves)
    return read(receipt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_id", choices=config()["fits"])
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    print(fit(args.fit_id, max_updates=args.max_updates))
