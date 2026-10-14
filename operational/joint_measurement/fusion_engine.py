"""Full-state fusion updates over resident inputs; no parent or recipe selection.

The scientific caller must admit a registered recipe, feature bytes, H-only
normalizers, labels and masses, and own the writer/GPU scope before calling.
Synthetic QA uses the identical optimizer/checkpoint path and is accounted.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import ttc_to_phase
from operational.range_transfer.common import digest, freeze, read, sha
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, accounting, require_resources
from .fusion_data import FusionTensorBank
from .fusion_model import RGBAnchorComplement
from .train import setup


@dataclass(frozen=True)
class FusionRecipe:
    """Numerical fields to register before updates; no scientific defaults."""

    seed: int
    mode: str
    updates: int
    batch_size: int
    hidden: int
    residual_scale: float
    learning_rate: float
    weight_decay: float
    clip_norm: float

    def validate(self) -> None:
        """Reject malformed recipes before model/device allocation."""
        for value in (self.seed, self.updates, self.batch_size, self.hidden):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Fusion seeds and dimensions must be integers")
        if self.seed < 0 or min(self.updates, self.batch_size, self.hidden) < 1:
            raise ValueError("Invalid fusion seed, endpoint or dimensions")
        values = (self.residual_scale, self.learning_rate, self.weight_decay, self.clip_norm)
        if (
            self.mode not in {"TRUE", "ZERO"}
            or not all(math.isfinite(value) for value in values)
            or not 0 < self.residual_scale <= 1
            or min(self.learning_rate, self.clip_norm) <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid numerical fusion recipe")


def sources() -> dict[str, str]:
    """Bind the actual engine, resident features, model and checkpoint writer."""
    paths = [
        f"operational/joint_measurement/{name}.py"
        for name in ("fusion_engine", "fusion_data", "fusion_features", "fusion_model", "train")
    ] + [
        "operational/rgb_port/train_producers.py",
        "src/e_jepa_ttc/simplex_t/phase.py",
        "src/e_jepa_ttc/rgb_port/normalization.py",
    ]
    return {path: sha(ROOT / path) for path in paths}


def array_digest(value: np.ndarray) -> str:
    """Bind exact array shape, dtype and bytes, once per durable admission."""
    array = np.ascontiguousarray(value)
    checksum = hashlib.sha256()
    checksum.update(str(array.dtype).encode())
    checksum.update(str(array.shape).encode())
    checksum.update(array.tobytes())
    return checksum.hexdigest()


def backward(
    model: RGBAnchorComplement,
    batch: dict[str, torch.Tensor],
    target_phase: torch.Tensor,
    masses: torch.Tensor,
    population: int,
) -> dict[str, float]:
    """Uniform-sampler estimate of global-mass emitted/raw phase L1.

    Missing RGB outputs remain exact event fallback and contribute constant loss.
    Explicit zero gradients keep the same AdamW/scheduler update semantics even
    for an all-missing batch; no samples are silently redrawn or removed.
    """
    output = model(**batch)
    if (
        target_phase.shape != output["raw_phase"].shape
        or masses.shape != target_phase.shape
        or population < 1
        or not torch.isfinite(target_phase).all()
        or not torch.isfinite(masses).all()
        or (masses < 0).any()
    ):
        raise ValueError("Invalid aligned fusion supervision")
    emitted = (output["point_phase"] - target_phase).abs() / 0.03
    raw = 0.1 * (output["raw_phase"] - target_phase).abs() / 0.03
    weight = population * masses
    loss = ((emitted + raw) * weight).mean()
    # Every parameter participates, including the all-fallback constant case.
    for parameter in model.parameters():
        loss = loss + parameter.sum() * 0.0
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite fusion objective")
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "emitted": float((emitted * weight).mean().detach()),
        "raw": float((raw * weight).mean().detach()),
        "rgb_rows": int(batch["rgb_available"].sum()),
    }


def segment(
    directory: Path,
    bank: FusionTensorBank,
    target_ttc: np.ndarray,
    masses: np.ndarray,
    recipe: FusionRecipe,
    *,
    binding: dict,
    limit: int,
    deadline: float,
) -> dict:
    """Advance an admitted H head or synthetic QA, preserving every state on resume.

    Binding must come from the registered caller, never from feature metadata
    alone. This engine additionally freezes the actual normalized tensors,
    ordered population, supervision, recipe and code. It does not choose masses.
    """
    recipe.validate()
    if bank.role != "H" or not binding or not 0 <= limit <= recipe.updates:
        raise ValueError("Fusion training requires admitted H inputs and a bounded endpoint")
    target_ttc = np.asarray(target_ttc)
    masses = np.asarray(masses)
    if (
        target_ttc.shape != (bank.population,)
        or masses.shape != target_ttc.shape
        or not np.isfinite(target_ttc).all()
        or np.any(target_ttc == 0)
        or not np.isfinite(masses).all()
        or np.any(masses < 0)
        or not np.isclose(masses.sum(), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("Invalid full-population fusion labels or normalized masses")
    identity = {
        "fit_id": f"FUSION-{recipe.mode}-S{recipe.seed}",
        "binding": binding,
        "recipe": asdict(recipe),
        "sources": sources(),
        "role": bank.role,
        "manifest_sha256": bank.manifest_sha256,
        "sample_tokens_sha256": array_digest(bank.sample_tokens),
        "input_sha256": {
            name: array_digest(value.detach().cpu().numpy()) for name, value in bank.inputs.items()
        },
        "target_sha256": array_digest(target_ttc),
        "mass_sha256": array_digest(masses),
        "device": str(bank.device),
        "objective": "global mass emitted phase L1 + 0.1 raw L1, divided by 0.03",
        "sampler": "uniform with replacement over every H row",
        "all_missing_batch": "constant fallback loss; zero gradients; regular AdamW step",
        "scheduler": "CosineAnnealingLR T_max=endpoint eta_min=0",
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    setup(recipe.seed)
    model = (
        RGBAnchorComplement(
            bank.inputs["rgb_features"].shape[1],
            bank.inputs["event_features"].shape[1],
            hidden=recipe.hidden,
            residual_scale=recipe.residual_scale,
            mode=recipe.mode,
        )
        .to(bank.device)
        .train()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.updates)
    sampler = torch.Generator().manual_seed(recipe.seed)
    state = ProducerCheckpoint(directory, identity, recipe.updates)
    cursor = state.restore(model, optimizer, scheduler, sampler) if state.pointer.exists() else None
    curve = cursor["curve"] if cursor else []
    timings = cursor["timings"] if cursor else []
    accounting()
    if cursor is None:
        state.save(
            model, optimizer, scheduler, sampler, {"curve": [], "timings": []}, status="RUNNING"
        )
    targets = ttc_to_phase(torch.tensor(target_ttc, dtype=torch.float32, device=bank.device))
    weights = torch.tensor(masses, dtype=torch.float32, device=bank.device)
    while state.completed < limit and time.perf_counter() < deadline:
        if state.completed % 20 == 0:
            try:
                require_resources()
            except RuntimeError as error:
                if str(error).startswith("RESOURCE_WAIT"):
                    break
                raise
        start = time.perf_counter()
        indices = torch.randint(bank.population, (recipe.batch_size,), generator=sampler)
        selected = indices.to(bank.device)
        batch = bank.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        losses = backward(model, batch, targets[selected], weights[selected], bank.population)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), recipe.clip_norm, error_if_nonfinite=True
        )
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        curve.append(
            {"update": state.completed, "ids": indices.tolist(), **losses, "norm": float(norm)}
        )
        timings.append(time.perf_counter() - start)
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
    # Exceptions never save possibly partial optimizer state. The existing
    # producer journal accounts recovery from the previous durable checkpoint.
    state.save(
        model,
        optimizer,
        scheduler,
        sampler,
        {"curve": curve, "timings": timings},
        status="COMPLETE" if state.completed == recipe.updates else "PAUSED",
    )
    accounting()
    return read(directory / "CHECKPOINT_RECEIPT.json")
