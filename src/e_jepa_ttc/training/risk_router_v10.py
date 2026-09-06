"""Fixed endpoint CPU float64 selector fits with complete deterministic resume."""

from __future__ import annotations

import os
import platform
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    PhaseLedger,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.frozen_expert_tables_v10 import (
    FrozenTable,
    ModelInputs,
    array_hash,
    feature_mask,
)
from e_jepa_ttc.models.simplex_risk_router import (
    SimplexRiskRouter,
    relative_cost_targets,
    weighted_cost_loss,
)


def batch_schedule(n: int, seed: int, outer: int, updates: int = 1500) -> np.ndarray:
    """Uniform repeated permutations, generated independently of optimizer RNG."""
    rng = np.random.default_rng(10 * seed + outer)
    count = updates * 256
    return np.concatenate([rng.permutation(n) for _ in range((count + n - 1) // n)])[
        :count
    ].reshape(updates, 256)


def scaler(features: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = mass / mass.sum()
    mean = np.sum(w[:, None] * features, axis=0)
    std = np.maximum(np.sqrt(np.sum(w[:, None] * (features - mean) ** 2, axis=0)), 1e-8)
    return mean, std


def normalized(
    inputs: ModelInputs, mean: np.ndarray, std: np.ndarray, arm: str
) -> tuple[np.ndarray, np.ndarray]:
    z = (inputs.features - mean) / std
    clipping = np.mean(np.abs(z) > 8, axis=0)
    return np.clip(z, -8, 8) * feature_mask(arm, z.shape[1]), clipping


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    receipt = binding(path)
    atomic_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def train_steps(
    *,
    inputs: ModelInputs,
    target_phase: np.ndarray,
    mass: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    arm: str,
    seed: int,
    outer: int,
    output: Path,
    identity: dict[str, Any],
    total_updates: int = 1500,
    stop_at: int | None = None,
    resume: bool = False,
    check: Callable[[], None] = lambda: None,
    ledger: PhaseLedger | None = None,
) -> dict[str, Any]:
    """Low-level engine; synthetic smoke is explicitly separated by identity."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if identity.get("purpose") not in ("qa", "scientific"):
        raise ValueError("missing fit purpose")
    if identity["purpose"] == "scientific" and total_updates != 1500:
        raise ValueError("scientific update budget is exactly 1500")
    if identity["purpose"] == "scientific":
        if ledger is None or ledger.state != "FITTING":
            raise ValueError("scientific engine requires active FITTING ledger")
        ledger.require("FITTING")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SimplexRiskRouter(
        inputs.features.shape[1], constrained="UNCONSTRAINED" not in arm
    ).double()
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    schedule = batch_schedule(len(mass), seed, outer, total_updates)
    z, clipping = normalized(inputs, mean, std, arm)
    x = torch.from_numpy(z.copy())
    phases = torch.from_numpy(inputs.phases.copy())
    targets = relative_cost_targets(torch.from_numpy(target_phase.copy()), phases)
    weights = torch.from_numpy(mass.copy())
    full_identity = {
        **identity,
        "arm": arm,
        "seed": seed,
        "outer": outer,
        "updates": total_updates,
        "schedule": array_hash(schedule),
        "x": array_hash(z),
        "phases": array_hash(inputs.phases),
        "target": array_hash(target_phase),
        "mass": array_hash(mass),
        "mean": array_hash(mean),
        "std": array_hash(std),
        "environment": dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device="cpu",
            dtype="float64",
        ),
    }
    lock = object_digest(full_identity)
    update = 0
    started_ns = time.time_ns()
    history: list[float] = []
    full_losses: dict[str, float] = {}
    pointer = output / "RESUME.json"
    if pointer.exists():
        if not resume:
            raise ValueError("fit exists; explicit resume required")
        import json

        saved = torch.load(
            verify(json.loads(pointer.read_text())), map_location="cpu", weights_only=False
        )
        if saved["identity"] != full_identity or saved["lock"] != lock:
            raise ValueError("resume scientific/input/optimizer identity changed")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if (
            not np.array_equal(saved["schedule"], schedule)
            or saved["schedule_offset"] != saved["update"]
        ):
            raise ValueError("resume schedule/offset mismatch")
        update = saved["update"]
        started_ns = saved["started_ns"]
        history = saved["history"]
        full_losses = saved["full_losses"]
    elif any(output.iterdir()):
        raise ValueError("partial output without complete resume receipt")
    end = total_updates if stop_at is None else stop_at
    if not update <= end <= total_updates:
        raise ValueError("invalid resume endpoint")

    def snapshot() -> dict[str, Any]:
        check()
        with torch.no_grad():
            if str(update) not in full_losses and (
                update in (0, 100, 500, 1000, 1500) or identity["purpose"] == "qa"
            ):
                full_losses[str(update)] = float(
                    weighted_cost_loss(model(x, phases), targets, weights, len(mass))
                )
        payload = dict(
            started_ns=started_ns,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            update=update,
            identity=full_identity,
            lock=lock,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            schedule=schedule,
            schedule_offset=update,
            mean=mean,
            std=std,
            history=history,
            full_losses=full_losses,
            clipping=clipping,
            torch_version=torch.__version__,
        )
        record = save_checkpoint(output / f"update{update:04d}.pt", payload)
        atomic_json(pointer, record)
        return record

    if update == 0:
        snapshot()
    for k in range(update, end):
        check()
        idx = torch.from_numpy(schedule[k])
        opt.zero_grad(set_to_none=True)
        loss = weighted_cost_loss(model(x[idx], phases[idx]), targets[idx], weights[idx], len(mass))
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite optimizer loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise ArithmeticError("nonfinite gradients")
        opt.step()
        update = k + 1
        history.append(float(loss.detach()))
        if update % 100 == 0:
            snapshot()
    endpoint = snapshot()
    atomic_json(
        output / "TRAINING_SUMMARY.json",
        dict(
            started_ns=started_ns,
            ended_ns=time.time_ns(),
            update=update,
            losses=history,
            full_losses=full_losses,
            identity=full_identity,
            endpoint=endpoint,
        ),
    )
    return endpoint


def fit(
    table: FrozenTable,
    inputs: ModelInputs,
    *,
    ledger: PhaseLedger,
    mean: np.ndarray,
    std: np.ndarray,
    arm: str,
    seed: int,
    outer: int,
    output: Path,
    identity: dict[str, Any],
    resume: bool,
    check: Callable[[], None],
) -> dict[str, Any]:
    """Only verified inner-OOF tables may enter the scientific engine."""
    ledger.require("FITTING")
    table.validate("inner_oof")
    return train_steps(
        inputs=inputs,
        target_phase=table.supervision.target_phase,
        mass=table.supervision.global_mass,
        mean=mean,
        std=std,
        arm=arm,
        seed=seed,
        outer=outer,
        output=output,
        identity=identity,
        resume=resume,
        check=check,
        ledger=ledger,
    )


def predict(
    endpoint: dict[str, Any], inputs: ModelInputs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply exact hard selection from verified tiny checkpoint bytes."""
    saved = torch.load(verify(endpoint), map_location="cpu", weights_only=False)
    identity = saved["identity"]
    arm = identity["arm"]
    if saved["update"] != identity["updates"]:
        raise ValueError("incomplete endpoint")
    model = SimplexRiskRouter(
        inputs.features.shape[1], constrained="UNCONSTRAINED" not in arm
    ).double()
    model.load_state_dict(saved["model"])
    model.eval()
    z, clipping = normalized(inputs, saved["mean"], saved["std"], arm)
    with torch.no_grad():
        cost = model(torch.from_numpy(z), torch.from_numpy(inputs.phases.copy())).numpy()
    selected = cost.argmin(1)
    result = inputs.expert_ttc[np.arange(len(cost)), selected]
    if not np.isfinite(cost).all() or not np.all((result <= -0.1) | (result > 0.1)):
        raise ValueError("invalid endpoint prediction")
    return cost, selected, result, clipping
