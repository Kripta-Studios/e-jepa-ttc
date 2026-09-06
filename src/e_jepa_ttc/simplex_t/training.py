"""FP32 fixed-update head engine with deterministic sampler and exact CPU resume.

The caller must validate real caches, lineage, frozen identities and resource
leases. This engine has no raw loader and cannot make those assertions itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .model import TemporalConfig, TemporalRefiner, training_loss


class QuerySource(Protocol):
    """Uniform-query gather from prevalidated deduplicated features, never HDF5."""

    population: int
    identity_sha256: str

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return features, timing, valid, original experts, target and global mass."""
        ...


def learning_rate(update: int) -> float:
    """One-based update: warmup100, cosine to exactly 3e-5 at update2500."""
    if not 1 <= update <= 2500:
        raise ValueError("update outside frozen endpoint")
    if update <= 100:
        return 3e-4 * update / 100
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * (update - 100) / 2400)) / 2


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Publish a complete checkpoint with durable bytes before atomic replace."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fit(
    source: QuerySource,
    config: TemporalConfig,
    output: Path,
    *,
    seed: int,
    freeze_sha256: str,
    resource_ok: Callable[[], bool],
    resume: bool = False,
    stop_after: int = 2500,
) -> dict[str, Any]:
    """Train to an exact endpoint or save a resource pause; never score partials.

    stop_after is for accounted technical/resume probes only. Production callers
    must keep 2500. This implementation is CPU-only; changing device requires an
    explicit pre-freeze numerical recipe and device-specific resume validation.
    """
    if not 1 <= stop_after <= 2500 or seed not in {7, 13, 23} or source.population < 1:
        raise ValueError("invalid registered fit arguments")
    if len(freeze_sha256) != 64 or len(source.identity_sha256) != 64:
        raise ValueError("frozen source and implementation identity required")
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = TemporalRefiner(config).float().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = {
        "source": source.identity_sha256,
        "freeze": freeze_sha256,
        "config": asdict(config),
        "seed": seed,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint_last.pt"
    completed = 0
    losses: list[float] = []
    sampler_hashes: list[str] = []
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["identity_sha256"] != identity_hash:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        completed = state["completed_updates"]
        losses = state["losses"]
        sampler_hashes = state["sampler_hashes"]
    elif checkpoint.exists():
        raise FileExistsError("existing fit requires explicit resume")
    if completed > stop_after:
        raise ValueError("cannot rewind fixed fit")

    def save(status: str) -> dict[str, Any]:
        state = {
            "identity": identity,
            "identity_sha256": identity_hash,
            "completed_updates": completed,
            "status": status,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "losses": losses,
            "sampler_hashes": sampler_hashes,
        }
        atomic_checkpoint(checkpoint, state)
        return {
            "status": status,
            "completed_updates": completed,
            "scientific_endpoint": completed == 2500,
            "identity_sha256": identity_hash,
        }

    model.train()
    for update in range(completed + 1, stop_after + 1):
        if not resource_ok():
            return save("PAUSED_RESOURCE")
        ids = torch.randint(source.population, (128,), generator=generator)
        x, timing, valid, experts, truth, mass = source.gather(ids)
        if any(value.device.type != "cpu" for value in (x, timing, valid, experts, truth, mass)):
            raise ValueError("CPU recipe only")
        for value in (x, timing, experts, truth, mass):
            if value.dtype != torch.float32:
                raise ValueError("FP32 training tensors required")
        optimizer.param_groups[0]["lr"] = learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(
            model(x, timing, valid, experts),
            truth,
            experts,
            mass,
            source.population,
            selector_only=config.output_mode == "selector",
        )
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = update
        losses.append(float(loss.detach()))
        sampler_hashes.append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
        if update % 100 == 0:
            save("IN_PROGRESS")
        if update % 25 == 0:
            print(json.dumps({"update": update, "loss": losses[-1]}), flush=True)
    return save("COMPLETED" if completed == 2500 else "TECHNICAL_PARTIAL")
