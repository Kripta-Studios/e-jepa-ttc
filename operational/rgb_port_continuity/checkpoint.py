"""Idempotent lifecycle saves outside the complete stack of checkpoint wrappers."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file


def equal(left: Any, right: Any) -> bool:  # noqa: ANN401
    """Compare complete nested state, including tensor dtypes and RNG arrays."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and bool(left == right)


def reuse_checkpoint(
    state: training.ProducerCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    cursor: Mapping[str, Any],
    *,
    status: str,
) -> bool:
    """Update only lifecycle receipt if the entire durable training state is identical.

    A changed state at the same update is rejected before any native or auxiliary
    write. In particular, this never hides changed weights or discards advanced RNG.
    """
    version = state.versions / f"checkpoint_{state.completed:06d}.pt"
    if not version.exists():
        return False
    if state.pending or state.completed != state.durable:
        raise RuntimeError("Existing checkpoint is not a complete durable optimizer boundary")
    pointer, receipt = read_json_shared(state.pointer), read_json_shared(state.receipt)
    digest = sha256_file(version)
    if (
        pointer.get("version") != version.name
        or pointer.get("checkpoint_sha256") != digest
        or receipt.get("checkpoint_sha256") != digest
        or sha256_file(state.path) != digest
        or pointer.get("identity_sha256") != state.identity_sha256
        or receipt.get("identity_sha256") != state.identity_sha256
        or pointer.get("completed_updates") != state.completed
        or receipt.get("completed_updates") != state.completed
        or not receipt.get("complete_state")
        or receipt.get("accumulation_index") != 0
    ):
        raise RuntimeError("Idempotent save requires matching native bytes and receipts")
    if status not in {"READY", "RUNNING", "PAUSED_RESOURCE", "PAUSED_REQUESTED", "COMPLETE"}:
        raise ValueError("Unknown checkpoint lifecycle status")
    if status == "COMPLETE" and state.completed != state.updates:
        raise ValueError("Cannot label an incomplete training endpoint complete")
    payload = torch.load(version, map_location="cpu", weights_only=False)
    cuda = any(p.device.type == "cuda" for p in model.parameters())
    current = {
        "identity": state.identity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sampler_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if cuda else None,
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "cursor": dict(cursor),
        "completed_updates": state.completed,
        "accumulation_index": 0,
    }
    changed = [key for key, value in current.items() if not equal(payload.get(key), value)]
    if changed:
        raise RuntimeError(f"Same-update checkpoint state changed; original preserved: {changed}")
    state._write_journal()
    atomic_write_json(
        state.receipt,
        {
            **receipt,
            "status": status,
            "scientific_endpoint": status == "COMPLETE" and state.completed == state.updates,
            "recovery_upper": state.recovery_upper,
        },
    )
    atomic_write_json(
        state.directory / "IDEMPOTENT_SAVE.json",
        {
            "schema": "rgb_port_idempotent_lifecycle_save_v1",
            "completed_updates": state.completed,
            "checkpoint_sha256": digest,
            "payload_status": payload["status"],
            "lifecycle_status": status,
            "exact_state_fields": list(current),
            "optimizer_updates": 0,
            "auxiliary_wrappers_called": False,
        },
    )
    return True


@contextmanager
def installed() -> Iterator[None]:
    """Install only after every acceleration/pipeline wrapper has entered."""
    original = training.ProducerCheckpoint.save

    def save(state: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if not reuse_checkpoint(state, *args, **kwargs):
            original(state, *args, **kwargs)

    with patch.object(training.ProducerCheckpoint, "save", save):
        yield
