"""Accounted synthetic update/resume checks for each V18 optimization path."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
from pathlib import Path

import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import RUN, accounting, config, gpu_charge, require_resources
from .train import backward, initialize, make_optimizer, setup, sources


def fixture(size: int = 32) -> ObjectEventV4Batch:
    """A separate seeded synthetic population; no real sensor or TTC labels."""
    gen = torch.Generator().manual_seed(20261012)
    n = 8
    return ObjectEventV4Batch(
        events=torch.rand(n, 2, 12, size, size, generator=gen),
        delta_t_s=torch.full((n,), 0.1),
        observable_motion=torch.zeros(n, 18),
        visible_heights_px=torch.full((n, 2), float(size // 2)),
        target_ttc_s=torch.tensor([0.8, 2.0, 7.0, -3.0] * 2),
        boxes_xyxy=torch.tensor([2.0, 2.0, size - 2.0, size - 2.0]).repeat(n, 2, 1),
        common_square_xyxy=torch.zeros(n, 4),
        sequence_ids=["synthetic"] * n,
        sample_tokens=[f"synthetic-{i}" for i in range(n)],
        track_ids=["synthetic"] * n,
        dinov3_relation_targets=torch.rand(n, 2, 6, size // 4, size // 4, generator=gen),
        dinov3_relation_valid=torch.ones(n, 2, 6, size // 4, size // 4, dtype=torch.bool),
    )


def subset(batch: ObjectEventV4Batch, ids: list[int]) -> ObjectEventV4Batch:
    """Index tensors and identities by the same sampled IDs."""
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = (
            value[ids]
            if isinstance(value, torch.Tensor)
            else [value[i] for i in ids]
            if isinstance(value, list)
            else None
        )
    return ObjectEventV4Batch(**values)


def trajectory(fit_id: str, mode: str, device: str, limit: int) -> dict:
    """Execute/resume a ten-step synthetic trajectory with the real checkpoint writer."""
    setup(7)
    directory = RUN / "qa" / device / fit_id / mode
    identity = {
        "fit_id": f"QA-{device}-{fit_id}-{mode}",
        "seed": 7,
        "sources": sources(),
        "qa_source": sha(Path(__file__)),
        "population": "synthetic8",
        "endpoint": 10,
        "device": device,
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    model = initialize(fit_id, random_parent=True).to(device).train()
    initial = copy.deepcopy(model.encoder.state_dict())
    optimizer, scheduler = make_optimizer(model, updates=10)
    generator = torch.Generator().manual_seed(7)
    state = ProducerCheckpoint(directory, identity, 10)
    cursor = (
        state.restore(model, optimizer, scheduler, generator) if state.pointer.exists() else None
    )
    curve = cursor["curve"] if cursor is not None else []
    population = fixture()
    if cursor is None:
        state.save(model, optimizer, scheduler, generator, {"curve": curve}, status="RUNNING")
    while state.completed < limit:
        ids = torch.randint(8, (4,), generator=generator).tolist()
        batches = [subset(population, ids[i : i + 2]).to(torch.device(device)) for i in (0, 2)]
        optimizer.zero_grad(set_to_none=True)
        losses = backward(model, batches, torch.full((4,), 1 / 8), 8)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        curve.append({"ids": ids, "losses": losses, "norm": float(norm)})
    state.save(
        model,
        optimizer,
        scheduler,
        generator,
        {"curve": curve},
        status="COMPLETE" if limit == 10 else "PAUSED",
    )
    difference = max(
        float((value - initial[key]).abs().max())
        for key, value in model.encoder.state_dict().items()
    )
    if limit > 0 and (difference > 0) == ("FROZEN" in fit_id):
        raise AssertionError("Synthetic encoder update contract failed")
    accounting()
    return torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)


def exact(left: object, right: object) -> bool:
    """Compare nested RNG, optimizer, sampler and tensor state exactly."""
    import numpy as np

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact(v, right[k]) for k, v in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def run(device: str) -> None:
    """Check 10 versus 5+restore+5, charging all 20 updates per path."""
    from contextlib import nullcontext

    require_resources(starting=True)
    for fit_id in config()["fits"]:
        receipt = RUN / "qa" / device / fit_id / "PASS.json"
        if receipt.exists():
            if read(receipt)["source_hash"] != digest(sources()):
                raise ValueError(
                    "QA sources changed: archive prior evidence before a new QA version"
                )
            continue
        with (
            gpu_charge("synthetic_qa/" + fit_id, reserve=300) if device == "cuda" else nullcontext()
        ):
            continuous = trajectory(fit_id, "continuous", device, 10)
            trajectory(fit_id, "split", device, 5)
            restored = trajectory(fit_id, "split", device, 10)
            keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "sampler_generator_state",
                "torch_rng_state",
                "cuda_rng_state_all",
                "numpy_random_state",
                "python_random_state",
                "cursor",
            ]
            comparisons = {key: exact(continuous[key], restored[key]) for key in keys}
            if not all(comparisons.values()):
                write(receipt.with_name("FAIL.json"), comparisons)
                raise AssertionError(f"Resume mismatch: {comparisons}")
            write(
                receipt,
                {
                    "status": "PASS",
                    "device": device,
                    "comparisons": comparisons,
                    "physical_updates": 20,
                    "source_hash": digest(sources()),
                    "utc": utc(),
                },
            )
            print(f"QA {device} {fit_id} PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run(parser.parse_args().device)
