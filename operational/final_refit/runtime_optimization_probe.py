"""Disposable phase profiler and optimizer parity probe for a paused RGB fit."""

# ruff: noqa: ANN401 -- checkpoint/source adapters retain dynamic protocols

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.eager_probe import _aggregate_difference, _atomic_json
from operational.final_refit.producers import (
    build_rgb_recipe,
    make_effective_batch,
    producer_objective,
)
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.train_producers import build_producer

SCHEMA = "final_refit_rgb_runtime_optimization_probe_v1"


def _restore(payload: dict[str, Any]) -> None:
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng_state_all"]])
    np.random.set_state(payload["numpy_random_state"])
    import random

    random.setstate(payload["python_random_state"])


def _optimizer(model: torch.nn.Module, recipe: Any, mode: str) -> torch.optim.AdamW:
    flags: dict[str, Any] = {}
    if mode == "foreach":
        flags["foreach"] = True
    elif mode == "fused":
        flags["fused"] = True
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
        **flags,
    )


def _run_arm(
    payload: dict[str, Any], recipe: Any, device_batches: list[Any], mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Pay lazy kernel/library initialization on a disposable copy so every arm
    # starts its measured trajectory from the exact checkpoint state.
    warm_model = build_producer(recipe).cuda().train()
    warm_model.load_state_dict(payload["model_state_dict"])
    warm_optimizer = _optimizer(warm_model, recipe, mode)
    warm_optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
    _restore(payload)
    warm_optimizer.zero_grad(set_to_none=True)
    with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
        warm_loss, _ = producer_objective(warm_model, device_batches[0], recipe, 1)
    warm_loss.backward()
    torch.nn.utils.clip_grad_norm_(warm_model.parameters(), 1.0, error_if_nonfinite=True)
    warm_optimizer.step()
    torch.cuda.synchronize()
    del warm_loss, warm_optimizer, warm_model
    torch.cuda.empty_cache()

    model = build_producer(recipe).cuda().train()
    model.load_state_dict(payload["model_state_dict"])
    optimizer = _optimizer(model, recipe, mode)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
    _restore(payload)
    totals: dict[str, float] = {
        name: 0.0
        for name in ("forward_ms", "backward_ms", "clip_ms", "optimizer_ms", "step_wall_ms")
    }
    first: dict[str, Any] = {}
    for index, batch in enumerate(device_batches):
        optimizer.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        torch.cuda.synchronize()
        wall = time.perf_counter()
        events[0].record()
        with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
            loss, components = producer_objective(model, batch, recipe, 1)
        events[1].record()
        loss.backward()
        events[2].record()
        clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(recipe.training_config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
        events[3].record()
        optimizer.step()
        events[4].record()
        torch.cuda.synchronize()
        totals["step_wall_ms"] += (time.perf_counter() - wall) * 1000
        totals["forward_ms"] += events[0].elapsed_time(events[1])
        totals["backward_ms"] += events[1].elapsed_time(events[2])
        totals["clip_ms"] += events[2].elapsed_time(events[3])
        totals["optimizer_ms"] += events[3].elapsed_time(events[4])
        if index == 0:
            first = {
                "loss": loss.detach().clone(),
                "components": {name: value.detach().clone() for name, value in components.items()},
                "clip": torch.as_tensor(clip).detach().clone(),
                "gradients": {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                },
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
    count = len(device_batches)
    result: dict[str, Any] = {name: value / count for name, value in totals.items()}
    result.update(
        mode=mode,
        steps=count,
        updates_per_second=1000.0 / result["step_wall_ms"],
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )
    del optimizer, model
    torch.cuda.empty_cache()
    return result, first


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Profile the paused checkpoint on copied batches; no fit state is written."""
    checkpoint = args.checkpoint.resolve(True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    descriptor, teacher = args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    source = load_all_train_source(descriptor, teacher)
    try:
        recipe = build_rgb_recipe(args.config.resolve(True), "R_A5", descriptor, source)
        order = payload["cursor"]["order"]
        position = int(payload["cursor"]["position"])
        host_batches = []
        signatures = []
        for offset in range(args.steps):
            start = position + offset * recipe.effective_batch_size
            indices = order[start : start + recipe.effective_batch_size].tolist()
            batch = make_effective_batch(source, indices, recipe)
            host_batches.append(batch)
            signatures.append([list(part.events.shape) for part in batch.parts])
        transfer_start = time.perf_counter()
        device_batches = [batch.to(torch.device("cuda")) for batch in host_batches]
        torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - transfer_start) * 1000 / len(host_batches)
        arms = []
        evidence = {}
        torch.cuda.reset_peak_memory_stats()
        for mode in ("default", "foreach", "fused"):
            try:
                metrics, first = _run_arm(payload, recipe, device_batches, mode)
                arms.append(metrics)
                evidence[mode] = first
            except (RuntimeError, TypeError, ValueError) as error:
                arms.append({"mode": mode, "unsupported": True, "error": str(error)})
        baseline = evidence["default"]
        parity = {}
        for mode in ("foreach", "fused"):
            if mode not in evidence:
                continue
            parity[mode] = {
                name: _aggregate_difference(
                    baseline[name], evidence[mode][name], atol=2e-5, rtol=2e-4
                )
                for name in baseline
            }
        journal_samples = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.json"
            for index in range(20):
                start = time.perf_counter()
                _atomic_json(path, {"completed": index, "pending": bool(index % 2)})
                journal_samples.append((time.perf_counter() - start) * 1000)
        return {
            "schema": SCHEMA,
            "status": "PROFILED",
            "scientific_updates": 0,
            "technical_optimizer_steps": sum(
                int(item.get("steps", 0)) + 1
                for item in arms
                if not item.get("unsupported")
            ),
            "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_completed_updates": payload["completed_updates"],
            "recipe_identity_sha256": recipe.identity_sha256,
            "batch_signatures": signatures,
            "host_to_device_ms_mean": transfer_ms,
            "arms": arms,
            "parity_vs_default_first_step": parity,
            "journal_atomic_write_ms": {
                "mean": sum(journal_samples) / len(journal_samples),
                "max": max(journal_samples),
            },
            "cuda_graphs": "NOT_RETRIED_PRIOR_CUDAGRAPH_MEMORY_FAILURE_AND_DYNAMIC_SIGNATURES",
        }
    finally:
        source.close()
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--teacher-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--steps", type=int, default=5, choices=range(2, 11))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {"status": result["status"], "technical": result["technical_optimizer_steps"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
