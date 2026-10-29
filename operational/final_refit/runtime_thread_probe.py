"""Disposable concurrent-prefetch and intra-op thread probe for final RGB fitting."""

# ruff: noqa: ANN401 -- retained checkpoint/source protocols are dynamic

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.eager_probe import _aggregate_difference, _atomic_json
from operational.final_refit.producers import (
    BoundedRGBPrefetch,
    ParallelRGBDecoder,
    _validate_batch,
    build_rgb_recipe,
    producer_objective,
)
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.train_producers import _fast_resource_guard, build_producer

SCHEMA = "final_refit_rgb_runtime_thread_probe_v1"


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng_state_all"]])


def _optimizer(model: torch.nn.Module, recipe: Any, payload: Mapping[str, Any]) -> Any:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
    return optimizer


def _fresh_cursor(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **copy.deepcopy(payload["cursor"]),
        "order": payload["cursor"]["order"].clone(),
    }


def _one_arm(
    *,
    payload: Mapping[str, Any],
    recipe: Any,
    source: Any,
    threads: int,
    steps: int,
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.set_num_threads(threads)
    model = build_producer(recipe).cuda().train()
    model.load_state_dict(payload["model_state_dict"])
    optimizer = _optimizer(model, recipe, payload)
    _restore_rng(payload)
    cursor = _fresh_cursor(payload)
    decoder = ParallelRGBDecoder(source, workers=4, max_pending_bytes=128 * 1024**2)
    prefetch = BoundedRGBPrefetch(source, recipe, cursor, depth=2)
    totals = {
        name: 0.0
        for name in (
            "resource_guard_ms",
            "prefetch_wait_ms",
            "h2d_ms",
            "validation_ms",
            "forward_ms",
            "finite_check_ms",
            "backward_ms",
            "clip_ms",
            "optimizer_ms",
            "loss_transfer_ms",
            "wall_ms",
        )
    }
    evidence = []
    try:
        for update in range(steps + 1):
            start = int(cursor["position"])
            stop = start + recipe.effective_batch_size
            indices = cursor["order"][start:stop].tolist()
            wall_start = time.perf_counter()
            guard_start = time.perf_counter()
            _fast_resource_guard(run_root, checkpoint_reservation=1 * 1024**3)
            guard_ms = (time.perf_counter() - guard_start) * 1000
            wait_start = time.perf_counter()
            host = prefetch.get(indices)
            wait_ms = (time.perf_counter() - wait_start) * 1000
            cursor["position"] = stop
            transfer_start = time.perf_counter()
            batch = host.to(torch.device("cuda"))
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - transfer_start) * 1000
            validation_start = time.perf_counter()
            for part in batch.parts:
                _validate_batch(part, recipe)
            validation_ms = (time.perf_counter() - validation_start) * 1000
            optimizer.zero_grad(set_to_none=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
            with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
                loss, components = producer_objective(
                    model, batch, recipe, int(payload["cursor"]["epoch"])
                )
            events[1].record()
            finite_start = time.perf_counter()
            finite = bool(torch.isfinite(loss))
            finite_ms = (time.perf_counter() - finite_start) * 1000
            if not finite:
                raise FloatingPointError("Nonfinite technical loss")
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
            transfer_loss_start = time.perf_counter()
            scalar_values = torch.stack(
                [loss.detach().float(), *(value.detach().float() for value in components.values())]
            ).cpu()
            transfer_loss_ms = (time.perf_counter() - transfer_loss_start) * 1000
            measured = update > 0
            if measured:
                totals["resource_guard_ms"] += guard_ms
                totals["prefetch_wait_ms"] += wait_ms
                totals["h2d_ms"] += h2d_ms
                totals["validation_ms"] += validation_ms
                totals["forward_ms"] += events[0].elapsed_time(events[1])
                totals["finite_check_ms"] += finite_ms
                totals["backward_ms"] += events[1].elapsed_time(events[2])
                totals["clip_ms"] += events[2].elapsed_time(events[3])
                totals["optimizer_ms"] += events[3].elapsed_time(events[4])
                totals["loss_transfer_ms"] += transfer_loss_ms
                totals["wall_ms"] += (time.perf_counter() - wall_start) * 1000
                if len(evidence) < 2:
                    evidence.append(
                        {
                            "loss": loss.detach().clone(),
                            "components": {
                                name: value.detach().clone() for name, value in components.items()
                            },
                            "clip": torch.as_tensor(clip).detach().clone(),
                            "gradients": {
                                name: parameter.grad.detach().clone()
                                for name, parameter in model.named_parameters()
                                if parameter.grad is not None
                            },
                            "model": copy.deepcopy(model.state_dict()),
                            "optimizer": copy.deepcopy(optimizer.state_dict()),
                            "rng": {
                                "cpu": torch.get_rng_state().clone(),
                                "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
                            },
                            "scalars": scalar_values.clone(),
                            "indices": list(indices),
                        }
                    )
        result: dict[str, Any] = {name: value / steps for name, value in totals.items()}
        result.update(
            threads=threads,
            steps=steps,
            warmup_steps=1,
            updates_per_second=1000.0 / result["wall_ms"],
            prefetch=prefetch.snapshot(),
            decoder=decoder.snapshot(),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        return result, evidence
    finally:
        prefetch.close()
        decoder.close()
        del optimizer, model
        torch.cuda.empty_cache()


def _parity(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    result = []
    for left, right in zip(reference, candidate, strict=True):
        fields = {
            name: _aggregate_difference(left[name], right[name], atol=2e-5, rtol=2e-4)
            for name in ("loss", "components", "clip", "gradients", "model", "optimizer")
        }
        exact = {
            "rng_cpu": torch.equal(left["rng"]["cpu"], right["rng"]["cpu"]),
            "rng_cuda": all(
                torch.equal(a, b)
                for a, b in zip(left["rng"]["cuda"], right["rng"]["cuda"], strict=True)
            ),
            "indices": left["indices"] == right["indices"],
        }
        result.append(
            {
                "fields": fields,
                "exact": exact,
                "passed": all(item["original_allclose"] for item in fields.values())
                and all(exact.values()),
            }
        )
    return {"updates": result, "passed": all(item["passed"] for item in result)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Profile thread settings from one immutable checkpoint copy."""
    checkpoint = args.checkpoint.resolve(True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    descriptor, teacher = args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    source = load_all_train_source(descriptor, teacher)
    original_threads = torch.get_num_threads()
    try:
        recipe = build_rgb_recipe(args.config.resolve(True), "R_A5", descriptor, source)
        modes = [original_threads, 1, 2, 4, original_threads]
        arms = []
        evidence = []
        for threads in modes:
            torch.cuda.reset_peak_memory_stats()
            arm, trace = _one_arm(
                payload=payload,
                recipe=recipe,
                source=source,
                threads=threads,
                steps=args.steps,
                run_root=args.run_root.resolve(),
            )
            arms.append(arm)
            evidence.append(trace)
            _atomic_json(
                args.output.resolve(),
                {
                    "schema": SCHEMA,
                    "status": "RUNNING",
                    "scientific_updates": 0,
                    "technical_optimizer_steps": len(arms) * (args.steps + 1),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "checkpoint_completed_updates": payload["completed_updates"],
                    "recipe_identity_sha256": recipe.identity_sha256,
                    "original_threads": original_threads,
                    "completed_arms": arms,
                    "discarded_prior_probe": {
                        "technical_optimizer_steps": 33,
                        "reason": (
                            "No receipt survived profiler post-processing failure; timings also "
                            "omitted ParallelRGBDecoder and are not used."
                        ),
                    },
                },
            )
        parity = [_parity(evidence[0], candidate) for candidate in evidence[1:]]
        admitted_candidates = [index for index in range(1, 4) if parity[index - 1]["passed"]]
        best_index = min(admitted_candidates, key=lambda index: arms[index]["wall_ms"])
        profiler: dict[str, Any]
        profiler_steps = 0
        try:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                _one_arm(
                    payload=payload,
                    recipe=recipe,
                    source=source,
                    threads=int(arms[best_index]["threads"]),
                    steps=2,
                    run_root=args.run_root.resolve(),
                )
            # The workspace can contain non-ASCII path components, which Kineto's
            # Windows trace writer does not support reliably. Key averages retain
            # the CPU/CUDA profile needed by this bounded admission probe.
            averages = sorted(
                profile.key_averages(),
                key=lambda item: float(
                    getattr(item, "self_cuda_time_total", item.self_device_time_total)
                ),
                reverse=True,
            )
            profiler = {
                "status": "CAPTURED",
                "threads": arms[best_index]["threads"],
                "trace": None,
                "trace_export": "SKIPPED_UNICODE_PATH_KINETO_WINDOWS",
                "top_self_cuda": [
                    {
                        "key": item.key,
                        "self_cuda_time_total_us": float(
                            getattr(item, "self_cuda_time_total", item.self_device_time_total)
                        ),
                        "self_cpu_time_total_us": item.self_cpu_time_total,
                        "count": item.count,
                    }
                    for item in averages[:20]
                ],
            }
            profiler_steps = 3
        except (RuntimeError, AssertionError, AttributeError, OSError) as error:
            profiler = {"status": "UNAVAILABLE", "error": str(error)}
        return {
            "schema": SCHEMA,
            "status": "PROFILED",
            "scientific_updates": 0,
            "technical_optimizer_steps": len(modes) * (args.steps + 1) + profiler_steps,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_completed_updates": payload["completed_updates"],
            "recipe_identity_sha256": recipe.identity_sha256,
            "original_threads": original_threads,
            "interop_threads": torch.get_num_interop_threads(),
            "arms": arms,
            "parity_vs_first_baseline": parity,
            "baseline_repeat_parity": parity[-1],
            "best_admitted_index": best_index,
            "profiler": profiler,
            "discarded_prior_probe": {
                "technical_optimizer_steps": 33,
                "reason": (
                    "No receipt survived profiler post-processing failure; timings also omitted "
                    "ParallelRGBDecoder and are not used."
                ),
            },
        }
    finally:
        torch.set_num_threads(original_threads)
        source.close()
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--teacher-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--trace", type=Path, required=True)
    result.add_argument("--steps", type=int, default=5, choices=range(3, 9))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    value = run(args)
    _atomic_json(args.output.resolve(), value)
    print(json.dumps({"status": value["status"], "technical": value["technical_optimizer_steps"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
