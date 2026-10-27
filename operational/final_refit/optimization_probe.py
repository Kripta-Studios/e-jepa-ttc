"""Bounded numerical admission for prospective final RGB runtime optimizations."""

# ruff: noqa: ANN401 -- retained source/checkpoint protocols are dynamic

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import io
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import target_log_ratio_from_ttc
from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.eager_probe import _atomic_json
from operational.final_refit.producer_plan import runtime_source_paths
from operational.final_refit.producers import (
    ProducerRecipe,
    build_rgb_recipe,
    configure_rgb_runtime,
    make_effective_batch,
    producer_objective,
)
from operational.final_refit.rgb import AllTrainRGBSource, load_all_train_source
from operational.rgb_port.train_producers import build_producer

SCHEMA = "final_refit_rgb_optimization_admission_v1"
ATOL = 2e-5
RTOL = 2e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _snapshot(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Tensor | Any]:
    if isinstance(value, Tensor):
        return {prefix: value}
    if isinstance(value, np.ndarray):
        return {prefix: torch.from_numpy(value.copy())}
    if isinstance(value, Mapping):
        result: dict[str, Tensor | Any] = {}
        for key in sorted(value):
            result.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def compare_values(left: Any, right: Any, *, exact: bool = False) -> dict[str, Any]:
    """Compare nested tensors with finite/name/shape guards and fixed tolerances."""
    first, second = _flatten(left), _flatten(right)
    names_equal = first.keys() == second.keys()
    shapes_equal = names_equal
    finite = True
    allclose = names_equal
    maximum, numerator, denominator = 0.0, 0.0, 0.0
    if names_equal:
        for name in first:
            a, b = first[name], second[name]
            if isinstance(a, Tensor) != isinstance(b, Tensor):
                shapes_equal = allclose = False
                continue
            if not isinstance(a, Tensor):
                allclose = allclose and a == b
                continue
            if a.shape != b.shape or a.dtype != b.dtype:
                shapes_equal = allclose = False
                continue
            if a.is_floating_point() or a.is_complex():
                fa, fb = a.detach().float(), b.detach().float()
                finite = (
                    finite and bool(torch.isfinite(fa).all()) and bool(torch.isfinite(fb).all())
                )
                difference = (fa - fb).abs()
                maximum = max(maximum, float(difference.max().cpu()) if difference.numel() else 0.0)
                numerator += float(difference.square().sum().cpu())
                denominator += float(fa.square().sum().cpu())
                allclose = allclose and (
                    torch.equal(fa, fb) if exact else torch.allclose(fa, fb, atol=ATOL, rtol=RTOL)
                )
            else:
                allclose = allclose and torch.equal(a, b)
    relative = math.sqrt(numerator / max(denominator, 1e-30))
    return {
        "names_equal": names_equal,
        "shapes_equal": shapes_equal,
        "finite": finite,
        "max_abs": maximum,
        "relative_l2": relative,
        "atol": ATOL,
        "rtol": RTOL,
        "exact": exact,
        "passed": bool(names_equal and shapes_equal and finite and allclose),
    }


def _optimizer(model: nn.Module, recipe: ProducerRecipe) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, recipe: ProducerRecipe
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(recipe.epochs),
        eta_min=float(recipe.training_config["minimum_learning_rate"]),
    )


def _candidate_batch(batch: Any, recipe: ProducerRecipe, enabled: bool) -> tuple[Any, Any, bool]:
    if not enabled:
        return batch, None, True
    try:
        from operational.final_refit import host_loss
    except ImportError:
        return batch, {"status": "UNAVAILABLE"}, False
    prepare = getattr(host_loss, "prepare_effective_batch", None)
    if prepare is None:
        return batch, {"status": "UNAVAILABLE"}, False
    value = prepare(batch, recipe)
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1], True
    return value, getattr(value, "optimization_metadata", None), True


def _collect_scalars(
    loss: Tensor, components: Mapping[str, Tensor], enabled: bool
) -> tuple[Any, bool]:
    if enabled:
        try:
            from operational.final_refit import host_loss

            collect = getattr(host_loss, "collect_loss_scalars", None)
            if collect is not None:
                return collect(loss, components), True
        except ImportError:
            pass
    return (
        {
            **{name: float(value.detach().float().cpu()) for name, value in components.items()},
            "total": float(loss.detach().float().cpu()),
        },
        not enabled,
    )


def _slice_batch_output(value: Any, start: int, stop: int, total: int) -> Any:
    snapped = _snapshot(value)
    if isinstance(snapped, Tensor):
        return snapped[start:stop] if snapped.ndim and snapped.shape[0] == total else snapped
    if isinstance(snapped, Mapping):
        return {key: _slice_batch_output(item, start, stop, total) for key, item in snapped.items()}
    if isinstance(snapped, tuple):
        return tuple(_slice_batch_output(item, start, stop, total) for item in snapped)
    if isinstance(snapped, list):
        return [_slice_batch_output(item, start, stop, total) for item in snapped]
    return snapped


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    recipe: ProducerRecipe,
    epoch: int,
    *,
    optimized_host_loss: bool,
    forward_plan: list[list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    forwards: list[Any] = [] if forward_plan is None else [None] * sum(map(len, forward_plan))
    call_index = 0

    def capture(_module: nn.Module, _args: Any, output: Any) -> None:
        nonlocal call_index
        if forward_plan is None:
            forwards.append(_snapshot(output))
            return
        plan = forward_plan[call_index]
        total = sum(size for _, size in plan)
        offset = 0
        for original, size in plan:
            forwards[original] = _slice_batch_output(output, offset, offset + size, total)
            offset += size
        call_index += 1

    handle = model.register_forward_hook(capture)
    optimizer.zero_grad(set_to_none=True)
    try:
        with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
            loss, components = producer_objective(model, batch, recipe, epoch)
    finally:
        handle.remove()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite optimization-probe loss")
    loss.backward()
    raw = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    clip = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(recipe.training_config["grad_clip_norm"]),
        error_if_nonfinite=True,
    )
    clipped = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    loss_scalars, scalar_optimization_available = _collect_scalars(
        loss, components, optimized_host_loss
    )
    return {
        "forwards": forwards,
        "loss": loss.detach().clone(),
        "components": _snapshot(components),
        "raw_gradients": raw,
        "clip_norm": torch.as_tensor(clip).detach().clone(),
        "clipped_gradients": clipped,
        "loss_scalars": loss_scalars,
        "scalar_optimization_available": scalar_optimization_available,
        "model": _snapshot(model.state_dict()),
        "optimizer": _snapshot(optimizer.state_dict()),
        "rng": _rng(),
    }


def _resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    generator: torch.Generator,
    recipe: ProducerRecipe,
    *,
    transport_reuse: bool,
) -> tuple[nn.Module, torch.optim.Optimizer, Any, torch.Generator]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "generator": generator.get_state(),
        "rng": _rng(),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cuda", weights_only=False)
    resumed_model = build_producer(recipe).cuda().train()
    resumed_model.runtime_reuse_reverse_sample = transport_reuse
    resumed_optimizer = _optimizer(resumed_model, recipe)
    resumed_scheduler = _scheduler(resumed_optimizer, recipe)
    resumed_generator = torch.Generator().manual_seed(0)
    resumed_model.load_state_dict(restored["model"])
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed_scheduler.load_state_dict(restored["scheduler"])
    resumed_generator.set_state(restored["generator"].cpu())
    _restore_rng(restored["rng"])
    return resumed_model, resumed_optimizer, resumed_scheduler, resumed_generator


def _indices(source: AllTrainRGBSource, signature: str) -> tuple[int, ...]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, key in enumerate(source.batch_keys):
        groups.setdefault((str(key[0]), int(key[1])), []).append(index)
    roles = sorted({role for role, _ in groups})
    counts = (2,) if signature == "T2" else (3,) if signature == "T3" else (2, 3)
    per_group = 1 if signature != "MIXED_B32" else 6
    selected: list[int] = []
    for role in roles:
        for count in counts:
            rows = groups.get((role, count), [])
            if len(rows) < per_group:
                raise ValueError(f"Insufficient observed rows for {role}/T{count}")
            selected.extend(rows[:per_group])
    if signature == "MIXED_B32":
        selected = selected[:32]
        if len(selected) < 32:
            raise ValueError("Observed mixed signature cannot form B32")
    return tuple(selected)


def _merge_part_group(parts: list[Any]) -> Any:
    first = parts[0]
    values: dict[str, Any] = {}
    for field in dataclasses.fields(first):
        items = [getattr(part, field.name) for part in parts]
        if all(item is None for item in items):
            values[field.name] = None
        elif all(isinstance(item, Tensor) for item in items):
            values[field.name] = torch.cat(items, dim=0)
        elif all(isinstance(item, list) for item in items):
            values[field.name] = [entry for item in items for entry in item]
        else:
            raise TypeError(f"Cannot merge batch field {field.name}")
    return type(first)(**values)


def _merge_parts_by_t(batch: Any) -> tuple[Any, list[list[tuple[int, int]]]]:
    grouped: dict[int, list[tuple[int, Any]]] = {}
    for index, part in enumerate(batch.parts):
        grouped.setdefault(int(part.events.shape[1]), []).append((index, part))
    merged, plan = [], []
    for _, entries in sorted(grouped.items()):
        merged.append(_merge_part_group([part for _, part in entries]))
        plan.append([(index, int(part.events.shape[0])) for index, part in entries])
    return type(batch)(merged), plan


def _one_step_arm(
    *,
    recipe: ProducerRecipe,
    batch: Any,
    epoch: int,
    initial_state: Mapping[str, Tensor],
    initial_rng: Mapping[str, Any],
    merge_parts: bool,
) -> dict[str, Any]:
    _restore_rng(initial_rng)
    model = build_producer(recipe).cuda().train()
    model.runtime_reuse_reverse_sample = False
    model.load_state_dict(initial_state)
    optimizer = _optimizer(model, recipe)
    device_batch = batch.to(torch.device("cuda"))
    plan = None
    if merge_parts:
        device_batch, plan = _merge_parts_by_t(device_batch)
    result = _step(
        model,
        optimizer,
        device_batch,
        recipe,
        epoch,
        optimized_host_loss=False,
        forward_plan=plan,
    )
    del optimizer, model, device_batch
    torch.cuda.empty_cache()
    return result


def _run_arm(
    *,
    recipe: ProducerRecipe,
    batch: Any,
    epoch: int,
    initial_state: Mapping[str, Tensor],
    initial_rng: Mapping[str, Any],
    optimized_host_loss: bool,
    transport_reuse: bool,
    resume_after_first: bool = True,
) -> list[dict[str, Any]]:
    _restore_rng(initial_rng)
    model = build_producer(recipe).cuda().train()
    model.runtime_reuse_reverse_sample = transport_reuse
    model.load_state_dict(initial_state)
    optimizer = _optimizer(model, recipe)
    scheduler = _scheduler(optimizer, recipe)
    generator = torch.Generator().manual_seed(recipe.seed)
    prepared_host, metadata, host_optimization_available = _candidate_batch(
        batch, recipe, optimized_host_loss
    )
    prepared = prepared_host.to(torch.device("cuda"))
    results = [
        _step(model, optimizer, prepared, recipe, epoch, optimized_host_loss=optimized_host_loss)
    ]
    torch.randperm(len(batch.parts), generator=generator)
    if resume_after_first:
        model, optimizer, scheduler, generator = _resume(
            model,
            optimizer,
            scheduler,
            generator,
            recipe,
            transport_reuse=transport_reuse,
        )
    results.append(
        _step(model, optimizer, prepared, recipe, epoch, optimized_host_loss=optimized_host_loss)
    )
    torch.randperm(len(batch.parts), generator=generator)
    results[-1]["scheduler"] = _snapshot(scheduler.state_dict())
    results[-1]["generator"] = generator.get_state().clone()
    results[-1]["host_metadata"] = _snapshot(metadata)
    results[-1]["host_optimization_available"] = host_optimization_available
    del optimizer, scheduler, generator, model
    torch.cuda.empty_cache()
    return results


def _timed_arm(
    *,
    recipe: ProducerRecipe,
    batch: Any,
    epoch: int,
    initial_state: Mapping[str, Tensor],
    initial_rng: Mapping[str, Any],
    optimized_host_loss: bool,
    transport_reuse: bool,
) -> dict[str, Any]:
    _restore_rng(initial_rng)
    model = build_producer(recipe).cuda().train()
    model.runtime_reuse_reverse_sample = transport_reuse
    model.load_state_dict(initial_state)
    optimizer = _optimizer(model, recipe)
    prepared_host, _, available = _candidate_batch(batch, recipe, optimized_host_loss)
    prepared = prepared_host.to(torch.device("cuda"))
    elapsed = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
            loss, components = producer_objective(model, prepared, recipe, epoch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite timed optimization-probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(recipe.training_config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        _, scalar_available = _collect_scalars(loss, components, optimized_host_loss)
        elapsed.append((time.perf_counter() - started) * 1000)
    del optimizer, model, prepared
    torch.cuda.empty_cache()
    return {
        "steps": 2,
        "wall_ms": elapsed,
        "mean_wall_ms": sum(elapsed) / len(elapsed),
        "host_optimization_available": available,
        "scalar_optimization_available": scalar_available,
    }


def _compare_steps(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    checks = []
    numerical_names = (
        "forwards",
        "loss",
        "components",
        "raw_gradients",
        "clip_norm",
        "clipped_gradients",
        "loss_scalars",
    )
    state_names = ("model", "optimizer", "scheduler", "generator")
    for left, right in zip(reference, candidate, strict=True):
        numerical = {
            name: compare_values(left.get(name), right.get(name)) for name in numerical_names
        }
        state = {name: compare_values(left.get(name), right.get(name)) for name in state_names}
        rng = compare_values(left["rng"], right["rng"], exact=True)
        checks.append(
            {
                "numerical": numerical,
                "state": state,
                "rng": rng,
                "passed": all(value["passed"] for value in numerical.values())
                and all(value["passed"] for value in state.values())
                and rng["passed"],
            }
        )
    return {"steps": checks, "passed": all(value["passed"] for value in checks)}


def _resume_parity(
    resumed: list[dict[str, Any]], uninterrupted: list[dict[str, Any]]
) -> dict[str, Any]:
    names = ("model", "optimizer", "scheduler", "generator", "rng")
    steps = []
    for left, right in zip(resumed, uninterrupted, strict=True):
        fields = {
            name: compare_values(left.get(name), right.get(name), exact=True) for name in names
        }
        steps.append(
            {"fields": fields, "passed": all(value["passed"] for value in fields.values())}
        )
    return {"steps": steps, "passed": all(value["passed"] for value in steps)}


def _source_hashes(repository: Path) -> dict[str, str]:
    members = {
        **runtime_source_paths(repository),
        "operational/final_refit/optimization_probe.py": repository
        / "operational/final_refit/optimization_probe.py",
        "operational/final_refit/producer_plan.py": repository
        / "operational/final_refit/producer_plan.py",
    }
    return {name: _sha256(path.resolve(strict=True)) for name, path in sorted(members.items())}


def _population_physical_mask_parity(source: AllTrainRGBSource) -> dict[str, Any]:
    """Compare the target-dependent physical branch on every TRAIN row, without images."""
    if source.population_size != 88_744 or len(source._rows) != source.population_size:
        raise ValueError("Physical-mask parity requires the complete 88,744-row TRAIN union")
    targets = np.asarray([float(row["target_ttc"]) for _, _, row in source._rows], dtype=np.float32)
    deltas = np.asarray(
        [
            (int(row["producer_sensor_times_us"][-1]) - int(row["producer_sensor_times_us"][-2]))
            / 1_000_000.0
            for _, _, row in source._rows
        ],
        dtype=np.float32,
    )
    cpu_target, cpu_mask = target_log_ratio_from_ttc(
        torch.from_numpy(targets), torch.from_numpy(deltas)
    )
    gpu_target, gpu_mask = target_log_ratio_from_ttc(
        torch.from_numpy(targets).cuda(), torch.from_numpy(deltas).cuda()
    )
    gpu_target_cpu = gpu_target.cpu()
    gpu_mask_cpu = gpu_mask.cpu()
    target_comparison = compare_values(cpu_target, gpu_target_cpu)
    mask_exact = torch.equal(cpu_mask, gpu_mask_cpu)
    digest = hashlib.sha256(cpu_mask.numpy().tobytes(order="C")).hexdigest()
    return {
        "population_size": source.population_size,
        "target_dtype": str(targets.dtype),
        "delta_dtype": str(deltas.dtype),
        "physical_valid_count": int(cpu_mask.sum()),
        "physical_mask_sha256": digest,
        "mask_exact": mask_exact,
        "target_comparison": target_comparison,
        "passed": bool(mask_exact and target_comparison["passed"]),
    }


def _baseline_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the bounded parity matrix without touching live fit state."""
    runtime_settings = configure_rgb_runtime()
    repository = Path(__file__).resolve().parents[2]
    before = _source_hashes(repository)
    baseline_root = args.baseline_root.resolve(True)
    baseline_before = _baseline_hashes(baseline_root)
    source = load_all_train_source(
        args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    )
    population_physical_mask_parity = _population_physical_mask_parity(source)
    cases: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    batching_experiments: list[dict[str, Any]] = []
    technical = 0
    host_optimization_enabled = not args.disable_host_metadata
    transport_reuse_enabled = not args.disable_transport_reuse
    try:
        signatures = ("T2", "T3", "MIXED", "MIXED_B32")
        for fit_id in ("R_A5", "R_C2F"):
            base_recipe = build_rgb_recipe(
                args.config.resolve(True), fit_id, args.source_descriptor.resolve(True), source
            )
            reference_recipe = base_recipe
            candidate_recipe = base_recipe
            torch.manual_seed(base_recipe.seed)
            torch.cuda.manual_seed_all(base_recipe.seed)
            initial_model = build_producer(reference_recipe)
            initial_state = copy.deepcopy(initial_model.state_dict())
            del initial_model
            for epoch in (2, 4):
                for signature in signatures:
                    print(f"BEGIN {fit_id} epoch={epoch} signature={signature}", flush=True)
                    indices = _indices(source, signature)
                    host = make_effective_batch(source, indices, reference_recipe)
                    seed_state = _rng()
                    reference = _run_arm(
                        recipe=reference_recipe,
                        batch=host,
                        epoch=epoch,
                        initial_state=initial_state,
                        initial_rng=seed_state,
                        optimized_host_loss=False,
                        transport_reuse=False,
                    )
                    candidate = _run_arm(
                        recipe=candidate_recipe,
                        batch=host,
                        epoch=epoch,
                        initial_state=initial_state,
                        initial_rng=seed_state,
                        optimized_host_loss=host_optimization_enabled,
                        transport_reuse=transport_reuse_enabled,
                    )
                    candidate_uninterrupted = _run_arm(
                        recipe=candidate_recipe,
                        batch=host,
                        epoch=epoch,
                        initial_state=initial_state,
                        initial_rng=seed_state,
                        optimized_host_loss=host_optimization_enabled,
                        transport_reuse=transport_reuse_enabled,
                        resume_after_first=False,
                    )
                    comparison = _compare_steps(reference, candidate)
                    resume_comparison = _resume_parity(candidate, candidate_uninterrupted)
                    host_optimization_available = bool(
                        candidate[-1].get("host_optimization_available")
                    )
                    scalar_optimization_available = bool(
                        candidate[-1].get("scalar_optimization_available")
                    )
                    comparison["host_optimization_available"] = host_optimization_available
                    comparison["scalar_optimization_available"] = scalar_optimization_available
                    comparison["resume_parity"] = resume_comparison
                    comparison["passed"] = (
                        comparison["passed"]
                        and host_optimization_available
                        and scalar_optimization_available
                        and resume_comparison["passed"]
                    )
                    cases.append(
                        {
                            "fit_id": fit_id,
                            "epoch": epoch,
                            "signature": signature,
                            "effective_batch_size": len(indices),
                            "physical_shapes": [list(part.events.shape) for part in host.parts],
                            "comparison": comparison,
                            "passed": comparison["passed"],
                        }
                    )
                    technical += 6
                    if signature == "MIXED_B32" and comparison["passed"]:
                        baseline_timing = _timed_arm(
                            recipe=reference_recipe,
                            batch=host,
                            epoch=epoch,
                            initial_state=initial_state,
                            initial_rng=seed_state,
                            optimized_host_loss=False,
                            transport_reuse=False,
                        )
                        candidate_timing = _timed_arm(
                            recipe=candidate_recipe,
                            batch=host,
                            epoch=epoch,
                            initial_state=initial_state,
                            initial_rng=seed_state,
                            optimized_host_loss=host_optimization_enabled,
                            transport_reuse=transport_reuse_enabled,
                        )
                        performance.append(
                            {
                                "fit_id": fit_id,
                                "epoch": epoch,
                                "signature": signature,
                                "baseline": baseline_timing,
                                "candidate": candidate_timing,
                            }
                        )
                        technical += 4
                    _atomic_json(
                        args.output.resolve(),
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING",
                            "scientific_optimizer_updates": 0,
                            "technical_optimizer_updates": technical,
                            "cases": cases,
                            "performance": performance,
                        },
                    )
                    del host, reference, candidate, candidate_uninterrupted
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(
                        f"END {fit_id} epoch={epoch} signature={signature} "
                        f"passed={comparison['passed']}",
                        flush=True,
                    )
            for signature in ("T2", "MIXED"):
                indices = _indices(source, signature)
                host = make_effective_batch(source, indices, reference_recipe)
                seed_state = _rng()
                reference_step = _one_step_arm(
                    recipe=reference_recipe,
                    batch=host,
                    epoch=4,
                    initial_state=initial_state,
                    initial_rng=seed_state,
                    merge_parts=False,
                )
                merged_step = _one_step_arm(
                    recipe=reference_recipe,
                    batch=host,
                    epoch=4,
                    initial_state=initial_state,
                    initial_rng=seed_state,
                    merge_parts=True,
                )
                experiment_comparison = _compare_steps([reference_step], [merged_step])
                batching_experiments.append(
                    {
                        "fit_id": fit_id,
                        "epoch": 4,
                        "signature": signature,
                        "effective_batch_size": len(indices),
                        "original_parts": len(host.parts),
                        "merged_parts": len({int(part.events.shape[1]) for part in host.parts}),
                        "comparison": experiment_comparison,
                        "feasibility_passed": experiment_comparison["passed"],
                        "admitted": False,
                        "decision": (
                            "REJECT_RNG_OR_NUMERICAL_DRIFT"
                            if not experiment_comparison["passed"]
                            else "NOT_ADMITTED_SINGLE_STEP_FEASIBILITY_ONLY"
                        ),
                    }
                )
                technical += 2
                del host, reference_step, merged_step
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        source.close()
    after = _source_hashes(repository)
    baseline_after = _baseline_hashes(baseline_root)
    numerical_ok = population_physical_mask_parity["passed"] and all(
        all(
            all(field["passed"] for field in step["numerical"].values())
            for step in case["comparison"]["steps"]
        )
        for case in cases
    )
    state_ok = all(
        all(
            all(field["passed"] for field in step["state"].values())
            for step in case["comparison"]["steps"]
        )
        for case in cases
    )
    rng_ok = all(
        all(step["rng"]["passed"] for step in case["comparison"]["steps"]) for case in cases
    )
    resume_ok = all(case["comparison"]["resume_parity"]["passed"] for case in cases)
    host_available = all(case["comparison"]["host_optimization_available"] for case in cases)
    scalar_available = all(case["comparison"]["scalar_optimization_available"] for case in cases)
    all_cases = (
        numerical_ok and state_ok and rng_ok and resume_ok and host_available and scalar_available
    )
    sources_stable = before == after and baseline_before == baseline_after
    checks = {
        "numerical_parity": (
            "PASS" if numerical_ok and host_available and scalar_available else "FAIL"
        ),
        "state_parity": "PASS" if state_ok else "FAIL",
        "resume_parity": "PASS" if resume_ok else "FAIL",
        "rng_parity": "PASS" if rng_ok else "FAIL",
    }
    source_names = sorted(set(baseline_before) | set(after))
    source_changes = {
        name: {
            "before_sha256": baseline_before.get(name),
            "after_sha256": after.get(name),
        }
        for name in source_names
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "PASS" if all_cases and sources_stable and host_optimization_enabled else "FAIL"
        ),
        "scientific_optimizer_updates": 0,
        "technical_optimizer_updates": technical,
        "admitted_options": {
            "shared_resource_snapshot": False,
            "host_loss_metadata": all_cases and host_optimization_enabled,
            "transport_reuse": all_cases and transport_reuse_enabled,
            "scalar_transfer": all_cases and host_optimization_enabled,
            "merge_parts_by_t": False,
            "cudagraph_local_transport": False,
        },
        "checks": checks,
        "population_physical_mask_parity": population_physical_mask_parity,
        "configured_options": {
            "host_loss_metadata": host_optimization_enabled,
            "transport_reuse": transport_reuse_enabled,
        },
        "runtime_settings": runtime_settings,
        "cases": cases,
        "performance": performance,
        "experimental_merge_parts_by_t": batching_experiments,
        "source_sha256_before_after": source_changes,
        "source_stability": {
            "passed": sources_stable,
            "baseline_before": baseline_before,
            "baseline_after": baseline_after,
            "candidate_before": before,
            "candidate_after": after,
        },
        "limits": {
            "merge_parts_by_t": "Not admitted by this exact-trajectory matrix.",
            "cuda_graphs": (
                "Not admitted without fixed-signature graph, RNG, resume and memory proof."
            ),
            "live_fit_state": "Never read or modified.",
        },
    }
    receipt["admission_sha256"] = _canonical_sha256(receipt)
    if receipt["status"] == "PASS":
        validate_receipt(receipt)
    _atomic_json(args.output.resolve(), receipt)
    return receipt


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Fail closed when an optimization admission receipt is incomplete or self-inconsistent."""
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("Optimization admission receipt is not a PASS receipt")
    if value.get("scientific_optimizer_updates") != 0:
        raise ValueError("Optimization admission must contain zero scientific updates")
    if (
        not isinstance(value.get("technical_optimizer_updates"), int)
        or int(value["technical_optimizer_updates"]) < 96
    ):
        raise ValueError("Optimization admission technical update count is incomplete")

    expected_checks = {
        "numerical_parity",
        "state_parity",
        "resume_parity",
        "rng_parity",
    }
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != expected_checks:
        raise ValueError("Optimization admission checks do not match the schema")
    if any(checks[name] != "PASS" for name in expected_checks):
        raise ValueError("Optimization admission contains a failed parity check")
    population_parity = value.get("population_physical_mask_parity")
    if (
        not isinstance(population_parity, Mapping)
        or population_parity.get("population_size") != 88_744
        or population_parity.get("mask_exact") is not True
        or population_parity.get("passed") is not True
    ):
        raise ValueError("Whole-population CPU/GPU physical-mask parity is missing or failed")
    target_comparison = population_parity.get("target_comparison")
    if (
        not isinstance(target_comparison, Mapping)
        or target_comparison.get("finite") is not True
        or target_comparison.get("passed") is not True
    ):
        raise ValueError("Whole-population physical targets differ between CPU and GPU")

    expected_options = {
        "shared_resource_snapshot",
        "host_loss_metadata",
        "transport_reuse",
        "scalar_transfer",
        "merge_parts_by_t",
        "cudagraph_local_transport",
    }
    options = value.get("admitted_options")
    if not isinstance(options, Mapping) or set(options) != expected_options:
        raise ValueError("Optimization option set does not match the schema")
    if any(not isinstance(options[name], bool) for name in expected_options):
        raise ValueError("Optimization admission options must be booleans")
    if not all(options[name] for name in ("host_loss_metadata", "scalar_transfer")):
        raise ValueError("Core host runtime optimizations were not admitted")
    configured = value.get("configured_options")
    if not isinstance(configured, Mapping) or set(configured) != {
        "host_loss_metadata",
        "transport_reuse",
    }:
        raise ValueError("Configured optimization set does not match the schema")
    if options["transport_reuse"] is not configured["transport_reuse"]:
        raise ValueError("Transport admission disagrees with the configured isolation arm")
    if options["merge_parts_by_t"]:
        raise ValueError("Single-step merge-parts experiment cannot admit batching")

    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise ValueError("Optimization admission must contain exactly 16 core cases")
    expected_cases = {
        (fit_id, epoch, signature)
        for fit_id in ("R_A5", "R_C2F")
        for epoch in (2, 4)
        for signature in ("T2", "T3", "MIXED", "MIXED_B32")
    }
    observed_cases: set[tuple[Any, Any, Any]] = set()
    for case in cases:
        observed_cases.add((case.get("fit_id"), case.get("epoch"), case.get("signature")))
        comparison = case.get("comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("Core case has no comparison")
        steps = comparison.get("steps")
        if not isinstance(steps, list) or len(steps) != 2:
            raise ValueError("Each core case must compare two optimizer steps")
        step_passes: list[bool] = []
        for step in steps:
            numerical = step.get("numerical")
            state = step.get("state")
            rng = step.get("rng")
            if not isinstance(numerical, Mapping) or not isinstance(state, Mapping):
                raise ValueError("Core comparison fields are missing")
            fields = [*numerical.values(), *state.values(), rng]
            recomputed = all(
                isinstance(field, Mapping)
                and field.get("passed") is True
                and field.get("finite") is True
                for field in fields
            )
            if step.get("passed") is not recomputed:
                raise ValueError("Core step pass flag disagrees with field results")
            step_passes.append(recomputed)
        resume = comparison.get("resume_parity")
        if not isinstance(resume, Mapping) or resume.get("passed") is not True:
            raise ValueError("Resume parity is missing or failed")
        resume_steps = resume.get("steps")
        if not isinstance(resume_steps, list) or len(resume_steps) != 2:
            raise ValueError("Resume parity must cover both optimizer steps")
        for resume_step in resume_steps:
            fields = resume_step.get("fields")
            recomputed = (
                isinstance(fields, Mapping)
                and bool(fields)
                and all(
                    isinstance(field, Mapping)
                    and field.get("passed") is True
                    and field.get("finite") is True
                    and field.get("exact") is True
                    for field in fields.values()
                )
            )
            if resume_step.get("passed") is not recomputed:
                raise ValueError("Resume pass flag disagrees with exact field results")
        recomputed_case = (
            all(step_passes)
            and comparison.get("host_optimization_available") is True
            and comparison.get("scalar_optimization_available") is True
            and resume.get("passed") is True
        )
        if (
            comparison.get("passed") is not recomputed_case
            or case.get("passed") is not recomputed_case
        ):
            raise ValueError("Core case pass flag disagrees with its evidence")
    if observed_cases != expected_cases:
        raise ValueError("Optimization admission case matrix is incomplete or duplicated")

    stability = value.get("source_stability")
    if not isinstance(stability, Mapping) or stability.get("passed") is not True:
        raise ValueError("Candidate and baseline source snapshots were not stable")
    source_changes = value.get("source_sha256_before_after")
    if not isinstance(source_changes, Mapping) or not source_changes:
        raise ValueError("Optimization source bindings are missing")
    for hashes in source_changes.values():
        if not isinstance(hashes, Mapping) or set(hashes) != {
            "before_sha256",
            "after_sha256",
        }:
            raise ValueError("Optimization source binding has an invalid shape")
        if hashes["after_sha256"] is None:
            raise ValueError("Candidate source binding is missing an after hash")

    expected_hash = value.get("admission_sha256")
    unsigned = dict(value)
    unsigned.pop("admission_sha256", None)
    if expected_hash != _canonical_sha256(unsigned):
        raise ValueError("Optimization admission hash does not match its contents")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--teacher-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--baseline-root", type=Path, required=True)
    result.add_argument("--disable-host-metadata", action="store_true")
    result.add_argument("--disable-transport-reuse", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    value = run(parser().parse_args(argv))
    print(json.dumps({"status": value["status"], "admission_sha256": value["admission_sha256"]}))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
