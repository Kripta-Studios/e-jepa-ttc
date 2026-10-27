"""Disposable RGB producer parity and CUDA-graph admission probe.

This module is deliberately outside the scientific training path.  It may run
forward/backward/optimizer work on disposable model copies, but it never opens
or writes a producer checkpoint.  A graph result is admitted only when the
eager and candidate paths agree through one complete step and the steady-state
benchmark proves that the fixed-signature route is both bounded and useful.
"""

# ruff: noqa: ANN401 -- runtime adapters bridge the retained producer protocol

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

SCHEMA = "final_refit_rgb_runtime_probe_v1"
FIT_IDS = ("R_A5", "R_C2F")
REGIMES = ("foreground_warmup", "full_objective")
ADMITTED_BATCH = 32
ADMITTED_FRAMES = 3


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one technical receipt without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _host_memory_snapshot() -> dict[str, int | None]:
    """Read host availability and Windows commit headroom without allocating tensors."""
    import psutil

    from operational.rgb_port.train_producers import _windows_commit_headroom

    return {
        "host_available_bytes": int(psutil.virtual_memory().available),
        "windows_commit_headroom_bytes": _windows_commit_headroom(),
    }


def _release_graph_runtime() -> None:
    """Release disposable compiler/model caches between independent cases."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


@dataclass(frozen=True)
class ForwardSignature:
    """Exact tensor and keyword signature of one producer forward."""

    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    input_dtype: str
    input_device: str
    delta_shape: tuple[int, ...]
    delta_stride: tuple[int, ...]
    delta_dtype: str
    delta_device: str
    return_dense_features: bool

    @classmethod
    def from_call(
        cls, inputs: Tensor, delta_t_s: Tensor, *, return_dense_features: bool
    ) -> ForwardSignature:
        """Build a signature without synchronizing or copying tensor data."""
        return cls(
            input_shape=tuple(inputs.shape),
            input_stride=tuple(inputs.stride()),
            input_dtype=str(inputs.dtype),
            input_device=str(inputs.device),
            delta_shape=tuple(delta_t_s.shape),
            delta_stride=tuple(delta_t_s.stride()),
            delta_dtype=str(delta_t_s.dtype),
            delta_device=str(delta_t_s.device),
            return_dense_features=bool(return_dense_features),
        )

    @property
    def batch_frames(self) -> tuple[int, int]:
        if len(self.input_shape) < 2:
            raise ValueError("Producer input must have batch and temporal dimensions")
        return int(self.input_shape[0]), int(self.input_shape[1])

    def label(self) -> str:
        """Return a stable compact label for receipts."""
        encoded = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class BoundedGraphDispatcher:
    """Route only registered exact signatures through graph-backed callables."""

    def __init__(self, eager: Callable[..., Any], *, max_graphs: int = 1) -> None:
        if max_graphs <= 0:
            raise ValueError("max_graphs must be positive")
        self.eager = eager
        self.max_graphs = int(max_graphs)
        self._graphs: dict[ForwardSignature, Callable[..., Any]] = {}
        self.graph_calls = 0
        self.eager_calls = 0
        self.fallback_signatures: Counter[str] = Counter()
        self.graph_signatures: Counter[str] = Counter()
        self.last_output: Any = None

    def register(self, signature: ForwardSignature, graph_callable: Callable[..., Any]) -> None:
        """Register one already-built graph callable without eviction."""
        if signature in self._graphs:
            raise ValueError("Graph signature is already registered")
        if len(self._graphs) >= self.max_graphs:
            raise ValueError("Bounded graph cache is full")
        if signature.batch_frames != (ADMITTED_BATCH, ADMITTED_FRAMES):
            raise ValueError("Only the measured B32/T3 signature may be registered")
        self._graphs[signature] = graph_callable

    def __call__(
        self, inputs: Tensor, delta_t_s: Tensor, *, return_dense_features: bool = False
    ) -> Any:
        signature = ForwardSignature.from_call(
            inputs, delta_t_s, return_dense_features=return_dense_features
        )
        selected = self._graphs.get(signature)
        if selected is None:
            self.eager_calls += 1
            self.fallback_signatures[signature.label()] += 1
            output = self.eager(inputs, delta_t_s, return_dense_features=return_dense_features)
        else:
            self.graph_calls += 1
            self.graph_signatures[signature.label()] += 1
            output = selected(inputs, delta_t_s, return_dense_features=return_dense_features)
        self.last_output = output
        return output

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-compatible bounded-routing evidence."""
        return {
            "max_graphs": self.max_graphs,
            "registered_graphs": len(self._graphs),
            "graph_calls": self.graph_calls,
            "eager_calls": self.eager_calls,
            "graph_signatures": dict(sorted(self.graph_signatures.items())),
            "fallback_signatures": dict(sorted(self.fallback_signatures.items())),
        }


def schedule_signature_histogram(
    source: Any,
    order: Sequence[int] | Tensor,
    *,
    effective_batch_size: int = ADMITTED_BATCH,
    maximum_updates: int | None = None,
) -> dict[str, Any]:
    """Count exact role/T microbatch sizes in an already-fixed epoch order.

    Grouping mirrors the source contract: a microbatch never crosses a source
    role or temporal length.  It does not pad terminal or heterogeneous parts.
    """
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    values = order.tolist() if isinstance(order, Tensor) else list(order)
    if sorted(map(int, values)) != list(range(int(source.population_size))):
        raise ValueError("Schedule order must be a permutation of the source population")
    keys = getattr(source, "batch_keys", None)
    if keys is None:
        keys = [("unknown", int(value)) for value in source.frame_counts]
    if len(keys) != int(source.population_size):
        raise ValueError("Source batch keys do not cover the population")
    parts: Counter[str] = Counter()
    logical: Counter[str] = Counter()
    graph_examples = 0
    total_examples = 0
    update_count = 0
    for start in range(0, len(values), effective_batch_size):
        if maximum_updates is not None and update_count >= maximum_updates:
            break
        selected = [int(value) for value in values[start : start + effective_batch_size]]
        grouped: dict[tuple[str, int], int] = {}
        for index in selected:
            role, frames = keys[index]
            key = str(role), int(frames)
            grouped[key] = grouped.get(key, 0) + 1
        labels: list[str] = []
        for (role, frames), size in sorted(grouped.items()):
            label = f"{role}:B{size}:T{frames}"
            parts[label] += 1
            labels.append(label)
            total_examples += size
            if (size, frames) == (ADMITTED_BATCH, ADMITTED_FRAMES):
                graph_examples += size
        logical["+".join(labels)] += 1
        update_count += 1
    return {
        "effective_batch_size": effective_batch_size,
        "updates": update_count,
        "parts": dict(sorted(parts.items())),
        "logical_batches": dict(sorted(logical.items())),
        "graph_eligible_examples": graph_examples,
        "total_examples": total_examples,
        "graph_eligible_example_fraction": (
            graph_examples / total_examples if total_examples else 0.0
        ),
        "padding_or_reordering": False,
    }


def _same(left: Any, right: Any, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.is_floating_point() or left.is_complex():
            return bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _same(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _named_tensors(value: Any, prefix: str = "output") -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    if isinstance(value, Tensor):
        result[prefix] = value.detach().clone()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            result.update(_named_tensors(getattr(value, field.name), f"{prefix}.{field.name}"))
    elif isinstance(value, Mapping):
        for name, item in value.items():
            result.update(_named_tensors(item, f"{prefix}.{name}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_named_tensors(item, f"{prefix}.{index}"))
    return result


def _comparison(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor], *, atol: float, rtol: float
) -> dict[str, Any]:
    names_equal = tuple(left) == tuple(right)
    failures: list[str] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    if names_equal:
        for name in left:
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                failures.append(name)
                continue
            if a.numel() and (a.is_floating_point() or a.is_complex()):
                absolute = (a.float() - b.float()).abs()
                maximum_absolute = max(maximum_absolute, float(absolute.max().item()))
                denominator = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-12)
                maximum_relative = max(
                    maximum_relative, float((absolute / denominator).max().item())
                )
            if not _same(a, b, atol=atol, rtol=rtol):
                failures.append(name)
    return {
        "names_equal": names_equal,
        "passed": names_equal and not failures,
        "failed_names": failures,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "atol": atol,
        "rtol": rtol,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        _same(left["python"], right["python"])
        and _same(left["numpy"], right["numpy"])
        and torch.equal(left["cpu"], right["cpu"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"], strict=True))
    )


def _gradients(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.grad.detach().clone()
        for name, value in model.named_parameters()
        if value.grad is not None
    }


def _synchronize() -> None:
    torch.cuda.current_stream().synchronize()


def _autocast() -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _optimizer(model: nn.Module, recipe: Any) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )


def _scheduler(optimizer: torch.optim.Optimizer, recipe: Any) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(recipe.epochs),
        eta_min=float(recipe.training_config["minimum_learning_rate"]),
    )


def _run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    objective: Callable[[nn.Module], tuple[Tensor, Mapping[str, Tensor]]],
    *,
    clip_norm: float,
    update: bool,
    counters: dict[str, int] | None = None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    if counters is not None:
        counters["forward_backward_passes_started"] += 1
    with _autocast():
        loss, components = objective(model)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Runtime probe produced nonfinite loss")
    loss.backward()
    before = _gradients(model)
    clip = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    after = _gradients(model)
    if update:
        optimizer.step()
        scheduler.step()
        if counters is not None:
            counters["optimizer_steps_completed"] += 1
    _synchronize()
    return {
        "loss": loss.detach().clone(),
        "components": {name: value.detach().clone() for name, value in components.items()},
        "gradients_before_clip": before,
        "clip_norm": torch.as_tensor(clip).detach().clone(),
        "gradients_after_clip": after,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }


def _dynamo_snapshot() -> dict[str, dict[str, int]]:
    from torch._dynamo.utils import counters

    return {
        section: {str(name): int(value) for name, value in values.items()}
        for section, values in counters.items()
    }


def _dynamo_delta(
    before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for section in set(before) | set(after):
        names = set(before.get(section, {})) | set(after.get(section, {}))
        values = {
            name: int(after.get(section, {}).get(name, 0))
            - int(before.get(section, {}).get(name, 0))
            for name in names
        }
        result[section] = {name: value for name, value in values.items() if value}
    return {name: values for name, values in result.items() if values}


def _timed_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Callable[[nn.Module], tuple[Tensor, Mapping[str, Tensor]]],
    *,
    clip_norm: float,
    steps: int,
    counters: dict[str, int] | None = None,
) -> list[float]:
    values: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _synchronize()
        started = time.perf_counter()
        if counters is not None:
            counters["forward_backward_passes_started"] += 1
        with _autocast():
            loss, _ = objective(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if counters is not None:
            counters["optimizer_steps_completed"] += 1
        _synchronize()
        values.append(time.perf_counter() - started)
    return values


def _timing(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    return {
        "steps": float(len(ordered)),
        "total_seconds": float(sum(ordered)),
        "mean_ms": float(statistics.fmean(ordered) * 1000),
        "median_ms": float(statistics.median(ordered) * 1000),
        "p95_ms": float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] * 1000),
    }


def validate_case(
    case: Mapping[str, Any], *, min_speedup: float, max_peak_bytes: int
) -> tuple[bool, list[str]]:
    """Fail closed over a single arm/loss-regime result."""
    reasons: list[str] = []
    parity = case.get("parity", {})
    for name in (
        "output",
        "loss",
        "components",
        "gradients_before_clip",
        "clip_norm",
        "gradients_after_clip",
        "model_after_step",
        "optimizer_after_step",
        "scheduler_after_step",
    ):
        if not parity.get(name, {}).get("passed", False):
            reasons.append(f"parity_failed:{name}")
    if not parity.get("rng_after_step_equal", False):
        reasons.append("rng_after_step_differs")
    if not case.get("warmup_complete_state_restored", False):
        reasons.append("warmup_state_not_restored")
    if not case.get("sampler_generator_preserved", False):
        reasons.append("sampler_generator_changed")
    routing = case.get("routing", {})
    if int(routing.get("graph_calls", 0)) <= 0:
        reasons.append("no_graph_routed_call")
    compile_evidence = case.get("compile", {})
    if int(compile_evidence.get("unique_graphs", 0)) <= 0:
        reasons.append("no_unique_graph")
    if int(compile_evidence.get("steady_state_new_unique_graphs", 0)) != 0:
        reasons.append("steady_state_recompiled")
    if not bool(compile_evidence.get("observed_graph_launch", False)):
        reasons.append("no_observed_graph_launch")
    raw_peak = case.get("peak_cuda_memory_bytes")
    if not isinstance(raw_peak, int):
        reasons.append("peak_cuda_memory_unavailable")
    elif raw_peak > max_peak_bytes:
        reasons.append("peak_cuda_memory_exceeded")
    throughput = case.get("throughput") or {}
    speedup = float(throughput.get("speedup", 0.0))
    if speedup < min_speedup:
        reasons.append("speedup_below_threshold")
    return not reasons, reasons


def validate_probe(
    payload: Mapping[str, Any], *, min_speedup: float, max_peak_bytes: int
) -> dict[str, Any]:
    """Validate that both arms and both loss regimes passed independently."""
    cases = payload.get("cases", [])
    seen = {(item.get("fit_id"), item.get("regime")) for item in cases}
    required = {(fit, regime) for fit in FIT_IDS for regime in REGIMES}
    reasons: list[str] = []
    if seen != required:
        reasons.append("arm_or_regime_coverage_differs")
    decisions = []
    for case in cases:
        passed, case_reasons = validate_case(
            case, min_speedup=min_speedup, max_peak_bytes=max_peak_bytes
        )
        decisions.append(
            {
                "fit_id": case.get("fit_id"),
                "regime": case.get("regime"),
                "passed": passed,
                "reasons": case_reasons,
            }
        )
        reasons.extend(
            f"{case.get('fit_id')}:{case.get('regime')}:{reason}" for reason in case_reasons
        )
    return {
        "status": "ADMITTED" if not reasons else "REJECTED",
        "graph_training_enabled": not reasons,
        "reasons": reasons,
        "cases": decisions,
        "fallback": "eager_exact_recipe",
    }


def _observed_graph_launch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Callable[[nn.Module], tuple[Tensor, Mapping[str, Tensor]]],
    clip_norm: float,
    counters: dict[str, int] | None = None,
) -> tuple[bool, list[str]]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        optimizer.zero_grad(set_to_none=True)
        if counters is not None:
            counters["forward_backward_passes_started"] += 1
        with _autocast():
            loss, _ = objective(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        _synchronize()
    names = sorted({str(item.key) for item in profile.key_averages()})
    observed = any("cudagraph" in name.lower() or "cuda graph" in name.lower() for name in names)
    return observed, names


def _representative_indices(source: Any, order: Tensor) -> list[int]:
    keys = getattr(source, "batch_keys", None)
    if keys is None:
        keys = [("unknown", int(value)) for value in source.frame_counts]
    for start in range(0, len(order), ADMITTED_BATCH):
        selected = [int(value) for value in order[start : start + ADMITTED_BATCH].tolist()]
        if len(selected) != ADMITTED_BATCH:
            continue
        if len({keys[index] for index in selected}) == 1 and int(keys[selected[0]][1]) == 3:
            return selected
    raise RuntimeError("No schedule-native homogeneous B32/T3 RGB batch exists")


def _case(
    *,
    fit_id: str,
    regime: str,
    source: Any,
    recipe: Any,
    indices: Sequence[int],
    steps: int,
    warmups: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from operational.final_refit import producers
    from operational.rgb_port.train_producers import _pair_intervals, build_producer

    epoch = 1 if regime == "foreground_warmup" else 4
    device = torch.device("cuda")
    host_batch = producers.make_effective_batch(source, indices, recipe)
    batch = host_batch.to(device)
    torch.manual_seed(int(recipe.seed))
    torch.cuda.manual_seed_all(int(recipe.seed))
    model = build_producer(recipe).to(device).train()
    initial_model = copy.deepcopy(model.state_dict())
    optimizer = _optimizer(model, recipe)
    scheduler = _scheduler(optimizer, recipe)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_scheduler = copy.deepcopy(scheduler.state_dict())
    initial_rng = _rng_state()
    sampler = torch.Generator().manual_seed(int(recipe.seed))
    initial_sampler = sampler.get_state().clone()
    clip_norm = float(recipe.training_config["grad_clip_norm"])

    def objective(selected: nn.Module) -> tuple[Tensor, Mapping[str, Tensor]]:
        return producers.producer_objective(selected, batch, recipe, epoch)

    counters = {
        "forward_backward_passes_started": 0,
        "optimizer_steps_completed": 0,
    }
    original_forward = model.forward
    eager_recorder = BoundedGraphDispatcher(original_forward, max_graphs=1)
    cast(Any, model).forward = eager_recorder
    eager_step = _run_step(
        model,
        optimizer,
        scheduler,
        objective,
        clip_norm=clip_norm,
        update=True,
        counters=counters,
    )
    eager_step["output"] = _named_tensors(eager_recorder.last_output)
    eager_rng = _rng_state()

    # Establish the usable eager baseline before attempting graph capture.  A
    # capture failure therefore remains an explicit rejection with timing.
    cast(Any, model).forward = original_forward
    model.load_state_dict(initial_model)
    eager_optimizer = _optimizer(model, recipe)
    _restore_rng(initial_rng)
    torch.cuda.reset_peak_memory_stats()
    eager_times = _timed_steps(
        model,
        eager_optimizer,
        objective,
        clip_norm=clip_norm,
        steps=steps,
        counters=counters,
    )
    eager_timing = _timing(eager_times)
    if progress is not None:
        progress(
            {
                "schema": "final_refit_rgb_runtime_case_progress_v1",
                "status": "EAGER_BASELINE_COMPLETE",
                "fit_id": fit_id,
                "regime": regime,
                "scientific_optimizer_updates": 0,
                "recoverable_technical_work": dict(counters),
                "throughput": {
                    "scope": "prepared_device_batch_forward_backward_clip_optimizer",
                    "steps": steps,
                    "eager": eager_timing,
                },
                "graph_admitted": False,
            }
        )

    dispatcher: BoundedGraphDispatcher | None = None
    signature: ForwardSignature | None = None
    compile_before = _dynamo_snapshot()
    compile_started = time.perf_counter()
    try:
        model.load_state_dict(initial_model)
        optimizer.load_state_dict(initial_optimizer)
        scheduler.load_state_dict(initial_scheduler)
        _restore_rng(initial_rng)
        compiled = torch.compile(
            original_forward, backend="cudagraphs", dynamic=False, fullgraph=False
        )
        sample_part = batch.parts[0]
        sample_delta = _pair_intervals(
            sample_part.delta_t_s,
            batch_size=int(sample_part.events.shape[0]),
            steps=int(sample_part.events.shape[1]),
        )
        signature = ForwardSignature.from_call(
            sample_part.events,
            sample_delta,
            return_dense_features=True,
        )
        dispatcher = BoundedGraphDispatcher(original_forward, max_graphs=1)
        dispatcher.register(signature, compiled)
        cast(Any, model).forward = dispatcher
        warm_initial = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": _rng_state(),
        }
        for _ in range(warmups):
            _run_step(
                model,
                optimizer,
                scheduler,
                objective,
                clip_norm=clip_norm,
                update=False,
                counters=counters,
            )
        compile_seconds = time.perf_counter() - compile_started
        model.load_state_dict(warm_initial["model"])
        optimizer.load_state_dict(warm_initial["optimizer"])
        scheduler.load_state_dict(warm_initial["scheduler"])
        _restore_rng(warm_initial["rng"])
        optimizer.zero_grad(set_to_none=True)
        restored = (
            _same(model.state_dict(), warm_initial["model"])
            and _same(optimizer.state_dict(), warm_initial["optimizer"])
            and _same(scheduler.state_dict(), warm_initial["scheduler"])
            and _rng_equal(_rng_state(), warm_initial["rng"])
            and torch.equal(sampler.get_state(), initial_sampler)
            and all(value.grad is None for value in model.parameters())
        )
        compile_after = _dynamo_snapshot()
        graph_step = _run_step(
            model,
            optimizer,
            scheduler,
            objective,
            clip_norm=clip_norm,
            update=True,
            counters=counters,
        )
        graph_step["output"] = _named_tensors(dispatcher.last_output)
        graph_rng = _rng_state()
    except Exception as error:
        compile_seconds = time.perf_counter() - compile_started
        failure = {"passed": False, "error": "graph_candidate_unavailable"}
        return {
            "fit_id": fit_id,
            "regime": regime,
            "epoch": epoch,
            "scientific_optimizer_updates": 0,
            "disposable_technical_work": dict(counters),
            "signature": None if signature is None else dataclasses.asdict(signature),
            "routing": (
                {"graph_calls": 0, "eager_calls": 0, "registered_graphs": 0}
                if dispatcher is None
                else dispatcher.snapshot()
            ),
            "warmup_backward_passes_requested": warmups,
            "warmup_optimizer_updates": 0,
            "warmup_complete_state_restored": False,
            "sampler_generator_preserved": torch.equal(sampler.get_state(), initial_sampler),
            "parity": {
                name: dict(failure)
                for name in (
                    "output",
                    "loss",
                    "components",
                    "gradients_before_clip",
                    "clip_norm",
                    "gradients_after_clip",
                    "model_after_step",
                    "optimizer_after_step",
                    "scheduler_after_step",
                )
            }
            | {"rng_after_step_equal": False},
            "compile": {
                "backend": "cudagraphs",
                "seconds_before_failure": compile_seconds,
                "error_type": type(error).__name__,
                "error": str(error),
                "unique_graphs": 0,
                "steady_state_new_unique_graphs": 0,
                "observed_graph_launch": False,
            },
            "throughput": {
                "scope": "prepared_device_batch_forward_backward_clip_optimizer",
                "steps_per_arm": steps,
                "eager": eager_timing,
                "graph": None,
                "speedup": 0.0,
            },
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }

    tolerance = {"atol": 2e-5, "rtol": 2e-4}
    parity = {
        "output": _comparison(eager_step["output"], graph_step["output"], **tolerance),
        "loss": _comparison(
            {"loss": eager_step["loss"]}, {"loss": graph_step["loss"]}, **tolerance
        ),
        "components": _comparison(eager_step["components"], graph_step["components"], **tolerance),
        "gradients_before_clip": _comparison(
            eager_step["gradients_before_clip"],
            graph_step["gradients_before_clip"],
            **tolerance,
        ),
        "clip_norm": _comparison(
            {"clip_norm": eager_step["clip_norm"]},
            {"clip_norm": graph_step["clip_norm"]},
            **tolerance,
        ),
        "gradients_after_clip": _comparison(
            eager_step["gradients_after_clip"],
            graph_step["gradients_after_clip"],
            **tolerance,
        ),
        "model_after_step": _comparison(
            cast(Mapping[str, Tensor], eager_step["model"]),
            cast(Mapping[str, Tensor], graph_step["model"]),
            **tolerance,
        ),
        "optimizer_after_step": {
            "passed": _same(eager_step["optimizer"], graph_step["optimizer"], **tolerance)
        },
        "scheduler_after_step": {
            "passed": _same(eager_step["scheduler"], graph_step["scheduler"], **tolerance)
        },
        "rng_after_step_equal": _rng_equal(eager_rng, graph_rng),
    }

    # Candidate timing uses the same initial model bytes as the eager baseline.
    model.load_state_dict(initial_model)
    graph_optimizer = _optimizer(model, recipe)
    cast(Any, model).forward = dispatcher
    steady_before = _dynamo_snapshot()
    graph_times = _timed_steps(
        model,
        graph_optimizer,
        objective,
        clip_norm=clip_norm,
        steps=steps,
        counters=counters,
    )
    steady_after = _dynamo_snapshot()
    peak_bytes = int(torch.cuda.max_memory_allocated())
    observed, profiler_events = _observed_graph_launch(
        model, graph_optimizer, objective, clip_norm, counters
    )
    graph_timing = _timing(graph_times)
    steady_delta = _dynamo_delta(steady_before, steady_after)
    compile_delta = _dynamo_delta(compile_before, compile_after)
    unique_graphs = int(compile_delta.get("stats", {}).get("unique_graphs", 0))
    steady_unique = int(steady_delta.get("stats", {}).get("unique_graphs", 0))
    return {
        "fit_id": fit_id,
        "regime": regime,
        "epoch": epoch,
        "scientific_optimizer_updates": 0,
        "disposable_technical_work": dict(counters),
        "signature": dataclasses.asdict(signature),
        "routing": dispatcher.snapshot(),
        "warmup_backward_passes_requested": warmups,
        "warmup_optimizer_updates": 0,
        "warmup_complete_state_restored": restored,
        "sampler_generator_preserved": torch.equal(sampler.get_state(), initial_sampler),
        "parity": parity,
        "compile": {
            "backend": "cudagraphs",
            "dynamic": False,
            "fullgraph": False,
            "seconds": compile_seconds,
            "counter_delta": compile_delta,
            "unique_graphs": unique_graphs,
            "steady_state_counter_delta": steady_delta,
            "steady_state_new_unique_graphs": steady_unique,
            "observed_graph_launch": observed,
            "profiler_event_names": profiler_events,
        },
        "throughput": {
            "scope": "prepared_device_batch_forward_backward_clip_optimizer",
            "steps_per_arm": steps,
            "eager": eager_timing,
            "graph": graph_timing,
            "speedup": eager_timing["median_ms"] / graph_timing["median_ms"],
        },
        "peak_cuda_memory_bytes": peak_bytes,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete disposable probe; never open a scientific checkpoint."""
    if args.steps < 1 or args.warmups < 1:
        raise ValueError("steps and warmups must be positive")
    memory = _host_memory_snapshot()
    commit = memory["windows_commit_headroom_bytes"]
    if int(memory["host_available_bytes"] or 0) < args.min_host_available_bytes or (
        commit is not None and commit < args.min_windows_commit_headroom_bytes
    ):
        return {
            "schema": SCHEMA,
            "status": "RESOURCE_BLOCKED",
            "scope": "disposable_technical_probe_zero_scientific_updates",
            "scientific_optimizer_updates": 0,
            "cases": [],
            "host_memory_preflight": memory,
            "thresholds": {
                "min_host_available_bytes": args.min_host_available_bytes,
                "min_windows_commit_headroom_bytes": (args.min_windows_commit_headroom_bytes),
            },
            "decision": {
                "status": "REJECTED",
                "graph_training_enabled": False,
                "reasons": ["host_memory_preflight_failed"],
                "cases": [],
                "fallback": "eager_exact_recipe",
            },
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime probe requested but CUDA is unavailable")
    from operational.final_refit import producers
    from operational.rgb_port.train_producers import _epoch_order

    descriptor = args.source_descriptor.resolve(strict=True)
    source = producers.load_all_train_source(descriptor, args.teacher_manifest)
    try:
        generator = torch.Generator().manual_seed(7)
        order = _epoch_order(source, generator)
        histogram = schedule_signature_histogram(source, order)
        indices = _representative_indices(source, order)
        cases: list[dict[str, Any]] = []
        resource_failure: dict[str, Any] | None = None
        for fit_id in FIT_IDS:
            recipe = producers.build_rgb_recipe(
                args.config.resolve(strict=True),
                fit_id,
                descriptor,
                source,
                microbatch_size=ADMITTED_BATCH,
            )
            for regime in REGIMES:
                progress_path = args.output.resolve().with_name(
                    f"{args.output.stem}.{fit_id}.{regime}.progress.json"
                )

                def save_progress(
                    value: Mapping[str, Any], *, path: Path = progress_path
                ) -> None:
                    _atomic_json(path, value)

                try:
                    result = _case(
                        fit_id=fit_id,
                        regime=regime,
                        source=source,
                        recipe=recipe,
                        indices=indices,
                        steps=args.steps,
                        warmups=args.warmups,
                        progress=save_progress,
                    )
                except MemoryError as error:
                    progress_value = (
                        json.loads(progress_path.read_text(encoding="utf-8"))
                        if progress_path.is_file()
                        else None
                    )
                    failed = {"passed": False, "error": "resource_failure"}
                    result = {
                        "fit_id": fit_id,
                        "regime": regime,
                        "scientific_optimizer_updates": 0,
                        "recoverable_progress": progress_value,
                        "technical_work_after_last_receipt": "UNKNOWN_NOT_RECOVERABLE",
                        "warmup_complete_state_restored": False,
                        "sampler_generator_preserved": False,
                        "routing": {"graph_calls": 0},
                        "parity": {
                            name: dict(failed)
                            for name in (
                                "output",
                                "loss",
                                "components",
                                "gradients_before_clip",
                                "clip_norm",
                                "gradients_after_clip",
                                "model_after_step",
                                "optimizer_after_step",
                                "scheduler_after_step",
                            )
                        }
                        | {"rng_after_step_equal": False},
                        "compile": {
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "unique_graphs": 0,
                            "steady_state_new_unique_graphs": 0,
                            "observed_graph_launch": False,
                        },
                        "throughput": (
                            None if progress_value is None else progress_value.get("throughput")
                        ),
                        "peak_cuda_memory_bytes_unavailable": True,
                    }
                    resource_failure = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "fit_id": fit_id,
                        "regime": regime,
                    }
                finally:
                    _release_graph_runtime()
                cases.append(result)
                _atomic_json(
                    args.output.resolve(),
                    {
                        "schema": SCHEMA,
                        "status": "IN_PROGRESS" if resource_failure is None else "RESOURCE_FAILURE",
                        "scope": "disposable_technical_probe_zero_scientific_updates",
                        "scientific_optimizer_updates": 0,
                        "source_identity": dict(source.identity),
                        "host_memory_preflight": memory,
                        "cases": cases,
                        "resource_failure": resource_failure,
                        "graph_training_enabled": False,
                    },
                )
                if resource_failure is not None:
                    break
            if resource_failure is not None:
                break
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "COMPLETE" if resource_failure is None else "RESOURCE_FAILURE",
            "scope": "disposable_technical_probe_zero_scientific_updates",
            "scientific_optimizer_updates": 0,
            "source_identity": dict(source.identity),
            "host_memory_preflight": memory,
            "schedule_signature_histogram": histogram,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cases": cases,
            "resource_failure": resource_failure,
            "thresholds": {
                "min_speedup": args.min_speedup,
                "max_peak_cuda_memory_bytes": args.max_peak_cuda_memory_bytes,
            },
        }
        payload["decision"] = validate_probe(
            payload,
            min_speedup=args.min_speedup,
            max_peak_bytes=args.max_peak_cuda_memory_bytes,
        )
        return payload
    finally:
        source.close()


def parser() -> argparse.ArgumentParser:
    """Create the CLI without reading data or touching CUDA."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--teacher-manifest", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--warmups", type=int, default=3)
    result.add_argument("--min-speedup", type=float, default=1.03)
    result.add_argument("--max-peak-cuda-memory-bytes", type=int, default=10 * 1024**3)
    result.add_argument("--min-host-available-bytes", type=int, default=8 * 1024**3)
    result.add_argument("--min-windows-commit-headroom-bytes", type=int, default=12 * 1024**3)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Execute and atomically publish one technical probe result."""
    args = parser().parse_args(argv)
    output = args.output.resolve()
    try:
        payload = run_probe(args)
    except MemoryError as error:
        payload = {
            "schema": SCHEMA,
            "status": "RESOURCE_FAILURE",
            "scope": "disposable_technical_probe_zero_scientific_updates",
            "scientific_optimizer_updates": 0,
            "technical_work": "UNKNOWN_NOT_RECOVERABLE",
            "cases": [],
            "resource_failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "decision": {
                "status": "REJECTED",
                "graph_training_enabled": False,
                "reasons": ["unhandled_host_memory_failure"],
                "cases": [],
                "fallback": "eager_exact_recipe",
            },
        }
    _atomic_json(output, payload)
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0 if payload["decision"]["graph_training_enabled"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedGraphDispatcher",
    "ForwardSignature",
    "main",
    "parser",
    "schedule_signature_histogram",
    "validate_case",
    "validate_probe",
]
