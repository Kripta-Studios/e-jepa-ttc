"""Zero-science eager-runtime admission for the two final RGB producers.

The probe uses disposable fresh models and a bounded number of optimizer steps.
It compares the final all-TRAIN effective-batch route with the retained V13
objective, validates a serialized resume, and never opens a scientific fit.
CUDA graphs and ``torch.compile`` are intentionally outside this module.
"""

# ruff: noqa: ANN401 -- the retained RGB source and recipe protocols are dynamic

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.training.causal_scale_eap import _autocast, _foreground_only_loss_config
from operational.final_refit.producer_plan import runtime_source_paths
from operational.final_refit.producers import (
    BoundedRGBPrefetch,
    ParallelRGBDecoder,
    _final_memory_allowed,
    build_rgb_recipe,
    configure_rgb_runtime,
    make_effective_batch,
    producer_objective,
)
from operational.final_refit.rgb import AllTrainRGBSource, load_all_train_source
from operational.rgb_port.recipe import ProducerRecipe, file_sha256
from operational.rgb_port.train_producers import (
    _producer_loss,
    _resource_guard,
    build_producer,
)
from operational.rgb_port_revision.loss import EffectiveBatch

SCHEMA = "final_refit_rgb_eager_runtime_admission_v1"
FIT_IDS = ("R_A5", "R_C2F")
REGIMES = {"foreground_warmup": 1, "full_objective": 4}
CASE_KINDS = ("T2", "T3", "MIXED_T2_T3")
DIAGNOSTIC_SHA256 = "13e37aba488fabde7c9b2f18cf317ba6af37def654f2446c3360b434fc71fd06"
CONTROL_POLICY = {
    "schema": "final_refit_rgb_deterministic_transport_strict_policy_v1",
    "diagnostic_sha256": DIAGNOSTIC_SHA256,
    "strict_atol": 2e-5,
    "strict_rtol": 2e-4,
    "numerical_rule": "strict_candidate_reference_and_strict_reference_reference_control",
    "exact_gates": [
        "effective_batch",
        "initial_state",
        "rng",
        "sampler_generator",
        "serialized_resume",
    ],
    "scope": "same_device_same_recipe_same_seed_same_case_bf16_eager_only",
}
CONTROL_POLICY_SHA256 = hashlib.sha256(
    json.dumps(CONTROL_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class CaseSelection:
    """Observed native rows used by one bounded runtime case."""

    kind: str
    indices: tuple[int, ...]
    observed_keys: tuple[tuple[str, int], ...]


def select_observed_cases(source: Any, *, batch_size: int) -> tuple[CaseSelection, ...]:
    """Select deterministic observed T2, T3 and mixed rows without targets."""
    if batch_size < 2 or batch_size > 8:
        raise ValueError("Technical case batch size must be in [2,8]")
    by_key: dict[tuple[str, int], list[int]] = {}
    for index, raw in enumerate(source.batch_keys):
        key = str(raw[0]), int(raw[1])
        if key[1] in {2, 3}:
            by_key.setdefault(key, []).append(index)
    choices: dict[int, tuple[tuple[str, int], list[int]]] = {}
    for frames in (2, 3):
        matches = [
            (key, values)
            for key, values in sorted(by_key.items())
            if key[1] == frames and len(values) >= batch_size
        ]
        if not matches:
            raise ValueError(f"No observed homogeneous T{frames} technical batch")
        choices[frames] = matches[0]
    half = batch_size // 2
    other = batch_size - half
    key2, rows2 = choices[2]
    key3, rows3 = choices[3]
    selections = (
        CaseSelection("T2", tuple(rows2[:batch_size]), (key2,)),
        CaseSelection("T3", tuple(rows3[:batch_size]), (key3,)),
        CaseSelection(
            "MIXED_T2_T3",
            tuple(rows2[:half] + rows3[:other]),
            tuple(sorted({key2, key3})),
        ),
    )
    if tuple(item.kind for item in selections) != CASE_KINDS:
        raise AssertionError("Technical case coverage changed")
    return selections


def _reference_batch(
    source: AllTrainRGBSource, indices: Sequence[int], recipe: ProducerRecipe
) -> EffectiveBatch:
    """Build the retained V13 grouping independently of the final wrapper."""
    groups: dict[tuple[str, int], list[int]] = {}
    for index in indices:
        groups.setdefault(source.batch_keys[int(index)], []).append(int(index))
    parts = [
        source.batch(group[start : start + recipe.microbatch_size], "rgb")
        for _, group in sorted(groups.items())
        for start in range(0, len(group), recipe.microbatch_size)
    ]
    return EffectiveBatch(parts)


def _batch_tensors(batch: EffectiveBatch) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    names = (
        "events",
        "delta_t_s",
        "target_ttc_s",
        "boxes_xyxy",
        "dinov3_relation_targets",
        "dinov3_relation_valid",
    )
    for part_index, part in enumerate(batch.parts):
        for name in names:
            value = getattr(part, name)
            if value is not None:
                result[f"part{part_index}.{name}"] = value.detach().cpu()
    return result


def _batch_identity(batch: EffectiveBatch) -> list[dict[str, Any]]:
    return [
        {
            "sample_tokens": list(part.sample_tokens),
            "sequence_ids": list(part.sequence_ids),
            "track_ids": list(part.track_ids),
            "shape": list(part.events.shape),
        }
        for part in batch.parts
    ]


def _same(left: Any, right: Any, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                return False
            return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _same(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, np.floating)):
        if not isinstance(right, (float, np.floating)):
            return False
        return bool(math.isfinite(float(left)) and math.isfinite(float(right)) and left == right)
    if isinstance(right, (float, np.floating)):
        return False
    return bool(left == right)


def _rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _optimizer(model: nn.Module, recipe: ProducerRecipe) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )


def _scheduler(optimizer: torch.optim.Optimizer, recipe: ProducerRecipe) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=recipe.epochs,
        eta_min=float(recipe.training_config["minimum_learning_rate"]),
    )


def _reference_objective(
    model: nn.Module, batch: EffectiveBatch, recipe: ProducerRecipe, epoch: int
) -> tuple[Tensor, Mapping[str, Tensor]]:
    config = CausalScaleTTCLossConfig(**recipe.loss_config)
    selected = _foreground_only_loss_config(config) if epoch <= 3 else config
    return _producer_loss(cast(Any, model), batch, selected)


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Any,
    *,
    precision: str,
    clip_norm: float,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    with _autocast(torch.device("cuda"), precision):
        loss, components = objective(model)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Eager admission produced nonfinite loss")
    loss.backward()
    raw = {
        name: value.grad.detach().clone()
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    clip = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    clipped = {
        name: value.grad.detach().clone()
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "loss": loss.detach().clone(),
        "components": {name: value.detach().clone() for name, value in components.items()},
        "raw_gradients": raw,
        "clip_norm": torch.as_tensor(clip).detach().clone(),
        "clipped_gradients": clipped,
    }


def _compare(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    return {"passed": _same(left, right, atol=atol, rtol=rtol), "atol": atol, "rtol": rtol}


def _tensor_leaves(value: Any, prefix: str = "root") -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    if isinstance(value, Tensor):
        # Optimizer state mixes CPU step counters with CUDA moment tensors.
        # Normalize diagnostic copies only; this never changes the training state.
        result[prefix] = value.detach().float().cpu()
    elif isinstance(value, Mapping):
        for name, item in value.items():
            result.update(_tensor_leaves(item, f"{prefix}.{name}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_tensor_leaves(item, f"{prefix}.{index}"))
    return result


def _aggregate_difference(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    a, b = _tensor_leaves(left), _tensor_leaves(right)
    if set(a) != set(b):
        return {
            "names_equal": False,
            "original_allclose": False,
            "max_abs": math.inf,
            "relative_l2": math.inf,
        }
    deltas: list[Tensor] = []
    left_values: list[Tensor] = []
    right_values: list[Tensor] = []
    original = True
    finite = True
    for name in sorted(a):
        if a[name].shape != b[name].shape:
            return {
                "names_equal": True,
                "shapes_equal": False,
                "original_allclose": False,
                "max_abs": math.inf,
                "relative_l2": math.inf,
            }
        x, y = a[name].reshape(-1), b[name].reshape(-1)
        finite = finite and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        original = original and bool(torch.allclose(x, y, atol=atol, rtol=rtol))
        if x.numel():
            deltas.append(x - y)
            left_values.append(x)
            right_values.append(y)
    if not deltas:
        exact = _same(left, right, atol=atol, rtol=rtol)
        return {
            "names_equal": True,
            "shapes_equal": True,
            "original_allclose": exact,
            "max_abs": 0.0 if exact else math.inf,
            "relative_l2": 0.0 if exact else math.inf,
            "finite": exact,
        }
    if not finite:
        return {
            "names_equal": True,
            "shapes_equal": True,
            "original_allclose": False,
            "max_abs": math.inf,
            "relative_l2": math.inf,
            "finite": False,
        }
    delta = torch.cat(deltas)
    joined_left, joined_right = torch.cat(left_values), torch.cat(right_values)
    denominator = max(
        float(torch.linalg.vector_norm(joined_left)),
        float(torch.linalg.vector_norm(joined_right)),
        1e-30,
    )
    return {
        "names_equal": True,
        "shapes_equal": True,
        "original_allclose": original,
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta)) / denominator,
        "finite": True,
    }


def _controlled_compare(
    candidate: Any,
    reference: Any,
    control_left: Any,
    control_right: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Require strict parity on both candidate/reference and the informative A/A control."""
    observed = _aggregate_difference(candidate, reference, atol=atol, rtol=rtol)
    control = _aggregate_difference(control_left, control_right, atol=atol, rtol=rtol)
    controlled = (
        bool(observed.get("names_equal"))
        and bool(observed.get("shapes_equal", True))
        and bool(observed.get("finite"))
        and bool(control.get("names_equal"))
        and bool(control.get("shapes_equal", True))
        and bool(control.get("finite"))
        and math.isfinite(float(observed["max_abs"]))
        and math.isfinite(float(observed["relative_l2"]))
        and math.isfinite(float(control["max_abs"]))
        and math.isfinite(float(control["relative_l2"]))
        and float(observed["max_abs"]) <= float(control["max_abs"])
        and float(observed["relative_l2"]) <= float(control["relative_l2"])
    )
    return {
        "passed": bool(observed["original_allclose"]) and bool(control["original_allclose"]),
        "strict_original_threshold_passed": bool(observed["original_allclose"]),
        "strict_control_threshold_passed": bool(control["original_allclose"]),
        "control_dominance_passed": controlled,
        "candidate_reference": observed,
        "reference_reference_control": control,
        "policy": CONTROL_POLICY["numerical_rule"],
        "atol": atol,
        "rtol": rtol,
    }


def _recompute_reported_numerical_pass(value: Mapping[str, Any]) -> bool:
    """Recompute a serialized comparison decision instead of trusting its flags."""
    observed = value.get("candidate_reference", {})
    control = value.get("reference_reference_control", {})
    if not isinstance(observed, Mapping) or not isinstance(control, Mapping):
        return False
    try:
        metrics = tuple(
            float(item[name]) for item in (observed, control) for name in ("max_abs", "relative_l2")
        )
    except (KeyError, TypeError, ValueError):
        return False
    valid = all(
        item.get("names_equal") is True
        and item.get("shapes_equal", True) is True
        and item.get("finite") is True
        for item in (observed, control)
    ) and all(math.isfinite(metric) for metric in metrics)
    controlled = valid and metrics[0] <= metrics[2] and metrics[1] <= metrics[3]
    strict = (
        value.get("strict_original_threshold_passed") is True
        and value.get("strict_control_threshold_passed") is True
        and valid
    )
    recomputed = strict
    return (
        value.get("control_dominance_passed") is controlled
        and value.get("passed") is recomputed
        and recomputed
    )


def _serialize_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    generator: torch.Generator,
    *,
    recipe: ProducerRecipe,
    cursor: Mapping[str, int],
) -> tuple[nn.Module, torch.optim.Optimizer, Any, torch.Generator, dict[str, int]]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "generator": generator.get_state(),
        "rng": _rng(),
        "cursor": dict(cursor),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cuda", weights_only=False)
    restored_model = build_producer(recipe).cuda().train()
    restored_optimizer = _optimizer(restored_model, recipe)
    restored_scheduler = _scheduler(restored_optimizer, recipe)
    restored_generator = torch.Generator().manual_seed(0)
    restored_model.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler.load_state_dict(loaded["scheduler"])
    restored_generator.set_state(loaded["generator"].cpu())
    _restore_rng(loaded["rng"])
    return (
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_generator,
        dict(loaded["cursor"]),
    )


def _run_case(
    *,
    source: AllTrainRGBSource,
    recipe: ProducerRecipe,
    selection: CaseSelection,
    regime: str,
    updates: int,
) -> dict[str, Any]:
    epoch = REGIMES[regime]
    candidate_host = make_effective_batch(source, selection.indices, recipe)
    reference_host = _reference_batch(source, selection.indices, recipe)
    batch_parity = _same(_batch_tensors(candidate_host), _batch_tensors(reference_host)) and _same(
        _batch_identity(candidate_host), _batch_identity(reference_host)
    )
    candidate_batch = candidate_host.to(torch.device("cuda"))
    reference_batch = reference_host.to(torch.device("cuda"))

    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    candidate = build_producer(recipe).cuda().train()
    initial = copy.deepcopy(candidate.state_dict())
    reference = build_producer(recipe).cuda().train()
    reference.load_state_dict(initial)
    control_left = build_producer(recipe).cuda().train()
    control_left.load_state_dict(initial)
    control_right = build_producer(recipe).cuda().train()
    control_right.load_state_dict(initial)
    initial_state_equal = all(
        _same(candidate.state_dict(), model.state_dict())
        for model in (reference, control_left, control_right)
    )
    candidate_optimizer = _optimizer(candidate, recipe)
    reference_optimizer = _optimizer(reference, recipe)
    control_left_optimizer = _optimizer(control_left, recipe)
    control_right_optimizer = _optimizer(control_right, recipe)
    candidate_scheduler = _scheduler(candidate_optimizer, recipe)
    reference_scheduler = _scheduler(reference_optimizer, recipe)
    control_left_scheduler = _scheduler(control_left_optimizer, recipe)
    control_right_scheduler = _scheduler(control_right_optimizer, recipe)
    candidate_generator = torch.Generator().manual_seed(recipe.seed)
    reference_generator = torch.Generator().manual_seed(recipe.seed)
    control_left_generator = torch.Generator().manual_seed(recipe.seed)
    control_right_generator = torch.Generator().manual_seed(recipe.seed)
    precision = str(recipe.training_config["precision"])
    clip_norm = float(recipe.training_config["grad_clip_norm"])
    tolerance = {"atol": 2e-5, "rtol": 2e-4}
    comparisons: list[dict[str, Any]] = []
    resume_exact = False
    cursor = {"epoch": epoch, "technical_update": 0}

    for update in range(updates):
        state = _rng()
        candidate_result = _step(
            candidate,
            candidate_optimizer,
            lambda model: producer_objective(model, candidate_batch, recipe, epoch),
            precision=precision,
            clip_norm=clip_norm,
        )
        candidate_rng = _rng()
        _restore_rng(state)
        reference_result = _step(
            reference,
            reference_optimizer,
            lambda model: _reference_objective(model, reference_batch, recipe, epoch),
            precision=precision,
            clip_norm=clip_norm,
        )
        reference_rng = _rng()
        _restore_rng(state)
        control_left_result = _step(
            control_left,
            control_left_optimizer,
            lambda model: _reference_objective(model, reference_batch, recipe, epoch),
            precision=precision,
            clip_norm=clip_norm,
        )
        control_left_rng = _rng()
        _restore_rng(state)
        control_right_result = _step(
            control_right,
            control_right_optimizer,
            lambda model: _reference_objective(model, reference_batch, recipe, epoch),
            precision=precision,
            clip_norm=clip_norm,
        )
        control_right_rng = _rng()
        torch.randperm(len(selection.indices), generator=candidate_generator)
        torch.randperm(len(selection.indices), generator=reference_generator)
        torch.randperm(len(selection.indices), generator=control_left_generator)
        torch.randperm(len(selection.indices), generator=control_right_generator)
        cursor["technical_update"] = update + 1
        comparisons.append(
            {
                "update": update + 1,
                "loss": _controlled_compare(
                    candidate_result["loss"],
                    reference_result["loss"],
                    control_left_result["loss"],
                    control_right_result["loss"],
                    **tolerance,
                ),
                "components": _controlled_compare(
                    candidate_result["components"],
                    reference_result["components"],
                    control_left_result["components"],
                    control_right_result["components"],
                    **tolerance,
                ),
                "raw_gradients": _controlled_compare(
                    candidate_result["raw_gradients"],
                    reference_result["raw_gradients"],
                    control_left_result["raw_gradients"],
                    control_right_result["raw_gradients"],
                    **tolerance,
                ),
                "clip_norm": _controlled_compare(
                    candidate_result["clip_norm"],
                    reference_result["clip_norm"],
                    control_left_result["clip_norm"],
                    control_right_result["clip_norm"],
                    **tolerance,
                ),
                "clipped_gradients": _controlled_compare(
                    candidate_result["clipped_gradients"],
                    reference_result["clipped_gradients"],
                    control_left_result["clipped_gradients"],
                    control_right_result["clipped_gradients"],
                    **tolerance,
                ),
                "model_after_step": _controlled_compare(
                    candidate.state_dict(),
                    reference.state_dict(),
                    control_left.state_dict(),
                    control_right.state_dict(),
                    **tolerance,
                ),
                "optimizer_after_step": _controlled_compare(
                    candidate_optimizer.state_dict(),
                    reference_optimizer.state_dict(),
                    control_left_optimizer.state_dict(),
                    control_right_optimizer.state_dict(),
                    **tolerance,
                ),
                "scheduler_after_step": _controlled_compare(
                    candidate_scheduler.state_dict(),
                    reference_scheduler.state_dict(),
                    control_left_scheduler.state_dict(),
                    control_right_scheduler.state_dict(),
                    **tolerance,
                ),
                "rng_equal": all(
                    _same(candidate_rng, value)
                    for value in (reference_rng, control_left_rng, control_right_rng)
                ),
                "sampler_generator_equal": all(
                    torch.equal(candidate_generator.get_state(), value.get_state())
                    for value in (
                        reference_generator,
                        control_left_generator,
                        control_right_generator,
                    )
                ),
            }
        )
        if update == 0:
            before = {
                "model": copy.deepcopy(candidate.state_dict()),
                "optimizer": copy.deepcopy(candidate_optimizer.state_dict()),
                "scheduler": copy.deepcopy(candidate_scheduler.state_dict()),
                "generator": candidate_generator.get_state().clone(),
                "rng": _rng(),
                "cursor": dict(cursor),
            }
            restored = _serialize_resume(
                candidate,
                candidate_optimizer,
                candidate_scheduler,
                candidate_generator,
                recipe=recipe,
                cursor=cursor,
            )
            del candidate, candidate_optimizer, candidate_scheduler, candidate_generator
            candidate, candidate_optimizer, candidate_scheduler, candidate_generator, cursor = (
                restored
            )
            resume_exact = (
                _same(candidate.state_dict(), before["model"])
                and _same(candidate_optimizer.state_dict(), before["optimizer"])
                and _same(candidate_scheduler.state_dict(), before["scheduler"])
                and torch.equal(candidate_generator.get_state(), before["generator"])
                and _same(_rng(), before["rng"])
                and cursor == before["cursor"]
            )

    fields = (
        "loss",
        "components",
        "raw_gradients",
        "clip_norm",
        "clipped_gradients",
        "model_after_step",
        "optimizer_after_step",
        "scheduler_after_step",
    )
    passed = (
        batch_parity
        and initial_state_equal
        and resume_exact
        and all(
            all(bool(item[field]["passed"]) for field in fields)
            and bool(item["rng_equal"])
            and bool(item["sampler_generator_equal"])
            for item in comparisons
        )
    )
    return {
        "fit_id": recipe.fit_id,
        "regime": regime,
        "epoch": epoch,
        "case_kind": selection.kind,
        "indices": list(selection.indices),
        "observed_keys": [list(item) for item in selection.observed_keys],
        "batch_parity": batch_parity,
        "initial_state_equal": initial_state_equal,
        "resume_exact": resume_exact,
        "seed": recipe.seed,
        "recipe_identity_sha256": recipe.identity_sha256,
        "comparisons": comparisons,
        "candidate_technical_optimizer_steps": updates,
        "reference_technical_optimizer_steps": updates,
        "control_technical_optimizer_steps": 2 * updates,
        "control_policy_sha256": CONTROL_POLICY_SHA256,
        "scientific_updates": 0,
        "passed": passed,
    }


def _runtime_route_admission(
    source: AllTrainRGBSource, recipes: Mapping[str, ProducerRecipe]
) -> dict[str, Any]:
    """Exercise the production decoder/prefetch and one full-B32 forward per arm."""
    by_key: dict[tuple[str, int], list[int]] = {}
    for index, raw in enumerate(source.batch_keys):
        by_key.setdefault((str(raw[0]), int(raw[1])), []).append(index)
    eligible = [
        (key, rows)
        for key, rows in sorted(by_key.items(), key=lambda item: (-item[0][1], item[0][0]))
        if len(rows) >= 32
    ]
    if not eligible:
        raise ValueError("Runtime route admission requires an observed physical B32 batch")
    admitted_key, admitted_rows = eligible[0]
    indices = tuple(admitted_rows[:32])
    selected = set(indices)
    order = torch.tensor(
        [*indices, *(index for index in range(source.population_size) if index not in selected)],
        dtype=torch.int64,
    )
    reference = make_effective_batch(source, indices, recipes["R_A5"])
    cursor: dict[str, Any] = {"order": order, "position": 0}
    decoder = ParallelRGBDecoder(source, workers=4, max_pending_bytes=128 * 1024**2)
    prefetch = BoundedRGBPrefetch(source, recipes["R_A5"], cursor, depth=2)
    try:
        actual = prefetch.get(indices)
        batch_parity = _same(_batch_tensors(actual), _batch_tensors(reference)) and _same(
            _batch_identity(actual), _batch_identity(reference)
        )
        device_batch = actual.to(torch.device("cuda"))
        torch.cuda.reset_peak_memory_stats()
        forwards: dict[str, Any] = {}
        for fit_id, recipe in recipes.items():
            torch.manual_seed(recipe.seed)
            torch.cuda.manual_seed_all(recipe.seed)
            model = build_producer(recipe).cuda().train()
            with (
                torch.no_grad(),
                _autocast(torch.device("cuda"), str(recipe.training_config["precision"])),
            ):
                loss, components = producer_objective(model, device_batch, recipe, 4)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(loss)) and all(
                bool(torch.isfinite(value).all()) for value in components.values()
            )
            forwards[fit_id] = {
                "finite": finite,
                "loss": float(loss.detach().float().cpu()),
                "component_names": sorted(components),
            }
            del model, loss, components
            torch.cuda.empty_cache()
        peak = int(torch.cuda.max_memory_allocated())
        passed = batch_parity and all(item["finite"] for item in forwards.values())
        return {
            "passed": passed,
            "effective_batch_size": 32,
            "physical_batch_key": list(admitted_key),
            "physical_part_shapes": [list(part.events.shape) for part in actual.parts],
            "indices": list(indices),
            "batch_parity": batch_parity,
            "decoder": decoder.snapshot(),
            "prefetch": prefetch.snapshot(),
            "forward_only": forwards,
            "technical_forward_only_passes": len(forwards),
            "technical_optimizer_steps": 0,
            "scientific_updates": 0,
            "peak_cuda_memory_bytes": peak,
            "scope": "full_B32_decode_prefetch_forward_only_no_backward",
        }
    finally:
        prefetch.close()
        decoder.close()
        torch.cuda.empty_cache()
        gc.collect()


def validate_receipt(payload: Mapping[str, Any]) -> None:
    """Fail closed unless every arm/regime/observed-time case passed."""
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("scientific_updates") != 0
    ):
        raise ValueError("Eager admission schema or science accounting differs")
    cases = payload.get("cases", [])
    expected = {
        (fit_id, regime, kind) for fit_id in FIT_IDS for regime in REGIMES for kind in CASE_KINDS
    }
    observed = {(item.get("fit_id"), item.get("regime"), item.get("case_kind")) for item in cases}
    if observed != expected or len(cases) != len(expected):
        raise ValueError("Eager admission case coverage or parity failed")
    fields = (
        "loss",
        "components",
        "raw_gradients",
        "clip_norm",
        "clipped_gradients",
        "model_after_step",
        "optimizer_after_step",
        "scheduler_after_step",
    )
    expected_updates = int(payload.get("updates_per_case_per_path", 0))
    if expected_updates not in {2, 3}:
        raise ValueError("Eager admission update count differs")
    for case in cases:
        comparisons = case.get("comparisons", [])
        exact = (
            case.get("batch_parity") is True
            and case.get("initial_state_equal") is True
            and case.get("resume_exact") is True
            and case.get("scientific_updates") == 0
            and len(comparisons) == expected_updates
            and case.get("candidate_technical_optimizer_steps") == expected_updates
            and case.get("reference_technical_optimizer_steps") == expected_updates
            and case.get("control_technical_optimizer_steps") == 2 * expected_updates
        )
        numerical = all(
            item.get("update") == index
            and item.get("rng_equal") is True
            and item.get("sampler_generator_equal") is True
            and all(
                isinstance(item.get(field), Mapping)
                and item[field].get("policy") == CONTROL_POLICY["numerical_rule"]
                and _recompute_reported_numerical_pass(item[field])
                for field in fields
            )
            for index, item in enumerate(comparisons, start=1)
        )
        if not exact or not numerical or case.get("passed") is not True:
            raise ValueError("Eager admission case exact or numerical gate failed")
    if payload.get("technical_optimizer_steps") != sum(
        int(item["candidate_technical_optimizer_steps"])
        + int(item["reference_technical_optimizer_steps"])
        + int(item["control_technical_optimizer_steps"])
        for item in cases
    ):
        raise ValueError("Eager admission technical accounting differs")
    if payload.get("control_policy") != CONTROL_POLICY:
        raise ValueError("Eager admission numerical control policy object differs")
    if payload.get("control_policy_sha256") != CONTROL_POLICY_SHA256 or any(
        item.get("control_policy_sha256") != CONTROL_POLICY_SHA256 for item in cases
    ):
        raise ValueError("Eager admission numerical control policy differs")
    runtime_route = payload.get("runtime_route", {})
    if (
        payload.get("runtime_admission") is not True
        or payload.get("producer_plan_runtime_admission") != "PASS"
        or runtime_route.get("passed") is not True
        or runtime_route.get("effective_batch_size") != 32
        or runtime_route.get("technical_optimizer_steps") != 0
        or runtime_route.get("scientific_updates") != 0
        or runtime_route.get("technical_forward_only_passes") != len(FIT_IDS)
    ):
        raise ValueError("Production decoder/prefetch B32 forward admission failed")


def _executed_hashes(repository: Path) -> dict[str, str]:
    paths = {
        "operational/final_refit/eager_probe.py": Path(__file__).resolve(),
        **runtime_source_paths(repository),
        "configs/rgb_port/producers.json": repository / "configs/rgb_port/producers.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute bounded disposable eager cases and persist progress after each."""
    runtime_settings = configure_rgb_runtime()
    if args.updates_per_case not in {2, 3}:
        raise ValueError("updates-per-case must be 2 or 3")
    repository = Path(__file__).resolve().parents[2]
    descriptor = args.source_descriptor.resolve(strict=True)
    teacher = args.teacher_manifest.resolve(strict=True)
    config = args.config.resolve(strict=True)
    allowed, resources = _resource_guard(
        args.output.resolve().parent, checkpoint_reservation=512 * 1024**2
    )
    if not _final_memory_allowed(allowed, resources):
        return {
            "schema": SCHEMA,
            "status": "RESOURCE_BLOCKED",
            "scientific_updates": 0,
            "technical_optimizer_steps": 0,
            "technical_forward_only_passes": 0,
            "cases": [],
            "resources": resources,
            "runtime_admission": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Eager CUDA admission requested but CUDA is unavailable")
    source = load_all_train_source(descriptor, teacher)
    cases: list[dict[str, Any]] = []
    technical = 0
    runtime_route: dict[str, Any] | None = None
    try:
        selections = select_observed_cases(source, batch_size=args.case_batch_size)
        recipes = {
            fit_id: build_rgb_recipe(config, fit_id, descriptor, source, microbatch_size=32)
            for fit_id in FIT_IDS
        }
        base: dict[str, Any] = {
            "schema": SCHEMA,
            "runtime_settings": runtime_settings,
            "status": "RUNNING",
            "backend": "eager_bf16_no_compile_no_graphs",
            "scientific_updates": 0,
            "technical_optimizer_steps": 0,
            "technical_forward_only_passes": 0,
            "updates_per_case_per_path": args.updates_per_case,
            "control_policy": CONTROL_POLICY,
            "control_policy_sha256": CONTROL_POLICY_SHA256,
            "source_descriptor_sha256": file_sha256(descriptor),
            "teacher_manifest_sha256": file_sha256(teacher),
            "source_identity": dict(source.identity),
            "executed_source_sha256": _executed_hashes(repository),
            "resources": resources,
            "cases": cases,
            "runtime_admission": False,
        }
        _atomic_json(args.output.resolve(), base)
        runtime_route = _runtime_route_admission(source, recipes)
        _atomic_json(
            args.output.resolve(),
            {
                **base,
                "technical_forward_only_passes": runtime_route["technical_forward_only_passes"],
                "runtime_route": runtime_route,
            },
        )
        if not runtime_route["passed"]:
            return {
                **base,
                "status": "FAILED",
                "technical_forward_only_passes": runtime_route["technical_forward_only_passes"],
                "runtime_route": runtime_route,
                "runtime_admission": False,
                "failure": "Production decoder/prefetch B32 route parity failed",
            }
        for fit_id in FIT_IDS:
            recipe = recipes[fit_id]
            for regime in REGIMES:
                for selection in selections:
                    result = _run_case(
                        source=source,
                        recipe=recipe,
                        selection=selection,
                        regime=regime,
                        updates=args.updates_per_case,
                    )
                    cases.append(result)
                    technical += 4 * args.updates_per_case
                    _atomic_json(
                        args.output.resolve(),
                        {
                            **base,
                            "technical_optimizer_steps": technical,
                            "technical_forward_only_passes": runtime_route[
                                "technical_forward_only_passes"
                            ],
                            "runtime_route": runtime_route,
                            "cases": cases,
                        },
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
        payload = {
            **base,
            "status": "PASS",
            "technical_optimizer_steps": technical,
            "technical_forward_only_passes": runtime_route["technical_forward_only_passes"],
            "runtime_route": runtime_route,
            "cases": cases,
            "runtime_admission": True,
            "producer_plan_runtime_admission": "PASS",
        }
        try:
            validate_receipt(payload)
        except ValueError as error:
            return {
                **payload,
                "status": "FAILED",
                "runtime_admission": False,
                "producer_plan_runtime_admission": "FAILED",
                "failure": str(error),
            }
        return payload
    except MemoryError as error:
        return {
            "schema": SCHEMA,
            "status": "RESOURCE_FAILURE",
            "backend": "eager_bf16_no_compile_no_graphs",
            "scientific_updates": 0,
            "technical_optimizer_steps": technical,
            "technical_forward_only_passes": (
                0
                if runtime_route is None
                else runtime_route.get("technical_forward_only_passes", 0)
            ),
            "runtime_route": runtime_route,
            "technical_work_after_last_receipt": "UNKNOWN_NOT_RECOVERABLE",
            "cases": cases,
            "runtime_admission": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    finally:
        source.close()
        torch.cuda.empty_cache()
        gc.collect()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--teacher-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--updates-per-case", type=int, choices=(2, 3), default=2)
    result.add_argument("--case-batch-size", type=int, choices=range(2, 9), default=4)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = run(args)
    _atomic_json(args.output.resolve(), payload)
    print(
        json.dumps({"status": payload["status"], "runtime_admission": payload["runtime_admission"]})
    )
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CaseSelection", "main", "parser", "select_observed_cases", "validate_receipt"]
