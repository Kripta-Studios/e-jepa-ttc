"""Bounded eager BF16 reproducibility diagnostic for one failed admission case."""

# ruff: noqa: ANN401 -- diagnostic bridges retained runtime protocols

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from operational.final_refit.eager_probe import (
    _atomic_json,
    _optimizer,
    _reference_batch,
    _reference_objective,
    _restore_rng,
    _rng,
    _step,
    select_observed_cases,
)
from operational.final_refit.producers import (
    _final_memory_allowed,
    build_rgb_recipe,
    make_effective_batch,
    producer_objective,
)
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.recipe import ProducerRecipe, file_sha256
from operational.rgb_port.train_producers import _resource_guard, build_producer

SCHEMA = "final_refit_rgb_eager_bf16_diagnostic_v1"
ATOL = 2e-5
RTOL = 2e-4


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_leaves(value: Any, prefix: str = "root") -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    if isinstance(value, Tensor):
        result[prefix] = value.detach().float().cpu()
    elif isinstance(value, Mapping):
        for key, item in value.items():
            result.update(_tensor_leaves(item, f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_tensor_leaves(item, f"{prefix}.{index}"))
    return result


def tensor_differences(left: Any, right: Any, *, limit: int = 30) -> dict[str, Any]:
    """Quantify parameter-local absolute, relative-L2 and cosine differences."""
    a = _tensor_leaves(left)
    b = _tensor_leaves(right)
    shared = sorted(set(a) & set(b))
    rows: list[dict[str, Any]] = []
    aggregate_left: list[Tensor] = []
    aggregate_right: list[Tensor] = []
    for name in shared:
        x, y = a[name], b[name]
        if x.shape != y.shape:
            rows.append({"name": name, "shape_equal": False})
            continue
        flat_x, flat_y = x.reshape(-1), y.reshape(-1)
        delta = flat_x - flat_y
        left_norm = float(torch.linalg.vector_norm(flat_x))
        right_norm = float(torch.linalg.vector_norm(flat_y))
        delta_norm = float(torch.linalg.vector_norm(delta))
        denominator = max(left_norm, right_norm, 1e-30)
        cosine = (
            float(torch.nn.functional.cosine_similarity(flat_x, flat_y, dim=0))
            if flat_x.numel() and left_norm > 0 and right_norm > 0
            else None
        )
        passed = bool(torch.allclose(x, y, atol=ATOL, rtol=RTOL, equal_nan=True))
        rows.append(
            {
                "name": name,
                "shape_equal": True,
                "elements": int(x.numel()),
                "allclose_original_threshold": passed,
                "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                "delta_l2": delta_norm,
                "left_l2": left_norm,
                "right_l2": right_norm,
                "relative_l2": delta_norm / denominator,
                "cosine": cosine,
            }
        )
        if flat_x.numel():
            aggregate_left.append(flat_x)
            aggregate_right.append(flat_y)
    ordered = sorted(
        rows,
        key=lambda item: (
            not bool(item.get("shape_equal", False)),
            float(item.get("relative_l2", math.inf)),
            float(item.get("max_abs", math.inf)),
        ),
        reverse=True,
    )
    joined_left = torch.cat(aggregate_left) if aggregate_left else torch.zeros(0)
    joined_right = torch.cat(aggregate_right) if aggregate_right else torch.zeros(0)
    joined_delta = joined_left - joined_right
    aggregate_denominator = max(
        float(torch.linalg.vector_norm(joined_left)),
        float(torch.linalg.vector_norm(joined_right)),
        1e-30,
    )
    return {
        "names_equal": set(a) == set(b),
        "tensor_count": len(shared),
        "failed_tensor_count_original_threshold": sum(
            not bool(item.get("allclose_original_threshold", False)) for item in rows
        ),
        "aggregate": {
            "elements": int(joined_left.numel()),
            "max_abs": float(joined_delta.abs().max()) if joined_delta.numel() else 0.0,
            "delta_l2": float(torch.linalg.vector_norm(joined_delta)),
            "relative_l2": float(torch.linalg.vector_norm(joined_delta)) / aggregate_denominator,
            "cosine": (
                float(torch.nn.functional.cosine_similarity(joined_left, joined_right, dim=0))
                if joined_left.numel()
                and bool(torch.linalg.vector_norm(joined_left) > 0)
                and bool(torch.linalg.vector_norm(joined_right) > 0)
                else None
            ),
        },
        "worst_tensors": ordered[:limit],
        "original_threshold": {"atol": ATOL, "rtol": RTOL},
    }


def _pair(
    *,
    recipe: ProducerRecipe,
    left_batch: Any,
    right_batch: Any,
    left_kind: str,
    right_kind: str,
) -> dict[str, Any]:
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    left = build_producer(recipe).cuda().train()
    initial = copy.deepcopy(left.state_dict())
    right = build_producer(recipe).cuda().train()
    right.load_state_dict(initial)
    initial_state_exact = all(
        torch.equal(left.state_dict()[name], right.state_dict()[name]) for name in initial
    )
    left_optimizer = _optimizer(left, recipe)
    right_optimizer = _optimizer(right, recipe)
    precision = str(recipe.training_config["precision"])
    clip_norm = float(recipe.training_config["grad_clip_norm"])

    def objective(kind: str, model: nn.Module, batch: Any) -> Any:
        if kind == "candidate":
            return producer_objective(model, batch, recipe, 4)
        if kind == "reference":
            return _reference_objective(model, batch, recipe, 4)
        raise ValueError(f"Unknown diagnostic objective kind: {kind}")

    rng = _rng()
    left_result = _step(
        left,
        left_optimizer,
        lambda model: objective(left_kind, model, left_batch),
        precision=precision,
        clip_norm=clip_norm,
    )
    left_rng = _rng()
    _restore_rng(rng)
    right_result = _step(
        right,
        right_optimizer,
        lambda model: objective(right_kind, model, right_batch),
        precision=precision,
        clip_norm=clip_norm,
    )
    right_rng = _rng()
    return {
        "left": left_kind,
        "right": right_kind,
        "initial_state_exact": initial_state_exact,
        "rng_equal": all(
            torch.equal(a, b) for a, b in zip(left_rng["cuda"], right_rng["cuda"], strict=True)
        )
        and torch.equal(left_rng["cpu"], right_rng["cpu"]),
        "loss": tensor_differences({"loss": left_result["loss"]}, {"loss": right_result["loss"]}),
        "components": tensor_differences(left_result["components"], right_result["components"]),
        "raw_gradients": tensor_differences(
            left_result["raw_gradients"], right_result["raw_gradients"]
        ),
        "clipped_gradients": tensor_differences(
            left_result["clipped_gradients"], right_result["clipped_gradients"]
        ),
        "clip_norm": tensor_differences(
            {"clip_norm": left_result["clip_norm"]},
            {"clip_norm": right_result["clip_norm"]},
        ),
        "model_after_step": tensor_differences(left.state_dict(), right.state_dict()),
        "optimizer_after_step": tensor_differences(
            left_optimizer.state_dict(), right_optimizer.state_dict()
        ),
        "technical_optimizer_steps": 2,
        "scientific_updates": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
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
            "resources": resources,
        }
    source = load_all_train_source(descriptor, teacher)
    try:
        selection = next(
            item for item in select_observed_cases(source, batch_size=4) if item.kind == "T2"
        )
        recipe = build_rgb_recipe(config, "R_A5", descriptor, source, microbatch_size=32)
        candidate = make_effective_batch(source, selection.indices, recipe).to(torch.device("cuda"))
        reference = _reference_batch(source, selection.indices, recipe).to(torch.device("cuda"))
        candidate_reference = _pair(
            recipe=recipe,
            left_batch=candidate,
            right_batch=reference,
            left_kind="candidate",
            right_kind="reference",
        )
        reference_reference = _pair(
            recipe=recipe,
            left_batch=reference,
            right_batch=reference,
            left_kind="reference",
            right_kind="reference",
        )
        aa_noise = reference_reference["raw_gradients"]["aggregate"]
        ab_noise = candidate_reference["raw_gradients"]["aggregate"]
        return {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "scope": "one_R_A5_full_objective_T2_case_plus_reference_reference_control",
            "scientific_updates": 0,
            "technical_optimizer_steps": 4,
            "recipe_unchanged": True,
            "precision_unchanged": "bf16",
            "loss_unchanged": True,
            "seed": recipe.seed,
            "selection": asdict(selection),
            "source_descriptor_sha256": file_sha256(descriptor),
            "teacher_manifest_sha256": file_sha256(teacher),
            "executed_source_sha256": {
                "operational/final_refit/eager_diagnostic.py": _sha256(Path(__file__)),
                "operational/final_refit/eager_probe.py": _sha256(
                    Path(__file__).with_name("eager_probe.py")
                ),
            },
            "resources": resources,
            "candidate_reference": candidate_reference,
            "reference_reference_control": reference_reference,
            "interpretation_inputs": {
                "aa_raw_gradient_relative_l2": aa_noise["relative_l2"],
                "ab_raw_gradient_relative_l2": ab_noise["relative_l2"],
                "aa_raw_gradient_max_abs": aa_noise["max_abs"],
                "ab_raw_gradient_max_abs": ab_noise["max_abs"],
                "policy_frozen": False,
                "no_tolerance_change_authorized": True,
            },
        }
    finally:
        source.close()
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-descriptor", type=Path, required=True)
    result.add_argument("--teacher-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = run(args)
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "scientific_updates": 0}))
    return 0 if payload["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
