"""Test deterministic RGB backward twice on one reset-in-place model instance."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from operational.final_refit.eager_probe import (
    _aggregate_difference,
    _atomic_json,
    _optimizer,
    _reference_objective,
    _restore_rng,
    _rng,
    _same,
    _step,
    select_observed_cases,
)
from operational.final_refit.producers import build_rgb_recipe, make_effective_batch
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.train_producers import build_producer

SCHEMA = "final_refit_rgb_same_instance_backward_diagnostic_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute two disposable strict-deterministic steps after exact in-place reset."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG before process start")
    descriptor, teacher = args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    source = load_all_train_source(descriptor, teacher)
    try:
        recipe = build_rgb_recipe(args.config.resolve(True), "R_A5", descriptor, source)
        selection = next(
            item
            for item in select_observed_cases(source, batch_size=4)
            if item.kind == "MIXED_T2_T3"
        )
        batch = make_effective_batch(source, selection.indices, recipe).to(torch.device("cuda"))
        torch.manual_seed(recipe.seed)
        torch.cuda.manual_seed_all(recipe.seed)
        model = build_producer(recipe).cuda().train()
        if args.deterministic_transport:
            model.config = replace(model.config, deterministic_transport_sampling=True)
        initial = copy.deepcopy(model.state_dict())
        state = _rng()
        results = []
        models = []
        optimizers = []
        flags = []
        for replay in range(2):
            if replay:
                model.load_state_dict(initial)
                _restore_rng(state)
            optimizer = _optimizer(model, recipe)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_deterministic_debug_mode("error")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            with torch.autograd.set_multithreading_enabled(False):
                flags.append(
                    {
                        "deterministic": torch.are_deterministic_algorithms_enabled(),
                        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                        "autograd_multithreading": torch.autograd.is_multithreading_enabled(),
                    }
                )
                result = _step(
                    model,
                    optimizer,
                    lambda value: _reference_objective(value, batch, recipe, 4),
                    precision=str(recipe.training_config["precision"]),
                    clip_norm=float(recipe.training_config["grad_clip_norm"]),
                )
            results.append(result)
            models.append(copy.deepcopy(model.state_dict()))
            optimizers.append(copy.deepcopy(optimizer.state_dict()))
        fields = {
            name: _aggregate_difference(results[0][name], results[1][name], atol=0.0, rtol=0.0)
            for name in ("loss", "components", "raw_gradients", "clip_norm", "clipped_gradients")
        }
        fields["model_after_step"] = _aggregate_difference(models[0], models[1], atol=0, rtol=0)
        fields["optimizer_after_step"] = _aggregate_difference(
            optimizers[0], optimizers[1], atol=0, rtol=0
        )
        exact = all(item["original_allclose"] for item in fields.values())
        return {
            "schema": SCHEMA,
            "status": "SAME_INSTANCE_EXACT" if exact else "SAME_INSTANCE_NONEXACT",
            "scientific_updates": 0,
            "technical_optimizer_steps": 2,
            "technical_backward_passes": 2,
            "recipe_precision": recipe.training_config["precision"],
            "diagnostic_deterministic_transport_sampling": args.deterministic_transport,
            "initial_reset_exact": _same(model.state_dict(), models[1]),
            "effective_flags": flags,
            "comparisons": fields,
            "same_model_object_id": True,
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
    result.add_argument("--deterministic-transport", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0 if result["status"] == "SAME_INSTANCE_EXACT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
