"""Bounded deterministic-algorithm diagnosis for the failed RGB eager case."""

# ruff: noqa: ANN401 -- diagnostic wrapper preserves the probed step signature

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from operational.final_refit import eager_probe
from operational.final_refit.eager_probe import (
    _atomic_json,
    _run_case,
    _sha256,
    select_observed_cases,
)
from operational.final_refit.producers import build_rgb_recipe
from operational.final_refit.rgb import load_all_train_source

SCHEMA = "final_refit_rgb_deterministic_backward_diagnostic_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one disposable R_A5/full/MIXED case with strict deterministic diagnostics."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start")
    descriptor = args.source_descriptor.resolve(strict=True)
    teacher = args.teacher_manifest.resolve(strict=True)
    config = args.config.resolve(strict=True)
    source = load_all_train_source(descriptor, teacher)
    completed = 0
    attempted_backward_passes = 0
    effective_flags: list[dict[str, Any]] = []
    try:
        recipe = build_rgb_recipe(config, "R_A5", descriptor, source, microbatch_size=32)
        selection = next(
            item
            for item in select_observed_cases(source, batch_size=args.case_batch_size)
            if item.kind == "MIXED_T2_T3"
        )
        original_step = eager_probe._step

        def strict_step(*step_args: Any, **step_kwargs: Any) -> dict[str, Any]:
            # build_producer reseeds with warn_only=True; restore strict error mode
            # after every construction and immediately before forward/backward.
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_deterministic_debug_mode("error")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            with torch.autograd.set_multithreading_enabled(False):
                effective_flags.append(
                    {
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                        "debug_mode": int(torch.get_deterministic_debug_mode()),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "autograd_multithreading": torch.autograd.is_multithreading_enabled(),
                    }
                )
                return original_step(*step_args, **step_kwargs)

        eager_probe._step = strict_step
        attempted_backward_passes = 4
        try:
            result = _run_case(
                source=source,
                recipe=recipe,
                selection=selection,
                regime="full_objective",
                updates=1,
            )
        finally:
            eager_probe._step = original_step
        completed = 4
        comparison = result["comparisons"][0]
        exact_numerics = all(
            comparison[name]["candidate_reference"]["max_abs"] == 0.0
            and comparison[name]["reference_reference_control"]["max_abs"] == 0.0
            for name in (
                "loss",
                "components",
                "raw_gradients",
                "clip_norm",
                "clipped_gradients",
                "model_after_step",
                "optimizer_after_step",
                "scheduler_after_step",
            )
        )
        return {
            "schema": SCHEMA,
            "status": "DETERMINISTIC_EXACT" if exact_numerics else "DETERMINISTIC_NONEXACT",
            "scientific_updates": 0,
            "technical_optimizer_steps": completed,
            "technical_backward_passes_attempted": attempted_backward_passes,
            "deterministic_algorithms": True,
            "deterministic_debug_mode": "error",
            "autograd_multithreading": False,
            "effective_flags_before_each_forward_backward": effective_flags,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "recipe_precision": recipe.training_config["precision"],
            "recipe_identity_sha256": recipe.identity_sha256,
            "source_descriptor_sha256": _sha256(descriptor),
            "case": result,
            "exact_numerics": exact_numerics,
        }
    except RuntimeError as error:
        return {
            "schema": SCHEMA,
            "status": "DETERMINISTIC_OPERATOR_ERROR",
            "scientific_updates": 0,
            "technical_optimizer_steps": completed,
            "technical_backward_passes_attempted": attempted_backward_passes,
            "deterministic_algorithms": True,
            "deterministic_debug_mode": "error",
            "autograd_multithreading": False,
            "effective_flags_before_each_forward_backward": effective_flags,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
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
    result.add_argument("--case-batch-size", type=int, default=4, choices=range(2, 9))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = run(args)
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0 if payload["status"] == "DETERMINISTIC_EXACT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
