"""One disposable full-objective B32 backward per final RGB producer."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch

from operational.final_refit.eager_probe import _atomic_json, _optimizer, _step
from operational.final_refit.producers import (
    build_rgb_recipe,
    make_effective_batch,
    producer_objective,
)
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.train_producers import build_producer

SCHEMA = "final_refit_rgb_b32_backward_admission_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Measure finite full-objective B32 backward and memory without input loading."""
    descriptor, teacher = args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    source = load_all_train_source(descriptor, teacher)
    results = []
    physical_shapes: list[list[int]] = []
    try:
        groups: dict[tuple[str, int], list[int]] = {}
        for index, key in enumerate(source.batch_keys):
            groups.setdefault((str(key[0]), int(key[1])), []).append(index)
        eligible = [(key, rows) for key, rows in groups.items() if key[1] == 3 and len(rows) >= 32]
        if not eligible:
            raise ValueError("No observed homogeneous T3 B32 batch")
        key, rows = sorted(eligible)[0]
        indices = tuple(rows[:32])
        for fit_id in ("R_A5", "R_C2F"):
            recipe = build_rgb_recipe(args.config.resolve(True), fit_id, descriptor, source)
            host = make_effective_batch(source, indices, recipe)
            if not physical_shapes:
                physical_shapes = [list(part.events.shape) for part in host.parts]
            batch = host.to(torch.device("cuda"))
            torch.manual_seed(recipe.seed)
            torch.cuda.manual_seed_all(recipe.seed)
            model = build_producer(recipe).cuda().train()
            optimizer = _optimizer(model, recipe)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            step = _step(
                model,
                optimizer,
                partial(producer_objective, batch=batch, recipe=recipe, epoch=4),
                precision=str(recipe.training_config["precision"]),
                clip_norm=float(recipe.training_config["grad_clip_norm"]),
            )
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "fit_id": fit_id,
                    "recipe_identity_sha256": recipe.identity_sha256,
                    "deterministic_transport_sampling": recipe.model_config[
                        "deterministic_transport_sampling"
                    ],
                    "finite_loss": bool(torch.isfinite(step["loss"])),
                    "finite_raw_gradients": all(
                        bool(torch.isfinite(value).all())
                        for value in step["raw_gradients"].values()
                    ),
                    "finite_clipped_gradients": all(
                        bool(torch.isfinite(value).all())
                        for value in step["clipped_gradients"].values()
                    ),
                    "elapsed_step_seconds_excluding_input_loading": elapsed,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "technical_optimizer_steps": 1,
                }
            )
            del step, optimizer, model, batch, host
            torch.cuda.empty_cache()
            gc.collect()
        passed = all(
            item["finite_loss"]
            and item["finite_raw_gradients"]
            and item["finite_clipped_gradients"]
            and item["deterministic_transport_sampling"] is True
            for item in results
        )
        return {
            "schema": SCHEMA,
            "status": "PASS" if passed else "FAILED",
            "scientific_updates": 0,
            "technical_optimizer_steps": sum(item["technical_optimizer_steps"] for item in results),
            "batch_key": list(key),
            "effective_batch_size": 32,
            "physical_shapes": physical_shapes,
            "scope": "full_objective_epoch4_backward_excludes_input_loading",
            "results": results,
            "runtime_admission": passed,
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
    result = run(args)
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps({"status": result["status"], "runtime_admission": result["runtime_admission"]})
    )
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
