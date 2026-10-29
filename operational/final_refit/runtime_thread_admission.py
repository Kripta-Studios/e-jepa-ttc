"""Admit a CPU intra-op thread setting across both final RGB recipes and regimes."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.final_refit.eager_probe import _atomic_json
from operational.final_refit.producers import build_rgb_recipe
from operational.final_refit.rgb import load_all_train_source
from operational.final_refit.runtime_thread_probe import _one_arm, _parity
from operational.rgb_port.train_producers import build_producer


def _rng() -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-descriptor", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint.resolve(True), map_location="cpu", weights_only=False)
    source = load_all_train_source(
        args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    )
    original_threads = torch.get_num_threads()
    cases: list[dict[str, Any]] = []
    technical = 0
    try:
        for fit_id in ("R_A5", "R_C2F"):
            recipe = build_rgb_recipe(
                Path("configs/rgb_port/producers.json"),
                fit_id,
                args.source_descriptor.resolve(True),
                source,
            )
            torch.manual_seed(recipe.seed)
            torch.cuda.manual_seed_all(recipe.seed)
            model = build_producer(recipe)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(recipe.training_config["learning_rate"]),
                weight_decay=float(recipe.training_config["weight_decay"]),
            )
            initial = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "cursor": {
                    **copy.deepcopy(checkpoint["cursor"]),
                    "order": checkpoint["cursor"]["order"].clone(),
                },
                **_rng(),
            }
            del optimizer, model
            for epoch in (2, 4):
                payload = copy.deepcopy(initial)
                payload["cursor"]["epoch"] = epoch
                baseline, baseline_trace = _one_arm(
                    payload=payload,
                    recipe=recipe,
                    source=source,
                    threads=original_threads,
                    steps=1,
                    run_root=args.run_root.resolve(),
                )
                candidate, candidate_trace = _one_arm(
                    payload=payload,
                    recipe=recipe,
                    source=source,
                    threads=1,
                    steps=1,
                    run_root=args.run_root.resolve(),
                )
                parity = _parity(baseline_trace, candidate_trace)
                cases.append(
                    {
                        "fit_id": fit_id,
                        "epoch": epoch,
                        "baseline": baseline,
                        "candidate": candidate,
                        "parity": parity,
                        "passed": parity["passed"],
                    }
                )
                technical += 4
                _atomic_json(
                    args.output.resolve(),
                    {
                        "schema": "final_refit_runtime_thread_admission_v1",
                        "status": "RUNNING",
                        "scientific_updates": 0,
                        "technical_optimizer_steps": technical,
                        "cases": cases,
                    },
                )
        receipt = {
            "schema": "final_refit_runtime_thread_admission_v1",
            "status": "PASS" if all(case["passed"] for case in cases) else "FAIL",
            "runtime_admission": all(case["passed"] for case in cases),
            "scientific_updates": 0,
            "technical_optimizer_steps": technical,
            "baseline_threads": original_threads,
            "candidate_threads": 1,
            "cases": cases,
            "limits": "One measured update after one disposable warmup per arm and case.",
        }
        _atomic_json(args.output.resolve(), receipt)
        print(json.dumps({"status": receipt["status"], "technical": technical}))
        return 0 if receipt["runtime_admission"] else 1
    finally:
        torch.set_num_threads(original_threads)
        source.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
