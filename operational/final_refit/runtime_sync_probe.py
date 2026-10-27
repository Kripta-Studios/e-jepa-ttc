"""Compare the production RGB step with an explicitly synchronized timing step."""

# ruff: noqa: ANN401 -- retained checkpoint/source protocols are dynamic

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.eager_probe import _atomic_json
from operational.final_refit.producers import (
    BoundedRGBPrefetch,
    ParallelRGBDecoder,
    _validate_batch,
    build_rgb_recipe,
    producer_objective,
)
from operational.final_refit.rgb import load_all_train_source
from operational.final_refit.runtime_thread_probe import (
    _fresh_cursor,
    _optimizer,
    _parity,
    _restore_rng,
)
from operational.rgb_port.train_producers import _fast_resource_guard, build_producer


def _arm(
    payload: Mapping[str, Any],
    recipe: Any,
    source: Any,
    run_root: Path,
    *,
    synchronized: bool,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = build_producer(recipe).cuda().train()
    model.load_state_dict(payload["model_state_dict"])
    optimizer = _optimizer(model, recipe, payload)
    _restore_rng(payload)
    cursor = _fresh_cursor(payload)
    decoder = ParallelRGBDecoder(source, workers=4, max_pending_bytes=128 * 1024**2)
    prefetch = BoundedRGBPrefetch(source, recipe, cursor, depth=2)
    elapsed: list[float] = []
    evidence: list[dict[str, Any]] = []
    try:
        for update in range(steps + 1):
            start = int(cursor["position"])
            stop = start + recipe.effective_batch_size
            indices = cursor["order"][start:stop].tolist()
            wall = time.perf_counter()
            _fast_resource_guard(run_root, checkpoint_reservation=1 * 1024**3)
            batch = prefetch.get(indices).to(torch.device("cuda"))
            if synchronized:
                torch.cuda.synchronize()
            for part in batch.parts:
                _validate_batch(part, recipe)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch.device("cuda"), str(recipe.training_config["precision"])):
                loss, components = producer_objective(
                    model, batch, recipe, int(payload["cursor"]["epoch"])
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite technical loss")
            loss.backward()
            clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(recipe.training_config["grad_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            if synchronized:
                torch.cuda.synchronize()
                scalars = torch.stack(
                    [loss.detach().float(), *(x.detach().float() for x in components.values())]
                ).cpu()
            else:
                scalars = torch.tensor(
                    [
                        float(loss.detach().float().cpu()),
                        *(float(x.detach().float().cpu()) for x in components.values()),
                    ]
                )
            cursor["position"] = stop
            if update:
                elapsed.append((time.perf_counter() - wall) * 1000)
                if len(evidence) < 2:
                    evidence.append(
                        {
                            "loss": loss.detach().clone(),
                            "components": {k: v.detach().clone() for k, v in components.items()},
                            "clip": torch.as_tensor(clip).detach().clone(),
                            "gradients": {
                                k: p.grad.detach().clone()
                                for k, p in model.named_parameters()
                                if p.grad is not None
                            },
                            "model": copy.deepcopy(model.state_dict()),
                            "optimizer": copy.deepcopy(optimizer.state_dict()),
                            "rng": {
                                "cpu": torch.get_rng_state().clone(),
                                "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()],
                            },
                            "scalars": scalars.clone(),
                            "indices": list(indices),
                        }
                    )
        values = torch.tensor(elapsed)
        return (
            {
                "path": "instrumented" if synchronized else "production_equivalent",
                "threads": torch.get_num_threads(),
                "steps": steps,
                "mean_ms": float(values.mean()),
                "median_ms": float(values.median()),
                "min_ms": min(elapsed),
                "max_ms": max(elapsed),
                "updates_per_second": 1000.0 / (sum(elapsed) / len(elapsed)),
                "prefetch": prefetch.snapshot(),
                "decoder": decoder.snapshot(),
            },
            evidence,
        )
    finally:
        prefetch.close()
        decoder.close()
        del optimizer, model
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-descriptor", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    payload = torch.load(args.checkpoint.resolve(True), map_location="cpu", weights_only=False)
    source = load_all_train_source(
        args.source_descriptor.resolve(True), args.teacher_manifest.resolve(True)
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        recipe = build_rgb_recipe(
            Path("configs/rgb_port/producers.json"),
            "R_A5",
            args.source_descriptor.resolve(True),
            source,
        )
        arms: list[dict[str, Any]] = []
        traces: list[list[dict[str, Any]]] = []
        for synchronized in (False, True, False, True):
            arm, trace = _arm(
                payload, recipe, source, args.run_root.resolve(),
                synchronized=synchronized, steps=args.steps,
            )
            arms.append(arm)
            traces.append(trace)
            _atomic_json(
                args.output.resolve(),
                {"schema": "final_refit_runtime_sync_probe_v1", "status": "RUNNING",
                 "scientific_updates": 0, "technical_optimizer_steps": len(arms) * (args.steps + 1),
                 "arms": arms},
            )
        parity = [_parity(traces[0], trace) for trace in traces[1:]]
        receipt = {
            "schema": "final_refit_runtime_sync_probe_v1",
            "status": "PASS" if all(x["passed"] for x in parity) else "FAIL",
            "scientific_updates": 0,
            "technical_optimizer_steps": len(arms) * (args.steps + 1),
            "arms": arms,
            "parity_vs_first": parity,
            "scope": (
                "Step path only; durable journal, recovery scan, and checkpoint writes omitted."
            ),
        }
        _atomic_json(args.output.resolve(), receipt)
        print(json.dumps({"status": receipt["status"], "arms": arms}))
        return 0 if receipt["status"] == "PASS" else 1
    finally:
        torch.set_num_threads(previous_threads)
        source.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
