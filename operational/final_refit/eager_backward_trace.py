"""Trace module-boundary gradients for one failed RGB eager case."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

import e_jepa_ttc.models.causal_scale_ttc as causal_module
import e_jepa_ttc.models.local_transport as transport_module
from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.eager_probe import (
    _atomic_json,
    _reference_objective,
    _restore_rng,
    _rng,
    select_observed_cases,
)
from operational.final_refit.producers import build_rgb_recipe, make_effective_batch
from operational.final_refit.rgb import load_all_train_source
from operational.rgb_port.train_producers import build_producer

SCHEMA = "final_refit_rgb_backward_module_trace_v1"


def _fingerprint(values: tuple[Tensor | None, ...]) -> list[dict[str, Any] | None]:
    result = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        cpu = value.detach().float().cpu().contiguous()
        result.append(
            {
                "shape": list(value.shape),
                "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                "max_abs": float(cpu.abs().max()) if cpu.numel() else 0.0,
                "l2": float(torch.linalg.vector_norm(cpu)) if cpu.numel() else 0.0,
                "finite": bool(torch.isfinite(cpu).all()),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run two same-instance backwards and find the first module-boundary difference."""
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
        initial = copy.deepcopy(model.state_dict())
        state = _rng()
        traces: list[list[dict[str, Any]]] = []
        flags = []
        for replay in range(2):
            if replay:
                model.load_state_dict(initial)
                _restore_rng(state)
            model.zero_grad(set_to_none=True)
            trace: list[dict[str, Any]] = []
            handles = []
            for name, module in model.named_modules():
                if not name:
                    continue

                def hook(
                    current: nn.Module,
                    grad_input: tuple[Tensor | None, ...],
                    grad_output: tuple[Tensor | None, ...],
                    *,
                    module_name: str = name,
                    trace_target: list[dict[str, Any]] = trace,
                ) -> None:
                    trace_target.append(
                        {
                            "module": module_name,
                            "type": type(current).__qualname__,
                            "grad_input": _fingerprint(grad_input),
                            "grad_output": _fingerprint(grad_output),
                        }
                    )

                handles.append(module.register_full_backward_hook(cast(Any, hook)))
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_deterministic_debug_mode("error")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                with (
                    torch.autograd.set_multithreading_enabled(False),
                    _autocast(torch.device("cuda"), str(recipe.training_config["precision"])),
                ):
                    loss, _ = _reference_objective(model, batch, recipe, 4)
                    flags.append(
                        {
                            "after_forward_deterministic": (
                                torch.are_deterministic_algorithms_enabled()
                            ),
                            "after_forward_warn_only": (
                                torch.is_deterministic_algorithms_warn_only_enabled()
                            ),
                            "after_forward_debug_mode": int(torch.get_deterministic_debug_mode()),
                            "after_forward_autograd_multithreading": (
                                torch.autograd.is_multithreading_enabled()
                            ),
                        }
                    )
                    loss.backward()
                torch.cuda.synchronize()
            finally:
                for handle in handles:
                    handle.remove()
            traces.append(trace)
        first = None
        for index, (left, right) in enumerate(zip(*traces, strict=False)):
            if left != right:
                first = {"backward_index": index, "left": left, "right": right}
                break
        if first is None and len(traces[0]) != len(traces[1]):
            first = {"backward_index": min(map(len, traces)), "reason": "trace_length_differs"}
        return {
            "schema": SCHEMA,
            "status": "TRACE_EXACT" if first is None else "TRACE_DIVERGED",
            "scientific_updates": 0,
            "technical_optimizer_steps": 0,
            "technical_backward_passes": 2,
            "first_module_boundary_difference": first,
            "trace_lengths": list(map(len, traces)),
            "effective_flags_after_forward_before_backward": flags,
            "environment": {
                "torch_version": torch.__version__,
                "torch_file": inspect.getfile(torch),
                "causal_scale_ttc_file": inspect.getfile(causal_module),
                "local_transport_file": inspect.getfile(transport_module),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
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
    result = run(args)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
