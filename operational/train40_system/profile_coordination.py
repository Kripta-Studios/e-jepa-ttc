"""Read-only real TRAIN forward/backward profiling and exact coordination admission."""

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Callable
from pathlib import Path

import torch

from operational.efficient_context.common import Lease, digest
from operational.train40_system.checkpoint import payload_digest
from operational.train40_system.contracts import read
from operational.train40_system.coordination import DevicePrefetch, relational_diagnostic
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine import SealedInputs


def run(output: Path, arm: str) -> None:
    """Compare identical restored weights/RNG with zero optimizer updates and export a trace."""
    import e_jepa_ttc.training.causal_scale_eap as training
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.reproducibility import seed_everything

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(7, deterministic=True)
    directory = output / "coordination_admission"
    directory.mkdir(exist_ok=True)
    checkpoint = output / "fits" / f"{arm}_seed7" / "checkpoint_last.pt"
    receipt = read(checkpoint.with_name("CHECKPOINT_RECEIPT.json"))
    if digest(checkpoint) != receipt["sha256"]:
        raise ValueError("Profile requires a stable verified checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    full_digest = saved.pop("full_state_sha256")
    if payload_digest(saved) != full_digest:
        raise ValueError("Complete checkpoint integrity differs")
    recipe = read(output / "TRAINING_PROTOCOL.json")["producers"][arm]
    model = CausalScaleTTC(CausalScaleTTCConfig(**recipe["model_config"])).to("cuda").train()
    loss_config = CausalScaleTTCLossConfig(**recipe["loss_config"])
    original = training._record_relational_fg_bg_diagnostic
    data = SealedInputs(output)
    order = saved["cursor"]["order"]
    position = saved["cursor"]["position"]
    ids = order[position : position + 32].tolist()
    host = data.batch(ids)
    batch = host.to(torch.device("cuda"))
    initial_cpu_rng = saved["torch_rng_state"]
    initial_cuda_rng = saved["cuda_rng_state_all"]
    if initial_cuda_rng is None:
        raise ValueError("Actual CUDA resume RNG is required for the parity admission")

    def step(
        diagnostic: Callable[..., None],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        float,
        dict[str, torch.Tensor],
    ]:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        torch.set_rng_state(initial_cpu_rng)
        torch.cuda.set_rng_state_all(initial_cuda_rng)
        model.zero_grad(set_to_none=True)
        training._record_relational_fg_bg_diagnostic = diagnostic
        torch.cuda.synchronize()
        start = time.perf_counter()
        with training._autocast(torch.device("cuda"), "bf16"):
            total, components, _ = training._loss(
                model,
                batch,
                loss_config,
                mask_t0_as_proxy=True,
                foreground_supervision="bbox_geometry",
                representation_supervision="dinov3_local_relational",
                representation_distillation_weight=8.0,
            )
        total.backward()
        torch.cuda.synchronize()
        copy_start = time.perf_counter()
        raw_gradients = {
            k: p.grad.detach().cpu().clone()
            for k, p in model.named_parameters()
            if p.grad is not None
        }
        raw_copy_seconds = time.perf_counter() - copy_start
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - raw_copy_seconds
        return (
            total.detach().cpu(),
            {k: v.detach().float().cpu() for k, v in components.items()},
            {
                k: p.grad.detach().cpu().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            },
            elapsed,
            raw_gradients,
        )

    try:
        step(original)  # GPU initialization warm-up; no optimizer.
        # A5's original CUDA grid_sample backward uses atomic accumulation; audit its
        # own repeatability before attributing floating-point drift to coordination.
        controls = [
            (name, step(fn))
            for name, fn in (
                ("original", original),
                ("original", original),
                ("optimized", relational_diagnostic),
                ("original", original),
                ("optimized", relational_diagnostic),
                ("optimized", relational_diagnostic),
            )
        ]
        expected, actual = controls[0][1], controls[2][1]
        for _, sample in controls:
            torch.testing.assert_close(sample[0], expected[0], rtol=0, atol=0)
            if expected[1].keys() != sample[1].keys() or expected[2].keys() != sample[2].keys():
                raise ValueError("Diagnostic or gradient coverage differs")
            for name in expected[1]:
                torch.testing.assert_close(
                    sample[1][name], expected[1][name], rtol=0, atol=0, equal_nan=True
                )
            for left, right in ((expected[2], sample[2]), (expected[4], sample[4])):
                for name in left:
                    if right[name].shape != left[name].shape:
                        raise ValueError("Gradient shape differs: " + name)
                    if not torch.isfinite(right[name]).all():
                        raise ValueError("Nonfinite parameter gradient: " + name)
        repeated = {}
        for title in ("pre_clip", "post_clip"):
            vectors = [
                torch.cat(
                    [
                        tensor.flatten().double()
                        for tensor in (item[1][4] if title == "pre_clip" else item[1][2]).values()
                    ]
                )
                for item in controls
            ]
            pairs = []
            for a, b in itertools.combinations(range(len(vectors)), 2):
                left, right = vectors[a], vectors[b]
                norm = max(float(left.norm()), float(right.norm()), 1e-30)
                delta = right - left
                pairs.append(
                    {
                        "kind": "cross" if controls[a][0] != controls[b][0] else controls[a][0],
                        "left": a,
                        "right": b,
                        "max_abs": float(delta.abs().max()),
                        "relative_L2": float(delta.norm()) / norm,
                        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
                    }
                )
            thresholds = {}
            for metric in ("max_abs", "relative_L2"):
                original_max = max(pair[metric] for pair in pairs if pair["kind"] == "original")
                optimized_max = max(pair[metric] for pair in pairs if pair["kind"] == "optimized")
                cross_max = max(pair[metric] for pair in pairs if pair["kind"] == "cross")
                floor = 8 * torch.finfo(torch.float32).eps
                if metric == "max_abs":
                    floor *= max(float(vector.abs().max()) for vector in vectors)
                limit = original_max + optimized_max + floor
                thresholds[metric] = {
                    "original_self_max": original_max,
                    "optimized_self_max": optimized_max,
                    "cross_max": cross_max,
                    "FP32_rounding_floor": floor,
                    "limit": limit,
                    "passed": cross_max <= limit,
                }
            repeated[title] = {
                "pairs": pairs,
                "thresholds": thresholds,
                "criterion": (
                    "cross <= original self maximum + optimized self maximum + 8 FP32 ulps"
                ),
            }
        atomic_json(directory / "REPEATED_GRADIENT_ADMISSION.json", repeated)
        if not all(
            value["passed"]
            for section in repeated.values()
            for value in section["thresholds"].values()
        ):
            raise ValueError("Gradient difference exceeds measured CUDA repeatability envelope")
        baseline_seconds, optimized_seconds = [], []
        for _ in range(3):
            baseline_seconds.append(step(original)[3])
            optimized_seconds.append(step(relational_diagnostic)[3])
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            step(original)
        baseline_trace = directory / "baseline_trace.json"
        # Kineto's Windows C++ exporter cannot open this user's accented absolute path.
        profiler.export_chrome_trace(str(baseline_trace.relative_to(Path.cwd())))
        if not baseline_trace.is_file() or baseline_trace.stat().st_size == 0:
            raise ValueError("Baseline trace export failed; keep the checkpoint preserved")
        (directory / "baseline_cpu_profile.txt").write_text(
            profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30),
            encoding="utf-8",
        )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            step(relational_diagnostic)
        optimized_trace = directory / "optimized_trace.json"
        profiler.export_chrome_trace(str(optimized_trace.relative_to(Path.cwd())))
        if not optimized_trace.is_file() or optimized_trace.stat().st_size == 0:
            raise ValueError("Optimized trace export failed; keep the checkpoint preserved")
        (directory / "optimized_cpu_profile.txt").write_text(
            profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30),
            encoding="utf-8",
        )
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]
        pipeline = DevicePrefetch(data)
        try:
            for offset in (position, position + len(ids)):
                ticket = pipeline.take(order, offset, 32)
                reference = data.batch(order[offset : offset + 32].tolist())
                result = ticket.consume()
                for name in vars(reference):
                    left, right = getattr(reference, name), getattr(result, name)
                    if isinstance(left, torch.Tensor):
                        torch.testing.assert_close(
                            right.cpu(),
                            left.float() if right.is_floating_point() else left,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                    elif left != right:
                        raise ValueError(f"Prefetch metadata differs: {name}")
        finally:
            pipeline.close()
        if not torch.equal(cpu_before, torch.get_rng_state()) or not all(
            torch.equal(a, b)
            for a, b in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise ValueError("Prefetch unexpectedly consumed training RNG")
        atomic_json(
            directory / "REAL_COORDINATION_ADMISSION.json",
            {
                "status": "PASSED",
                "arm": arm,
                "checkpoint_sha256": receipt["sha256"],
                "checkpoint_full_state_sha256": full_digest,
                "checkpoint_updates": saved["committed_updates"],
                "exact_loss_all_components_and_all_parameter_gradients": all(
                    torch.equal(actual[2][key], expected[2][key]) for key in expected[2]
                ),
                "exact_loss_and_all_components": True,
                "all_gradient_keys_shapes_and_finite_audited": True,
                "gradient_parity_within_baseline_repeatability": True,
                "repeated_gradient_admission": repeated,
                "gradient_tensors_compared": len(expected[2]),
                "prefetch_exact_all_tensor_fields_and_metadata": True,
                "prefetch_preserves_CPU_and_CUDA_RNG": True,
                "baseline_forward_backward_clip_seconds": baseline_seconds,
                "optimized_forward_backward_clip_seconds": optimized_seconds,
                "optimizer_updates": 0,
                "training_checkpoint_modified": False,
                "source_sha256": digest(Path(__file__)),
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                "trace_SHA256": {
                    "baseline": digest(baseline_trace),
                    "optimized": digest(optimized_trace),
                },
            },
        )
    finally:
        training._record_relational_fg_bg_diagnostic = original
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), default="a5")
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
