"""Zero-update real-CUDA admission for PyTorch's built-in CUDA-graphs backend."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.efficient_context.common import Lease, digest
from operational.train40_system.checkpoint import payload_digest
from operational.train40_system.contracts import read
from operational.train40_system.coordination import relational_diagnostic
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine import SealedInputs

ADMITTED_BATCH_SIZE = 32
BACKEND = "cudagraphs"


class StepResult(NamedTuple):
    """Immutable CPU evidence from one forward/backward without an optimizer."""

    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    output_identity: dict[str, Any]
    raw: dict[str, torch.Tensor]
    clipped: dict[str, torch.Tensor]
    cpu_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    elapsed_seconds: float


class ForwardDispatcher:
    """Select the compiled B32 callable while retaining exact eager tail behavior."""

    def __init__(
        self, original: Callable[..., object], compiled: Callable[..., object]
    ) -> None:
        self.original = original
        self.compiled = compiled
        self.compiled_enabled = False
        self.compiled_calls = 0
        self.eager_calls = 0
        self.eager_tail_calls = 0

    def __call__(
        self,
        inputs: torch.Tensor,
        delta_t_s: torch.Tensor,
        *,
        return_dense_features: bool = False,
    ) -> object:
        if self.compiled_enabled and inputs.shape[0] == ADMITTED_BATCH_SIZE:
            self.compiled_calls += 1
            return self.compiled(
                inputs,
                delta_t_s,
                return_dense_features=return_dense_features,
            )
        self.eager_calls += 1
        if inputs.shape[0] != ADMITTED_BATCH_SIZE:
            self.eager_tail_calls += 1
        return self.original(
            inputs,
            delta_t_s,
            return_dense_features=return_dense_features,
        )

    def snapshot(self) -> dict[str, int]:
        """Return bounded call evidence without retaining graph tensors."""

        return {
            "compiled_calls": self.compiled_calls,
            "eager_calls": self.eager_calls,
            "eager_tail_calls": self.eager_tail_calls,
        }


def _compile_forward(
    forward: Callable[..., object],
    compiler: Callable[..., Callable[..., object]] = torch.compile,
) -> tuple[Callable[..., object], float]:
    started = time.perf_counter()
    compiled = compiler(forward, backend=BACKEND, dynamic=False, fullgraph=False)
    return compiled, time.perf_counter() - started


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _rng_identity(cpu: torch.Tensor, cuda: list[torch.Tensor]) -> dict[str, Any]:
    return {
        "cpu": _tensor_sha256(cpu),
        "cuda": [_tensor_sha256(value) for value in cuda],
    }


def _output_identity(output: object) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for name, value in vars(output).items():
        if isinstance(value, torch.Tensor):
            identity[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        elif isinstance(value, dict):
            identity[name] = {
                child_name: {
                    "shape": list(child.shape),
                    "dtype": str(child.dtype),
                    "sha256": _tensor_sha256(child),
                }
                for child_name, child in value.items()
            }
        elif value is not None:
            raise TypeError(f"Unexpected model output field type: {name}={type(value).__name__}")
        else:
            identity[name] = None
    return identity


def _gradient_comparison(controls: list[tuple[str, StepResult]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for title in ("raw", "clipped"):
        vectors = [
            torch.cat(
                [tensor.flatten().double() for tensor in getattr(sample, title).values()]
            )
            for _, sample in controls
        ]
        pairs = []
        for left_index, right_index in itertools.combinations(range(len(vectors)), 2):
            left, right = vectors[left_index], vectors[right_index]
            norm = max(float(left.norm()), float(right.norm()), 1.0e-30)
            delta = right - left
            left_kind, right_kind = controls[left_index][0], controls[right_index][0]
            pairs.append(
                {
                    "kind": left_kind if left_kind == right_kind else "cross",
                    "left": left_index,
                    "right": right_index,
                    "max_abs": float(delta.abs().max()),
                    "relative_L2": float(delta.norm()) / norm,
                    "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
                }
            )
        thresholds = {}
        for metric in ("max_abs", "relative_L2"):
            baseline_max = max(pair[metric] for pair in pairs if pair["kind"] == "baseline")
            compiled_max = max(pair[metric] for pair in pairs if pair["kind"] == "compiled")
            cross_max = max(pair[metric] for pair in pairs if pair["kind"] == "cross")
            floor = 8 * torch.finfo(torch.float32).eps
            if metric == "max_abs":
                floor *= max(float(vector.abs().max()) for vector in vectors)
            limit = baseline_max + compiled_max + floor
            thresholds[metric] = {
                "baseline_self_max": baseline_max,
                "compiled_self_max": compiled_max,
                "cross_max": cross_max,
                "FP32_rounding_floor": floor,
                "limit": limit,
                "passed": cross_max <= limit,
            }
        result[title] = {
            "pairs": pairs,
            "thresholds": thresholds,
            "criterion": "cross <= baseline self max + compiled self max + 8 FP32 ulps",
        }
    return result


def _counter_snapshot(counters: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {
        category: {name: int(value) for name, value in values.items()}
        for category, values in counters.items()
        if values
    }


def _counter_delta(
    after: Mapping[str, Mapping[str, int]], before: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    result = {}
    for category in set(after) | set(before):
        values = {}
        for name in set(after.get(category, {})) | set(before.get(category, {})):
            delta = int(after.get(category, {}).get(name, 0)) - int(
                before.get(category, {}).get(name, 0)
            )
            if delta:
                values[name] = delta
        if values:
            result[category] = values
    return result


def _launch_profile(step: Callable[[], StepResult]) -> dict[str, dict[str, float | int]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        step()
    launches = {}
    for event in profiler.key_averages():
        if "Launch" in event.key or "launch" in event.key:
            launches[event.key] = {
                "count": int(event.count),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
            }
    return launches


def run(output: Path, arm: str, checkpoint: Path | None = None) -> None:
    """Compare eager and CUDA-graph forward/backward from one verified checkpoint."""
    from torch._dynamo import list_backends
    from torch._dynamo.utils import counters

    import e_jepa_ttc.training.causal_scale_eap as training
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.reproducibility import seed_everything

    directory = output / "graph_launch_admission"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "REAL_GRAPH_LAUNCH_ADMISSION.json"
    backend_source = Path(torch.__file__).parent / "_dynamo" / "backends" / "cudagraphs.py"
    checkpoint = checkpoint or output / "fits" / f"{arm}_seed7" / "checkpoint_last.pt"
    checkpoint = checkpoint.resolve()
    checkpoint_sha = digest(checkpoint)
    common: dict[str, Any] = {
        "arm": arm,
        "backend": BACKEND,
        "source_sha256": digest(Path(__file__)),
        "backend_source": str(backend_source),
        "backend_source_sha256": digest(backend_source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "optimizer_updates": 0,
        "openmp_environment": {
            "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
            "KMP_BLOCKTIME": os.environ.get("KMP_BLOCKTIME"),
        },
    }
    data: SealedInputs | None = None
    model: Any = None
    original_forward: Callable[..., object] | None = None
    original_diagnostic: Callable[..., None] | None = None
    try:
        if os.environ.get("OMP_WAIT_POLICY", "").upper() == "PASSIVE" or os.environ.get(
            "KMP_BLOCKTIME"
        ) == "0":
            raise RuntimeError("Graph admission requires the original non-idle-wait environment")
        source = backend_source.read_text(encoding="utf-8")
        if "register_backend(name=\"cudagraphs\"" not in source or "aot_autograd(" not in source:
            raise RuntimeError("Installed Torch CUDA-graphs backend source is incompatible")
        if BACKEND not in list_backends():
            raise RuntimeError("Installed Torch does not register the cudagraphs backend")
        receipt_path = checkpoint.with_name("CHECKPOINT_RECEIPT.json")
        receipt = read(receipt_path)
        if receipt["sha256"] != checkpoint_sha:
            raise ValueError("Graph admission requires a stable verified checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        full_digest = saved.pop("full_state_sha256")
        if payload_digest(saved) != full_digest:
            raise ValueError("Complete checkpoint integrity differs")

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_everything(7, deterministic=True)
        recipe = read(output / "TRAINING_PROTOCOL.json")["producers"][arm]
        original_diagnostic = training._record_relational_fg_bg_diagnostic
        training._record_relational_fg_bg_diagnostic = relational_diagnostic
        model = CausalScaleTTC(CausalScaleTTCConfig(**recipe["model_config"])).to("cuda").train()
        model.load_state_dict(saved["model_state_dict"], strict=True)
        loss_config = CausalScaleTTCLossConfig(**recipe["loss_config"])
        bound_forward: Callable[..., object] = model.forward
        original_forward = bound_forward
        parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
        state_keys = tuple(model.state_dict())
        compiled_forward, wrapper_creation_seconds = _compile_forward(bound_forward)
        dispatcher = ForwardDispatcher(bound_forward, compiled_forward)
        model.forward = dispatcher

        data = SealedInputs(output)
        order = saved["cursor"]["order"]
        position = int(saved["cursor"]["position"])
        if len(order) - position < ADMITTED_BATCH_SIZE:
            position = 0
        full_ids = order[position : position + ADMITTED_BATCH_SIZE].tolist()
        tail_ids = order[-8:].tolist()
        full_batch = data.batch(full_ids).to(torch.device("cuda"))
        tail_batch = data.batch(tail_ids).to(torch.device("cuda"))
        initial_cpu_rng = saved["torch_rng_state"]
        initial_cuda_rng = saved["cuda_rng_state_all"]
        if initial_cuda_rng is None:
            raise ValueError("Actual CUDA resume RNG is required")

        def reset() -> None:
            model.load_state_dict(saved["model_state_dict"], strict=True)
            torch.set_rng_state(initial_cpu_rng)
            torch.cuda.set_rng_state_all(initial_cuda_rng)
            model.zero_grad(set_to_none=True)

        def step(
            batch: ObjectEventV4Batch, compiled: bool, *, restore: bool = True
        ) -> StepResult:
            if restore:
                reset()
            dispatcher.compiled_enabled = compiled
            torch.cuda.synchronize()
            started = time.perf_counter()
            with training._autocast(torch.device("cuda"), "bf16"):
                total, components, output = training._loss(
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
            raw_copy_started = time.perf_counter()
            raw = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            raw_copy_seconds = time.perf_counter() - raw_copy_started
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started - raw_copy_seconds
            clipped = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            return StepResult(
                total.detach().cpu(),
                {name: value.detach().float().cpu() for name, value in components.items()},
                _output_identity(output),
                raw,
                clipped,
                torch.get_rng_state().clone(),
                [state.clone() for state in torch.cuda.get_rng_state_all()],
                elapsed,
            )

        counters.clear()
        reset()
        dispatcher.compiled_enabled = True
        compile_started = time.perf_counter()
        step(full_batch, True, restore=False)
        first_compiled_step_seconds = time.perf_counter() - compile_started
        compile_counters = _counter_snapshot(counters)
        # Compilation/capture warm-up is engineering work. Restore the complete
        # model and training RNG before every scientific parity observation.
        controls = [
            (kind, step(full_batch, kind == "compiled"))
            for kind in ("baseline", "baseline", "compiled", "baseline", "compiled", "compiled")
        ]
        expected = controls[0][1]
        for _, sample in controls:
            torch.testing.assert_close(sample.loss, expected.loss, rtol=0, atol=0)
            if sample.output_identity != expected.output_identity:
                raise ValueError("Model output tensor identity differs")
            if sample.components.keys() != expected.components.keys():
                raise ValueError("Loss component coverage differs")
            for name in expected.components:
                torch.testing.assert_close(
                    sample.components[name],
                    expected.components[name],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
            for gradients, reference in (
                (sample.raw, expected.raw),
                (sample.clipped, expected.clipped),
            ):
                if gradients.keys() != reference.keys():
                    raise ValueError("Gradient coverage differs")
                for name, gradient in gradients.items():
                    if (
                        gradient.shape != reference[name].shape
                        or not torch.isfinite(gradient).all()
                    ):
                        raise ValueError("Invalid gradient: " + name)
        gradient_count = len(expected.raw)
        if arm == "a5" and gradient_count != 59:
            raise ValueError(f"Expected all 59 A5 gradients, observed {gradient_count}")
        repeated = _gradient_comparison(controls)
        if not all(
            metric["passed"]
            for section in repeated.values()
            for metric in section["thresholds"].values()
        ):
            raise ValueError("Compiled gradient exceeds eager CUDA repeatability")

        eager_tail = step(tail_batch, False)
        compiled_tail = step(tail_batch, True)
        torch.testing.assert_close(compiled_tail.loss, eager_tail.loss, rtol=0, atol=0)
        if compiled_tail.output_identity != eager_tail.output_identity:
            raise ValueError("B8 eager fallthrough output differs")
        for name in eager_tail.components:
            torch.testing.assert_close(
                compiled_tail.components[name],
                eager_tail.components[name],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        if dispatcher.eager_tail_calls < 2:
            raise ValueError("B8 tail did not remain on the original eager forward")

        def trajectory(compiled: bool) -> list[StepResult]:
            reset()
            samples = []
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                samples.append(step(full_batch, compiled, restore=False))
            return samples

        eager_trajectory = trajectory(False)
        compiled_trajectory = trajectory(True)

        def compare_trajectory(
            eager_samples: list[StepResult], compiled_samples: list[StepResult]
        ) -> None:
            for eager, candidate in zip(eager_samples, compiled_samples, strict=True):
                torch.testing.assert_close(candidate.loss, eager.loss, rtol=0, atol=0)
                if candidate.output_identity != eager.output_identity:
                    raise ValueError("Compiled dropout output trajectory differs")
                for name in eager.components:
                    torch.testing.assert_close(
                        candidate.components[name],
                        eager.components[name],
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                if not torch.equal(candidate.cpu_rng, eager.cpu_rng) or not all(
                    torch.equal(left, right)
                    for left, right in zip(candidate.cuda_rng, eager.cuda_rng, strict=True)
                ):
                    raise ValueError("Compiled dropout changed the CPU/CUDA RNG trajectory")

        compare_trajectory(eager_trajectory, compiled_trajectory)

        def mixed_trajectory(compiled: bool) -> list[StepResult]:
            reset()
            samples = []
            for batch in (full_batch, tail_batch, full_batch):
                model.zero_grad(set_to_none=True)
                samples.append(step(batch, compiled, restore=False))
            return samples

        eager_mixed = mixed_trajectory(False)
        compiled_mixed = mixed_trajectory(True)
        compare_trajectory(eager_mixed, compiled_mixed)

        torch.cuda.reset_peak_memory_stats()
        baseline_profile = _launch_profile(lambda: step(full_batch, False))
        eager_peak_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        compiled_profile = _launch_profile(lambda: step(full_batch, True))
        compiled_peak_reserved = torch.cuda.max_memory_reserved()
        if not any("GraphLaunch" in name for name in compiled_profile):
            raise RuntimeError("cudagraphs backend produced no observed CUDA graph launch")
        post_steady_counters = _counter_snapshot(counters)
        steady_counter_delta = _counter_delta(post_steady_counters, compile_counters)
        steady_recompiles = steady_counter_delta.get("stats", {}).get("unique_graphs", 0)
        if steady_recompiles:
            raise RuntimeError(f"Steady replay compiled {steady_recompiles} additional graphs")
        if tuple(model.state_dict()) != state_keys or {
            name: id(parameter) for name, parameter in model.named_parameters()
        } != parameter_ids:
            raise ValueError("Compilation changed model keys or parameter identity")
        if digest(checkpoint) != checkpoint_sha:
            raise ValueError("Admission changed the training checkpoint")

        atomic_json(directory / "REPEATED_GRADIENT_ADMISSION.json", repeated)
        atomic_json(
            report_path,
            common
            | {
                "status": "PASSED",
                "compatible": True,
                "checkpoint_full_state_sha256": full_digest,
                "checkpoint_updates": saved["committed_updates"],
                "batch32_ordinals": full_ids,
                "batch8_ordinals": tail_ids,
                "torch_version": torch.__version__,
                "wrapper_creation_seconds": wrapper_creation_seconds,
                "first_compiled_forward_backward_seconds": first_compiled_step_seconds,
                "timing_scope": (
                    "loss forward + backward + gradient clip; excludes raw-gradient CPU copy, "
                    "optimizer, journal, checkpoint and input collation"
                ),
                "steady_eager_seconds": [
                    sample.elapsed_seconds for kind, sample in controls if kind == "baseline"
                ],
                "steady_compiled_seconds": [
                    sample.elapsed_seconds for kind, sample in controls if kind == "compiled"
                ],
                "compile_counters": compile_counters,
                "post_steady_counters": post_steady_counters,
                "steady_counter_delta": steady_counter_delta,
                "steady_recompiles": steady_recompiles,
                "launch_profiles": {"eager": baseline_profile, "compiled": compiled_profile},
                "peak_reserved_vram_bytes": {
                    "eager_after_graph_capture": eager_peak_reserved,
                    "compiled_replay": compiled_peak_reserved,
                },
                "dispatcher_calls": dispatcher.snapshot(),
                "exact_loss_and_all_components": True,
                "exact_all_model_output_tensors": True,
                "reference_output_identity": expected.output_identity,
                "all_gradient_keys_shapes_and_finite_audited": True,
                "gradient_tensors_compared": gradient_count,
                "gradient_parity_within_baseline_repeatability": True,
                "repeated_gradient_admission": repeated,
                "repeated_dropout_CPU_and_CUDA_RNG_trajectory_exact": True,
                "B32_B8_B32_output_loss_components_and_RNG_trajectory_exact": True,
                "rng_trajectory": [
                    _rng_identity(sample.cpu_rng, sample.cuda_rng)
                    for sample in compiled_trajectory
                ],
                "batch8_uses_original_eager_forward": True,
                "model_parameter_identity_and_state_dict_keys_unchanged": True,
                "original_shape_and_delta_t_guards_retained": True,
                "training_checkpoint_modified": False,
            },
        )
    except Exception as exc:
        unchanged = checkpoint.is_file() and digest(checkpoint) == checkpoint_sha
        atomic_json(
            report_path,
            common
            | {
                "status": "FAILED_ENGINEERING_TRIAL",
                "compatible": False,
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "training_checkpoint_modified": not unchanged,
            },
        )
        raise
    finally:
        if original_diagnostic is not None:
            training._record_relational_fg_bg_diagnostic = original_diagnostic
        if model is not None and original_forward is not None:
            model.forward = original_forward
        if data is not None:
            data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), default="a5")
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    with Lease(arguments.output.resolve()):
        run(arguments.output.resolve(), arguments.arm, arguments.checkpoint)
