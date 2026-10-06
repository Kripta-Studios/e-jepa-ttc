"""Zero-update real-CUDA admission for exact same-loss relation-map reuse."""

# ruff: noqa: E402 -- the admitted OpenMP environment must precede Torch imports.

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import NamedTuple, cast

from operational.train40_system.engine_idle_wait import (
    IDLE_WAIT_ENVIRONMENT,
    configure_environment,
)

configure_environment()

import torch

from operational.efficient_context.common import Lease, digest
from operational.train40_system.checkpoint import payload_digest
from operational.train40_system.contracts import read
from operational.train40_system.coordination import relational_diagnostic
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine import SealedInputs
from operational.train40_system.relation_reuse import (
    ReuseCounters,
    TrainingModule,
    scoped_relation_reuse,
)


class StepResult(NamedTuple):
    """One loss evaluation with immutable CPU evidence."""

    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    clipped: dict[str, torch.Tensor]
    raw: dict[str, torch.Tensor]
    cpu_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]


def _gradient_comparison(
    controls: list[tuple[str, StepResult]],
) -> dict:
    result = {}
    for title in ("raw", "clipped"):
        vectors = [
            torch.cat(
                [tensor.flatten().double() for tensor in getattr(sample, title).values()]
            )
            for _, sample in controls
        ]
        pairs = []
        for a, b in itertools.combinations(range(len(vectors)), 2):
            left, right = vectors[a], vectors[b]
            norm = max(float(left.norm()), float(right.norm()), 1.0e-30)
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
            baseline_max = max(pair[metric] for pair in pairs if pair["kind"] == "baseline")
            reuse_max = max(pair[metric] for pair in pairs if pair["kind"] == "reuse")
            cross_max = max(pair[metric] for pair in pairs if pair["kind"] == "cross")
            floor = 8 * torch.finfo(torch.float32).eps
            if metric == "max_abs":
                floor *= max(float(vector.abs().max()) for vector in vectors)
            limit = baseline_max + reuse_max + floor
            thresholds[metric] = {
                "baseline_self_max": baseline_max,
                "reuse_self_max": reuse_max,
                "cross_max": cross_max,
                "FP32_rounding_floor": floor,
                "limit": limit,
                "passed": cross_max <= limit,
            }
        result[title] = {
            "pairs": pairs,
            "thresholds": thresholds,
            "criterion": "cross <= baseline self max + reuse self max + 8 FP32 ulps",
        }
    return result


def run(output: Path, arm: str) -> None:
    """Compare a real paused checkpoint and cursor batch without optimizer updates."""
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
    directory = output / "relation_reuse_admission"
    directory.mkdir(exist_ok=True)
    checkpoint = output / "fits" / f"{arm}_seed7" / "checkpoint_last.pt"
    idle_freeze = output / "IDLE_WAIT_FREEZE.json"
    from operational.train40_system.contracts import verify_sources

    verify_sources(read(idle_freeze))
    receipt = read(checkpoint.with_name("CHECKPOINT_RECEIPT.json"))
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != receipt["sha256"]:
        raise ValueError("Relation reuse admission requires a stable verified checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    full_digest = saved.pop("full_state_sha256")
    if payload_digest(saved) != full_digest:
        raise ValueError("Complete checkpoint integrity differs")
    recipe = read(output / "TRAINING_PROTOCOL.json")["producers"][arm]
    model = CausalScaleTTC(CausalScaleTTCConfig(**recipe["model_config"])).to("cuda").train()
    loss_config = CausalScaleTTCLossConfig(**recipe["loss_config"])
    data = SealedInputs(output)
    order = saved["cursor"]["order"]
    position = saved["cursor"]["position"]
    ids = order[position : position + 32].tolist()
    batch = data.batch(ids).to(torch.device("cuda"))
    initial_cuda_rng = saved["cuda_rng_state_all"]
    if initial_cuda_rng is None:
        raise ValueError("Actual CUDA resume RNG is required")
    original_diagnostic = training._record_relational_fg_bg_diagnostic
    training._record_relational_fg_bg_diagnostic = relational_diagnostic
    counters = ReuseCounters()

    def step(reuse: bool) -> StepResult:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(initial_cuda_rng)
        model.zero_grad(set_to_none=True)

        def execute() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
            return total, components

        if reuse:
            with scoped_relation_reuse(cast(TrainingModule, training), counters):
                total, components = execute()
        else:
            total, components = execute()
        total.backward()
        torch.cuda.synchronize()
        raw = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize()
        clipped = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return StepResult(
            total.detach().cpu(),
            {name: value.detach().float().cpu() for name, value in components.items()},
            clipped,
            raw,
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()],
        )

    try:
        step(False)
        controls = [
            (name, step(name == "reuse"))
            for name in ("baseline", "baseline", "reuse", "baseline", "reuse", "reuse")
        ]
        expected = controls[0][1]
        for _, sample in controls:
            torch.testing.assert_close(sample.loss, expected.loss, rtol=0, atol=0)
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
            for gradients, expected_gradients in (
                (sample.clipped, expected.clipped),
                (sample.raw, expected.raw),
            ):
                if gradients.keys() != expected_gradients.keys():
                    raise ValueError("Gradient coverage differs")
                for name, gradient in gradients.items():
                    if gradient.shape != expected_gradients[name].shape or not torch.isfinite(
                        gradient
                    ).all():
                        raise ValueError("Invalid gradient: " + name)
            if not torch.equal(sample.cpu_rng, expected.cpu_rng) or not all(
                torch.equal(left, right)
                for left, right in zip(sample.cuda_rng, expected.cuda_rng, strict=True)
            ):
                raise ValueError("Relation reuse changed the training RNG trajectory")
        gradient_count = len(expected.clipped)
        if arm == "a5" and gradient_count != 59:
            raise ValueError(f"Expected all 59 A5 gradients, observed {gradient_count}")
        repeated = _gradient_comparison(controls)
        if not all(
            metric["passed"]
            for section in repeated.values()
            for metric in section["thresholds"].values()
        ):
            raise ValueError("Reuse gradient exceeds original CUDA repeatability")
        if counters.loss_calls != 3 or counters.original_map_calls != 3:
            raise ValueError("Each reuse loss must execute exactly one original map")
        if counters.reuse_hits != 3 or counters.reuse_misses != 0:
            raise ValueError("Each reuse loss must serve exactly one same-call diagnostic")
        if digest(checkpoint) != checkpoint_sha:
            raise ValueError("Admission changed the training checkpoint")
        actual = controls[2][1]
        gradient_bit_equality = all(
            torch.equal(actual_gradients[name], expected_gradients[name])
            for actual_gradients, expected_gradients in (
                (actual.clipped, expected.clipped),
                (actual.raw, expected.raw),
            )
            for name in expected_gradients
        )
        atomic_json(directory / "REPEATED_GRADIENT_ADMISSION.json", repeated)
        atomic_json(
            directory / "REAL_RELATION_REUSE_ADMISSION.json",
            {
                "status": "PASSED",
                "arm": arm,
                "checkpoint_sha256": checkpoint_sha,
                "idle_wait_freeze_sha256": digest(idle_freeze),
                "environment": dict(IDLE_WAIT_ENVIRONMENT),
                "checkpoint_full_state_sha256": full_digest,
                "checkpoint_updates": saved["committed_updates"],
                "exact_loss_and_all_components": True,
                "all_gradient_keys_shapes_and_finite_audited": True,
                "gradient_tensors_compared": gradient_count,
                "gradient_parity_within_baseline_repeatability": True,
                "gradient_bit_equality_observed": gradient_bit_equality,
                "repeated_gradient_admission": repeated,
                "reuse_counters": counters.snapshot(),
                "first_grad_enabled_map_calls_original": True,
                "same_call_diagnostic_receives_detached_map": True,
                "CPU_and_CUDA_RNG_trajectory_exact": True,
                "optimizer_updates": 0,
                "training_checkpoint_modified": False,
                "source_sha256": digest(Path(__file__)),
            },
        )
    finally:
        training._record_relational_fg_bg_diagnostic = original_diagnostic
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), default="a5")
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
