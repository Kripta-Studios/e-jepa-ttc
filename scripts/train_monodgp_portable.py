"""Bounded real-RGB experiments for the explicitly declared portable MonoDGP variant."""

# ruff: noqa: E402 -- allow direct script execution from an uninstalled worktree.

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from smoke_monodgp_portable import (
    VARIANT,
    VELOCITY_VARIANT,
    build_detector,
    prepare_source,
    require_finite,
    sha256,
)

from e_jepa_ttc.detection.development_metrics import evaluate_detections
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample
from e_jepa_ttc.detection.pilot_admission import validate_pilot_admission


def write_json(path: Path, value: Any) -> None:
    """Write finite machine-readable experiment artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class PreparedCache:
    """Keep a bounded set of resized CPU inputs, not the entire raw image dataset."""

    def __init__(
        self,
        dataset: EAPRGBDataset,
        width: int,
        height: int,
        capacity: int = 32,
        *,
        include_velocity: bool = False,
    ) -> None:
        self.dataset, self.width, self.height, self.capacity = dataset, width, height, capacity
        self.include_velocity = include_velocity
        self.cache: OrderedDict[int, tuple] = OrderedDict()
        self.receipts: dict[str, dict] = {}

    def get(self, index: int) -> tuple:
        """Return prepared input, extrinsic, and token with a stable data receipt."""
        if index not in self.cache:
            sample = self.dataset[index]
            prepared = prepare_sample(
                sample, width=self.width, height=self.height, include_velocity=self.include_velocity
            )
            self.cache[index] = prepared, sample.event_from_ego, sample.sample_token
            self.receipts[sample.sample_token] = {
                **prepared.report,
                "rgb_member_sha256": sample.member_sha256,
            }
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def inputs_on_gpu(prepared: Any) -> tuple:
    """Transfer a complete image and supervision through separate arguments."""
    return (
        prepared.image.unsqueeze(0).cuda(),
        prepared.calibration.unsqueeze(0).cuda(),
        [{key: value.cuda() for key, value in prepared.target.items()}],
        prepared.image_size.unsqueeze(0).cuda(),
    )


def collate_on_gpu(samples: list[Any]) -> tuple:
    """Stack full RGB images into contiguous NCHW while retaining variable targets."""
    return (
        torch.stack([sample.image for sample in samples]).contiguous().cuda(),
        torch.stack([sample.calibration for sample in samples]).cuda(),
        [{key: value.cuda() for key, value in sample.target.items()} for sample in samples],
        torch.stack([sample.image_size for sample in samples]).cuda(),
    )


def evaluate(
    model: Any,
    criterion: Any,
    cache: PreparedCache,
    output: Path,
    *,
    predict_velocity: bool = False,
) -> dict:
    """Infer with targets=None and compute losses/metrics only after inference."""
    model.eval()
    criterion.eval()
    predictions, truth, counters, losses = {}, {}, {}, []
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(cache.dataset)):
            prepared, extrinsic, token = cache.get(index)
            images, calibration, targets, sizes = inputs_on_gpu(prepared)
            outputs = model(images, calibration, None, sizes)
            require_finite(outputs)
            measured = criterion(outputs, targets)
            require_finite(measured, "evaluation_losses")
            losses.append({key: float(value) for key, value in measured.items()})
            predictions[token], counters[token] = decode_eap(
                outputs,
                0,
                prepared.calibration,
                extrinsic,
                prepared.image_size,
            )
            truth[token] = prepared.native_ground_truth
    torch.cuda.synchronize()
    metrics = evaluate_detections(predictions, truth)
    if predict_velocity:
        from e_jepa_ttc.detection.velocity_diagnostics import evaluate_velocity_labels

        metrics["native_velocity"] = evaluate_velocity_labels(predictions, truth)
    metrics.update(
        {
            "loss_center_mean": statistics.mean(item["loss_center"] for item in losses),
            "loss_center_median": statistics.median(item["loss_center"] for item in losses),
            "loss_depth_mean": statistics.mean(item["loss_depth"] for item in losses),
            "total_weighted_loss_mean": statistics.mean(
                sum(
                    value * criterion.weight_dict[key]
                    for key, value in item.items()
                    if key in criterion.weight_dict
                )
                for item in losses
            ),
            "inference_and_scoring_seconds": time.perf_counter() - started,
            "model_targets_argument": None,
        }
    )
    write_json(
        output,
        {
            "metrics": metrics,
            "predictions": predictions,
            "ground_truth": truth,
            "decode_counts": counters,
            "losses_by_frame": losses,
            "data_receipts": cache.receipts,
        },
    )
    return metrics


def run(args: argparse.Namespace, report: dict) -> None:
    """Predeclare, train to a fixed budget, reload the checkpoint, and evaluate."""
    if args.output.exists():
        raise FileExistsError(f"Use a new run directory; refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    admission = validate_pilot_admission(args.manifest, args.candidate_split, args.data_root)
    report["metadata_admission"] = admission
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokens = manifest[args.role]["sample_tokens"]
    if args.batch_size > len(tokens):
        raise ValueError("Batch size cannot exceed the number of distinct training frames.")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is not supported by this CUDA device.")
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)
    variant = VELOCITY_VARIANT if args.predict_velocity else VARIANT
    source = prepare_source(
        args.source_root,
        args.portable_root,
        args.output / "upstream.patch",
        predict_velocity=args.predict_velocity,
    )
    model, criterion, config = build_detector(
        args.portable_root,
        args.group_num,
        predict_velocity=args.predict_velocity,
        velocity_loss_coef=args.velocity_loss_coef,
    )
    backbone_receipt = None
    if args.backbone_weights:
        if args.resume:
            raise ValueError("A resumed run must not overwrite its backbone initialization.")
        from e_jepa_ttc.detection.backbone_initialization import (
            OFFICIAL_SHA256,
            load_imagenet_resnet50_backbone,
        )

        backbone_receipt = load_imagenet_resnet50_backbone(
            model, args.backbone_weights, OFFICIAL_SHA256
        )
    resume_payload = None
    base_steps = 0
    reference_initial = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        if resume_payload["config"] != config or resume_payload["variant"] != variant:
            raise ValueError("Checkpoint architecture or declared variant differs from this run.")
        parent_prereg_path = args.resume.parent / "preregistration.json"
        parent_prereg = json.loads(parent_prereg_path.read_text(encoding="utf-8"))
        if resume_payload["preregistration_sha256"] != sha256(parent_prereg_path):
            raise ValueError("Checkpoint provenance does not match the adjacent preregistration.")
        if (
            parent_prereg["sample_tokens"] != tokens
            or parent_prereg["image_hw"] != [args.height, args.width]
            or parent_prereg["source"]["patch_sha256"] != source["patch_sha256"]
        ):
            raise ValueError(
                "Resume requires the same training tokens, image size, and source patch."
            )
        model.load_state_dict(resume_payload["model"], strict=True)
        base_steps = int(resume_payload["steps"])
    if args.acceptance_reference:
        if (
            not args.resume
            or args.acceptance_reference.resolve()
            != (args.resume.parent / "initial_train.json").resolve()
        ):
            raise ValueError("Acceptance reference must be the resume parent's initial_train.json.")
        if (
            not args.acceptance_reference_sha256
            or sha256(args.acceptance_reference) != args.acceptance_reference_sha256.lower()
        ):
            raise ValueError("Acceptance reference must match the explicitly supplied SHA256.")
        reference = json.loads(args.acceptance_reference.read_text(encoding="utf-8"))
        if set(reference["predictions"]) != set(tokens):
            raise ValueError("Acceptance reference tokens differ from this run.")
        reference_initial = float(reference["metrics"]["loss_center_mean"])
        if not np.isfinite(reference_initial) or reference_initial <= 0:
            raise ValueError("Acceptance reference must contain a positive finite initial loss.")
    file_paths = [Path(__file__), ROOT / "scripts/smoke_monodgp_portable.py"] + [
        ROOT / "src/e_jepa_ttc/detection" / name
        for name in (
            "portable_attention.py",
            "monodgp_eap_adapter.py",
            "development_metrics.py",
            "evaluation_ap_fast.py",
            "eap_rgb_data.py",
            "geometry_contract.py",
            "velocity_extension.py",
            "velocity_diagnostics.py",
            "backbone_initialization.py",
            "pilot_admission.py",
        )
    ]
    preregistration = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "variant": variant,
        "role": args.role,
        "sample_tokens": tokens,
        "seed": args.seed,
        "model_config": config,
        "image_hw": [args.height, args.width],
        "batch_size": args.batch_size,
        "training_image_layout": "contiguous_NCHW",
        "maximum_image_exposures": args.steps * args.batch_size,
        "training_precision": args.precision,
        "inference_precision": "fp32",
        "matmul_precision": args.matmul_precision,
        "learning_rate": args.lr,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "weight_decay": 1e-4,
            "parameter_groups": "all_trainable_parameters_same_lr_and_decay",
            "gradient_clip_norm": 1.0,
        },
        "maximum_steps": args.steps,
        "step_budget_semantics": "additional_optimizer_updates",
        "base_checkpoint_steps": base_steps,
        "maximum_total_steps": base_steps + args.steps,
        "resume_checkpoint_sha256": sha256(args.resume) if args.resume else None,
        "resume_rng_policy": "restart_seeded_sampler_and_torch_rng" if args.resume else None,
        "acceptance_reference_sha256": sha256(args.acceptance_reference)
        if args.acceptance_reference
        else None,
        "reference_initial_center_mean": reference_initial,
        "absolute_center_acceptance_maximum": reference_initial * 0.5
        if reference_initial
        else None,
        "maximum_training_seconds": args.max_seconds,
        "acceptance": {
            "overfit16_mean_center_loss_ratio_at_most": 0.5,
            "finite_parameters": True,
            "active_class_head_updated": True,
            "criterion_fixed_before_first_real_step": True,
        },
        "pilot_interpretation": (
            "bounded_feasibility_run_without_preregistered_accuracy_pass_threshold; "
            "report_all_train_and_sequence_disjoint_selection_metrics"
            if args.role == "train"
            else None
        ),
        "manifest_sha256": sha256(args.manifest),
        "metadata_admission": admission,
        "source": source,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in file_paths},
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "environment_lock_sha256": sha256(
            ROOT / "docs/detection_local_20261029/evidence/portable_dependency_lock.txt"
        ),
        "initialization": "resume_public_train_experiment"
        if args.resume
        else (
            "imagenet1k_v1_backbone_random_detection_heads"
            if args.backbone_weights
            else "random_no_pretrained_weights"
        ),
        "backbone_initialization": backbone_receipt,
        "selection_tokens": manifest["selection"]["sample_tokens"] if args.role == "train" else [],
    }
    write_json(args.output / "preregistration.json", preregistration)
    for path in file_paths:
        snapshot = args.output / "code" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        if sha256(snapshot) != preregistration["code_sha256"][str(path.relative_to(ROOT))]:
            raise RuntimeError("Code changed while creating the preregistered snapshot.")
    report.update(
        preregistration_sha256=sha256(args.output / "preregistration.json"),
        source=source,
        role=args.role,
        status="RUNNING",
        training_steps=0,
    )
    train_dataset = EAPRGBDataset(args.data_root, tokens)
    train_cache = PreparedCache(
        train_dataset, args.width, args.height, include_velocity=args.predict_velocity
    )
    initial = evaluate(
        model,
        criterion,
        train_cache,
        args.output / "initial_train.json",
        predict_velocity=args.predict_velocity,
    )
    report["initial_train"] = initial
    print(
        json.dumps(
            {
                "stage": "initial_eval",
                "center_loss": initial["loss_center_mean"],
                "depth_loss": initial["loss_depth_mean"],
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        del resume_payload
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    order: list[int] = []
    timings = []
    gradient_norms = []
    empty_target_exposures = 0
    valid_velocity_object_exposures = 0
    active_head_before = model.class_embed[config["dec_layers"] - 1].weight.detach().cpu().clone()
    velocity_before = (
        [head.layers[-1].weight.detach().cpu().clone() for head in model.velocity_embed]
        if args.predict_velocity
        else []
    )
    curve = args.output / "training.jsonl"
    with curve.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            if time.perf_counter() - started >= args.max_seconds:
                report["stop_reason"] = "time_budget"
                break
            batch = []
            batch_tokens = []
            for _ in range(args.batch_size):
                if not order:
                    order = list(range(len(tokens)))
                    rng.shuffle(order)
                prepared, _, token = train_cache.get(order.pop())
                batch.append(prepared)
                batch_tokens.append(token)
                empty_target_exposures += int(len(prepared.target["labels"]) == 0)
                if args.predict_velocity:
                    valid_velocity_object_exposures += int(
                        prepared.target["velocity_valid"].sum().item()
                    )
            images, calibration, targets, sizes = collate_on_gpu(batch)
            model.train()
            criterion.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                outputs = model(images, calibration, targets, sizes)
                require_finite(outputs)
                loss_values = criterion(outputs, targets)
                require_finite(loss_values, "training_losses")
                loss = sum(
                    value * criterion.weight_dict[key]
                    for key, value in loss_values.items()
                    if key in criterion.weight_dict
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            timings.append(duration)
            gradient_norms.append(float(norm))
            record = {
                "step": step,
                "total_step": base_steps + step,
                "sample_token": batch_tokens[0],
                "sample_tokens": batch_tokens,
                "loss": float(loss.detach()),
                "loss_center": float(loss_values["loss_center"].detach()),
                "step_seconds": duration,
                "gradient_norm_before_clip": float(norm),
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            report["training_steps"] = step + 1
            if step % 25 == 0 or step == args.steps - 1:
                print(json.dumps(record), flush=True)
            del outputs, loss_values, loss
    report["training_seconds"] = time.perf_counter() - started
    report["base_checkpoint_steps"] = base_steps
    report["total_training_steps"] = base_steps + report["training_steps"]
    report["image_exposures_this_stage"] = report["training_steps"] * args.batch_size
    report["empty_target_exposures"] = empty_target_exposures
    report["valid_velocity_object_exposures"] = valid_velocity_object_exposures
    if not timings:
        raise RuntimeError(
            "Time budget expired before any optimizer update; overfit gate not passed."
        )
    report["step_seconds_median"] = statistics.median(timings)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["active_head_updated"] = not torch.equal(
        active_head_before, model.class_embed[config["dec_layers"] - 1].weight.detach().cpu()
    )
    report["all_parameters_finite"] = all(
        torch.isfinite(p).all().item() for p in model.parameters()
    )
    if args.predict_velocity:
        report["velocity_heads_updated"] = [
            not torch.equal(before, head.layers[-1].weight.detach().cpu())
            for before, head in zip(velocity_before, model.velocity_embed, strict=True)
        ]
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "variant": variant,
            "seed": args.seed,
            "steps": report["total_training_steps"],
            "batch_size": args.batch_size,
            "image_exposures_this_stage": report["image_exposures_this_stage"],
            "training_precision": args.precision,
            "inference_precision": "fp32",
            "matmul_precision": args.matmul_precision,
            "preregistration_sha256": report["preregistration_sha256"],
        },
        checkpoint,
    )
    report["checkpoint_sha256"] = sha256(checkpoint)
    del model, optimizer
    torch.cuda.empty_cache()
    model, criterion, _ = build_detector(
        args.portable_root,
        args.group_num,
        predict_velocity=args.predict_velocity,
        velocity_loss_coef=args.velocity_loss_coef,
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(restored["model"], strict=True)
    del restored
    report["checkpoint_reload"] = "PASS_STRICT"
    final = evaluate(
        model,
        criterion,
        train_cache,
        args.output / "final_train.json",
        predict_velocity=args.predict_velocity,
    )
    report["final_train"] = final
    report["ratio_to_run_start_center_loss"] = (
        final["loss_center_mean"] / initial["loss_center_mean"]
    )
    ratio = final["loss_center_mean"] / (reference_initial or initial["loss_center_mean"])
    report["mean_center_loss_ratio"] = ratio
    report["overfit_acceptance"] = (
        "PASS"
        if (
            args.role == "overfit16"
            and ratio <= 0.5
            and report["all_parameters_finite"]
            and report["active_head_updated"]
        )
        else ("FAIL" if args.role == "overfit16" else "NOT_APPLICABLE_PILOT")
    )
    train_dataset.close()
    if args.role == "train":
        with EAPRGBDataset(args.data_root, manifest["selection"]["sample_tokens"]) as selection:
            report["selection"] = evaluate(
                model,
                criterion,
                PreparedCache(
                    selection, args.width, args.height, include_velocity=args.predict_velocity
                ),
                args.output / "selection.json",
                predict_velocity=args.predict_velocity,
            )
    report["status"] = "COMPLETED"
    report["ended_at_utc"] = datetime.now(UTC).isoformat()
    print(
        json.dumps(
            {
                "status": report["status"],
                "overfit_acceptance": report["overfit_acceptance"],
                "center_loss_ratio": ratio,
                "step_seconds_median": report["step_seconds_median"],
            }
        ),
        flush=True,
    )


def main() -> int:
    """CLI for explicitly budgeted research runs with failure receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-split",
        type=Path,
        required=True,
        help="Sequence split whose exact SHA256 is bound by the pilot manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("overfit16", "train"), default="overfit16")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--group-num", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--predict-velocity", action="store_true")
    parser.add_argument("--velocity-loss-coef", type=float, default=0.1)
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--acceptance-reference", type=Path)
    parser.add_argument("--acceptance-reference-sha256")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP"
    )
    parser.add_argument(
        "--portable-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP-portable"
    )
    args = parser.parse_args()
    if (
        args.predict_velocity
        and args.portable_root == ROOT / "output/detector_sources/MonoDGP-portable"
    ):
        args.portable_root = ROOT / "output/detector_sources/MonoDGP-portable-velocity"
    budget_values = (args.steps, args.max_seconds, args.group_num, args.batch_size, args.lr)
    if any(not math.isfinite(value) or value <= 0 for value in budget_values):
        parser.error("Steps, time budget, group count, and learning rate must be positive.")
    if args.predict_velocity and (
        not math.isfinite(args.velocity_loss_coef) or args.velocity_loss_coef <= 0
    ):
        parser.error("Learned velocity requires a finite positive velocity loss coefficient.")
    if args.output.exists():
        parser.error("Run output already exists; choose a new experiment directory.")
    report: dict[str, Any] = {"status": "FAILED"}
    code = 0
    try:
        run(args, report)
        if report.get("overfit_acceptance") == "FAIL":
            code = 3
    except Exception as exc:
        report.update(
            status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        traceback.print_exc()
        code = 2
    write_json(args.output / "report.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
