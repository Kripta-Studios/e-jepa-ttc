#!/usr/bin/env python3
"""Fine-tune Object Event v4.2 with the v4.5 paired reciprocal MiD objective.

This is a controlled loss-only successor to v4.4.  The model architecture and
common-coordinate event cache remain unchanged.  Each seed starts from its own
v4.2 best checkpoint, then receives train-only paired temporal reversal and a
paper-compatible log-eta objective.  Validation is used only for screen model
selection, exactly as in v4.2; no official eAP test or EvTTC labels are opened.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_2 import (  # noqa: E402
    EventBatch,
    MaterializedSplit,
    _autocast,
    _branch_metrics,
    _evaluate,
    _git_commit,
    _gradient_audit,
    _json_safe,
    _materialize,
    _resolve_device,
    _sampling_weights,
    _seed,
    _sha256,
)
from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config  # noqa: E402
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42  # noqa: E402
from e_jepa_ttc.object_event_v4_4 import official_eap_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_5 import (  # noqa: E402
    ObjectEventV45LossConfig,
    object_event_v4_5_loss,
    reciprocal_reverse_target,
)


@dataclass(frozen=True)
class ObjectEventV45TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    maximum_epochs: int = 10
    minimum_epochs: int = 4
    patience_epochs: int = 4
    backbone_learning_rate: float = 3.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    warmup_epochs: int = 1
    seed: int = 7
    precision: str = "fp32"
    shuffle_repeats_during_training: int = 2
    shuffle_repeats_final: int = 8
    bootstrap_repeats_during_training: int = 250
    bootstrap_repeats_final: int = 2000
    weighted_mid_gate: float = 190.0
    validation_pearson_gate: float = 0.52
    validation_balanced_sign_gate: float = 0.70
    validation_negative_accuracy_gate: float = 0.55
    validation_expansion_mae_gate: float = 0.020
    validation_saturation_gate: float = 0.08
    validation_min_sequence_negative_accuracy_gate: float = 0.10
    per_sequence_negative_min_count: int = 20
    zero_event_pearson_drop_gate: float = 0.20
    shuffled_event_pearson_drop_gate: float = 0.20

    def __post_init__(self) -> None:
        positive_ints = (
            self.batch_size,
            self.maximum_epochs,
            self.minimum_epochs,
            self.patience_epochs,
            self.warmup_epochs,
            self.shuffle_repeats_during_training,
            self.shuffle_repeats_final,
            self.bootstrap_repeats_during_training,
            self.bootstrap_repeats_final,
            self.per_sequence_negative_min_count,
            self.num_workers + 1,
        )
        if min(positive_ints) <= 0:
            raise ValueError("v4.5 integer controls must be positive")
        if self.minimum_epochs > self.maximum_epochs:
            raise ValueError("minimum_epochs exceeds maximum_epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if min(self.backbone_learning_rate, self.head_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(
    path: Path,
) -> tuple[ObjectEventV41Config, ObjectEventV45TrainConfig, ObjectEventV45LossConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.5 config must be a mapping")
    return (
        _construct(ObjectEventV41Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(ObjectEventV45TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV45LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
    )


def _load_initial_checkpoint(
    model: ObjectEventTTCV42,
    checkpoint_path: Path,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.2 checkpoint: {checkpoint_path}")
    checkpoint_seed = int(cast(Mapping[str, Any], payload.get("train_config", {})).get("seed", -1))
    if checkpoint_seed != expected_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {expected_seed}, got {checkpoint_seed}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return cast(dict[str, Any], payload)


def _augment_metrics(
    rows: pd.DataFrame,
    per_sequence: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    max_abs_expansion: float,
    per_sequence_negative_min_count: int,
) -> dict[str, Any]:
    target = rows["target_expansion"].to_numpy(dtype=np.float64)
    prediction = rows["prediction_expansion"].to_numpy(dtype=np.float64)
    reverse_prediction = rows["reverse_expansion"].to_numpy(dtype=np.float64)
    delta_t = rows["delta_t_s"].to_numpy(dtype=np.float64)
    target_ttc = rows["target_ttc_s"].to_numpy(dtype=np.float64)
    official = official_eap_metrics(
        target,
        prediction,
        delta_t,
        target_ttc,
        max_abs_expansion=max_abs_expansion,
    )
    reverse_target = reciprocal_reverse_target(
        torch.from_numpy(target), maximum=max_abs_expansion
    ).numpy()
    reciprocal = np.abs(
        np.log1p(-np.clip(prediction, -max_abs_expansion * 0.999, max_abs_expansion * 0.999))
        + np.log1p(
            -np.clip(
                reverse_prediction,
                -max_abs_expansion * 0.999,
                max_abs_expansion * 0.999,
            )
        )
    )
    reverse_metrics = _branch_metrics(
        reverse_target,
        reverse_prediction,
        delta_t,
        ttc_clip=60.0,
        min_expansion=1.0e-4,
    )
    sign_counts = (
        rows.assign(is_negative=rows["target_expansion"].to_numpy(dtype=np.float64) < 0.0)
        .groupby("sequence_id", sort=True)["is_negative"]
        .agg(["sum", "count"])
    )
    negative_by_sequence = sign_counts["sum"].astype(int).to_dict()
    count_by_sequence = sign_counts["count"].astype(int).to_dict()
    per_sequence["negative_count"] = per_sequence["sequence_id"].map(
        negative_by_sequence
    ).astype(int)
    per_sequence["positive_count"] = (
        per_sequence["sequence_id"].map(count_by_sequence).astype(int)
        - per_sequence["negative_count"]
    )
    eligible = per_sequence[
        per_sequence["negative_count"] >= per_sequence_negative_min_count
    ]
    min_sequence_negative = (
        float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    )
    result = dict(metrics)
    result["official_eap"] = official
    result["exact_reversal"] = {
        "target_metrics": reverse_metrics,
        "mean_abs_log_eta_reciprocity_error": float(np.mean(reciprocal)),
        "max_abs_log_eta_reciprocity_error": float(np.max(reciprocal)),
    }
    result["per_sequence"]["minimum_eligible_negative_accuracy"] = min_sequence_negative
    result["per_sequence"]["eligible_negative_sequence_count"] = int(len(eligible))
    return cast(dict[str, Any], _json_safe(result))


def _evaluate_v45(
    model: ObjectEventTTCV42,
    split: MaterializedSplit,
    *,
    train_config: ObjectEventV45TrainConfig,
    loss_config: ObjectEventV45LossConfig,
    device: torch.device,
    final: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows, per_sequence, metrics = _evaluate(
        model,
        split,
        batch_size=train_config.batch_size,
        device=device,
        max_abs_expansion=loss_config.max_abs_expansion,
        shuffle_repeats=(
            train_config.shuffle_repeats_final
            if final
            else train_config.shuffle_repeats_during_training
        ),
        bootstrap_repeats=(
            train_config.bootstrap_repeats_final
            if final
            else train_config.bootstrap_repeats_during_training
        ),
        seed=seed,
    )
    return (
        rows,
        per_sequence,
        _augment_metrics(
            rows,
            per_sequence,
            metrics,
            max_abs_expansion=loss_config.max_abs_expansion,
            per_sequence_negative_min_count=train_config.per_sequence_negative_min_count,
        ),
    )


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _selection_objective(metrics: Mapping[str, Any]) -> float:
    event = cast(Mapping[str, object], metrics["event"])
    official = cast(Mapping[str, object], metrics["official_eap"])
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    dependence = cast(Mapping[str, object], metrics["event_dependence"])
    weighted_mid = _finite(official.get("weighted_mid"), default=1.0e6)
    return (
        weighted_mid
        + 80.0 * (1.0 - _finite(event.get("balanced_sign_accuracy")))
        + 40.0 * (1.0 - _finite(event.get("negative_accuracy")))
        + 20.0 * (1.0 - _finite(event.get("pearson")))
        + 20.0
        * (
            1.0
            - _finite(sequence.get("minimum_eligible_negative_accuracy"))
        )
        - 10.0 * _finite(dependence.get("shuffled_event_pearson_drop"))
    )


def _gates(
    metrics: Mapping[str, Any], config: ObjectEventV45TrainConfig
) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    official = cast(Mapping[str, object], metrics["official_eap"])
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    dependence = cast(Mapping[str, object], metrics["event_dependence"])
    return {
        "weighted_mid": _finite(official.get("weighted_mid"), default=float("inf"))
        <= config.weighted_mid_gate,
        "pearson": _finite(event.get("pearson")) >= config.validation_pearson_gate,
        "balanced_sign": _finite(event.get("balanced_sign_accuracy"))
        >= config.validation_balanced_sign_gate,
        "negative_accuracy": _finite(event.get("negative_accuracy"))
        >= config.validation_negative_accuracy_gate,
        "expansion_mae": _finite(event.get("expansion_mae"), default=float("inf"))
        <= config.validation_expansion_mae_gate,
        "saturation": _finite(event.get("ttc_saturation_rate"), default=float("inf"))
        <= config.validation_saturation_gate,
        "min_sequence_negative_accuracy": _finite(
            sequence.get("minimum_eligible_negative_accuracy")
        )
        >= config.validation_min_sequence_negative_accuracy_gate,
        "zero_event_dependence": _finite(dependence.get("zero_event_pearson_drop"))
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": _finite(
            dependence.get("shuffled_event_pearson_drop")
        )
        >= config.shuffled_event_pearson_drop_gate,
    }


def run(
    *,
    cache_manifest: Path,
    config_path: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    device_name: str,
    seed_override: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    model_config, train_config, loss_config = _load_config(config_path)
    if seed_override is not None:
        train_config = ObjectEventV45TrainConfig(
            **{**asdict(train_config), "seed": seed_override}
        )
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=model_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=model_config.input_size
    )
    weights = _sampling_weights(train_split)
    model = ObjectEventTTCV42(model_config)
    initial_payload = _load_initial_checkpoint(
        model, initial_checkpoint, expected_seed=train_config.seed
    )
    model = model.to(device)

    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": train_config.backbone_learning_rate},
            {"params": head_parameters, "lr": train_config.head_learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_split) / train_config.batch_size)
    total_steps = steps_per_epoch * train_config.maximum_epochs
    warmup_steps = steps_per_epoch * train_config.warmup_epochs

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1.0e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and train_config.precision == "fp16"
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    best_path = output_dir / "best_observed.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.jsonl"

    # Epoch zero is the matching v4.2 checkpoint.  Keeping it as an eligible
    # candidate makes the loss-only experiment fail-safe: fine-tuning cannot
    # silently replace the established baseline with a worse model.
    _, _, initial_validation_metrics = _evaluate_v45(
        model,
        validation_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=False,
        seed=train_config.seed + 4500,
    )
    best_objective = _selection_objective(initial_validation_metrics)
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    torch.save(
        {
            "artifact_type": "object_event_v4_5_best_observed",
            "epoch": 0,
            "global_step": 0,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "loss_config": asdict(loss_config),
            "model_state_dict": model.state_dict(),
            "validation_metrics": initial_validation_metrics,
            "selection_objective": best_objective,
            "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
            "initial_checkpoint_sha256": _sha256(initial_checkpoint),
        },
        best_path,
    )

    for epoch in range(1, train_config.maximum_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(train_config.seed * 1000 + epoch)
        indices = torch.multinomial(
            weights,
            num_samples=len(train_split),
            replacement=True,
            generator=generator,
        )
        component_sums: Counter[str] = Counter()
        epoch_loss = 0.0
        epoch_examples = 0
        gradient_norm_sum = 0.0
        for start in range(0, len(indices), train_config.batch_size):
            batch_indices = indices[start : start + train_config.batch_size]
            events = train_split.events[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            flip_mask = torch.rand(events.shape[0], device=device) < 0.5
            if bool(flip_mask.any()):
                events[flip_mask] = torch.flip(events[flip_mask], dims=(-1,))
            batch = EventBatch(
                events=events,
                delta_t_s=train_split.delta_t_s[batch_indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
                target_ttc_s=train_split.target_ttc_s[batch_indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, train_config.precision):
                output = model(batch.events)
                loss_output = object_event_v4_5_loss(
                    output,
                    batch.delta_t_s,
                    batch.target_ttc_s,
                    config=loss_config,
                )
            if scaler is not None:
                scaler.scale(loss_output.total).backward()
                scaler.unscale_(optimizer)
            else:
                loss_output.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    train_config.max_grad_norm,
                )
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1
            size = int(events.shape[0])
            epoch_examples += size
            epoch_loss += float(loss_output.total.detach().cpu()) * size
            gradient_norm_sum += gradient_norm
            for name, value in loss_output.components.items():
                component_sums[name] += float(value.detach().cpu()) * size

        _, _, validation_metrics = _evaluate_v45(
            model,
            validation_split,
            train_config=train_config,
            loss_config=loss_config,
            device=device,
            final=False,
            seed=train_config.seed + epoch * 17,
        )
        objective = _selection_objective(validation_metrics)
        gates = _gates(validation_metrics, train_config)
        improved = objective < best_objective - 1.0e-6
        if improved:
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "artifact_type": "object_event_v4_5_best_observed",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "loss_config": asdict(loss_config),
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "selection_objective": objective,
                    "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
                    "initial_checkpoint_sha256": _sha256(initial_checkpoint),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "artifact_type": "object_event_v4_5_last",
                "epoch": epoch,
                "global_step": global_step,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_metrics": validation_metrics,
            },
            last_path,
        )
        event = cast(Mapping[str, object], validation_metrics["event"])
        official = cast(Mapping[str, object], validation_metrics["official_eap"])
        sequence = cast(Mapping[str, object], validation_metrics["per_sequence"])
        row = cast(
            dict[str, Any],
            _json_safe(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": epoch_loss / max(epoch_examples, 1),
                    "train_components": {
                        name: value / max(epoch_examples, 1)
                        for name, value in component_sums.items()
                    },
                    "gradient_norm_mean": gradient_norm_sum / max(steps_per_epoch, 1),
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "validation_selection_objective": objective,
                    "validation": validation_metrics,
                    "gates": gates,
                    "best_epoch": best_epoch,
                    "best_objective": best_objective,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            ),
        )
        history.append(row)
        history_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": row["train_loss"],
                    "weighted_mid": official.get("weighted_mid"),
                    "pearson": event.get("pearson"),
                    "balanced_sign": event.get("balanced_sign_accuracy"),
                    "negative_accuracy": event.get("negative_accuracy"),
                    "min_sequence_negative_accuracy": sequence.get(
                        "minimum_eligible_negative_accuracy"
                    ),
                    "gates": gates,
                    "best_epoch": best_epoch,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if (
            epoch >= train_config.minimum_epochs
            and epochs_without_improvement >= train_config.patience_epochs
        ):
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    train_predictions, train_per_sequence, train_metrics = _evaluate_v45(
        model,
        train_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 7001,
    )
    validation_predictions, validation_per_sequence, validation_metrics = _evaluate_v45(
        model,
        validation_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 9001,
    )
    train_predictions.to_csv(output_dir / "train_predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    final_gates = _gates(validation_metrics, train_config)
    screen_passed = all(final_gates.values())
    gradient_audit = _gradient_audit(
        model,
        train_split,
        device=device,
        batch_size=train_config.batch_size,
    )
    if screen_passed:
        torch.save(best_payload, output_dir / "eligible.pt")
    ended_at = datetime.now(UTC)
    summary = cast(
        dict[str, Any],
        _json_safe(
            {
                "artifact_type": "object_event_v4_5_paired_reciprocal_mid_screen",
                "status": "screen_passed" if screen_passed else "screen_failed",
                "created_at": ended_at.isoformat(),
                "started_at": started_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "git_commit": _git_commit(),
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "cache_manifest": cache_manifest.resolve().as_posix(),
                "cache_manifest_sha256": _sha256(cache_manifest),
                "config": config_path.resolve().as_posix(),
                "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
                "initial_checkpoint_sha256": _sha256(initial_checkpoint),
                "initial_checkpoint_epoch": int(initial_payload.get("epoch", 0)),
                "initial_validation_metrics": initial_validation_metrics,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "train_split": train_manifest,
                "validation_split": validation_manifest,
                "completed_epochs": len(history),
                "best_epoch": best_epoch,
                "best_selection_objective": best_objective,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "selection_gates": final_gates,
                "screen_passed": screen_passed,
                "gradient_audit": gradient_audit,
                "scientific_contract": {
                    "same_v4_2_architecture": True,
                    "event_only": True,
                    "receives_observable_motion": False,
                    "receives_boxes": False,
                    "receives_rgb": False,
                    "fine_tunes_matching_seed_v4_2_checkpoint": True,
                    "optimises_log_eta_mid": True,
                    "uses_exact_reciprocal_reverse_target": True,
                    "uses_log_eta_reciprocity": True,
                    "validation_is_not_official_eap_test": True,
                    "evttc_not_opened": True,
                },
            }
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output exists: {output_dir}; pass --force")
            shutil.rmtree(output_dir)
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            config_path=args.config.resolve(),
            initial_checkpoint=args.initial_checkpoint.resolve(),
            output_dir=output_dir,
            device_name=args.device,
            seed_override=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["screen_passed"]) else 2
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_5_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
