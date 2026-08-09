"""Synthetic learning loop for the event-only v5 causal foreground-scale arm."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from e_jepa_ttc.data.synthetic_causal_scale import (
    SyntheticCausalScaleDataset,
    SyntheticCausalScaleSample,
)
from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    CausalScaleTTCOutput,
)
from e_jepa_ttc.reproducibility import seed_everything


@dataclass(frozen=True)
class CausalScaleSyntheticTrainingConfig:
    """Bounded optimizer and selection settings for the synthetic learning gate."""

    seed: int = 7
    epochs: int = 24
    batch_size: int = 32
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    foreground_warmup_epochs: int = 0
    spatial_translation_pixels: int = 0

    def __post_init__(self) -> None:
        if self.seed < 0 or min(self.epochs, self.batch_size) <= 0 or self.num_workers < 0:
            raise ValueError("training seed/workers and epoch/batch controls are invalid")
        if not 0 <= self.foreground_warmup_epochs < self.epochs:
            raise ValueError("foreground_warmup_epochs must lie in [0, epochs)")
        if self.spatial_translation_pixels < 0:
            raise ValueError("spatial_translation_pixels must be non-negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError("optimizer controls are invalid")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must lie in [0, learning_rate]")


@dataclass
class SyntheticTrainingResult:
    """Best validation-selected state and its complete optimization history."""

    model: CausalScaleTTC
    history: list[dict[str, float | int | None]]
    best_epoch: int
    best_selection_score: float


def _tensor(batch: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch field {key!r} must be a tensor")
    return value.to(device, non_blocking=True)


def _forward_loss(
    model: CausalScaleTTC,
    batch: Mapping[str, Any],
    device: torch.device,
    loss_config: CausalScaleTTCLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], CausalScaleTTCOutput]:
    inputs = _tensor(batch, "inputs", device)
    delta = _tensor(batch, "delta_t_s", device)
    output = model(inputs, delta)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=_tensor(batch, "target_ttc_seconds", device),
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=_tensor(batch, "target_valid", device),
        target_masks=_tensor(batch, "target_masks", device),
        mask_valid=_tensor(batch, "mask_valid", device),
        config=loss_config,
    )
    return result.total, result.components, output


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if (
        left.numel() < 2
        or left.std(unbiased=False) <= 1.0e-12
        or right.std(unbiased=False) <= 1.0e-12
    ):
        return None
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())


def _finite_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("non-finite metric reached aggregation")
    return float(sum(values) / len(values))


def _zero_fill_roll(values: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    shifted = torch.roll(values, shifts=(shift_y, shift_x), dims=(-2, -1))
    if shift_y > 0:
        shifted[..., :shift_y, :] = 0
    elif shift_y < 0:
        shifted[..., shift_y:, :] = 0
    if shift_x > 0:
        shifted[..., :, :shift_x] = 0
    elif shift_x < 0:
        shifted[..., :, shift_x:] = 0
    return shifted


def _translate_batch(
    batch: SyntheticCausalScaleSample,
    maximum_pixels: int,
) -> SyntheticCausalScaleSample:
    if maximum_pixels == 0:
        return batch
    shift_y = int(torch.randint(-maximum_pixels, maximum_pixels + 1, ()).item())
    shift_x = int(torch.randint(-maximum_pixels, maximum_pixels + 1, ()).item())
    translated = dict(batch)
    translated["inputs"] = _zero_fill_roll(batch["inputs"], shift_y, shift_x)
    translated["target_masks"] = _zero_fill_roll(
        batch["target_masks"],
        shift_y,
        shift_x,
    )
    return cast(SyntheticCausalScaleSample, translated)


def train_one_epoch(
    model: CausalScaleTTC,
    loader: DataLoader[SyntheticCausalScaleSample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_config: CausalScaleTTCLossConfig,
    grad_clip_norm: float,
    spatial_translation_pixels: int = 0,
) -> dict[str, float]:
    """Run one deterministic FP32 epoch and aggregate example-weighted losses."""

    model.train()
    totals: dict[str, float] = {}
    examples = 0
    for batch in loader:
        batch = _translate_batch(batch, spatial_translation_pixels)
        optimizer.zero_grad(set_to_none=True)
        total, components, _ = _forward_loss(model, batch, device, loss_config)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite causal-scale training loss")
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = int(cast(torch.Tensor, batch["inputs"]).shape[0])
        examples += batch_size
        totals["total"] = totals.get("total", 0.0) + float(total.detach().cpu()) * batch_size
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    if examples == 0:
        raise ValueError("training loader is empty")
    return {key: value / examples for key, value in totals.items()}


@torch.inference_mode()
def evaluate_synthetic_causal_scale(
    model: CausalScaleTTC,
    loader: DataLoader[SyntheticCausalScaleSample],
    device: torch.device,
    *,
    loss_config: CausalScaleTTCLossConfig,
    controls: bool,
) -> dict[str, float | None]:
    """Evaluate predicted masks, physical scale, uncertainty, and optional controls."""

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    known: list[torch.Tensor] = []
    ttc_predictions: list[torch.Tensor] = []
    ttc_targets: list[torch.Tensor] = []
    ratio_covered: list[torch.Tensor] = []
    mask_ious: list[torch.Tensor] = []
    empty_unknown: list[torch.Tensor] = []
    empty_false_positive: list[torch.Tensor] = []
    oddness: list[torch.Tensor] = []
    translation: list[torch.Tensor] = []
    losses: list[float] = []
    for batch in loader:
        total, _, output = _forward_loss(model, batch, device, loss_config)
        inputs = _tensor(batch, "inputs", device)
        delta = _tensor(batch, "delta_t_s", device)
        valid = _tensor(batch, "target_valid", device).bool()
        target_ratio = _tensor(batch, "target_log_ratio", device)
        target_ttc = _tensor(batch, "target_ttc_seconds", device)
        losses.append(float(total.cpu()))
        if bool(valid.any()):
            prediction = output.log_height_ratio[:, -1][valid]
            target = target_ratio[valid]
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            current_known = output.known_mask[valid]
            known.append(current_known.cpu())
            if bool(current_known.any()):
                ttc_predictions.append(output.ttc_mean_seconds[valid][current_known].cpu())
                ttc_targets.append(target_ttc[valid][current_known].cpu())
            sigma = torch.exp(0.5 * output.log_ratio_log_variance[:, -1][valid])
            ratio_covered.append(((prediction - target).abs() <= 1.2815516 * sigma).cpu())
        predicted_masks = torch.sigmoid(output.foreground_logits) >= 0.5
        target_masks = _tensor(batch, "target_masks", device).bool()
        target_nonempty = target_masks.flatten(-3).any(dim=-1)
        intersection = (predicted_masks & target_masks).sum(dim=(-3, -2, -1)).float()
        union = (predicted_masks | target_masks).sum(dim=(-3, -2, -1)).float().clamp_min(1.0)
        if bool(target_nonempty.any()):
            mask_ious.append((intersection[target_nonempty] / union[target_nonempty]).cpu())
        empty_samples = inputs.abs().flatten(1).sum(dim=-1) == 0.0
        if bool(empty_samples.any()):
            empty_unknown.append((~output.known_mask[empty_samples]).cpu())
            empty_false_positive.append(
                predicted_masks[empty_samples].float().mean(dim=(-4, -3, -2, -1)).cpu()
            )
        if controls:
            pair_inputs = inputs[:, -2:]
            pair_delta = delta[:, -1:]
            reverse = model(pair_inputs.flip(1), pair_delta)
            forward_ratio = output.log_height_ratio[:, -1]
            reverse_ratio = reverse.log_height_ratio[:, -1]
            denominator = forward_ratio.abs() + reverse_ratio.abs()
            supported = denominator >= 2.0 * model.config.min_abs_log_ratio
            if bool(supported.any()):
                oddness.append(
                    (
                        (forward_ratio[supported] + reverse_ratio[supported]).abs()
                        / denominator[supported]
                    ).cpu()
                )
            shifted = model(_zero_fill_roll(inputs, 3, -4), delta)
            if bool(output.known_mask.any()):
                translation.append(
                    (
                        shifted.log_height_ratio[:, -1][output.known_mask]
                        - forward_ratio[output.known_mask]
                    )
                    .abs()
                    .cpu()
                )
    if not predictions or not mask_ious:
        raise ValueError("evaluation lacks valid scale or foreground examples")
    prediction_all = torch.cat(predictions).float()
    target_all = torch.cat(targets).float()
    known_all = torch.cat(known).bool()
    denominator = float(torch.dot(target_all, target_all).clamp_min(1.0e-12))
    slope = float(torch.dot(target_all, prediction_all) / denominator)
    sign_accuracy = float((torch.sign(target_all) == torch.sign(prediction_all)).float().mean())
    metrics: dict[str, float | None] = {
        "loss": _finite_mean(losses),
        "log_ratio_mae": float((prediction_all - target_all).abs().mean()),
        "analytic_pearson": _pearson(target_all, prediction_all),
        "slope": slope,
        "sign_accuracy": sign_accuracy,
        "known_coverage": float(known_all.float().mean()),
        "foreground_iou": float(torch.cat(mask_ious).mean()),
        "ratio_80_coverage": float(torch.cat(ratio_covered).float().mean()),
        "empty_unknown": float(torch.cat(empty_unknown).float().mean()) if empty_unknown else None,
        "empty_false_positive_fraction": (
            float(torch.cat(empty_false_positive).mean()) if empty_false_positive else None
        ),
    }
    if ttc_predictions:
        predicted_ttc = torch.cat(ttc_predictions).float()
        target_ttc_all = torch.cat(ttc_targets).float()
        metrics["ttc_symmetric_relative_error"] = float(
            (
                2.0
                * (predicted_ttc - target_ttc_all).abs()
                / (predicted_ttc.abs() + target_ttc_all.abs()).clamp_min(1.0e-6)
            ).mean()
        )
    else:
        metrics["ttc_symmetric_relative_error"] = None
    if controls:
        if not oddness or not translation:
            raise ValueError("evaluation controls lack supported causal pairs")
        oddness_all = torch.cat(oddness).float()
        translation_all = torch.cat(translation).float()
        metrics.update(
            {
                "oddness_median": float(oddness_all.median()),
                "oddness_p95": float(torch.quantile(oddness_all, 0.95)),
                "translation_leakage_p95": float(torch.quantile(translation_all, 0.95)),
            }
        )
    return metrics


def _selection_score(metrics: Mapping[str, float | None]) -> float:
    ratio_mae = metrics.get("log_ratio_mae")
    foreground_iou = metrics.get("foreground_iou")
    correlation = metrics.get("analytic_pearson")
    slope = metrics.get("slope")
    sign_accuracy = metrics.get("sign_accuracy")
    if any(
        value is None
        for value in (ratio_mae, foreground_iou, correlation, slope, sign_accuracy)
    ):
        raise ValueError("validation selection metrics are unavailable")
    score = (
        2.0 * float(cast(float, ratio_mae))
        + (1.0 - float(cast(float, foreground_iou)))
        + (1.0 - float(cast(float, correlation)))
        + abs(1.0 - float(cast(float, slope)))
        + (1.0 - float(cast(float, sign_accuracy)))
    )
    if not math.isfinite(score):
        raise FloatingPointError("validation selection score is non-finite")
    return score


def train_synthetic_causal_scale(
    model_config: CausalScaleTTCConfig,
    training_config: CausalScaleSyntheticTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
    train_dataset: SyntheticCausalScaleDataset,
    validation_dataset: SyntheticCausalScaleDataset,
    device: torch.device,
) -> SyntheticTrainingResult:
    """Train on one seed group and select only by a disjoint validation group."""

    seed_everything(training_config.seed, deterministic=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_config.seed)
    train_loader: DataLoader[SyntheticCausalScaleSample] = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training_config.num_workers,
    )
    validation_loader: DataLoader[SyntheticCausalScaleSample] = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    model = CausalScaleTTC(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs - training_config.foreground_warmup_epochs,
        eta_min=training_config.minimum_learning_rate,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    history: list[dict[str, float | int | None]] = []
    warmup_loss_config = replace(
        loss_config,
        log_ratio_nll_weight=0.0,
        log_ratio_huber_weight=0.0,
        risk_weight=0.0,
        auxiliary_inverse_ttc_weight=0.0,
        residual_regularization_weight=0.0,
        temporal_consistency_weight=0.0,
    )
    for epoch in range(1, training_config.epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_loss_config = (
            warmup_loss_config
            if epoch <= training_config.foreground_warmup_epochs
            else loss_config
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_config=epoch_loss_config,
            grad_clip_norm=training_config.grad_clip_norm,
            spatial_translation_pixels=training_config.spatial_translation_pixels,
        )
        validation_metrics = evaluate_synthetic_causal_scale(
            model,
            validation_loader,
            device,
            loss_config=loss_config,
            controls=False,
        )
        score = _selection_score(validation_metrics)
        record: dict[str, float | int | None] = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "foreground_warmup": int(epoch <= training_config.foreground_warmup_epochs),
            "selection_eligible": int(epoch > training_config.foreground_warmup_epochs),
            "selection_score": score,
        }
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(record)
        if epoch > training_config.foreground_warmup_epochs and score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch > training_config.foreground_warmup_epochs:
            scheduler.step()
    if best_state is None:
        raise RuntimeError("synthetic training produced no selectable checkpoint")
    model.load_state_dict(best_state, strict=True)
    return SyntheticTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_selection_score=best_score,
    )


@torch.inference_mode()
def calibrate_ratio_uncertainty(
    model: CausalScaleTTC,
    loader: DataLoader[SyntheticCausalScaleSample],
    device: torch.device,
    *,
    target_coverage: float = 0.8,
) -> dict[str, float | int]:
    """Fit one scalar variance scale using validation residuals only."""

    if not 0.5 < target_coverage < 1.0:
        raise ValueError("target_coverage must lie in (0.5,1)")
    model.eval()
    model.set_log_ratio_variance_offset(0.0)
    standardized: list[torch.Tensor] = []
    for batch in loader:
        inputs = _tensor(batch, "inputs", device)
        delta = _tensor(batch, "delta_t_s", device)
        valid = _tensor(batch, "target_valid", device).bool()
        if not bool(valid.any()):
            continue
        output = model(inputs, delta)
        target = _tensor(batch, "target_log_ratio", device)[valid]
        residual = (output.log_height_ratio[:, -1][valid] - target).abs()
        sigma = torch.exp(0.5 * output.log_ratio_log_variance[:, -1][valid])
        standardized.append((residual / sigma.clamp_min(1.0e-8)).cpu())
    if not standardized:
        raise ValueError("validation calibration has no valid ratios")
    values = torch.cat(standardized).float()
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    central_z = float(normal.icdf(torch.tensor((1.0 + target_coverage) / 2.0)))
    empirical_quantile = float(torch.quantile(values, target_coverage))
    scale = max(empirical_quantile / central_z, 1.0e-6)
    offset = 2.0 * math.log(scale)
    model.set_log_ratio_variance_offset(offset)
    return {
        "target_coverage": target_coverage,
        "valid_count": int(values.numel()),
        "standard_deviation_scale": scale,
        "log_variance_offset": offset,
    }


def checkpoint_payload(
    result: SyntheticTrainingResult,
    training_config: CausalScaleSyntheticTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
) -> dict[str, Any]:
    """Build a portable checkpoint mapping for ``torch.save``."""

    return {
        "artifact_type": "causal_scale_v5_synthetic_checkpoint_v1",
        "model_config": result.model.checkpoint_config(),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "best_epoch": result.best_epoch,
        "best_selection_score": result.best_selection_score,
        "model_state_dict": result.model.state_dict(),
    }


__all__ = [
    "CausalScaleSyntheticTrainingConfig",
    "SyntheticTrainingResult",
    "calibrate_ratio_uncertainty",
    "checkpoint_payload",
    "evaluate_synthetic_causal_scale",
    "train_one_epoch",
    "train_synthetic_causal_scale",
]
