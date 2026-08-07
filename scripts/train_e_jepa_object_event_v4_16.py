#!/usr/bin/env python3
"""Train Object Event TTC v4.16 causal temporal sign + magnitude heads."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_2 import (  # noqa: E402
    _git_commit,
    _json_safe,
    _resolve_device,
    _seed,
    _sha256,
)
from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import (  # noqa: E402
    IDENTITY_COLUMNS,
    _align_ensemble,
    _balanced_subset_indices,
    _load_backbone,
    _load_probe_config,
    _per_sequence,
    _read_ensemble,
    _sample_weights,
)
from e_jepa_ttc.models.object_event_v4_12 import ObjectEventTTCV412  # noqa: E402
from e_jepa_ttc.models.object_event_v4_15 import (  # noqa: E402
    ObjectEventTTCV415,
    ObjectEventV415Config,
)
from e_jepa_ttc.models.object_event_v4_16 import (  # noqa: E402
    ObjectEventTTCV416,
    ObjectEventV416Config,
)
from e_jepa_ttc.object_event_v4_4 import (  # noqa: E402
    branch_metrics,
    official_eap_metrics,
    pearson,
)
from e_jepa_ttc.training.object_event_v4_16 import (  # noqa: E402
    ObjectEventV416LossConfig,
    causal_window_indices,
    gather_scalar_windows,
    gather_windows,
    sign_statistics,
    temporal_dual_head_loss,
    uniform_epoch_indices,
    v416_screen_gates,
)


@dataclass(frozen=True)
class ObjectEventV416TrainConfig:
    seed: int = 1616
    descriptor_batch_size: int = 12
    head_batch_size: int = 64
    overfit_samples: int = 96
    overfit_epochs: int = 100
    fold_count: int = 3
    fold_epochs: int = 24
    final_epochs: int = 24
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    per_sequence_negative_min_count: int = 20

    def __post_init__(self) -> None:
        integers = (
            self.descriptor_batch_size,
            self.head_batch_size,
            self.overfit_samples,
            self.overfit_epochs,
            self.fold_count,
            self.fold_epochs,
            self.final_epochs,
            self.per_sequence_negative_min_count,
        )
        if min(integers) <= 0:
            raise ValueError("v4.16 integer controls must be positive")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("v4.16 optimizer controls must be positive")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(
    path: Path,
) -> tuple[
    ObjectEventV416Config,
    ObjectEventV416TrainConfig,
    ObjectEventV416LossConfig,
    dict[str, float],
]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.16 config must be a mapping")
    model = _construct(ObjectEventV416Config, cast(Mapping[str, Any], raw.get("model", {})))
    train = _construct(
        ObjectEventV416TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))
    )
    loss = _construct(
        ObjectEventV416LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))
    )
    gates_raw = raw.get("screen_gates", {})
    if not isinstance(gates_raw, dict) or not gates_raw:
        raise ValueError("screen_gates must be a non-empty mapping")
    gates = {str(key): float(value) for key, value in gates_raw.items()}
    return model, train, loss, gates


def _parse_checkpoints(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        result[seed] = Path(path_text)
    if sorted(result) != [7, 13, 23]:
        raise ValueError("v4.16 requires exact checkpoints for seeds 7, 13 and 23")
    return dict(sorted(result.items()))


def _build_frozen_consensus(
    *,
    checkpoint_paths: Mapping[int, Path],
    v48_config_path: Path,
    v412_config_path: Path,
    device: torch.device,
) -> tuple[ObjectEventTTCV415, dict[int, dict[str, Any]]]:
    probe_config, _, _ = _load_probe_config(v412_config_path)
    extractors: list[ObjectEventTTCV412] = []
    payloads: dict[int, dict[str, Any]] = {}
    for seed, path in checkpoint_paths.items():
        backbone, payload = _load_backbone(
            v48_config_path=v48_config_path,
            checkpoint_path=path,
        )
        extractors.append(ObjectEventTTCV412(backbone, probe_config))
        payloads[seed] = payload
    model = ObjectEventTTCV415(extractors, ObjectEventV415Config()).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, payloads


@torch.no_grad()
def _extract_features(
    frozen: ObjectEventTTCV415,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    branch: str,
    seed: int,
    maximum_magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if branch not in {"event", "zero", "shuffled"}:
        raise KeyError(branch)
    permutation = None
    if branch == "shuffled":
        permutation = torch.randperm(len(events), generator=torch.Generator().manual_seed(seed))
    descriptor_values: list[torch.Tensor] = []
    anchor_values: list[torch.Tensor] = []
    frozen.eval()
    for start in range(0, len(events), batch_size):
        end = min(start + batch_size, len(events))
        if branch == "zero":
            batch = torch.zeros_like(events[start:end])
        elif branch == "shuffled":
            assert permutation is not None
            batch = events[permutation[start:end]]
        else:
            batch = events[start:end]
        batch = batch.to(device=device, dtype=torch.float32)
        descriptors: list[torch.Tensor] = []
        magnitudes: list[torch.Tensor] = []
        for extractor in frozen.extractors:
            descriptor, pooled_log_eta = extractor._descriptor_and_base(batch)
            descriptor = functional.layer_norm(
                descriptor,
                (descriptor.shape[-1],),
                weight=None,
                bias=None,
                eps=frozen.config.epsilon,
            )
            expansion = (1.0 - torch.exp(pooled_log_eta)).abs().clamp_max(maximum_magnitude)
            descriptors.append(descriptor)
            magnitudes.append(expansion)
        stacked = torch.stack(descriptors, dim=0)
        consensus = torch.cat(
            (stacked.mean(dim=0), stacked.median(dim=0).values), dim=1
        )
        anchor = torch.stack(magnitudes, dim=0).median(dim=0).values
        descriptor_values.append(consensus.float().cpu())
        anchor_values.append(anchor.float().cpu())
    descriptor = torch.cat(descriptor_values, dim=0)
    anchor = torch.cat(anchor_values, dim=0)
    if descriptor.shape[1] != frozen.descriptor_dim:
        raise AssertionError("v4.16 consensus descriptor dimension mismatch")
    return descriptor, anchor


def _folds(sequence_ids: Sequence[str], fold_count: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(map(str, sequence_ids))), dtype=object)
    if len(unique) < fold_count:
        raise ValueError("not enough sequences for grouped folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    values = np.asarray(list(map(str, sequence_ids)), dtype=object)
    groups = [set(unique[index::fold_count].tolist()) for index in range(fold_count)]
    return [np.flatnonzero(np.isin(values, list(group))) for group in groups]


def _new_head(
    descriptor_dim: int,
    config: ObjectEventV416Config,
    *,
    seed: int,
    device: torch.device,
) -> ObjectEventTTCV416:
    _seed(seed)
    return ObjectEventTTCV416(descriptor_dim, config).to(device)


def _train_head(
    *,
    model: ObjectEventTTCV416,
    windows: torch.Tensor,
    window_mask: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    teacher_magnitude: torch.Tensor,
    history_target: torch.Tensor,
    sample_weights: torch.Tensor,
    train_indices: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    loss_config: ObjectEventV416LossConfig,
    device: torch.device,
    seed: int,
    epoch_callback: Any | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sampled = uniform_epoch_indices(train_indices, generator)
        totals = np.zeros(5, dtype=np.float64)
        seen = 0
        for start in range(0, len(sampled), batch_size):
            index = sampled[start : start + batch_size]
            x = windows[index].to(device=device, dtype=torch.float32)
            mask = window_mask[index].to(device=device)
            a = anchor[index].to(device=device, dtype=torch.float32)
            y = target[index].to(device=device, dtype=torch.float32)
            teacher = teacher_magnitude[index].to(device=device, dtype=torch.float32)
            hy = history_target[index].to(device=device, dtype=torch.float32)
            weight = sample_weights[index].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, mask, a)
            loss = temporal_dual_head_loss(
                sign_logit=output.sign_logit,
                instant_sign_logits=output.instant_sign_logits,
                magnitude=output.magnitude,
                target_expansion=y,
                teacher_magnitude=teacher,
                history_target_expansion=hy,
                history_mask=mask,
                sample_weight=weight,
                magnitude_floor=model.config.magnitude_floor,
                config=loss_config,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            size = len(index)
            totals += size * np.asarray(
                [
                    float(loss.total.detach().cpu()),
                    float(loss.final_sign.detach().cpu()),
                    float(loss.history_sign.detach().cpu()),
                    float(loss.magnitude_target.detach().cpu()),
                    float(loss.magnitude_teacher.detach().cpu()),
                ]
            )
            seen += size
        values = totals / max(seen, 1)
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(values[0]),
                "final_sign_loss": float(values[1]),
                "history_sign_loss": float(values[2]),
                "magnitude_target_loss": float(values[3]),
                "magnitude_teacher_loss": float(values[4]),
            }
        )
        if epoch_callback is not None and bool(epoch_callback(epoch, model, history[-1])):
            break
    return copy.deepcopy(model.state_dict()), history


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV416,
    windows: torch.Tensor,
    window_mask: torch.Tensor,
    anchor: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | float | list[float]]:
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    log_ratios: list[np.ndarray] = []
    odd_errors: list[np.ndarray] = []
    even_errors: list[np.ndarray] = []
    sign_weight_sum = None
    magnitude_weight_sum = None
    row_count = 0
    model.eval()
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        x = windows[start:end].to(device=device, dtype=torch.float32)
        mask = window_mask[start:end].to(device=device)
        a = anchor[start:end].to(device=device, dtype=torch.float32)
        output = model(x, mask, a)
        sign_error, magnitude_error = model.symmetry_errors(x, mask, a)
        predictions.append(output.signed_expansion.float().cpu().numpy())
        probabilities.append(output.negative_probability.float().cpu().numpy())
        logits.append(output.sign_logit.float().cpu().numpy())
        magnitudes.append(output.magnitude.float().cpu().numpy())
        log_ratios.append(output.magnitude_log_ratio.float().cpu().numpy())
        odd_errors.append(sign_error.float().cpu().numpy())
        even_errors.append(magnitude_error.float().cpu().numpy())
        sw = output.sign_temporal_weights.float().cpu().sum(dim=0)
        mw = output.magnitude_temporal_weights.float().cpu().sum(dim=0)
        sign_weight_sum = sw if sign_weight_sum is None else sign_weight_sum + sw
        magnitude_weight_sum = mw if magnitude_weight_sum is None else magnitude_weight_sum + mw
        row_count += len(x)
    assert sign_weight_sum is not None and magnitude_weight_sum is not None
    return {
        "prediction": np.concatenate(predictions),
        "probability": np.concatenate(probabilities),
        "logit": np.concatenate(logits),
        "magnitude": np.concatenate(magnitudes),
        "log_ratio": np.concatenate(log_ratios),
        "sign_oddness_max_abs": float(np.max(np.concatenate(odd_errors))),
        "magnitude_evenness_max_abs": float(np.max(np.concatenate(even_errors))),
        "mean_sign_temporal_weights": (sign_weight_sum / row_count).numpy().tolist(),
        "mean_magnitude_temporal_weights": (magnitude_weight_sum / row_count).numpy().tolist(),
    }


def _metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    minimum_negatives: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    result: dict[str, Any] = branch_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
    )
    per_sequence = _per_sequence(frame, prediction, minimum_negatives=minimum_negatives)
    result["minimum_sequence_negative_accuracy"] = per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    result["minimum_sequence_pearson"] = per_sequence.attrs["minimum_sequence_pearson"]
    result["magnitude_mae"] = float(np.mean(np.abs(np.abs(prediction) - np.abs(target))))
    official = official_eap_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
    )
    result["official_eap"] = official
    result["weighted_mid"] = float(official["weighted_mid"])
    result["weighted_rte_percent"] = float(official["weighted_rte_percent"])
    return result, per_sequence


def _overfit_gates(metrics: Mapping[str, Any], diagnostics: Mapping[str, float]) -> dict[str, bool]:
    return {
        "balanced_sign": float(metrics["balanced_sign_accuracy"]) >= 0.98,
        "negative_accuracy": float(metrics["negative_accuracy"]) >= 0.98,
        "expansion_mae": float(metrics["expansion_mae"]) <= 0.008,
        "exact_sign_oddness": diagnostics["sign_oddness_max_abs"] <= 1.0e-5,
        "exact_magnitude_evenness": diagnostics["magnitude_evenness_max_abs"] <= 1.0e-6,
    }


def _overfit_rank(metrics: Mapping[str, Any], gates: Mapping[str, bool]) -> tuple[float, ...]:
    """Rank failed overfit checkpoints without weakening any gate."""
    return (
        float(sum(bool(value) for value in gates.values())),
        float(metrics["balanced_sign_accuracy"]),
        float(metrics["negative_accuracy"]),
        -float(metrics["expansion_mae"]),
        float(metrics["pearson"]),
    )


def run(
    *,
    cache_manifest: Path,
    v48_config_path: Path,
    v412_config_path: Path,
    config_path: Path,
    checkpoint_paths: Mapping[int, Path],
    ensemble_train_path: Path,
    ensemble_validation_path: Path,
    output_dir: Path,
    device_name: str,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    if mode not in {"overfit", "screen"}:
        raise ValueError("mode must be overfit or screen")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    started = time.perf_counter()

    model_config, train_config, loss_config, gate_config = _load_config(config_path)
    device = _resolve_device(device_name)
    frozen, checkpoint_payloads = _build_frozen_consensus(
        checkpoint_paths=checkpoint_paths,
        v48_config_path=v48_config_path,
        v412_config_path=v412_config_path,
        device=device,
    )
    base_config, _, _, _, _ = _load_v48_config(v48_config_path)
    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=base_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=base_config.input_size
    )
    train_frame = _align_ensemble(train_split, _read_ensemble(ensemble_train_path))
    validation_frame = _align_ensemble(
        validation_split, _read_ensemble(ensemble_validation_path)
    )

    train_descriptor, train_anchor = _extract_features(
        frozen,
        train_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    train_window_index, train_window_mask_np, train_history_length = causal_window_indices(
        train_frame, window_size=model_config.window_size
    )
    train_windows = gather_windows(train_descriptor, train_window_index, train_window_mask_np)
    train_window_mask = torch.as_tensor(train_window_mask_np, dtype=torch.bool)
    train_target = torch.as_tensor(
        train_frame["target_expansion"].to_numpy(dtype=np.float32)
    )
    train_history_target = gather_scalar_windows(
        train_target, train_window_index, train_window_mask_np
    )
    train_teacher = torch.as_tensor(
        np.abs(train_frame["fused_prediction_expansion"].to_numpy(dtype=np.float32))
    )
    train_weights = _sample_weights(train_frame)

    if mode == "overfit":
        selected = _balanced_subset_indices(
            train_frame, train_config.overfit_samples, train_config.seed
        )
        frame = train_frame.iloc[selected.numpy()].reset_index(drop=True)
        model = _new_head(
            frozen.descriptor_dim, model_config, seed=train_config.seed, device=device
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_rank: tuple[float, ...] | None = None
        best_epoch = 0
        best_metrics: dict[str, Any] | None = None
        best_diagnostics: dict[str, float] | None = None
        best_gates: dict[str, bool] | None = None
        gate_state: dict[str, torch.Tensor] | None = None
        gate_epoch = 0
        gate_metrics: dict[str, Any] | None = None
        gate_diagnostics: dict[str, float] | None = None
        gate_gates: dict[str, bool] | None = None

        def _evaluate_overfit_epoch(
            epoch: int,
            candidate: ObjectEventTTCV416,
            history_row: dict[str, float],
        ) -> bool:
            nonlocal best_state, best_rank, best_epoch
            nonlocal best_metrics, best_diagnostics, best_gates
            nonlocal gate_state, gate_epoch, gate_metrics, gate_diagnostics, gate_gates

            subset_prediction = _predict(
                candidate,
                train_windows[selected],
                train_window_mask[selected],
                train_anchor[selected],
                batch_size=train_config.head_batch_size,
                device=device,
            )
            metrics, _ = _metrics(
                frame,
                cast(np.ndarray, subset_prediction["prediction"]),
                minimum_negatives=train_config.per_sequence_negative_min_count,
            )
            diagnostics = {
                "sign_oddness_max_abs": float(subset_prediction["sign_oddness_max_abs"]),
                "magnitude_evenness_max_abs": float(
                    subset_prediction["magnitude_evenness_max_abs"]
                ),
            }
            gates = _overfit_gates(metrics, diagnostics)
            passed = all(gates.values())
            rank = _overfit_rank(metrics, gates)

            history_row.update(
                {
                    "eval_balanced_sign_accuracy": float(
                        metrics["balanced_sign_accuracy"]
                    ),
                    "eval_negative_accuracy": float(metrics["negative_accuracy"]),
                    "eval_expansion_mae": float(metrics["expansion_mae"]),
                    "eval_pearson": float(metrics["pearson"]),
                    "eval_gate_count": float(sum(gates.values())),
                    "eval_passed": float(passed),
                }
            )

            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_state = copy.deepcopy(candidate.state_dict())
                best_epoch = epoch
                best_metrics = copy.deepcopy(metrics)
                best_diagnostics = copy.deepcopy(diagnostics)
                best_gates = copy.deepcopy(gates)

            if passed:
                gate_state = copy.deepcopy(candidate.state_dict())
                gate_epoch = epoch
                gate_metrics = copy.deepcopy(metrics)
                gate_diagnostics = copy.deepcopy(diagnostics)
                gate_gates = copy.deepcopy(gates)
                return True
            return False

        _, history = _train_head(
            model=model,
            windows=train_windows,
            window_mask=train_window_mask,
            anchor=train_anchor,
            target=train_target,
            teacher_magnitude=train_teacher,
            history_target=train_history_target,
            sample_weights=train_weights,
            train_indices=selected,
            epochs=train_config.overfit_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed,
            epoch_callback=_evaluate_overfit_epoch,
        )

        passed = gate_state is not None
        if passed:
            assert gate_metrics is not None and gate_diagnostics is not None and gate_gates is not None
            selected_state = gate_state
            selected_epoch = gate_epoch
            metrics = gate_metrics
            diagnostics = gate_diagnostics
            gates = gate_gates
            checkpoint_name = "best_gate_passing.pt"
            selection_reason = "first_gate_passing_epoch"
        else:
            assert best_state is not None
            assert best_metrics is not None and best_diagnostics is not None and best_gates is not None
            selected_state = best_state
            selected_epoch = best_epoch
            metrics = best_metrics
            diagnostics = best_diagnostics
            gates = best_gates
            checkpoint_name = "best_observed.pt"
            selection_reason = "best_observed_rank"

        torch.save(
            {
                "artifact_type": "object_event_v4_16_overfit_checkpoint",
                "model_state_dict": selected_state,
                "descriptor_dim": frozen.descriptor_dim,
                "model_config": asdict(model_config),
                "epoch": selected_epoch,
                "selection_reason": selection_reason,
            },
            output_dir / checkpoint_name,
        )
        pd.DataFrame(history).to_json(
            output_dir / "history.jsonl", orient="records", lines=True
        )
        result: dict[str, Any] = {
            "artifact_type": "object_event_v4_16_temporal_dual_head",
            "status": "overfit_passed" if passed else "overfit_failed",
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_epochs": len(history),
            "maximum_epochs": train_config.overfit_epochs,
            "selected_epoch": selected_epoch,
            "selected_checkpoint": checkpoint_name,
            "selection_reason": selection_reason,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "gates": gates,
            "passed": passed,
            "scientific_contract": {
                "overfit_evaluated_each_epoch": True,
                "overfit_gates_unchanged": True,
                "screen_schedule_unchanged": True,
                "validation_not_used_for_overfit_selection": True,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(_json_safe(result), indent=2), encoding="utf-8"
        )
        return result

    validation_descriptor, validation_anchor = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed + 1,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    validation_window_index, validation_window_mask_np, validation_history_length = causal_window_indices(
        validation_frame, window_size=model_config.window_size
    )
    validation_windows = gather_windows(
        validation_descriptor, validation_window_index, validation_window_mask_np
    )
    validation_window_mask = torch.as_tensor(validation_window_mask_np, dtype=torch.bool)

    fold_rows: list[dict[str, Any]] = []
    oof_prediction = np.full(len(train_frame), np.nan, dtype=np.float64)
    oof_probability = np.full(len(train_frame), np.nan, dtype=np.float64)
    folds = _folds(
        train_frame["sequence_id"].astype(str).tolist(),
        train_config.fold_count,
        train_config.seed,
    )
    all_indices = np.arange(len(train_frame), dtype=np.int64)
    for fold_index, held_out in enumerate(folds):
        train_indices_np = np.setdiff1d(all_indices, held_out, assume_unique=False)
        model = _new_head(
            frozen.descriptor_dim,
            model_config,
            seed=train_config.seed + 100 + fold_index,
            device=device,
        )
        _, fold_history = _train_head(
            model=model,
            windows=train_windows,
            window_mask=train_window_mask,
            anchor=train_anchor,
            target=train_target,
            teacher_magnitude=train_teacher,
            history_target=train_history_target,
            sample_weights=train_weights,
            train_indices=torch.as_tensor(train_indices_np, dtype=torch.long),
            epochs=train_config.fold_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed + 100 + fold_index,
        )
        held = torch.as_tensor(held_out, dtype=torch.long)
        fold_prediction = _predict(
            model,
            train_windows[held],
            train_window_mask[held],
            train_anchor[held],
            batch_size=train_config.head_batch_size,
            device=device,
        )
        oof_prediction[held_out] = cast(np.ndarray, fold_prediction["prediction"])
        oof_probability[held_out] = cast(np.ndarray, fold_prediction["probability"])
        held_frame = train_frame.iloc[held_out].reset_index(drop=True)
        fold_metrics, _ = _metrics(
            held_frame,
            cast(np.ndarray, fold_prediction["prediction"]),
            minimum_negatives=train_config.per_sequence_negative_min_count,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_sequences": sorted(held_frame["sequence_id"].astype(str).unique().tolist()),
                "epochs": train_config.fold_epochs,
                "final_loss": fold_history[-1]["loss"],
                "metrics": fold_metrics,
            }
        )
    if not np.isfinite(oof_prediction).all() or not np.isfinite(oof_probability).all():
        raise AssertionError("OOF predictions are incomplete")
    oof_metrics, _ = _metrics(
        train_frame,
        oof_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )

    final_model = _new_head(
        frozen.descriptor_dim,
        model_config,
        seed=train_config.seed,
        device=device,
    )
    final_state, final_history = _train_head(
        model=final_model,
        windows=train_windows,
        window_mask=train_window_mask,
        anchor=train_anchor,
        target=train_target,
        teacher_magnitude=train_teacher,
        history_target=train_history_target,
        sample_weights=train_weights,
        train_indices=torch.arange(len(train_frame), dtype=torch.long),
        epochs=train_config.final_epochs,
        batch_size=train_config.head_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,
        loss_config=loss_config,
        device=device,
        seed=train_config.seed,
    )
    validation_prediction = _predict(
        final_model,
        validation_windows,
        validation_window_mask,
        validation_anchor,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    prediction = cast(np.ndarray, validation_prediction["prediction"])
    validation_metrics, per_sequence = _metrics(
        validation_frame,
        prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(
        validation_frame,
        baseline_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )

    zero_descriptor, zero_anchor = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="zero",
        seed=train_config.seed + 2,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    shuffled_descriptor, shuffled_anchor = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="shuffled",
        seed=train_config.seed + 3,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    zero_windows = gather_windows(
        zero_descriptor, validation_window_index, validation_window_mask_np
    )
    shuffled_windows = gather_windows(
        shuffled_descriptor, validation_window_index, validation_window_mask_np
    )
    zero_result = _predict(
        final_model,
        zero_windows,
        validation_window_mask,
        zero_anchor,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    shuffled_result = _predict(
        final_model,
        shuffled_windows,
        validation_window_mask,
        shuffled_anchor,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    target_validation = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    temporal_probability = cast(np.ndarray, validation_prediction["probability"])
    temporal_magnitude = cast(np.ndarray, validation_prediction["magnitude"])
    temporal_sign = np.where(temporal_probability >= 0.5, -1.0, 1.0)
    baseline_sign = np.where(baseline_prediction < 0.0, -1.0, 1.0)
    temporal_sign_baseline_magnitude = temporal_sign * np.abs(baseline_prediction)
    baseline_sign_temporal_magnitude = baseline_sign * temporal_magnitude
    temporal_sign_baseline_magnitude_metrics, _ = _metrics(
        validation_frame,
        temporal_sign_baseline_magnitude,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    baseline_sign_temporal_magnitude_metrics, _ = _metrics(
        validation_frame,
        baseline_sign_temporal_magnitude,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    diagnostics = {
        "zero_event_pearson_drop": pearson(target_validation, prediction)
        - pearson(target_validation, cast(np.ndarray, zero_result["prediction"])),
        "shuffled_event_pearson_drop": pearson(target_validation, prediction)
        - pearson(target_validation, cast(np.ndarray, shuffled_result["prediction"])),
        "sign_oddness_max_abs": float(validation_prediction["sign_oddness_max_abs"]),
        "magnitude_evenness_max_abs": float(validation_prediction["magnitude_evenness_max_abs"]),
        "mean_history_length": float(np.mean(validation_history_length)),
        "full_history_fraction": float(np.mean(validation_history_length >= model_config.window_size)),
        "mean_sign_temporal_weights": validation_prediction["mean_sign_temporal_weights"],
        "mean_magnitude_temporal_weights": validation_prediction["mean_magnitude_temporal_weights"],
        "mean_anchor_magnitude": float(np.mean(validation_anchor.numpy())),
        "mean_predicted_magnitude": float(np.mean(cast(np.ndarray, validation_prediction["magnitude"]))),
        "train_true_negative_rate": float(np.mean(train_frame["target_expansion"].to_numpy(dtype=np.float64) < 0.0)),
        "oof_predicted_negative_rate": float(np.mean(oof_probability >= 0.5)),
        "validation_true_negative_rate": float(np.mean(target_validation < 0.0)),
        "validation_predicted_negative_rate": float(np.mean(temporal_probability >= 0.5)),
        "importance_weight_min": float(train_weights.min().item()),
        "importance_weight_max": float(train_weights.max().item()),
        "importance_weight_mean": float(train_weights.mean().item()),
    }
    gate_diagnostics = {
        key: float(value)
        for key, value in diagnostics.items()
        if isinstance(value, (int, float, np.floating))
    }
    gates = v416_screen_gates(
        oof=cast(Mapping[str, float], oof_metrics),
        validation=cast(Mapping[str, float], validation_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        diagnostics=gate_diagnostics,
        gates=gate_config,
    )
    passed = all(gates.values())

    validation_output = validation_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "delta_t_s",
        "target_ttc_s",
        "target_expansion",
    ]].copy()
    validation_output["history_length"] = validation_history_length
    validation_output["baseline_prediction_expansion"] = baseline_prediction
    validation_output["v48_anchor_magnitude"] = validation_anchor.numpy()
    validation_output["negative_probability"] = cast(np.ndarray, validation_prediction["probability"])
    validation_output["predicted_magnitude"] = cast(np.ndarray, validation_prediction["magnitude"])
    validation_output["magnitude_log_ratio"] = cast(np.ndarray, validation_prediction["log_ratio"])
    validation_output["prediction_expansion"] = prediction
    validation_output["zero_events_prediction_expansion"] = cast(np.ndarray, zero_result["prediction"])
    validation_output["shuffled_prediction_expansion"] = cast(np.ndarray, shuffled_result["prediction"])
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)

    oof_output = train_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "target_expansion",
        "fused_prediction_expansion",
    ]].copy()
    oof_output["history_length"] = train_history_length
    oof_output["oof_negative_probability"] = oof_probability
    oof_output["oof_prediction_expansion"] = oof_prediction
    oof_output.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    pd.DataFrame(final_history).to_json(
        output_dir / "final_history.jsonl", orient="records", lines=True
    )
    (output_dir / "folds.json").write_text(
        json.dumps(_json_safe(fold_rows), indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "artifact_type": "object_event_v4_16_temporal_dual_head_checkpoint",
            "model_state_dict": final_state,
            "descriptor_dim": frozen.descriptor_dim,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "checkpoint_paths": {
                seed: path.resolve().as_posix() for seed, path in checkpoint_paths.items()
            },
        },
        output_dir / "temporal_dual_head.pt",
    )

    result = {
        "artifact_type": "object_event_v4_16_temporal_dual_head",
        "status": "screen_passed" if passed else "screen_failed",
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": {
            "model": asdict(model_config),
            "train": asdict(train_config),
            "loss": asdict(loss_config),
            "screen_gates": gate_config,
        },
        "cache_manifest": cache_manifest.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "v48_checkpoints": {
            seed: {
                "path": path.resolve().as_posix(),
                "sha256": _sha256(path),
                "epoch": checkpoint_payloads[seed].get("epoch"),
            }
            for seed, path in checkpoint_paths.items()
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "folds": fold_rows,
        "oof_metrics": oof_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "temporal_validation_metrics": validation_metrics,
        "temporal_sign_baseline_magnitude_metrics": temporal_sign_baseline_magnitude_metrics,
        "baseline_sign_temporal_magnitude_metrics": baseline_sign_temporal_magnitude_metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "three_frozen_v48_backbones": True,
            "causal_track_windows": True,
            "track_and_sequence_ids_are_grouping_metadata_not_forward_features": True,
            "one_exact_odd_temporal_sign_head": True,
            "one_sign_even_positive_magnitude_head": True,
            "final_prediction_is_sign_times_magnitude": True,
            "no_probability_threshold_sweep": True,
            "no_posthoc_override_rule": True,
            "v410_magnitude_is_train_distillation_target_only": True,
            "sequence_sign_importance_weights_applied_once_in_loss": True,
            "epoch_sampling_is_uniform_without_replacement": True,
            "hybrid_component_metrics_are_diagnostic_only": True,
            "validation_not_used_for_epoch_or_hyperparameter_selection": True,
            "event_only_validation_inference": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_multiseed_or_partial_unfreeze": passed,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(result), indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument(
        "--v48-config",
        type=Path,
        default=Path("configs/experiment/e_jepa_garl_object_event_dense_motion_v4_8.yaml"),
    )
    parser.add_argument(
        "--v412-config",
        type=Path,
        default=Path("configs/experiment/e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/e_jepa_garl_object_event_temporal_dual_head_v4_16.yaml"),
    )
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("overfit", "screen"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            cache_manifest=args.cache_manifest,
            v48_config_path=args.v48_config,
            v412_config_path=args.v412_config,
            config_path=args.config,
            checkpoint_paths=_parse_checkpoints(args.v48_checkpoint),
            ensemble_train_path=args.ensemble_train,
            ensemble_validation_path=args.ensemble_validation,
            output_dir=args.output_dir,
            device_name=args.device,
            mode=args.mode,
            force=args.force,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_16_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(result), indent=2))
    return 0 if bool(result["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
