#!/usr/bin/env python3
"""Train Object Event TTC v4.15 shared odd sign projection."""
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
    OddConsensusHead,
)
from e_jepa_ttc.object_event_v4_4 import (  # noqa: E402
    branch_metrics,
    official_eap_metrics,
    pearson,
)
from e_jepa_ttc.training.object_event_v4_12 import (  # noqa: E402
    ObjectEventV412LossConfig,
    reversal_balanced_sign_loss,
)
from e_jepa_ttc.training.object_event_v4_15 import (  # noqa: E402
    positive_magnitude_ceiling,
    positive_tail_threshold,
    projected_numpy,
    select_oof_threshold,
    sign_statistics,
    v415_screen_gates,
)


@dataclass(frozen=True)
class V415TrainConfig:
    seed: int = 1515
    descriptor_batch_size: int = 12
    head_batch_size: int = 128
    overfit_samples: int = 64
    overfit_epochs: int = 80
    fold_count: int = 3
    fold_epochs: int = 36
    final_epochs: int = 36
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    tail_alpha_min: float = 0.001
    tail_alpha_max: float = 0.003
    tail_alpha_count: int = 5
    oof_positive_accuracy_floor: float = 0.995
    oof_override_rate_ceiling: float = 0.10
    magnitude_cap_positive_quantile: float = 0.25
    per_sequence_negative_min_count: int = 20

    def __post_init__(self) -> None:
        if min(
            self.descriptor_batch_size,
            self.head_batch_size,
            self.overfit_samples,
            self.overfit_epochs,
            self.fold_count,
            self.fold_epochs,
            self.final_epochs,
            self.tail_alpha_count,
            self.per_sequence_negative_min_count,
        ) <= 0:
            raise ValueError("v4.15 integer controls must be positive")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if not 0.0 < self.tail_alpha_min <= self.tail_alpha_max < 0.5:
            raise ValueError("invalid positive-tail alpha range")
        if not 0.0 < self.magnitude_cap_positive_quantile < 1.0:
            raise ValueError("magnitude_cap_positive_quantile must lie in (0,1)")
        if min(self.learning_rate, self.max_grad_norm) <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(
    path: Path,
) -> tuple[
    ObjectEventV415Config,
    V415TrainConfig,
    ObjectEventV412LossConfig,
    dict[str, float],
]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.15 config must be a mapping")
    model = _construct(ObjectEventV415Config, cast(Mapping[str, Any], raw.get("model", {})))
    train = _construct(V415TrainConfig, cast(Mapping[str, Any], raw.get("train", {})))
    loss = _construct(
        ObjectEventV412LossConfig,
        cast(Mapping[str, Any], raw.get("loss", {})),
    )
    gates_raw = raw.get("screen_gates", {})
    if not isinstance(gates_raw, dict) or not gates_raw:
        raise ValueError("screen_gates must be a non-empty mapping")
    gates = {str(key): float(value) for key, value in gates_raw.items()}
    return model, train, loss, gates


def _parse_checkpoints(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("checkpoint arguments must be SEED=PATH")
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        result[seed] = Path(path_text)
    if len(result) < 3:
        raise ValueError("v4.15 requires at least three checkpoints")
    return dict(sorted(result.items()))


def _build_model(
    *,
    checkpoint_paths: Mapping[int, Path],
    v48_config_path: Path,
    v412_config_path: Path,
    model_config: ObjectEventV415Config,
    device: torch.device,
) -> tuple[ObjectEventTTCV415, dict[int, dict[str, Any]]]:
    v412_probe_config, _, _ = _load_probe_config(v412_config_path)
    extractors: list[ObjectEventTTCV412] = []
    payloads: dict[int, dict[str, Any]] = {}
    for seed, path in checkpoint_paths.items():
        backbone, payload = _load_backbone(
            v48_config_path=v48_config_path,
            checkpoint_path=path,
        )
        extractors.append(ObjectEventTTCV412(backbone, v412_probe_config))
        payloads[seed] = payload
    model = ObjectEventTTCV415(extractors, model_config).to(device)
    return model, payloads


@torch.no_grad()
def _extract_descriptors(
    model: ObjectEventTTCV415,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    branch: str,
    seed: int,
) -> torch.Tensor:
    if branch not in {"event", "zero", "shuffled"}:
        raise KeyError(branch)
    permutation = None
    if branch == "shuffled":
        permutation = torch.randperm(len(events), generator=torch.Generator().manual_seed(seed))
    values: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(events), batch_size):
        end = min(start + batch_size, len(events))
        if branch == "zero":
            batch = torch.zeros_like(events[start:end])
        elif branch == "shuffled":
            assert permutation is not None
            batch = events[permutation[start:end]]
        else:
            batch = events[start:end]
        descriptor = model.consensus_descriptor(batch.to(device=device, dtype=torch.float32))
        values.append(descriptor.float().cpu())
    return torch.cat(values, dim=0)


def _new_head(
    descriptor_dim: int,
    config: ObjectEventV415Config,
    *,
    seed: int,
    device: torch.device,
) -> OddConsensusHead:
    _seed(seed)
    return OddConsensusHead(descriptor_dim, config).to(device)


def _train_head(
    *,
    head: OddConsensusHead,
    descriptors: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    loss_config: ObjectEventV412LossConfig,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_weights = weights[train_indices].double().clamp_min(1.0e-12)
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_objective = float("inf")
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        head.train()
        sampled_local = torch.multinomial(
            train_weights,
            len(train_indices),
            replacement=True,
            generator=generator,
        )
        sampled = train_indices[sampled_local]
        total = 0.0
        count = 0
        for start in range(0, len(sampled), batch_size):
            index = sampled[start : start + batch_size]
            x = descriptors[index].to(device=device, dtype=torch.float32)
            y = target[index].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = reversal_balanced_sign_loss(
                logits, -logits, y, config=loss_config
            ).total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(index)
            count += len(index)
        objective = total / max(count, 1)
        if validation_indices is not None and len(validation_indices):
            head.eval()
            with torch.no_grad():
                logits = head(
                    descriptors[validation_indices].to(device=device, dtype=torch.float32)
                ).cpu()
            labels = (target[validation_indices] < 0.0).numpy()
            pred = (torch.sigmoid(logits).numpy() >= 0.5)
            pos = ~labels
            neg = labels
            pos_acc = float(np.mean(~pred[pos])) if pos.any() else 0.0
            neg_acc = float(np.mean(pred[neg])) if neg.any() else 0.0
            objective = 2.0 - pos_acc - neg_acc
        history.append({"epoch": float(epoch), "objective": float(objective)})
        if objective < best_objective - 1.0e-10:
            best_objective = objective
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state, strict=True)
    return best_state, best_epoch, history


@torch.no_grad()
def _head_probability(
    head: OddConsensusHead,
    descriptors: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    logits_values: list[np.ndarray] = []
    head.eval()
    for start in range(0, len(descriptors), batch_size):
        x = descriptors[start : start + batch_size].to(device=device, dtype=torch.float32)
        logits = head(x)
        logits_values.append(logits.float().cpu().numpy())
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(logits_values)


def _folds(sequence_ids: Sequence[str], fold_count: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(map(str, sequence_ids))), dtype=object)
    if len(unique) < fold_count:
        raise ValueError("not enough train sequences for grouped folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_sequences = [set(unique[index::fold_count].tolist()) for index in range(fold_count)]
    values = np.asarray(list(map(str, sequence_ids)), dtype=object)
    return [np.flatnonzero(np.isin(values, list(group))) for group in fold_sequences]


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
    per_sequence = _per_sequence(
        frame, prediction, minimum_negatives=minimum_negatives
    )
    result["minimum_sequence_negative_accuracy"] = per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    result["minimum_sequence_pearson"] = per_sequence.attrs["minimum_sequence_pearson"]
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


def _overfit_gates(stats: Mapping[str, float], antisymmetry: float) -> dict[str, bool]:
    return {
        "balanced_sign": stats["balanced_sign_accuracy"] >= 0.98,
        "positive_accuracy": stats["positive_accuracy"] >= 0.98,
        "negative_accuracy": stats["negative_accuracy"] >= 0.98,
        "exact_antisymmetry": antisymmetry <= 1.0e-6,
    }


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
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    if mode not in {"overfit", "screen"}:
        raise ValueError("mode must be overfit or screen")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    model_config, train_config, loss_config, gate_config = _load_config(config_path)
    device = _resolve_device(device_name)
    model, checkpoint_payloads = _build_model(
        checkpoint_paths=checkpoint_paths,
        v48_config_path=v48_config_path,
        v412_config_path=v412_config_path,
        model_config=model_config,
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

    if mode == "overfit":
        indices = _balanced_subset_indices(
            train_frame, train_config.overfit_samples, train_config.seed
        )
        subset_events = train_split.events[indices]
        descriptors = _extract_descriptors(
            model,
            subset_events,
            batch_size=train_config.descriptor_batch_size,
            device=device,
            branch="event",
            seed=train_config.seed,
        )
        target = torch.as_tensor(
            train_frame["target_expansion"].to_numpy(dtype=np.float32)[indices.numpy()]
        )
        weights = torch.ones(len(indices), dtype=torch.float64)
        head = _new_head(
            model.descriptor_dim,
            model_config,
            seed=train_config.seed,
            device=device,
        )
        all_indices = torch.arange(len(indices))
        state, best_epoch, history = _train_head(
            head=head,
            descriptors=descriptors,
            target=target,
            weights=weights,
            train_indices=all_indices,
            validation_indices=all_indices,
            epochs=train_config.overfit_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed,
        )
        probability, logits = _head_probability(
            head,
            descriptors,
            batch_size=train_config.head_batch_size,
            device=device,
        )
        frame = train_frame.iloc[indices.numpy()].reset_index(drop=True)
        prediction, override = projected_numpy(
            frame["fused_prediction_expansion"].to_numpy(dtype=np.float64),
            probability,
            0.5,
        )
        stats = sign_statistics(
            frame,
            prediction,
            minimum_negatives=train_config.per_sequence_negative_min_count,
        )
        antisymmetry = float(np.max(np.abs(torch.sigmoid(torch.from_numpy(logits)).numpy() + torch.sigmoid(torch.from_numpy(-logits)).numpy() - 1.0)))
        gates = _overfit_gates(stats, antisymmetry)
        passed = all(gates.values())
        torch.save(
            {
                "artifact_type": "object_event_v4_15_overfit_checkpoint",
                "head_state_dict": state,
                "epoch": best_epoch,
                "descriptor_dim": model.descriptor_dim,
            },
            output_dir / "best_gate_passing.pt",
        )
        pd.DataFrame(history).to_json(output_dir / "history.jsonl", orient="records", lines=True)
        result: dict[str, Any] = {
            "artifact_type": "object_event_v4_15_shared_odd_projection",
            "status": "overfit_passed" if passed else "overfit_failed",
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "best_epoch": best_epoch,
            "sign_metrics": stats,
            "override_rate": float(np.mean(override)),
            "antisymmetry_max_abs": antisymmetry,
            "gates": gates,
            "passed": passed,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(_json_safe(result), indent=2), encoding="utf-8"
        )
        return result

    train_descriptors = _extract_descriptors(
        model,
        train_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed,
    )
    validation_descriptors = _extract_descriptors(
        model,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed + 1,
    )
    target_train = torch.as_tensor(
        train_frame["target_expansion"].to_numpy(dtype=np.float32)
    )
    weights = _sample_weights(train_frame)
    fold_indices = _folds(
        train_frame["sequence_id"].astype(str).tolist(),
        train_config.fold_count,
        train_config.seed,
    )
    oof_probability = np.full(len(train_frame), np.nan, dtype=np.float64)
    oof_logits = np.full(len(train_frame), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    all_indices_np = np.arange(len(train_frame), dtype=np.int64)
    best_epochs: list[int] = []
    for fold, validation_indices_np in enumerate(fold_indices):
        train_indices_np = np.setdiff1d(all_indices_np, validation_indices_np)
        head = _new_head(
            model.descriptor_dim,
            model_config,
            seed=train_config.seed + 100 + fold,
            device=device,
        )
        _, best_epoch, history = _train_head(
            head=head,
            descriptors=train_descriptors,
            target=target_train,
            weights=weights,
            train_indices=torch.as_tensor(train_indices_np, dtype=torch.long),
            validation_indices=torch.as_tensor(validation_indices_np, dtype=torch.long),
            epochs=train_config.fold_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed + 200 + fold,
        )
        _, training_logits = _head_probability(
            head,
            train_descriptors[train_indices_np],
            batch_size=train_config.head_batch_size,
            device=device,
        )
        _, held_out_logits = _head_probability(
            head,
            train_descriptors[validation_indices_np],
            batch_size=train_config.head_batch_size,
            device=device,
        )
        median_abs_logit = float(np.median(np.abs(training_logits)))
        logit_scale = float(np.clip(1.0 / max(median_abs_logit, 1.0e-6), 0.1, 10.0))
        calibrated_logits = held_out_logits * logit_scale
        probability = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -40.0, 40.0)))
        oof_probability[validation_indices_np] = probability
        oof_logits[validation_indices_np] = calibrated_logits
        best_epochs.append(best_epoch)
        fold_rows.append(
            {
                "fold": fold,
                "held_out_sequences": sorted(
                    train_frame.iloc[validation_indices_np]["sequence_id"].astype(str).unique().tolist()
                ),
                "best_epoch": best_epoch,
                "best_objective": min(row["objective"] for row in history),
                "training_median_abs_logit": median_abs_logit,
                "logit_scale": logit_scale,
            }
        )
    if not np.isfinite(oof_probability).all() or not np.isfinite(oof_logits).all():
        raise RuntimeError("OOF predictions are incomplete")

    tail_alphas = np.linspace(
        train_config.tail_alpha_min,
        train_config.tail_alpha_max,
        train_config.tail_alpha_count,
    )
    selected, sweep = select_oof_threshold(
        train_frame,
        oof_probability,
        tail_alphas=tail_alphas,
        positive_accuracy_floor=train_config.oof_positive_accuracy_floor,
        override_rate_ceiling=train_config.oof_override_rate_ceiling,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    oof_prediction, oof_override = projected_numpy(
        train_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64),
        oof_probability,
        selected.threshold,
    )
    oof_direct_sign = np.where(oof_probability >= selected.threshold, -1.0, 1.0)
    oof_stats = sign_statistics(
        train_frame,
        oof_direct_sign,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    oof_stats["override_rate"] = float(np.mean(oof_override))

    final_head = _new_head(
        model.descriptor_dim,
        model_config,
        seed=train_config.seed + 1000,
        device=device,
    )
    final_state, _, final_history = _train_head(
        head=final_head,
        descriptors=train_descriptors,
        target=target_train,
        weights=weights,
        train_indices=torch.arange(len(train_frame)),
        validation_indices=None,
        epochs=train_config.final_epochs,
        batch_size=train_config.head_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,
        loss_config=loss_config,
        device=device,
        seed=train_config.seed + 1100,
    )
    _, final_train_logits = _head_probability(
        final_head,
        train_descriptors,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    final_training_median_abs_logit = float(np.median(np.abs(final_train_logits)))
    final_logit_scale = float(
        np.clip(1.0 / max(final_training_median_abs_logit, 1.0e-6), 0.1, 10.0)
    )
    final_train_logits = final_train_logits * final_logit_scale
    final_train_probability = 1.0 / (
        1.0 + np.exp(-np.clip(final_train_logits, -40.0, 40.0))
    )
    train_positive = train_frame["target_expansion"].to_numpy(dtype=np.float64) >= 0.0
    final_threshold = positive_tail_threshold(
        final_train_probability[train_positive], selected.tail_alpha
    )
    train_positive_magnitude = np.abs(
        train_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)[
            train_positive
        ]
    )
    final_magnitude_ceiling = positive_magnitude_ceiling(
        train_positive_magnitude, train_config.magnitude_cap_positive_quantile
    )
    _, raw_validation_logits = _head_probability(
        final_head,
        validation_descriptors,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    validation_logits = raw_validation_logits * final_logit_scale
    validation_probability = 1.0 / (
        1.0 + np.exp(-np.clip(validation_logits, -40.0, 40.0))
    )
    baseline_prediction = validation_frame[
        "fused_prediction_expansion"
    ].to_numpy(dtype=np.float64)
    uncapped_prediction, uncapped_override = projected_numpy(
        baseline_prediction, validation_probability, final_threshold
    )
    prediction, override = projected_numpy(
        baseline_prediction,
        validation_probability,
        final_threshold,
        magnitude_ceiling=final_magnitude_ceiling,
    )
    validation_metrics, per_sequence = _metrics(
        validation_frame,
        prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    baseline_metrics, _ = _metrics(
        validation_frame,
        baseline_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )

    zero_descriptors = _extract_descriptors(
        model,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="zero",
        seed=train_config.seed + 2,
    )
    shuffled_descriptors = _extract_descriptors(
        model,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="shuffled",
        seed=train_config.seed + 3,
    )
    _, zero_logits_raw = _head_probability(
        final_head, zero_descriptors, batch_size=train_config.head_batch_size, device=device
    )
    _, shuffled_logits_raw = _head_probability(
        final_head,
        shuffled_descriptors,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    zero_probability = 1.0 / (
        1.0 + np.exp(-np.clip(zero_logits_raw * final_logit_scale, -40.0, 40.0))
    )
    shuffled_probability = 1.0 / (
        1.0
        + np.exp(-np.clip(shuffled_logits_raw * final_logit_scale, -40.0, 40.0))
    )
    zero_prediction, _ = projected_numpy(
        validation_frame["fused_zero_events_expansion"].to_numpy(dtype=np.float64),
        zero_probability,
        final_threshold,
        magnitude_ceiling=final_magnitude_ceiling,
    )
    shuffled_prediction, _ = projected_numpy(
        validation_frame["fused_shuffled_mean_expansion"].to_numpy(dtype=np.float64),
        shuffled_probability,
        final_threshold,
        magnitude_ceiling=final_magnitude_ceiling,
    )
    target_validation = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    diagnostics = {
        "selected_tail_alpha": selected.tail_alpha,
        "oof_selected_threshold": selected.threshold,
        "final_selected_threshold": final_threshold,
        "magnitude_cap_positive_quantile": train_config.magnitude_cap_positive_quantile,
        "magnitude_ceiling_expansion": final_magnitude_ceiling,
        "threshold_feasible": selected.feasible,
        "final_training_median_abs_logit": final_training_median_abs_logit,
        "final_logit_scale": final_logit_scale,
        "uncapped_override_count": int(uncapped_override.sum()),
        "suppressed_override_count": int(uncapped_override.sum() - override.sum()),
        "override_count": int(override.sum()),
        "override_rate": float(np.mean(override)),
        "override_abs_magnitude_mean": float(
            np.mean(np.abs(baseline_prediction[override])) if override.any() else 0.0
        ),
        "override_abs_magnitude_max": float(
            np.max(np.abs(baseline_prediction[override])) if override.any() else 0.0
        ),
        "zero_event_pearson_drop": pearson(target_validation, prediction)
        - pearson(target_validation, zero_prediction),
        "shuffled_event_pearson_drop": pearson(target_validation, prediction)
        - pearson(target_validation, shuffled_prediction),
        "antisymmetry_max_abs": float(
            np.max(
                np.abs(
                    torch.sigmoid(torch.from_numpy(validation_logits)).numpy()
                    + torch.sigmoid(torch.from_numpy(-validation_logits)).numpy()
                    - 1.0
                )
            )
        ),
    }
    gates = v415_screen_gates(
        oof=cast(Mapping[str, float], oof_stats),
        validation=cast(Mapping[str, float], validation_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        diagnostics=diagnostics,
        gates=gate_config,
    )
    passed = all(gates.values())

    output = validation_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "delta_t_s",
        "target_ttc_s",
        "target_expansion",
    ]].copy()
    output["baseline_prediction_expansion"] = baseline_prediction
    output["negative_probability"] = validation_probability
    output["selected_threshold"] = final_threshold
    output["magnitude_ceiling_expansion"] = final_magnitude_ceiling
    output["uncapped_override"] = uncapped_override
    output["override"] = override
    output["projected_prediction_expansion"] = prediction
    output["zero_events_prediction_expansion"] = zero_prediction
    output["shuffled_prediction_expansion"] = shuffled_prediction
    output.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    sweep.to_csv(output_dir / "oof_threshold_sweep.csv", index=False)
    oof_output = train_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "target_expansion",
        "fused_prediction_expansion",
    ]].copy()
    oof_output["oof_calibrated_logit"] = oof_logits
    oof_output["oof_negative_probability"] = oof_probability
    oof_output["oof_projected_prediction_expansion"] = oof_prediction
    oof_output["oof_override"] = oof_override
    oof_output.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    torch.save(
        {
            "artifact_type": "object_event_v4_15_shared_odd_projection_checkpoint",
            "head_state_dict": final_state,
            "selected_tail_alpha": selected.tail_alpha,
            "oof_selected_threshold": selected.threshold,
            "selected_threshold": final_threshold,
            "magnitude_cap_positive_quantile": train_config.magnitude_cap_positive_quantile,
            "magnitude_ceiling_expansion": final_magnitude_ceiling,
            "final_logit_scale": final_logit_scale,
            "descriptor_dim": model.descriptor_dim,
            "checkpoint_paths": {seed: path.resolve().as_posix() for seed, path in checkpoint_paths.items()},
        },
        output_dir / "shared_odd_projection.pt",
    )

    result = {
        "artifact_type": "object_event_v4_15_shared_odd_projection",
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
        "fold_best_epochs": best_epochs,
        "selected_threshold": selected.__dict__,
        "oof_metrics": oof_stats,
        "baseline_validation_metrics": baseline_metrics,
        "projected_validation_metrics": validation_metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "three_frozen_v48_backbones": True,
            "one_shared_odd_sign_head": True,
            "descriptor_mean_and_median_are_exactly_odd": True,
            "tail_alpha_selected_only_from_grouped_train_oof": True,
            "final_threshold_is_train_positive_tail_quantile": True,
            "magnitude_ceiling_value_computed_from_train_positives_only": True,
            "magnitude_cap_quantile_informed_by_v4151_development_validation": True,
            "v4152_is_not_independent_validation": True,
            "fold_and_final_logits_scaled_from_train_only": True,
            "validation_not_used_for_threshold_or_checkpoint_selection": True,
            "sign_times_frozen_magnitude_projection": True,
            "no_opposite_sign_blending": True,
            "no_new_near_zero_cancellation": True,
            "event_only_inference": True,
            "boxes_heights_sequence_ids_and_track_ids_not_forward_features": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_integrated_training": passed,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(result), indent=2), encoding="utf-8"
    )
    pd.DataFrame(final_history).to_json(
        output_dir / "final_head_history.jsonl", orient="records", lines=True
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
        default=Path("configs/experiment/e_jepa_garl_object_event_shared_odd_projection_v4_15.yaml"),
    )
    parser.add_argument(
        "--v48-checkpoint",
        action="append",
        required=True,
        help="Repeat SEED=PATH at least three times",
    )
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
            "artifact_type": "object_event_v4_15_failure",
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
