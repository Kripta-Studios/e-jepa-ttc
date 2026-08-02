"""TTC objective for eAP pretraining.

This module provides the training loop and head for the signed-TTC objective,
which combines the standard JEPA objective with an auxiliary bounding-box
conditioned TTC regression head on GarlTTC targets.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader

from e_jepa_ttc.data.garlttc_eap import GarlTTCBatch, GarlTTCEAPDataset, GarlTTCEAPIndex
from e_jepa_ttc.models import (
    TubeletTokenGeometry,
    infer_tubelet_token_geometry,
    pool_object_embeddings,
)
from e_jepa_ttc.reproducibility import resolve_device
from e_jepa_ttc.training.carla_jepa import (
    _atomic_torch_save,
)
from e_jepa_ttc.training.eap_jepa import (
    EAPJEPATrainerConfig,
    build_eap_jepa_models,
    compute_eap_jepa_objective,
    update_eap_jepa_ema,
)

PRECISION_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def hash_context_ids(
    context_ids: list[str],
) -> str:
    payload = "\n".join(sorted(context_ids)).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


class EAPSignedTTCHead(nn.Module):
    """Auxiliary TTC regression head."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects [N, D] object embeddings. Returns [N]."""
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x.squeeze(-1)


def gather_object_ttc_targets(
    *,
    batch: GarlTTCBatch,
    object_indices: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Gather TTC targets in the exact order of pooled objects."""

    if not object_indices:
        raise RuntimeError("Cannot gather TTC targets for an empty object list")

    targets: list[torch.Tensor] = []

    for batch_index, object_index in object_indices:
        if batch_index < 0 or batch_index >= len(batch.target_ttc):
            raise IndexError(f"Invalid batch index returned by object pooling: {batch_index}")

        sample_targets = batch.target_ttc[batch_index]

        if sample_targets.ndim != 1:
            raise RuntimeError(
                f"Each batch.target_ttc item must have shape [N], got {tuple(sample_targets.shape)}"
            )

        if object_index < 0 or object_index >= len(sample_targets):
            raise IndexError(
                "Invalid object index returned by object pooling: "
                f"batch={batch_index}, object={object_index}, "
                f"target_count={len(sample_targets)}"
            )

        targets.append(sample_targets[object_index])

    result = torch.stack(targets).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    if result.ndim != 1:
        raise RuntimeError(f"Expected flattened TTC targets [N], got {tuple(result.shape)}")

    return result


def run_ttc_epoch(
    *,
    epoch: int,
    dataloader: DataLoader[GarlTTCBatch],
    target_encoder: nn.Module,
    online_encoder: nn.Module,
    predictor: nn.Module,
    ttc_head: EAPSignedTTCHead,
    scaler: torch.amp.GradScaler | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    geometry: TubeletTokenGeometry,
    device: torch.device,
    config: EAPJEPATrainerConfig,
    start_time: float,
    train: bool = True,
    total_optimizer_steps: int,
    optimizer_step: int,
    train_median_ttc: float | None = None,
) -> tuple[dict[str, float], int, float | None]:
    """Execute one epoch of the signed-TTC objective."""
    target_encoder.eval()

    if train:
        online_encoder.train()
        predictor.train()
        ttc_head.train()
        context_manager = torch.enable_grad()
    else:
        online_encoder.eval()
        predictor.eval()
        ttc_head.eval()
        context_manager = torch.no_grad()

    metrics_sum: dict[str, float] = defaultdict(float)
    count = 0
    total_samples = 0
    all_targets_for_median = []

    all_errors = []
    all_targets = []
    all_preds = []

    seq_track_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    group_size = config.gradient_accumulation
    batch_count = len(dataloader)

    with context_manager:
        for batch_index, batch in enumerate(dataloader):
            if train and batch_index % group_size == 0:
                optimizer.zero_grad(set_to_none=True)

            group_start = (batch_index // group_size) * group_size
            current_group_size = min(group_size, batch_count - group_start)

            context = batch.context.to(device, non_blocking=True)
            futures = batch.futures.to(device, non_blocking=True)
            future_valid = batch.future_valid.to(device, non_blocking=True)

            bsz = context.shape[0]
            if bsz == 0:
                continue

            total_samples += bsz

            use_autocast = device.type == "cuda" and config.precision in {"fp16", "bf16"}
            autocast_dtype = PRECISION_MAP.get(config.precision, torch.float32)

            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=autocast_dtype):
                jepa_out = compute_eap_jepa_objective(
                    online_encoder=online_encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    context=context,
                    futures=futures,
                    future_valid=future_valid,
                    config=config,
                )
                loss_jepa = jepa_out.loss
                cos_sim = torch.tensor(jepa_out.metrics.get("alignment_loss", 0.0), device=device)

                context_tokens = online_encoder.forward_tokens(context)

                object_embeddings, object_indices = pool_object_embeddings(
                    tokens=context_tokens,
                    bbox_masks=batch.bbox_masks,
                    geometry=geometry,
                )

                if len(object_indices) == 0:
                    raise RuntimeError("TTC batch contains no valid objects")

                target_ttc_norm = gather_object_ttc_targets(
                    batch=batch,
                    object_indices=object_indices,
                    device=device,
                )

                if object_embeddings.ndim != 2:
                    raise RuntimeError(
                        "Object embeddings must have shape [N, D], "
                        f"got {tuple(object_embeddings.shape)}"
                    )

                if object_embeddings.shape[0] != (target_ttc_norm.shape[0]):
                    raise RuntimeError(
                        "Object/target count mismatch: "
                        f"{object_embeddings.shape[0]} embeddings versus "
                        f"{target_ttc_norm.shape[0]} targets"
                    )

                pred_ttc_norm_raw = ttc_head(object_embeddings)

                if pred_ttc_norm_raw.shape != (target_ttc_norm.shape):
                    raise RuntimeError(
                        "Prediction/target shape mismatch: "
                        f"{tuple(pred_ttc_norm_raw.shape)} versus "
                        f"{tuple(target_ttc_norm.shape)}"
                    )

                loss_ttc = F.smooth_l1_loss(
                    pred_ttc_norm_raw,
                    target_ttc_norm,
                    beta=0.05,
                    reduction="mean",
                )

                total_loss = loss_jepa + config.ttc_loss_weight * loss_ttc
                total_loss_accum = total_loss / current_group_size

            if train:
                if scaler is not None and config.precision == "fp16":
                    scaler.scale(total_loss_accum).backward()
                else:
                    total_loss_accum.backward()

            group_finished = (batch_index + 1) % group_size == 0 or (batch_index + 1) == batch_count

            divergence: float | None = None
            ema_momentum: float | None = None

            if train and group_finished:
                trainable_params = (
                    list(online_encoder.parameters())
                    + list(predictor.parameters())
                    + list(ttc_head.parameters())
                )

                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    1.0,
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer_step += 1

                divergence, ema_momentum = update_eap_jepa_ema(
                    target_encoder=target_encoder,
                    online_encoder=online_encoder,
                    optimizer_step=optimizer_step,
                    total_optimizer_steps=(total_optimizer_steps),
                    config=config,
                )

            with torch.no_grad():
                pred_ttc_norm_metric = pred_ttc_norm_raw.detach().clamp(-1.0, 1.0)

                target_ttc_norm_metric = target_ttc_norm.detach()

                pred_ttc_seconds = pred_ttc_norm_metric * 10.0

                target_ttc_seconds = target_ttc_norm_metric * 10.0

                prediction_out_of_range_fraction = float(
                    (pred_ttc_norm_raw.detach().abs() > 1.0).float().mean().item()
                )

            metrics_sum["loss_total"] += total_loss.item()
            metrics_sum["loss_jepa"] += loss_jepa.item()
            metrics_sum["loss_ttc"] += loss_ttc.item()
            metrics_sum["cosine_similarity"] += cos_sim.item()
            metrics_sum["prediction_out_of_range_fraction"] += prediction_out_of_range_fraction

            if train and group_finished and divergence is not None and ema_momentum is not None:
                metrics_sum["target_encoder_divergence_l2"] += float(divergence)

                metrics_sum["ema_momentum"] += float(ema_momentum)

                metrics_sum["optimizer_update_count"] += 1.0

            count += 1

            predictions_cpu = pred_ttc_seconds.cpu().tolist()

            targets_cpu = target_ttc_seconds.cpu().tolist()

            for flat_index, (
                batch_index,
                object_index,
            ) in enumerate(object_indices):
                sequence_id = batch.sequence_ids[batch_index]

                track_id = batch.track_ids[batch_index][object_index]

                prediction = float(predictions_cpu[flat_index])

                target = float(targets_cpu[flat_index])

                absolute_error = abs(prediction - target)

                all_preds.append(prediction)
                all_targets.append(target)
                all_errors.append(absolute_error)

                seq_track_errors[sequence_id][track_id].append(absolute_error)

            if train:
                all_targets_for_median.extend(targets_cpu)

    if train and all_targets_for_median:
        train_median_ttc = float(np.median(all_targets_for_median))

    metrics: dict[str, float] = {}
    metrics["context_sample_count"] = float(total_samples)

    if count > 0:
        metrics["loss_total"] = metrics_sum["loss_total"] / count
        metrics["loss_jepa"] = metrics_sum["loss_jepa"] / count
        metrics["loss_ttc"] = metrics_sum["loss_ttc"] / count
        metrics["cosine_similarity"] = metrics_sum["cosine_similarity"] / count
        metrics["prediction_out_of_range_fraction"] = (
            metrics_sum["prediction_out_of_range_fraction"] / count
        )

    optimizer_update_count = int(metrics_sum["optimizer_update_count"])

    if optimizer_update_count > 0:
        metrics["target_encoder_divergence_l2"] = (
            metrics_sum["target_encoder_divergence_l2"] / optimizer_update_count
        )

        metrics["ema_momentum"] = metrics_sum["ema_momentum"] / optimizer_update_count
    else:
        metrics["target_encoder_divergence_l2"] = 0.0

        metrics["ema_momentum"] = 0.0

    metrics["optimizer_update_count"] = float(optimizer_update_count)

    if all_errors:
        all_err_arr = np.array(all_errors)
        all_tgt_arr = np.array(all_targets)
        all_pred_arr = np.array(all_preds)

        metrics["micro_mae"] = float(np.mean(all_err_arr))
        metrics["micro_rmse"] = float(np.sqrt(np.mean(all_err_arr**2)))

        ge_0_25_mask = np.abs(all_tgt_arr) >= 0.25
        if np.any(ge_0_25_mask):
            ge_targets = all_tgt_arr[ge_0_25_mask]
            ge_preds = all_pred_arr[ge_0_25_mask]
            ge_errors = all_err_arr[ge_0_25_mask]

            correct_signs = (ge_targets > 0) == (ge_preds > 0)
            metrics["sign_accuracy_abs_ge_0_25"] = float(np.mean(correct_signs))
            cov_frac = float(np.sum(ge_0_25_mask) / len(all_targets))
            metrics["sign_accuracy_coverage_fraction"] = cov_frac

            rel_errors = ge_errors / np.abs(ge_targets)
            metrics["mean_relative_error_abs_ge_0_25"] = float(np.mean(rel_errors))
            metrics["median_relative_error_abs_ge_0_25"] = float(np.median(rel_errors))

    track_maes, track_rmses = [], []
    seq_maes, seq_rmses = [], []
    for _seq_id, tracks in seq_track_errors.items():
        seq_errs = []
        for _track_id, errs in tracks.items():
            if errs:
                track_maes.append(np.mean(errs))
                track_rmses.append(np.sqrt(np.mean(np.array(errs) ** 2)))
                seq_errs.extend(errs)
        if seq_errs:
            seq_maes.append(np.mean(seq_errs))
            seq_rmses.append(np.sqrt(np.mean(np.array(seq_errs) ** 2)))

    if track_maes:
        metrics["macro_track_mae"] = float(np.mean(track_maes))
        metrics["macro_track_rmse"] = float(np.mean(track_rmses))
    if seq_maes:
        metrics["macro_seq_mae"] = float(np.mean(seq_maes))
        metrics["macro_seq_rmse"] = float(np.mean(seq_rmses))

    if train_median_ttc is not None and len(all_targets) > 0:
        naive_median_errs = np.abs(np.array(all_targets) - train_median_ttc)
        metrics["train_median_baseline_mae"] = float(np.mean(naive_median_errs))

    return metrics, optimizer_step, train_median_ttc


def _checkpoint(
    *,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    ttc_head: EAPSignedTTCHead,
    epoch: int,
    role: str,
    config: EAPJEPATrainerConfig,
    inventory_path: Path,
    split_path: Path,
    garlttc_index: GarlTTCEAPIndex,
    audit_json_path: Path,
    audit_json_sha256: str,
    audit_result: str,
    train_context_ids_sha256: str,
    validation_context_ids_sha256: str,
    train_context_count: int,
    validation_context_count: int,
    history: dict[str, Any] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
    optimizer_step: int | None = None,
    best_validation_macro_track_mae: float | None = None,
    best_validation_joint_loss: float | None = None,
) -> dict[str, Any]:
    if audit_result != "PASS":
        raise ValueError("Cannot build TTC checkpoint without PASS audit")

    if len(audit_json_sha256) != 64:
        raise ValueError("audit_json_sha256 must be a 64-character SHA256")
    from e_jepa_ttc.training.carla_jepa import (
        EVTTC_BASE_INPUT_CHANNELS,
        _artifact_hash,
        _git_commit,
        _source_tree_hash,
    )
    from e_jepa_ttc.training.eap_jepa import EAP_PRETRAINING_DATASET_ID

    train_ids = sorted(garlttc_index.train_sequences)
    val_ids = sorted(garlttc_index.validation_sequences)

    return {
        "external_pretraining": True,
        "pretraining_regime": "eap_ttc",
        "pretraining_dataset_id": EAP_PRETRAINING_DATASET_ID,
        "model_name": "event-tubelet-transformer",
        "in_channels": EVTTC_BASE_INPUT_CHANNELS,
        "event_bins": config.bins,
        "checkpoint_role": role,
        "checkpoint_selected_by": (
            "validation_macro_track_mae" if role == "best" else "final_epoch"
        ),
        "encoder_state_dict": encoder.state_dict(),
        "target_encoder_state_dict": target_encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "ttc_head_state_dict": ttc_head.state_dict(),
        "epoch": epoch,
        "seed": config.seed,
        "uses_ttc_labels": True,
        "uses_ttc_labels_for_loss": True,
        "uses_annotation_index_for_sampling": True,
        "uses_ttc_value_for_sampling": False,
        "uses_labels_for_window_sampling": False,
        "uses_object_bboxes": True,
        "uses_depth_track_derivatives": False,
        "ttc_head_transferable_to_evttc": False,
        "uses_collision_labels": False,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
        "garlttc_data_sha256": garlttc_index.data_sha256,
        "garlttc_annotations_sha256": garlttc_index.annotations_sha256,
        "garlttc_join_keys_sha256": garlttc_index.join_keys_sha256,
        "garlttc_full_joined_row_count": (garlttc_index.source_merged_row_count),
        "garlttc_selected_row_count": (garlttc_index.selected_row_count),
        "train_context_count": train_context_count,
        "validation_context_count": (validation_context_count),
        "garlttc_context_count": (train_context_count + validation_context_count),
        "train_sequences": train_ids,
        "validation_sequences": val_ids,
        "train_context_ids_sha256": train_context_ids_sha256,
        "validation_context_ids_sha256": validation_context_ids_sha256,
        "audit_json_path": audit_json_path.as_posix(),
        "audit_json_sha256": audit_json_sha256,
        "audit_result": audit_result,
        "inventory_artifact_sha256": _artifact_hash(inventory_path),
        "split_artifact_sha256": _artifact_hash(split_path),
        "transferred_components": ["encoder"],
        "discarded_pretraining_heads": ["predictor", "ttc_head"],
        "ttc_label_source": "GarlTTC-dataset/annotations/train.parquet",
        "trainer_config": asdict(config),
        "history": history or {},
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "optimizer_step": optimizer_step,
        "best_validation_macro_track_mae": best_validation_macro_track_mae,
        "best_validation_joint_loss": best_validation_joint_loss,
    }


def limit_dataset_contexts(
    dataset: GarlTTCEAPDataset,
    maximum: int | None,
) -> None:
    if maximum is None:
        return

    if maximum <= 0:
        raise ValueError("Dataset sample limit must be positive")

    if len(dataset.samples) <= maximum:
        return

    indices = np.linspace(
        0,
        len(dataset.samples) - 1,
        maximum,
        dtype=np.int64,
    )

    selected_indices = sorted(set(int(index) for index in indices))

    dataset.samples = [dataset.samples[index] for index in selected_indices]

    dataset.selected_context_ids = [
        dataset.selected_context_ids[index] for index in selected_indices
    ]

    dataset.selected_context_ids_hash = hash_context_ids(dataset.selected_context_ids)


def _restore_ttc_checkpoint(
    checkpoint_path: Path,
    *,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    ttc_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    expected_metadata: Mapping[str, object],
    expected_trainer_config: Mapping[str, object],
) -> tuple[int, int, float, float, list[dict[str, Any]]]:
    """Restore an epoch-boundary TTC checkpoint after provenance validation.

    The checkpoint is written atomically as ``resume.pt``. Resuming from a
    partially completed epoch would make the optimizer trajectory ambiguous,
    so this helper always resumes at ``saved_epoch + 1``.
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume TTC pretraining: checkpoint is missing: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("TTC resume checkpoint must contain a mapping.")

    for key, expected in expected_metadata.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"TTC resume provenance mismatch for {key!r}: "
                f"expected {expected!r}, got {actual!r}."
            )
    saved_config = payload.get("trainer_config")
    if not isinstance(saved_config, Mapping) or dict(saved_config) != dict(expected_trainer_config):
        raise ValueError("TTC resume trainer_config does not match the requested config.")

    state_specs = (
        (encoder, "encoder_state_dict"),
        (target_encoder, "target_encoder_state_dict"),
        (predictor, "predictor_state_dict"),
        (ttc_head, "ttc_head_state_dict"),
    )
    for module, key in state_specs:
        state = payload.get(key)
        if not isinstance(state, Mapping):
            raise ValueError(f"TTC resume checkpoint is missing {key}.")
        module.load_state_dict(state, strict=True)

    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("TTC resume checkpoint is missing optimizer_state_dict.")
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None:
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("TTC resume checkpoint is missing scheduler_state_dict.")
        scheduler.load_state_dict(scheduler_state)

    scaler_state = payload.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        if not isinstance(scaler_state, Mapping):
            raise ValueError("TTC resume scaler_state_dict is malformed.")
        scaler.load_state_dict(scaler_state)

    epoch = int(payload.get("epoch", 0))
    if epoch <= 0:
        raise ValueError("TTC resume checkpoint must represent a completed positive epoch.")
    optimizer_step = int(payload.get("optimizer_step", 0))
    best_ttc = float(payload.get("best_validation_macro_track_mae", float("inf")))
    best_joint = float(payload.get("best_validation_joint_loss", float("inf")))
    if not math.isfinite(best_ttc) and best_ttc != float("inf"):
        raise ValueError("TTC resume best_validation_macro_track_mae is not finite.")
    if not math.isfinite(best_joint) and best_joint != float("inf"):
        raise ValueError("TTC resume best_validation_joint_loss is not finite.")
    history_payload = payload.get("history", {})
    records = history_payload.get("records", []) if isinstance(history_payload, Mapping) else []
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("TTC resume history.records must be a list of mappings.")
    return epoch + 1, optimizer_step, best_ttc, best_joint, list(records)


def pretrain_eap_jepa_ttc(
    *,
    eap_root: str | Path,
    garlttc_root: str | Path,
    inventory_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: EAPJEPATrainerConfig,
    audit_json_path: Path,
    audit_result: str,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    if audit_result != "PASS":
        raise ValueError("TTC pretraining requires a PASS audit")

    from e_jepa_ttc.utils.hashing import sha256_file

    audit_json_sha256 = sha256_file(audit_json_path)
    """Run sequence-disjoint eAP TTC objective."""
    from dataclasses import replace

    from torch.utils.data import DataLoader

    from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
    from e_jepa_ttc.data.garlttc_eap import (
        collate_garlttc,
        load_garlttc_train_index,
        validate_garlttc_train_index,
    )
    from e_jepa_ttc.training.carla_jepa import _artifact_hash, _scheduler, _set_seed
    from e_jepa_ttc.utils.io import read_structured, write_structured

    eap_root = Path(eap_root).resolve()
    garlttc_root = Path(garlttc_root).resolve()
    inventory = Path(inventory_path).resolve()
    split = Path(split_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    assert_no_sealed_benchmark_paths((eap_root, garlttc_root, inventory, split, output))
    _set_seed(config.seed)

    device = resolve_device(device_name)

    split_payload = read_structured(split)
    assignments = split_payload.get("assignments", {})
    train_seqs = sorted(set(str(s) for s in assignments.get("train", [])))
    val_seqs = sorted(set(str(s) for s in assignments.get("validation", [])))

    if set(train_seqs) & set(val_seqs):
        raise ValueError(
            f"Train and validation sequences overlap: {set(train_seqs) & set(val_seqs)}"
        )

    index = load_garlttc_train_index(garlttc_root, train_seqs + val_seqs)
    index = replace(index, train_sequences=train_seqs, validation_sequences=val_seqs)

    validate_garlttc_train_index(
        index,
        expected_rows=(config.expected_garlttc_train_rows),
        allow_version_change=(config.allow_garlttc_version_change),
    )

    models = build_eap_jepa_models(config=config, device=device)
    encoder = models.online_encoder
    target_encoder = models.target_encoder
    predictor = models.predictor

    geometry = infer_tubelet_token_geometry(
        encoder,
        input_height=config.height,
        input_width=config.width,
    )

    train_dataset = GarlTTCEAPDataset(eap_root, index, train_seqs, config, geometry)
    val_dataset = GarlTTCEAPDataset(eap_root, index, val_seqs, config, geometry)

    limit_dataset_contexts(
        train_dataset,
        config.max_train_samples,
    )

    limit_dataset_contexts(
        val_dataset,
        config.max_validation_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_garlttc,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_garlttc,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    ttc_head = EAPSignedTTCHead(embed_dim=int(encoder.output_dim)).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters()) + list(ttc_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_steps = steps_per_epoch * config.epochs
    scheduler = _scheduler(
        optimizer, total_steps=total_steps, warmup_fraction=config.warmup_fraction
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and config.precision == "fp16"
        else None
    )

    last_path = output / "resume.pt"
    start_epoch = 1
    optimizer_step = 0
    best_val_ttc_mae = float("inf")
    best_val_joint_loss = float("inf")
    history: list[dict[str, Any]] = []
    train_median_ttc = None

    train_ctx_hash = train_dataset.selected_context_ids_hash

    val_ctx_hash = val_dataset.selected_context_ids_hash

    if len(train_ctx_hash) != 64:
        raise RuntimeError("Invalid train context SHA256")

    if len(val_ctx_hash) != 64:
        raise RuntimeError("Invalid validation context SHA256")

    if resume:
        (
            start_epoch,
            optimizer_step,
            best_val_ttc_mae,
            best_val_joint_loss,
            history,
        ) = _restore_ttc_checkpoint(
            last_path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ttc_head=ttc_head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_metadata={
                "seed": config.seed,
                "inventory_artifact_sha256": _artifact_hash(inventory),
                "split_artifact_sha256": _artifact_hash(split),
                "audit_json_sha256": audit_json_sha256,
                "garlttc_data_sha256": index.data_sha256,
                "garlttc_annotations_sha256": index.annotations_sha256,
                "garlttc_join_keys_sha256": index.join_keys_sha256,
                "train_context_ids_sha256": train_ctx_hash,
                "validation_context_ids_sha256": val_ctx_hash,
            },
            expected_trainer_config=asdict(config),
        )
        if start_epoch > config.epochs:
            raise ValueError(
                "TTC resume checkpoint is already at or beyond the requested epoch budget."
            )

    def _get_cp(role_name: str, current_epoch: int) -> dict[str, Any]:
        cp = _checkpoint(
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ttc_head=ttc_head,
            epoch=current_epoch,
            role=role_name,
            config=config,
            inventory_path=inventory,
            split_path=split,
            garlttc_index=index,
            audit_json_path=audit_json_path,
            audit_json_sha256=audit_json_sha256,
            audit_result=audit_result,
            train_context_ids_sha256=train_ctx_hash,
            validation_context_ids_sha256=val_ctx_hash,
            train_context_count=len(train_dataset),
            validation_context_count=len(val_dataset),
            history={"records": history},
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            scaler_state_dict=scaler.state_dict() if scaler else None,
            optimizer_step=optimizer_step,
            best_validation_macro_track_mae=best_val_ttc_mae,
            best_validation_joint_loss=best_val_joint_loss,
        )
        return cp

    start_time = time.time()

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics, optimizer_step, train_median_ttc = run_ttc_epoch(
            epoch=epoch,
            dataloader=train_loader,
            target_encoder=target_encoder,
            online_encoder=encoder,
            predictor=predictor,
            ttc_head=ttc_head,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            geometry=geometry,
            device=device,
            config=config,
            start_time=start_time,
            train=True,
            total_optimizer_steps=total_steps,
            optimizer_step=optimizer_step,
            train_median_ttc=train_median_ttc,
        )

        val_start = time.time()
        val_metrics, _, _ = run_ttc_epoch(
            epoch=epoch,
            dataloader=val_loader,
            target_encoder=target_encoder,
            online_encoder=encoder,
            predictor=predictor,
            ttc_head=ttc_head,
            scaler=None,
            optimizer=optimizer,
            scheduler=None,
            geometry=geometry,
            device=device,
            config=config,
            start_time=val_start,
            train=False,
            total_optimizer_steps=total_steps,
            optimizer_step=optimizer_step,
            train_median_ttc=train_median_ttc,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(record)

        val_ttc_mae = val_metrics.get("macro_track_mae", float("inf"))
        val_joint_loss = val_metrics.get("loss_total", float("inf"))
        best_val_joint_loss = min(
            best_val_joint_loss,
            val_joint_loss,
        )

        if val_ttc_mae < best_val_ttc_mae:
            best_val_ttc_mae = val_ttc_mae

            _atomic_torch_save(
                _get_cp("best", epoch),
                output / "eap_jepa_encoder_best.pt",
            )

        _atomic_torch_save(
            _get_cp("last", epoch),
            last_path,
        )

    _atomic_torch_save(_get_cp("final", config.epochs), output / "checkpoint_final.pt")

    train_dataset.close()
    val_dataset.close()

    summary = {
        "output_dir": output.as_posix(),
        "epochs_completed": config.epochs,
        "checkpoint_selected_by": "validation_macro_track_mae",
        "best_validation_macro_track_mae": best_val_ttc_mae,
        "best_validation_joint_loss": best_val_joint_loss,
        "history": history,
    }
    write_structured(output / "summary.json", summary)
    return summary


__all__ = [
    "EAPSignedTTCHead",
    "gather_object_ttc_targets",
    "run_ttc_epoch",
    "pretrain_eap_jepa_ttc",
]
