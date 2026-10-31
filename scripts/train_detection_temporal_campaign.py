"""Train/evaluate one preregistered S3 temporal control from a selected S1 parent.

This is a new producer.  It never mutates or imports state from an active S1 run
other than strict model tensors from the configured selected checkpoint.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from smoke_monodgp_portable import build_detector, require_finite

from e_jepa_ttc.detection.campaign_resources import load_protocol, resource_admission
from e_jepa_ttc.detection.campaign_training import UpdateLRScheduler
from e_jepa_ttc.detection.development_metrics import evaluate_detections
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample
from e_jepa_ttc.detection.temporal_campaign import (
    TemporalCampaignBatch,
    TemporalCampaignSampler,
    build_v5_temporal_campaign,
    load_temporal_admission_ledger,
)
from e_jepa_ttc.detection.temporal_detector import TemporalDetectorAdapter
from e_jepa_ttc.detection.temporal_training import (
    immutable_resource_contract,
    load_parent_model_only,
    load_temporal_checkpoint,
    load_temporal_training_config,
    publish_selected_candidate,
    save_temporal_checkpoint,
    seeded_fusion,
    temporal_eval_forward,
    temporal_optimizer_parameter_groups,
    temporal_train_step,
    validate_temporal_runtime,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _device_batch(batch: TemporalCampaignBatch, device: torch.device) -> TemporalCampaignBatch:
    targets = (
        tuple({key: value.to(device) for key, value in target.items()} for target in batch.targets)
        if batch.targets is not None
        else None
    )
    return replace(
        batch,
        current_images=batch.current_images.to(device),
        calibs=batch.calibs.to(device),
        targets=targets,
        image_sizes=batch.image_sizes.to(device),
        history_images=(
            batch.history_images.to(device) if batch.history_images is not None else None
        ),
        history_ages_us=(
            batch.history_ages_us.to(device) if batch.history_ages_us is not None else None
        ),
        history_mask=(batch.history_mask.to(device) if batch.history_mask is not None else None),
    )


def _evaluate(
    adapter: TemporalDetectorAdapter,
    dataset: Any,
    config: Any,
    protocol: EvaluationProtocol,
    device: torch.device,
) -> dict[str, object]:
    """Label-free temporal forward plus independent raw-label scoring."""
    adapter.eval()
    predictions: dict[str, list[dict[str, object]]] = {}
    raw_truth: dict[str, Any] = {}
    legacy_truth: dict[str, list[dict[str, object]]] = {}
    with EAPRGBDataset(config.baseline.data.root, dataset.current_tokens) as truth_reader:
        for index, token in enumerate(dataset.current_tokens):
            batch = _device_batch(dataset.collate([index]), device)
            outputs = temporal_eval_forward(adapter, batch)
            require_finite(outputs)
            raw = truth_reader[index]
            if raw.sample_token != token:
                raise ValueError("Independent GT reader order differs from temporal population")
            predictions[token] = decode_eap(
                outputs,
                0,
                batch.calibs[0],
                raw.event_from_ego,
                batch.image_sizes[0],
            )[0]
            raw_truth[token] = build_evaluation_ground_truth(
                raw.annotations,
                intrinsic=raw.intrinsic,
                event_from_ego=raw.event_from_ego,
                image_wh=(raw.image.shape[1], raw.image.shape[0]),
                protocol=protocol,
            )
            # Historical metric remains separate and uses only current raw GT.
            legacy_truth[token] = prepare_sample(
                raw,
                width=config.baseline.model.width,
                height=config.baseline.model.height,
                include_velocity=config.baseline.model.predict_velocity,
            ).native_ground_truth
    metrics = evaluate_detections_v2(predictions, raw_truth, protocol=protocol)
    return {
        "metrics_v2": metrics,
        "legacy_metrics_separate": evaluate_detections(predictions, legacy_truth),
        "model_forward_targets": None,
        "ground_truth_producer": "independent_EAPRGBDataset_after_forward",
    }


def run(config_path: Path, output: Path, resume: Path | None) -> None:
    if output.exists() and resume is None:
        raise FileExistsError("Use a new temporal output directory")
    config = load_temporal_training_config(config_path)
    runtime = validate_temporal_runtime(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Temporal runner requires exactly one visible CUDA device")
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(config.baseline.resources.num_cpu_threads)
    random.seed(config.baseline.seed)
    np.random.seed(config.baseline.seed)
    torch.manual_seed(config.baseline.seed)
    torch.cuda.manual_seed_all(config.baseline.seed)
    device = torch.device("cuda:0")
    resource_evidence = resource_admission(config.baseline)
    resource_contract = immutable_resource_contract(config, resource_evidence)
    # Retain bounded TAR indexes across random draws; no images or features are cached.
    open_shards = 128 if config.parent_config is not None else 2
    resource_contract["max_open_shards_per_reader"] = open_shards
    ledger = load_temporal_admission_ledger(
        config.runtime.temporal_media_admission, config.runtime.history_media_receipt
    )
    train_data = build_v5_temporal_campaign(
        dataset_root=config.baseline.data.root,
        manifest_path=config.baseline.data.manifest,
        population="train",
        config=config.temporal,
        admission_ledger=ledger,
        current_labels=True,
        max_label_sequences=config.baseline.data.label_cache_sequences,
        max_open_shards=open_shards,
    )
    monitor_data = build_v5_temporal_campaign(
        dataset_root=config.baseline.data.root,
        manifest_path=config.baseline.data.manifest,
        population="monitor_selection",
        config=config.temporal,
        admission_ledger=ledger,
        current_labels=False,
        max_label_sequences=config.baseline.data.label_cache_sequences,
        max_open_shards=open_shards,
    )
    backend, criterion, effective_model = build_detector(
        Path(config.runtime.portable_root),
        config.baseline.model.group_num,
        predict_velocity=config.baseline.model.predict_velocity,
        velocity_loss_coef=config.baseline.model.velocity_loss_coef,
    )
    load_parent_model_only(backend, config)
    adapter = TemporalDetectorAdapter(
        backend,
        seeded_fusion(config.baseline.seed),
        producer_version=config.temporal.producer_version,
        maximum_history_age_us=config.temporal.maximum_history_age_us,
    ).to(device)
    criterion = criterion.to(device)
    groups, parameter_ledger = temporal_optimizer_parameter_groups(adapter, config.baseline)
    optimizer = torch.optim.AdamW(groups)
    optimization = config.baseline.optimization
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=optimization.updates,
        kind=optimization.scheduler,
        warmup_fraction=optimization.warmup_fraction,
        minimum_ratio=optimization.minimum_lr_ratio,
    )
    sampler = TemporalCampaignSampler(train_data, seed=config.baseline.seed)
    runtime = {
        **runtime,
        "resource_contract": resource_contract,
        "effective_model": effective_model,
        "parameter_ledger": parameter_ledger,
        "train_population_contract": train_data.manifest_sha256,
        "monitor_population_contract": monitor_data.manifest_sha256,
    }
    start = 0
    committed_seconds = 0.0
    selected_lineage: dict[str, object] | None = None
    best_metric: float | None = None
    if resume is not None:
        start, best_metric, committed_seconds, selected_lineage = load_temporal_checkpoint(
            resume,
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=config,
            runtime_provenance=runtime,
        )
    output.mkdir(parents=True, exist_ok=resume is not None)
    _write(output / "runtime_admission.json", runtime)
    _write(output / "resource_admission_observed.json", resource_evidence)
    protocol = load_protocol(config.baseline.evaluation.protocol)
    segment_started = time.monotonic()
    for update in range(start + 1, optimization.updates + 1):
        step_started = time.monotonic()
        elapsed_seconds = committed_seconds + (time.monotonic() - segment_started)
        if elapsed_seconds >= config.baseline.resources.maximum_seconds:
            raise TimeoutError("Temporal campaign exceeded maximum_seconds")
        adapter.train()
        criterion.train()
        _, cpu_batch = sampler.next_batch(optimization.batch_size)
        losses = temporal_train_step(
            adapter,
            criterion,
            _device_batch(cpu_batch, device),
            optimizer,
            scheduler,
            update,
            optimization.gradient_clip_norm,
        )
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "update": update,
                        "batch_size": optimization.batch_size,
                        "image_exposures": update * optimization.batch_size,
                        "observed_at_unix": time.time(),
                        "step_seconds": time.monotonic() - step_started,
                        **losses,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        monitor_due = update in config.baseline.evaluation.updates
        full_due = update in config.baseline.evaluation.full_selection_updates
        if update % optimization.checkpoint_interval == 0 and not monitor_due:
            elapsed_seconds = committed_seconds + (time.monotonic() - segment_started)
            _write(output / f"training/update_{update:06d}.json", losses)
            save_temporal_checkpoint(
                output / "checkpoints/last.pt",
                adapter=adapter,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config,
                runtime_provenance=runtime,
                update=update,
                best_metric=best_metric,
                elapsed_seconds=elapsed_seconds,
                selected_lineage=selected_lineage,
            )
            committed_seconds = elapsed_seconds
            segment_started = time.monotonic()
        elif update % optimization.checkpoint_interval == 0:
            _write(output / f"training/update_{update:06d}.json", losses)
        improved = False
        monitor_complete = False
        full_complete = False
        if monitor_due:
            result = _evaluate(adapter, monitor_data, config, protocol, device)
            _write(output / f"evaluation/monitor_update_{update:06d}.json", result)
            metrics = cast(Mapping[str, object], result["metrics_v2"])
            value = float(cast(float, metrics[config.baseline.evaluation.selection_metric]))
            if best_metric is None or value > best_metric:
                best_metric = value
                improved = True
            monitor_complete = True
        if full_due:
            full = build_v5_temporal_campaign(
                dataset_root=config.baseline.data.root,
                manifest_path=config.baseline.data.manifest,
                population="full_selection",
                config=config.temporal,
                admission_ledger=ledger,
                current_labels=False,
                full_selection_path=config.baseline.data.full_selection_tokens,
                max_label_sequences=config.baseline.data.label_cache_sequences,
                max_open_shards=open_shards,
            )
            _write(
                output / f"evaluation/full_selection_update_{update:06d}.json",
                _evaluate(adapter, full, config, protocol, device),
            )
            full_complete = True
        if monitor_due:
            elapsed_seconds = committed_seconds + (time.monotonic() - segment_started)
            if improved:
                candidate = output / f"checkpoints/.selected_update_{update:06d}.candidate.pt"
                digest = save_temporal_checkpoint(
                    candidate,
                    adapter=adapter,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    config=config,
                    runtime_provenance=runtime,
                    update=update,
                    best_metric=best_metric,
                    monitor_complete=monitor_complete,
                    full_selection_complete=full_complete,
                    elapsed_seconds=elapsed_seconds,
                )
                destination = publish_selected_candidate(
                    candidate,
                    checkpoints_dir=output / "checkpoints",
                    update=update,
                    expected_sha256=digest,
                )
                selected_lineage = {
                    "path": str(destination.resolve()),
                    "sha256": digest,
                    "update": update,
                    "best_metric": best_metric,
                }
            save_temporal_checkpoint(
                output / "checkpoints/last.pt",
                adapter=adapter,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config,
                runtime_provenance=runtime,
                update=update,
                best_metric=best_metric,
                monitor_complete=monitor_complete,
                full_selection_complete=full_complete,
                elapsed_seconds=elapsed_seconds,
                selected_lineage=selected_lineage,
            )
            committed_seconds = elapsed_seconds
            segment_started = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
