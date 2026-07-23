"""Command line interface for the E-JEPA-TTC MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.baselines.causal_geometry import run_causal_geometry_baseline
from e_jepa_ttc.baselines.event_rate import run_event_rate_baseline
from e_jepa_ttc.baselines.geometric import run_geometric_baseline
from e_jepa_ttc.baselines.roi_events import run_roi_event_baseline
from e_jepa_ttc.baselines.trivial import run_trivial_baseline
from e_jepa_ttc.data.evttc import scan_evttc_root, validate_manifest, write_manifest
from e_jepa_ttc.data.index import build_temporal_index, write_index
from e_jepa_ttc.data.ml_cache import build_voxel_cache, remap_cache_splits
from e_jepa_ttc.data.official_protocol import evaluate_official_evttc_coverage
from e_jepa_ttc.data.split import write_splits
from e_jepa_ttc.data.synthetic import generate_synthetic_sequence, write_synthetic_hdf5
from e_jepa_ttc.models import MODEL_NAMES
from e_jepa_ttc.representations.corruptions import CORRUPTION_KINDS
from e_jepa_ttc.utils.io import write_structured


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _cmd_synthetic_generate(args: argparse.Namespace) -> int:
    sequence = generate_synthetic_sequence(
        width=args.width,
        height=args.height,
        windows=args.windows,
        context_ms=args.context_ms,
        stride_ms=args.stride_ms,
        horizons_ms=tuple(args.horizons_ms),
        seed=args.seed,
    )
    write_synthetic_hdf5(args.output, sequence)
    _print_json(
        {
            "output": str(args.output),
            "events": sequence.events.num_events,
            "windows": int(sequence.ttc_s.shape[0]),
            "ttc_min_s": float(sequence.ttc_s.min()),
            "ttc_max_s": float(sequence.ttc_s.max()),
        }
    )
    return 0


def _cmd_data_scan(args: argparse.Namespace) -> int:
    sequences = scan_evttc_root(args.root)
    write_manifest(args.output, sequences)
    _print_json(
        {
            "output": str(args.output),
            "sequence_count": len(sequences),
            "sequences": [sequence.sequence_id for sequence in sequences],
        }
    )
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    report = validate_manifest(args.manifest)
    _print_json(report)
    return 0


def _cmd_data_index(args: argparse.Namespace) -> int:
    entries = build_temporal_index(
        manifest_path=args.manifest,
        context_ms=args.context_ms,
        stride_ms=args.stride_ms,
        horizons_ms=tuple(args.horizons_ms),
        clip_ttc_seconds=(args.clip_ttc_min, args.clip_ttc_max),
    )
    write_index(args.output, entries)
    _print_json({"output": str(args.output), "window_count": len(entries)})
    return 0


def _cmd_data_official_coverage(args: argparse.Namespace) -> int:
    sequences = scan_evttc_root(args.root)
    report = evaluate_official_evttc_coverage(
        sequences,
        include_slider=args.include_slider,
    )
    if args.output is not None:
        write_structured(args.output, report)
    _print_json(report)
    return 0


def _cmd_data_verify_files(args: argparse.Namespace) -> int:
    from e_jepa_ttc.data.integrity import verify_file_manifest

    payload = verify_file_manifest(
        args.root,
        args.file_manifest,
        sha256=args.sha256,
    )
    if args.output is not None:
        write_structured(args.output, payload)
    _print_json(payload)
    return 0 if payload["valid"] else 2


def _cmd_split_create(args: argparse.Namespace) -> int:
    payload = write_splits(manifest_path=args.manifest, output_path=args.output, seed=args.seed)
    _print_json(payload)
    return 0


def _cmd_baseline_trivial(args: argparse.Namespace) -> int:
    payload = run_trivial_baseline(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
    )
    _print_json(payload)
    return 0


def _cmd_baseline_geometric(args: argparse.Namespace) -> int:
    payload = run_geometric_baseline(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
    )
    _print_json(payload)
    return 0


def _cmd_baseline_event_rate(args: argparse.Namespace) -> int:
    payload = run_event_rate_baseline(
        manifest_path=args.manifest,
        split_path=args.split,
        index_path=args.index,
        output_path=args.output,
    )
    _print_json(payload)
    return 0


def _cmd_baseline_causal_geometry(args: argparse.Namespace) -> int:
    payload = run_causal_geometry_baseline(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
        derivative_window=args.derivative_window,
    )
    _print_json(payload)
    return 0


def _cmd_baseline_roi_events(args: argparse.Namespace) -> int:
    payload = run_roi_event_baseline(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
        context_ms=args.context_ms,
        ridge_alpha=args.ridge_alpha,
        max_ttc_seconds=args.max_ttc_seconds,
        evaluation_splits=(
            tuple(args.evaluation_splits) if args.evaluation_splits is not None else None
        ),
    )
    _print_json(payload)
    return 0


def _cmd_baseline_eap_geometry(args: argparse.Namespace) -> int:
    from e_jepa_ttc.baselines.eap_geometry import evaluate_eap_geometry_baselines

    payload = evaluate_eap_geometry_baselines(
        cache_manifest_path=args.cache_manifest,
        splits=tuple(args.splits),
        output_path=args.output,
    )
    _print_json(payload)
    return 0


def _cmd_cache_voxel(args: argparse.Namespace) -> int:
    payload = build_voxel_cache(
        manifest_path=args.manifest,
        split_path=args.split,
        index_path=args.index,
        output_path=args.output,
        width=args.width,
        height=args.height,
        bins=args.bins,
        normalize=args.normalize,
        metadata_channels=args.metadata_channels,
        navigation_channels=args.navigation_channels,
        limit=args.limit,
    )
    _print_json(payload)
    return 0


def _cmd_cache_remap_splits(args: argparse.Namespace) -> int:
    payload = remap_cache_splits(
        cache_path=args.cache,
        split_path=args.split,
        output_path=args.output,
    )
    _print_json(payload)
    return 0


def _parse_sequence_splits(assignments: list[str]) -> dict[str, str]:
    sequence_splits: dict[str, str] = {}
    for assignment in assignments:
        sequence_id, separator, split_name = assignment.partition("=")
        if not separator or not sequence_id or not split_name:
            msg = f"Invalid sequence split {assignment!r}; expected SEQUENCE_ID=SPLIT."
            raise ValueError(msg)
        if sequence_id in sequence_splits:
            msg = f"Sequence {sequence_id!r} was assigned more than once."
            raise ValueError(msg)
        sequence_splits[sequence_id] = split_name
    return sequence_splits


def _cmd_cache_eap_object(args: argparse.Namespace) -> int:
    from e_jepa_ttc.data.eap_cache import (
        EAPObjectCacheConfig,
        materialize_eap_object_cache,
    )

    config = EAPObjectCacheConfig(
        history_frames=args.history_frames,
        prediction_horizons_ms=tuple(args.prediction_horizons_ms),
        event_window_ms=args.event_window_ms,
        adaptive_event_count=args.adaptive_event_count,
        minimum_adaptive_window_ms=args.minimum_adaptive_window_ms,
        maximum_target_slop_ms=args.maximum_target_slop_ms,
        maximum_history_gap_ms=args.maximum_history_gap_ms,
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        roi_expansion=args.roi_expansion,
        event_bins=args.event_bins,
        normalize_events=args.normalize_events,
        derivative_radius=args.derivative_radius,
        maximum_derivative_gap_s=args.maximum_derivative_gap_s,
        shard_size=args.shard_size,
        action_dim=args.action_dim,
        corruption_kind=args.corruption_kind,
        corruption_severity=args.corruption_severity,
        corruption_seed=args.corruption_seed,
        include_rgb=args.include_rgb,
        rgb_width=args.rgb_width,
        rgb_height=args.rgb_height,
    )
    payload = materialize_eap_object_cache(
        eap_root=args.eap_root,
        output_dir=args.output_dir,
        sequence_splits=_parse_sequence_splits(args.sequence_split),
        config=config,
        max_windows_per_sequence=args.max_windows_per_sequence,
        workers=args.workers,
    )
    _print_json(payload)
    return 0


def _cmd_cache_evttc_object(args: argparse.Namespace) -> int:
    from e_jepa_ttc.data.evttc_object_cache import (
        EvTTCObjectCacheConfig,
        materialize_evttc_object_cache,
    )

    payload = materialize_evttc_object_cache(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sequence_splits=_parse_sequence_splits(args.sequence_split),
        config=EvTTCObjectCacheConfig(
            history_frames=args.history_frames,
            prediction_horizons_ms=tuple(args.prediction_horizons_ms),
            event_window_ms=args.event_window_ms,
            maximum_target_slop_ms=args.maximum_target_slop_ms,
            maximum_history_gap_ms=args.maximum_history_gap_ms,
            width=args.width,
            height=args.height,
            event_bins=args.event_bins,
            normalize_events=args.normalize_events,
            shard_size=args.shard_size,
        ),
        max_windows_per_sequence=args.max_windows_per_sequence,
    )
    _print_json(payload)
    return 0


def _cmd_train_tiny_cnn(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.supervised import train_tiny_cnn

    payload = train_tiny_cnn(
        cache_path=args.cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        pretrained_encoder_path=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        train_fraction=args.train_fraction,
        subset_manifest_path=args.subset_manifest_path,
        model_name=args.model,
        navigation_mode=args.navigation_mode,
        evaluation_splits=tuple(args.evaluation_splits),
        train_splits=tuple(args.train_splits),
        validation_splits=tuple(args.validation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_train_evaluate(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint

    payload = evaluate_supervised_checkpoint(
        cache_path=args.cache,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        evaluation_splits=tuple(args.evaluation_splits),
        model_name=args.model,
    )
    _print_json(payload)
    return 0


def _cmd_train_latent_prober(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.prober import train_latent_ttc_prober

    payload = train_latent_ttc_prober(
        cache_path=args.cache,
        encoder_checkpoint_path=args.encoder_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        model_name=args.model,
        token_summary=args.token_summary,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        physics_prior=args.physics_prior,
        ridge_alpha=args.ridge_alpha,
        train_splits=tuple(args.train_splits),
        validation_splits=tuple(args.validation_splits),
        evaluation_splits=tuple(args.evaluation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_train_roi_latent_prober(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.prober import train_roi_latent_ttc_prober

    payload = train_roi_latent_ttc_prober(
        manifest_path=args.manifest,
        split_path=args.split,
        cache_path=args.cache,
        encoder_checkpoint_path=args.encoder_checkpoint,
        output_dir=args.output_dir,
        context_ms=args.context_ms,
        max_cache_slop_ms=args.max_cache_slop_ms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        model_name=args.model,
        token_summary=args.token_summary,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        physics_prior=args.physics_prior,
        ridge_alpha=args.ridge_alpha,
        train_splits=tuple(args.train_splits),
        validation_splits=tuple(args.validation_splits),
        evaluation_splits=tuple(args.evaluation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_evaluate_roi_latent_prober(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.prober import evaluate_roi_latent_ttc_prober_checkpoint

    payload = evaluate_roi_latent_ttc_prober_checkpoint(
        manifest_path=args.manifest,
        split_path=args.split,
        cache_path=args.cache,
        prober_checkpoint_path=args.checkpoint,
        output_path=args.output,
        context_ms=args.context_ms,
        max_cache_slop_ms=args.max_cache_slop_ms,
        batch_size=args.batch_size,
        device_name=args.device,
        model_name=args.model,
        evaluation_splits=tuple(args.evaluation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_train_roi_rollout_prober(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.prober import train_roi_rollout_ttc_prober

    payload = train_roi_rollout_ttc_prober(
        manifest_path=args.manifest,
        split_path=args.split,
        cache_path=args.cache,
        jepa_checkpoint_path=args.jepa_checkpoint,
        output_dir=args.output_dir,
        context_ms=args.context_ms,
        max_cache_slop_ms=args.max_cache_slop_ms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        model_name=args.model,
        rollout_token_summary=args.rollout_token_summary,
        rollout_include_context=not args.no_rollout_context,
        rollout_feature_mode=args.rollout_feature_mode,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        physics_prior=args.physics_prior,
        ridge_alpha=args.ridge_alpha,
        train_splits=tuple(args.train_splits),
        validation_splits=tuple(args.validation_splits),
        evaluation_splits=tuple(args.evaluation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_evaluate_roi_rollout_prober(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.prober import evaluate_roi_rollout_ttc_prober_checkpoint

    payload = evaluate_roi_rollout_ttc_prober_checkpoint(
        manifest_path=args.manifest,
        split_path=args.split,
        cache_path=args.cache,
        prober_checkpoint_path=args.checkpoint,
        output_path=args.output,
        context_ms=args.context_ms,
        max_cache_slop_ms=args.max_cache_slop_ms,
        batch_size=args.batch_size,
        device_name=args.device,
        model_name=args.model,
        evaluation_splits=tuple(args.evaluation_splits),
    )
    _print_json(payload)
    return 0


def _cmd_pretrain_jepa(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.jepa import pretrain_jepa

    payload = pretrain_jepa(
        cache_path=args.cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        pretrain_splits=tuple(args.pretrain_splits),
        validation_splits=tuple(args.validation_splits),
        temporal_horizons_ms=tuple(args.temporal_horizons_ms),
        max_target_slop_ms=args.max_target_slop_ms,
        mask_ratio=args.mask_ratio,
        block_count=args.block_count,
        mask_mode=args.mask_mode,
        ema_momentum=args.ema_momentum,
        regularizer=args.regularizer,
        variance_weight=args.variance_weight,
        min_std=args.min_std,
        visreg_center_weight=args.visreg_center_weight,
        visreg_sketch_weight=args.visreg_sketch_weight,
        visreg_projection_count=args.visreg_projection_count,
        temporal_straightening_weight=args.temporal_straightening_weight,
        dense_tokens=args.dense_tokens,
        motion_conditioning=args.motion_conditioning,
        deep_supervision_layers=tuple(args.deep_supervision_layers),
        dense_predictor=args.dense_predictor,
        context_token_weight=args.context_token_weight,
        model_name=args.model,
        navigation_mode=args.navigation_mode,
    )
    _print_json(payload)
    return 0


def _cmd_pretrain_object_jepa(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.object_jepa import pretrain_object_event_jepa

    payload = pretrain_object_event_jepa(
        cache_manifest_path=args.cache_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        embedding_dim=args.embedding_dim,
        feature_dim=args.feature_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        use_ego_actions=args.use_ego_actions,
        use_recurrence=args.use_recurrence,
        use_geometry=args.use_geometry,
    )
    _print_json(payload)
    return 0


def _cmd_train_object_ttc(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.object_jepa import fine_tune_object_ttc

    payload = fine_tune_object_ttc(
        cache_manifest_path=args.cache_manifest,
        output_dir=args.output_dir,
        pretrained_checkpoint_path=args.pretrained_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_fraction=args.label_fraction,
        seed=args.seed,
        device_name=args.device,
        use_ego_actions=args.use_ego_actions,
    )
    _print_json(payload)
    return 0


def _cmd_distill_object_jepa_dinov3(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.multimodal import distill_object_event_jepa_from_dinov3

    payload = distill_object_event_jepa_from_dinov3(
        cache_manifest_path=args.cache_manifest,
        event_checkpoint_path=args.event_checkpoint,
        output_dir=args.output_dir,
        teacher_model_name=args.teacher_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        distillation_weight=args.distillation_weight,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        seed=args.seed,
        device_name=args.device,
    )
    _print_json(payload)
    return 0


def _cmd_evaluate_object_ttc(args: argparse.Namespace) -> int:
    from e_jepa_ttc.training.object_jepa import evaluate_object_ttc_checkpoint

    payload = evaluate_object_ttc_checkpoint(
        cache_manifest_path=args.cache_manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        splits=tuple(args.splits),
        calibration_summary_path=args.calibration_summary,
        batch_size=args.batch_size,
        device_name=args.device,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        use_ego_actions=args.use_ego_actions,
    )
    _print_json(payload)
    return 0


def _load_object_runtime_example(
    cache_manifest: Path,
    checkpoint_path: Path,
    *,
    split: str,
    sample_index: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
    from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ObjectCentricEventJEPA(ObjectJEPAConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = EAPObjectCacheDataset(cache_manifest, splits=(split,))
    try:
        sample = dataset[sample_index]
        tensor_keys = (
            "context_events",
            "context_boxes",
            "context_object_mask",
            "context_sampling_boxes",
            "context_ego_actions",
            "context_ego_action_mask",
        )
        inputs = {
            name: sample[name][None]
            for name in tensor_keys
            if isinstance(sample[name], torch.Tensor)
        }
    finally:
        dataset.close()
    return model, inputs


def _cmd_runtime_export_object_ttc(args: argparse.Namespace) -> int:
    from e_jepa_ttc.runtime.export import export_object_ttc_onnx

    model, inputs = _load_object_runtime_example(
        args.cache_manifest,
        args.checkpoint,
        split=args.split,
        sample_index=args.sample_index,
    )
    payload = export_object_ttc_onnx(
        model,
        inputs,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
    )
    _print_json(payload)
    return 0


def _cmd_runtime_benchmark_object_ttc(args: argparse.Namespace) -> int:
    from e_jepa_ttc.runtime.benchmark import benchmark_object_ttc_model

    model, inputs = _load_object_runtime_example(
        args.cache_manifest,
        args.checkpoint,
        split=args.split,
        sample_index=args.sample_index,
    )
    payload = benchmark_object_ttc_model(
        model,
        inputs,
        device=args.device,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
    )
    payload.update(
        {
            "checkpoint": args.checkpoint.as_posix(),
            "cache_manifest": args.cache_manifest.as_posix(),
            "split": args.split,
            "sample_index": args.sample_index,
            "scope": "model_only_cached_preprocessing",
        }
    )
    write_structured(args.output, payload)
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser."""

    parser = argparse.ArgumentParser(prog="e-jepa-ttc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Synthetic event data commands.")
    synthetic_sub = synthetic.add_subparsers(dest="synthetic_command", required=True)
    synthetic_generate = synthetic_sub.add_parser("generate", help="Generate synthetic HDF5 data.")
    synthetic_generate.add_argument("--output", type=Path, required=True)
    synthetic_generate.add_argument("--width", type=int, default=64)
    synthetic_generate.add_argument("--height", type=int, default=48)
    synthetic_generate.add_argument("--windows", type=int, default=128)
    synthetic_generate.add_argument("--context-ms", type=int, default=100)
    synthetic_generate.add_argument("--stride-ms", type=int, default=20)
    synthetic_generate.add_argument("--horizons-ms", type=int, nargs="+", default=[50, 100])
    synthetic_generate.add_argument("--seed", type=int, default=0)
    synthetic_generate.set_defaults(func=_cmd_synthetic_generate)

    data = subparsers.add_parser("data", help="Dataset manifest and indexing commands.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_scan = data_sub.add_parser("scan", help="Scan an EvTTC local root.")
    data_scan.add_argument("--root", type=Path, required=True)
    data_scan.add_argument("--output", type=Path, required=True)
    data_scan.set_defaults(func=_cmd_data_scan)
    data_validate = data_sub.add_parser("validate", help="Validate a dataset manifest.")
    data_validate.add_argument("--manifest", type=Path, required=True)
    data_validate.set_defaults(func=_cmd_data_validate)
    data_index = data_sub.add_parser("index", help="Create a temporal window index.")
    data_index.add_argument("--manifest", type=Path, required=True)
    data_index.add_argument("--output", type=Path, required=True)
    data_index.add_argument("--context-ms", type=int, default=100)
    data_index.add_argument("--stride-ms", type=int, default=20)
    data_index.add_argument("--horizons-ms", type=int, nargs="+", default=[25, 50, 100, 250, 500])
    data_index.add_argument("--clip-ttc-min", type=float, default=0.1)
    data_index.add_argument("--clip-ttc-max", type=float, default=12.0)
    data_index.set_defaults(func=_cmd_data_index)
    data_official = data_sub.add_parser(
        "official-coverage",
        help="Check local assets against the official EvTTC bbox/ROI Table V sequence list.",
    )
    data_official.add_argument("--root", type=Path, required=True)
    data_official.add_argument("--output", type=Path)
    data_official.add_argument(
        "--include-slider",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Slider-750 and Slider-1000 rows required for complete Table V.",
    )
    data_official.set_defaults(func=_cmd_data_official_coverage)
    data_verify_files = data_sub.add_parser(
        "verify-files",
        help="Verify selective dataset files against pinned remote sizes and optional SHA-256.",
    )
    data_verify_files.add_argument("--root", type=Path, required=True)
    data_verify_files.add_argument("--file-manifest", type=Path, required=True)
    data_verify_files.add_argument("--sha256", action="store_true")
    data_verify_files.add_argument("--output", type=Path)
    data_verify_files.set_defaults(func=_cmd_data_verify_files)

    split = subparsers.add_parser("split", help="Split generation commands.")
    split_sub = split.add_subparsers(dest="split_command", required=True)
    split_create = split_sub.add_parser("create", help="Create sequence-level splits.")
    split_create.add_argument("--manifest", type=Path, required=True)
    split_create.add_argument("--output", type=Path, required=True)
    split_create.add_argument("--seed", type=int, default=42)
    split_create.set_defaults(func=_cmd_split_create)

    baseline = subparsers.add_parser("baseline", help="Baseline evaluation commands.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_trivial = baseline_sub.add_parser(
        "trivial", help="Evaluate mean/median TTC baselines."
    )
    baseline_trivial.add_argument("--manifest", type=Path, required=True)
    baseline_trivial.add_argument("--split", type=Path, required=True)
    baseline_trivial.add_argument("--output", type=Path)
    baseline_trivial.set_defaults(func=_cmd_baseline_trivial)
    baseline_geometric = baseline_sub.add_parser(
        "geometric",
        help="Evaluate bbox apparent-expansion TTC baseline.",
    )
    baseline_geometric.add_argument("--manifest", type=Path, required=True)
    baseline_geometric.add_argument("--split", type=Path, required=True)
    baseline_geometric.add_argument("--output", type=Path)
    baseline_geometric.set_defaults(func=_cmd_baseline_geometric)
    baseline_event_rate = baseline_sub.add_parser(
        "event-rate",
        help="Evaluate event-rate ridge baseline.",
    )
    baseline_event_rate.add_argument("--manifest", type=Path, required=True)
    baseline_event_rate.add_argument("--split", type=Path, required=True)
    baseline_event_rate.add_argument("--index", type=Path, required=True)
    baseline_event_rate.add_argument("--output", type=Path)
    baseline_event_rate.set_defaults(func=_cmd_baseline_event_rate)
    baseline_causal_geometry = baseline_sub.add_parser(
        "causal-geometry",
        help="Evaluate causal detection-assisted bbox expansion baseline.",
    )
    baseline_causal_geometry.add_argument("--manifest", type=Path, required=True)
    baseline_causal_geometry.add_argument("--split", type=Path, required=True)
    baseline_causal_geometry.add_argument("--output", type=Path)
    baseline_causal_geometry.add_argument("--derivative-window", type=int, default=15)
    baseline_causal_geometry.set_defaults(func=_cmd_baseline_causal_geometry)
    baseline_roi_events = baseline_sub.add_parser(
        "roi-events",
        help="Evaluate causal detection-assisted bbox/ROI event-feature baseline.",
    )
    baseline_roi_events.add_argument("--manifest", type=Path, required=True)
    baseline_roi_events.add_argument("--split", type=Path, required=True)
    baseline_roi_events.add_argument("--output", type=Path)
    baseline_roi_events.add_argument("--context-ms", type=int, default=100)
    baseline_roi_events.add_argument("--ridge-alpha", type=float, default=1.0)
    baseline_roi_events.add_argument("--max-ttc-seconds", type=float, default=60.0)
    baseline_roi_events.add_argument(
        "--evaluation-splits",
        nargs="+",
        help="Only evaluate these splits, while always fitting on train.",
    )
    baseline_roi_events.set_defaults(func=_cmd_baseline_roi_events)
    baseline_eap_geometry = baseline_sub.add_parser(
        "eap-geometry",
        help="Evaluate causal height/depth object-geometry baselines on an eAP object cache.",
    )
    baseline_eap_geometry.add_argument("--cache-manifest", type=Path, required=True)
    baseline_eap_geometry.add_argument("--splits", nargs="+", default=["validation"])
    baseline_eap_geometry.add_argument("--output", type=Path)
    baseline_eap_geometry.set_defaults(func=_cmd_baseline_eap_geometry)

    cache = subparsers.add_parser("cache", help="Feature cache commands.")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_voxel = cache_sub.add_parser("voxel", help="Build compact voxel-grid tensor cache.")
    cache_voxel.add_argument("--manifest", type=Path, required=True)
    cache_voxel.add_argument("--split", type=Path, required=True)
    cache_voxel.add_argument("--index", type=Path, required=True)
    cache_voxel.add_argument("--output", type=Path, required=True)
    cache_voxel.add_argument("--width", type=int, default=160)
    cache_voxel.add_argument("--height", type=int, default=90)
    cache_voxel.add_argument("--bins", type=int, default=5)
    cache_voxel.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Preserve raw voxel counts instead of robust per-window normalization.",
    )
    cache_voxel.add_argument(
        "--metadata-channels",
        action="store_true",
        help="Append log event-count and log event-rate channels to each voxel grid.",
    )
    cache_voxel.add_argument(
        "--navigation-channels",
        action="store_true",
        help="Append causal integrated-navigation motion channels when available.",
    )
    cache_voxel.add_argument("--limit", type=int)
    cache_voxel.set_defaults(func=_cmd_cache_voxel)
    cache_remap = cache_sub.add_parser(
        "remap-splits",
        help="Copy a tensor cache while assigning split labels from a split YAML.",
    )
    cache_remap.add_argument("--cache", type=Path, required=True)
    cache_remap.add_argument("--split", type=Path, required=True)
    cache_remap.add_argument("--output", type=Path, required=True)
    cache_remap.set_defaults(func=_cmd_cache_remap_splits)
    cache_eap = cache_sub.add_parser(
        "eap-object",
        help="Materialize causal object-ROI event histories and future targets from public eAP.",
    )
    cache_eap.add_argument("--eap-root", type=Path, required=True)
    cache_eap.add_argument("--output-dir", type=Path, required=True)
    cache_eap.add_argument(
        "--sequence-split",
        action="append",
        required=True,
        metavar="SEQUENCE_ID=SPLIT",
        help=(
            "Assign a complete eAP train sequence to train, validation, calibration or test; "
            "repeat once per sequence."
        ),
    )
    cache_eap.add_argument("--history-frames", type=int, default=3)
    cache_eap.add_argument(
        "--prediction-horizons-ms",
        type=int,
        nargs="+",
        default=[100, 250, 500],
    )
    cache_eap.add_argument("--event-window-ms", type=int, default=100)
    cache_eap.add_argument(
        "--adaptive-event-count",
        type=int,
        help=(
            "Use an ROI-local density-adaptive trailing window with approximately this "
            "many events instead of an always-fixed temporal support."
        ),
    )
    cache_eap.add_argument("--minimum-adaptive-window-ms", type=int, default=10)
    cache_eap.add_argument("--maximum-target-slop-ms", type=int, default=25)
    cache_eap.add_argument("--maximum-history-gap-ms", type=int, default=125)
    cache_eap.add_argument("--roi-width", type=int, default=64)
    cache_eap.add_argument("--roi-height", type=int, default=64)
    cache_eap.add_argument("--roi-expansion", type=float, default=1.25)
    cache_eap.add_argument("--event-bins", type=int, default=5)
    cache_eap.add_argument("--derivative-radius", type=int, default=2)
    cache_eap.add_argument("--maximum-derivative-gap-s", type=float, default=0.25)
    cache_eap.add_argument("--shard-size", type=int, default=512)
    cache_eap.add_argument("--action-dim", type=int, default=8)
    cache_eap.add_argument("--corruption-kind", choices=CORRUPTION_KINDS, default="none")
    cache_eap.add_argument("--corruption-severity", type=float, default=0.0)
    cache_eap.add_argument("--corruption-seed", type=int, default=0)
    cache_eap.add_argument(
        "--include-rgb",
        action="store_true",
        help="Also materialize synchronized object RGB crops for fusion/distillation ablations.",
    )
    cache_eap.add_argument("--rgb-width", type=int, default=112)
    cache_eap.add_argument("--rgb-height", type=int, default=112)
    cache_eap.add_argument("--max-windows-per-sequence", type=int)
    cache_eap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Materialize independent eAP sequences in parallel processes.",
    )
    cache_eap.add_argument(
        "--no-normalize-events",
        dest="normalize_events",
        action="store_false",
    )
    cache_eap.set_defaults(normalize_events=True, func=_cmd_cache_eap_object)
    cache_evttc = cache_sub.add_parser(
        "evttc-object",
        help="Materialize EvTTC object histories with causal integrated-navigation actions.",
    )
    cache_evttc.add_argument("--manifest", type=Path, required=True)
    cache_evttc.add_argument("--output-dir", type=Path, required=True)
    cache_evttc.add_argument(
        "--sequence-split",
        action="append",
        required=True,
        metavar="SEQUENCE_ID=SPLIT",
    )
    cache_evttc.add_argument("--history-frames", type=int, default=3)
    cache_evttc.add_argument(
        "--prediction-horizons-ms",
        type=int,
        nargs="+",
        default=[100, 250, 500],
    )
    cache_evttc.add_argument("--event-window-ms", type=int, default=100)
    cache_evttc.add_argument("--maximum-target-slop-ms", type=int, default=30)
    cache_evttc.add_argument("--maximum-history-gap-ms", type=int, default=80)
    cache_evttc.add_argument("--width", type=int, default=160)
    cache_evttc.add_argument("--height", type=int, default=90)
    cache_evttc.add_argument("--event-bins", type=int, default=5)
    cache_evttc.add_argument("--shard-size", type=int, default=256)
    cache_evttc.add_argument("--max-windows-per-sequence", type=int)
    cache_evttc.add_argument(
        "--no-normalize-events",
        dest="normalize_events",
        action="store_false",
    )
    cache_evttc.set_defaults(normalize_events=True, func=_cmd_cache_evttc_object)

    train = subparsers.add_parser("train", help="Training commands.")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_tiny = train_sub.add_parser("tiny-cnn", help="Train supervised TinyCNN on voxel cache.")
    train_tiny.add_argument("--cache", type=Path, required=True)
    train_tiny.add_argument("--output-dir", type=Path, required=True)
    train_tiny.add_argument("--epochs", type=int, default=80)
    train_tiny.add_argument("--batch-size", type=int, default=64)
    train_tiny.add_argument("--learning-rate", type=float, default=3e-4)
    train_tiny.add_argument("--seed", type=int, default=42)
    train_tiny.add_argument("--device", type=str, default="auto")
    train_tiny.add_argument("--model", choices=MODEL_NAMES, default="tiny-cnn")
    train_tiny.add_argument("--navigation-mode", choices=["enabled", "disabled"], default="enabled")
    train_tiny.add_argument("--pretrained-encoder", type=Path)
    train_tiny.add_argument("--train-fraction", type=float, default=1.0)
    train_tiny.add_argument("--subset-manifest-path", type=Path)
    train_tiny.add_argument("--train-splits", nargs="+", default=["train"])
    train_tiny.add_argument("--validation-splits", nargs="+", default=["validation"])
    train_tiny.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to evaluate after validation-selected training.",
    )
    train_tiny.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the TTC head after loading or initializing the encoder.",
    )
    train_tiny.set_defaults(func=_cmd_train_tiny_cnn)
    train_eval = train_sub.add_parser(
        "evaluate",
        help="Evaluate a saved supervised checkpoint without retraining.",
    )
    train_eval.add_argument("--cache", type=Path, required=True)
    train_eval.add_argument("--checkpoint", type=Path, required=True)
    train_eval.add_argument("--output", type=Path)
    train_eval.add_argument("--batch-size", type=int, default=64)
    train_eval.add_argument("--device", type=str, default="auto")
    train_eval.add_argument("--model", choices=MODEL_NAMES)
    train_eval.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["test"],
        help="Splits to evaluate from the saved checkpoint.",
    )
    train_eval.set_defaults(func=_cmd_train_evaluate)
    train_prober = train_sub.add_parser(
        "latent-prober",
        help="Train a frozen JEPA-latent residual TTC prober.",
    )
    train_prober.add_argument("--cache", type=Path, required=True)
    train_prober.add_argument("--encoder-checkpoint", type=Path, required=True)
    train_prober.add_argument("--output-dir", type=Path, required=True)
    train_prober.add_argument("--epochs", type=int, default=80)
    train_prober.add_argument("--batch-size", type=int, default=64)
    train_prober.add_argument("--learning-rate", type=float, default=3e-4)
    train_prober.add_argument("--seed", type=int, default=42)
    train_prober.add_argument("--device", type=str, default="auto")
    train_prober.add_argument("--model", choices=MODEL_NAMES)
    train_prober.add_argument("--token-summary", choices=["mean", "mean-std"], default="mean-std")
    train_prober.add_argument("--hidden-dim", type=int, default=256)
    train_prober.add_argument("--dropout", type=float, default=0.05)
    train_prober.add_argument("--physics-prior", choices=["none", "ridge"], default="ridge")
    train_prober.add_argument("--ridge-alpha", type=float, default=1.0)
    train_prober.add_argument("--train-splits", nargs="+", default=["train"])
    train_prober.add_argument("--validation-splits", nargs="+", default=["validation"])
    train_prober.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to evaluate after validation-selected prober training.",
    )
    train_prober.set_defaults(func=_cmd_train_latent_prober)
    train_roi_prober = train_sub.add_parser(
        "roi-latent-prober",
        help="Train a detection-assisted frozen JEPA-latent bbox/ROI TTC prober.",
    )
    train_roi_prober.add_argument("--manifest", type=Path, required=True)
    train_roi_prober.add_argument("--split", type=Path, required=True)
    train_roi_prober.add_argument("--cache", type=Path, required=True)
    train_roi_prober.add_argument("--encoder-checkpoint", type=Path, required=True)
    train_roi_prober.add_argument("--output-dir", type=Path, required=True)
    train_roi_prober.add_argument("--context-ms", type=int, default=100)
    train_roi_prober.add_argument("--max-cache-slop-ms", type=int, default=12)
    train_roi_prober.add_argument("--epochs", type=int, default=160)
    train_roi_prober.add_argument("--batch-size", type=int, default=64)
    train_roi_prober.add_argument("--learning-rate", type=float, default=3e-4)
    train_roi_prober.add_argument("--seed", type=int, default=42)
    train_roi_prober.add_argument("--device", type=str, default="auto")
    train_roi_prober.add_argument("--model", choices=MODEL_NAMES)
    train_roi_prober.add_argument(
        "--token-summary",
        choices=["mean", "mean-std"],
        default="mean-std",
    )
    train_roi_prober.add_argument("--hidden-dim", type=int, default=128)
    train_roi_prober.add_argument("--dropout", type=float, default=0.05)
    train_roi_prober.add_argument("--physics-prior", choices=["none", "ridge"], default="ridge")
    train_roi_prober.add_argument("--ridge-alpha", type=float, default=1.0)
    train_roi_prober.add_argument("--train-splits", nargs="+", default=["train"])
    train_roi_prober.add_argument("--validation-splits", nargs="+", default=["validation"])
    train_roi_prober.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to evaluate after validation-selected ROI prober training.",
    )
    train_roi_prober.set_defaults(func=_cmd_train_roi_latent_prober)
    train_roi_eval = train_sub.add_parser(
        "roi-latent-prober-evaluate",
        help="Evaluate a saved detection-assisted frozen JEPA-latent bbox/ROI TTC prober.",
    )
    train_roi_eval.add_argument("--manifest", type=Path, required=True)
    train_roi_eval.add_argument("--split", type=Path, required=True)
    train_roi_eval.add_argument("--cache", type=Path, required=True)
    train_roi_eval.add_argument("--checkpoint", type=Path, required=True)
    train_roi_eval.add_argument("--output", type=Path)
    train_roi_eval.add_argument("--context-ms", type=int, default=100)
    train_roi_eval.add_argument("--max-cache-slop-ms", type=int, default=12)
    train_roi_eval.add_argument("--batch-size", type=int, default=64)
    train_roi_eval.add_argument("--device", type=str, default="auto")
    train_roi_eval.add_argument("--model", choices=MODEL_NAMES)
    train_roi_eval.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["test"],
        help="Splits to evaluate from the saved ROI prober checkpoint.",
    )
    train_roi_eval.set_defaults(func=_cmd_evaluate_roi_latent_prober)
    train_roi_rollout = train_sub.add_parser(
        "roi-rollout-prober",
        help="Train a detection-assisted TTC prober on frozen JEPA-predicted rollouts.",
    )
    train_roi_rollout.add_argument("--manifest", type=Path, required=True)
    train_roi_rollout.add_argument("--split", type=Path, required=True)
    train_roi_rollout.add_argument("--cache", type=Path, required=True)
    train_roi_rollout.add_argument("--jepa-checkpoint", type=Path, required=True)
    train_roi_rollout.add_argument("--output-dir", type=Path, required=True)
    train_roi_rollout.add_argument("--context-ms", type=int, default=100)
    train_roi_rollout.add_argument("--max-cache-slop-ms", type=int, default=12)
    train_roi_rollout.add_argument("--epochs", type=int, default=160)
    train_roi_rollout.add_argument("--batch-size", type=int, default=64)
    train_roi_rollout.add_argument("--learning-rate", type=float, default=3e-4)
    train_roi_rollout.add_argument("--seed", type=int, default=42)
    train_roi_rollout.add_argument("--device", type=str, default="auto")
    train_roi_rollout.add_argument("--model", choices=MODEL_NAMES)
    train_roi_rollout.add_argument(
        "--rollout-token-summary",
        choices=["mean", "mean-std"],
        default="mean-std",
    )
    train_roi_rollout.add_argument(
        "--no-rollout-context",
        action="store_true",
        help="Use predicted future rollout features without the current context latent summary.",
    )
    train_roi_rollout.add_argument(
        "--rollout-feature-mode",
        choices=["flat", "dynamics"],
        default="flat",
        help="Rollout feature composition. 'dynamics' adds horizon deltas and velocities.",
    )
    train_roi_rollout.add_argument("--hidden-dim", type=int, default=128)
    train_roi_rollout.add_argument("--dropout", type=float, default=0.10)
    train_roi_rollout.add_argument("--physics-prior", choices=["none", "ridge"], default="ridge")
    train_roi_rollout.add_argument("--ridge-alpha", type=float, default=1.0)
    train_roi_rollout.add_argument("--train-splits", nargs="+", default=["train"])
    train_roi_rollout.add_argument("--validation-splits", nargs="+", default=["validation"])
    train_roi_rollout.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["train", "validation"],
        help="Splits to evaluate after validation-selected rollout prober training.",
    )
    train_roi_rollout.set_defaults(func=_cmd_train_roi_rollout_prober)
    train_roi_rollout_eval = train_sub.add_parser(
        "roi-rollout-prober-evaluate",
        help="Evaluate a saved detection-assisted JEPA-rollout TTC prober.",
    )
    train_roi_rollout_eval.add_argument("--manifest", type=Path, required=True)
    train_roi_rollout_eval.add_argument("--split", type=Path, required=True)
    train_roi_rollout_eval.add_argument("--cache", type=Path, required=True)
    train_roi_rollout_eval.add_argument("--checkpoint", type=Path, required=True)
    train_roi_rollout_eval.add_argument("--output", type=Path)
    train_roi_rollout_eval.add_argument("--context-ms", type=int, default=100)
    train_roi_rollout_eval.add_argument("--max-cache-slop-ms", type=int, default=12)
    train_roi_rollout_eval.add_argument("--batch-size", type=int, default=64)
    train_roi_rollout_eval.add_argument("--device", type=str, default="auto")
    train_roi_rollout_eval.add_argument("--model", choices=MODEL_NAMES)
    train_roi_rollout_eval.add_argument(
        "--evaluation-splits",
        nargs="+",
        default=["test"],
        help="Splits to evaluate from the saved rollout prober checkpoint.",
    )
    train_roi_rollout_eval.set_defaults(func=_cmd_evaluate_roi_rollout_prober)
    train_object_ttc = train_sub.add_parser(
        "object-ttc",
        help=(
            "Fine-tune and calibrate the object-centric TTC head with strict "
            "train/validation/calibration/test roles."
        ),
    )
    train_object_ttc.add_argument("--cache-manifest", type=Path, required=True)
    train_object_ttc.add_argument("--output-dir", type=Path, required=True)
    train_object_ttc.add_argument("--pretrained-checkpoint", type=Path)
    train_object_ttc.add_argument("--epochs", type=int, default=50)
    train_object_ttc.add_argument("--batch-size", type=int, default=32)
    train_object_ttc.add_argument("--learning-rate", type=float, default=1e-4)
    train_object_ttc.add_argument("--weight-decay", type=float, default=0.01)
    train_object_ttc.add_argument("--label-fraction", type=float, default=1.0)
    train_object_ttc.add_argument("--seed", type=int, default=42)
    train_object_ttc.add_argument("--device", type=str, default="auto")
    train_object_ttc.add_argument(
        "--no-ego-actions",
        dest="use_ego_actions",
        action="store_false",
        help="Ablate causal egoaction inputs for a matched downstream comparison.",
    )
    train_object_ttc.set_defaults(use_ego_actions=True, func=_cmd_train_object_ttc)

    pretrain = subparsers.add_parser("pretrain", help="Self-supervised pretraining commands.")
    pretrain_sub = pretrain.add_subparsers(dest="pretrain_command", required=True)
    pretrain_jepa = pretrain_sub.add_parser("jepa", help="Pretrain TinyCNN encoder with JEPA.")
    pretrain_jepa.add_argument("--cache", type=Path, required=True)
    pretrain_jepa.add_argument("--output-dir", type=Path, required=True)
    pretrain_jepa.add_argument("--epochs", type=int, default=120)
    pretrain_jepa.add_argument("--batch-size", type=int, default=128)
    pretrain_jepa.add_argument("--learning-rate", type=float, default=5e-4)
    pretrain_jepa.add_argument("--seed", type=int, default=42)
    pretrain_jepa.add_argument("--device", type=str, default="auto")
    pretrain_jepa.add_argument("--model", choices=MODEL_NAMES, default="tiny-cnn")
    pretrain_jepa.add_argument(
        "--navigation-mode", choices=["enabled", "disabled"], default="enabled"
    )
    pretrain_jepa.add_argument("--pretrain-splits", nargs="+", default=["train"])
    pretrain_jepa.add_argument("--validation-splits", nargs="+", default=["validation"])
    pretrain_jepa.add_argument(
        "--temporal-horizons-ms",
        type=int,
        nargs="*",
        default=[20, 60, 100, 240, 500],
        help=(
            "Future cache horizons for temporal JEPA. Pass the flag with no values to use "
            "same-window masked JEPA."
        ),
    )
    pretrain_jepa.add_argument("--max-target-slop-ms", type=int, default=10)
    pretrain_jepa.add_argument("--mask-ratio", type=float, default=0.45)
    pretrain_jepa.add_argument("--block-count", type=int, default=4)
    pretrain_jepa.add_argument(
        "--mask-mode",
        choices=["spatial", "tubelet"],
        default="spatial",
        help=(
            "Context masking mode. 'tubelet' masks spatio-temporal event-channel "
            "blocks while preserving metadata/navigation channels."
        ),
    )
    pretrain_jepa.add_argument("--ema-momentum", type=float, default=0.99)
    pretrain_jepa.add_argument(
        "--regularizer",
        choices=["variance", "visreg"],
        default="variance",
        help=(
            "Embedding regularizer for JEPA. 'variance' preserves the previous "
            "scale-only anti-collapse term; 'visreg' adds Gaussian SWD sketching."
        ),
    )
    pretrain_jepa.add_argument("--variance-weight", type=float, default=1.0)
    pretrain_jepa.add_argument("--min-std", type=float, default=0.05)
    pretrain_jepa.add_argument(
        "--visreg-center-weight",
        type=float,
        default=1.0,
        help="Centering loss weight used only with --regularizer visreg.",
    )
    pretrain_jepa.add_argument(
        "--visreg-sketch-weight",
        type=float,
        default=1.0,
        help="Sketching loss weight used only with --regularizer visreg.",
    )
    pretrain_jepa.add_argument(
        "--visreg-projection-count",
        type=int,
        default=32,
        help="Number of random Sliced-Wasserstein projections for VISReg sketching.",
    )
    pretrain_jepa.add_argument(
        "--temporal-straightening-weight",
        type=float,
        default=0.0,
        help=(
            "Optional curvature penalty on predicted multi-horizon latent trajectories, "
            "inspired by temporal straightening for training in imagination."
        ),
    )
    pretrain_jepa.add_argument(
        "--global-latent",
        dest="dense_tokens",
        action="store_false",
        help="Use the older global-latent temporal JEPA objective.",
    )
    pretrain_jepa.add_argument(
        "--no-motion-conditioning",
        dest="motion_conditioning",
        action="store_false",
        help="Disable causal context-motion conditioning for dense temporal JEPA.",
    )
    pretrain_jepa.add_argument(
        "--deep-supervision-layers",
        type=int,
        nargs="*",
        default=[],
        help=(
            "0-based token-transformer layer indices to supervise with dense JEPA. "
            "Use an empty list for final-layer-only JEPA."
        ),
    )
    pretrain_jepa.add_argument(
        "--dense-predictor",
        choices=["mlp", "transformer"],
        default="mlp",
        help="Dense JEPA predictor type; transformer enables token-token predictor attention.",
    )
    pretrain_jepa.add_argument(
        "--context-token-weight",
        type=float,
        default=0.0,
        help=(
            "Optional V-JEPA 2.1-style dense loss weight for predicting all current "
            "context tokens in addition to future tokens."
        ),
    )
    pretrain_jepa.set_defaults(dense_tokens=True, motion_conditioning=True)
    pretrain_jepa.set_defaults(func=_cmd_pretrain_jepa)
    pretrain_object = pretrain_sub.add_parser(
        "object-jepa",
        help="Pretrain the recurrent object-centric Event-JEPA without TTC labels.",
    )
    pretrain_object.add_argument("--cache-manifest", type=Path, required=True)
    pretrain_object.add_argument("--output-dir", type=Path, required=True)
    pretrain_object.add_argument("--epochs", type=int, default=50)
    pretrain_object.add_argument("--batch-size", type=int, default=32)
    pretrain_object.add_argument("--learning-rate", type=float, default=3e-4)
    pretrain_object.add_argument("--weight-decay", type=float, default=0.05)
    pretrain_object.add_argument("--seed", type=int, default=42)
    pretrain_object.add_argument("--device", type=str, default="auto")
    pretrain_object.add_argument("--embedding-dim", type=int, default=192)
    pretrain_object.add_argument("--feature-dim", type=int, default=128)
    pretrain_object.add_argument("--predictor-depth", type=int, default=3)
    pretrain_object.add_argument("--predictor-heads", type=int, default=6)
    pretrain_object.add_argument("--ema-start", type=float, default=0.99)
    pretrain_object.add_argument("--ema-end", type=float, default=0.9999)
    pretrain_object.add_argument(
        "--no-ego-actions",
        dest="use_ego_actions",
        action="store_false",
        help="Ablate causal egoaction conditioning while keeping architecture matched.",
    )
    pretrain_object.add_argument(
        "--no-recurrence",
        dest="use_recurrence",
        action="store_false",
        help="Ablate recurrent object memory and encode each step independently.",
    )
    pretrain_object.add_argument(
        "--no-geometry",
        dest="use_geometry",
        action="store_false",
        help="Ablate box geometry both in the encoder and future predictor loss.",
    )
    pretrain_object.set_defaults(
        use_ego_actions=True,
        use_recurrence=True,
        use_geometry=True,
        func=_cmd_pretrain_object_jepa,
    )
    pretrain_dinov3 = pretrain_sub.add_parser(
        "object-jepa-dinov3",
        help="Continue Event-JEPA with frozen DINOv3 RGB feature distillation and no TTC labels.",
    )
    pretrain_dinov3.add_argument("--cache-manifest", type=Path, required=True)
    pretrain_dinov3.add_argument("--event-checkpoint", type=Path, required=True)
    pretrain_dinov3.add_argument("--output-dir", type=Path, required=True)
    pretrain_dinov3.add_argument(
        "--teacher-model",
        default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    )
    pretrain_dinov3.add_argument("--epochs", type=int, default=20)
    pretrain_dinov3.add_argument("--batch-size", type=int, default=16)
    pretrain_dinov3.add_argument("--learning-rate", type=float, default=1e-4)
    pretrain_dinov3.add_argument("--weight-decay", type=float, default=0.05)
    pretrain_dinov3.add_argument("--distillation-weight", type=float, default=0.25)
    pretrain_dinov3.add_argument("--ema-start", type=float, default=0.99)
    pretrain_dinov3.add_argument("--ema-end", type=float, default=0.9999)
    pretrain_dinov3.add_argument("--seed", type=int, default=42)
    pretrain_dinov3.add_argument("--device", type=str, default="auto")
    pretrain_dinov3.set_defaults(func=_cmd_distill_object_jepa_dinov3)

    evaluate = subparsers.add_parser("evaluate", help="Fixed-checkpoint evaluation commands.")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_object = evaluate_sub.add_parser(
        "object-ttc",
        help="Evaluate a fixed object TTC checkpoint without fitting on evaluation data.",
    )
    evaluate_object.add_argument("--cache-manifest", type=Path, required=True)
    evaluate_object.add_argument("--checkpoint", type=Path, required=True)
    evaluate_object.add_argument("--output", type=Path, required=True)
    evaluate_object.add_argument("--calibration-summary", type=Path)
    evaluate_object.add_argument("--splits", nargs="+", default=["test"])
    evaluate_object.add_argument("--batch-size", type=int, default=32)
    evaluate_object.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate_object.add_argument("--seed", type=int, default=42)
    evaluate_object.add_argument("--device", type=str, default="auto")
    evaluate_object.add_argument(
        "--no-ego-actions",
        dest="use_ego_actions",
        action="store_false",
    )
    evaluate_object.set_defaults(use_ego_actions=True, func=_cmd_evaluate_object_ttc)

    runtime = subparsers.add_parser("runtime", help="Deployment export and timing commands.")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_export = runtime_sub.add_parser(
        "export-object-ttc",
        help="Export and numerically verify a batch-one ONNX TTC model.",
    )
    runtime_benchmark = runtime_sub.add_parser(
        "benchmark-object-ttc",
        help="Benchmark synchronized cached-input object TTC inference.",
    )
    for command in (runtime_export, runtime_benchmark):
        command.add_argument("--cache-manifest", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--split", default="test")
        command.add_argument("--sample-index", type=int, default=0)
    runtime_export.add_argument("--output-dir", type=Path, required=True)
    runtime_export.add_argument("--opset-version", type=int, default=18)
    runtime_export.set_defaults(func=_cmd_runtime_export_object_ttc)
    runtime_benchmark.add_argument("--output", type=Path, required=True)
    runtime_benchmark.add_argument("--device", type=str, default="auto")
    runtime_benchmark.add_argument("--warmup-iterations", type=int, default=20)
    runtime_benchmark.add_argument("--measured-iterations", type=int, default=100)
    runtime_benchmark.set_defaults(func=_cmd_runtime_benchmark_object_ttc)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
