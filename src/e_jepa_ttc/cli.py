"""Command line interface for the E-JEPA-TTC MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.baselines.causal_geometry import run_causal_geometry_baseline
from e_jepa_ttc.baselines.event_rate import run_event_rate_baseline
from e_jepa_ttc.baselines.geometric import run_geometric_baseline
from e_jepa_ttc.baselines.trivial import run_trivial_baseline
from e_jepa_ttc.data.evttc import scan_evttc_root, validate_manifest, write_manifest
from e_jepa_ttc.data.index import build_temporal_index, write_index
from e_jepa_ttc.data.ml_cache import build_voxel_cache
from e_jepa_ttc.data.split import write_splits
from e_jepa_ttc.data.synthetic import generate_synthetic_sequence, write_synthetic_hdf5


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
        limit=args.limit,
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
        ema_momentum=args.ema_momentum,
        variance_weight=args.variance_weight,
        min_std=args.min_std,
    )
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
    cache_voxel.add_argument("--limit", type=int)
    cache_voxel.set_defaults(func=_cmd_cache_voxel)

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
    train_tiny.add_argument("--pretrained-encoder", type=Path)
    train_tiny.set_defaults(func=_cmd_train_tiny_cnn)

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
    pretrain_jepa.add_argument("--ema-momentum", type=float, default=0.99)
    pretrain_jepa.add_argument("--variance-weight", type=float, default=1.0)
    pretrain_jepa.add_argument("--min-std", type=float, default=0.05)
    pretrain_jepa.set_defaults(func=_cmd_pretrain_jepa)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


