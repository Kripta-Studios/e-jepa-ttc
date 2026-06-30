"""Command line interface for the E-JEPA-TTC MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.data.evttc import (
    scan_evttc_root,
    validate_manifest,
    write_manifest,
)
from e_jepa_ttc.data.index import build_temporal_index, write_index
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
