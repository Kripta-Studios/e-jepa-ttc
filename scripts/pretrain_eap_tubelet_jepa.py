"""Preflight the bounded, label-free Dense Level--Dynamics Tubelet JEPA runner.

The runner accepts only an already signed matched manifest.  It intentionally does
not choose rows, inspect annotation files, or reconstruct a selection policy.  Raw
eAP batch decoding will be supplied by the dedicated manifest/data builder; until
then a valid manifest can be checked with ``--dry-run`` and normal execution fails
closed before data access.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.training.eap_highres_jepa import (  # noqa: E402
    load_signed_label_free_manifest,
)


def _bounded_positive(value: str, *, name: str, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer.") from exc
    if not 1 <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must lie in [1,{maximum}].")
    return parsed


def _load_label_free_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dense Level-Dynamics config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dense Level-Dynamics config must contain a mapping.")
    prohibited_fragments = (
        "ttc",
        "depth",
        "3d",
        "bbox",
        "box",
        "category",
        "mask",
        "rgb",
        "evttc",
    )
    prohibited_truthy: dict[str, Any] = {}

    def visit(item: object, path_parts: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key)
                child_path = (*path_parts, key_text)
                if (
                    any(fragment in key_text.lower() for fragment in prohibited_fragments)
                    and child is not False
                ):
                    prohibited_truthy[".".join(child_path)] = child
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, (*path_parts, f"[{index}]"))

    visit(value, ())
    if prohibited_truthy:
        raise ValueError(
            "Dense Level-Dynamics config enables prohibited SSL-Pure fields: "
            + ", ".join(sorted(prohibited_truthy))
        )
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    """Build a bounded CLI without annotation, benchmark, or supervised-label inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True, help="Read-only raw eAP root.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Already-signed label-free matched-manifest JSON; selection is not rebuilt here.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Label-free Dense Level-Dynamics YAML."
    )
    parser.add_argument("--seed", type=int, default=7, help="Controlled initial seed (default: 7).")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Bounded run output directory."
    )
    parser.add_argument(
        "--batch-size",
        type=lambda value: _bounded_positive(value, name="batch-size", maximum=2),
        default=2,
        help="Resident microbatch bound (1--2).",
    )
    parser.add_argument(
        "--temporal-steps",
        type=lambda value: _bounded_positive(value, name="temporal-steps", maximum=5),
        default=5,
        help="Resident context/future window step bound (1--5).",
    )
    parser.add_argument(
        "--horizons",
        type=lambda value: _bounded_positive(value, name="horizons", maximum=3),
        default=3,
        help="Serial future horizon bound (1--3).",
    )
    parser.add_argument(
        "--patch-query-chunk-size",
        type=lambda value: _bounded_positive(value, name="patch-query-chunk-size", maximum=60),
        default=60,
        help="Factorized predictor patch-query chunk bound (1--60).",
    )
    parser.add_argument(
        "--max-updates",
        type=lambda value: _bounded_positive(value, name="max-updates", maximum=100_000),
        default=1_000,
        help="Explicit finite update budget.",
    )
    parser.add_argument(
        "--workers",
        type=lambda value: _bounded_positive(value, name="workers", maximum=8),
        default=4,
        help="Raw-reader worker cap for the 32 GiB host.",
    )
    parser.add_argument("--device", default="auto", help="Requested bounded device identifier.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume only from a matching core checkpoint."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify root/config/signature/resource arguments without opening event data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a signed label-free run request and fail closed pending raw-batch wiring."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.eap_root.is_dir():
            raise FileNotFoundError(f"eAP root is missing or not a directory: {args.eap_root}")
        manifest, provenance = load_signed_label_free_manifest(args.manifest)
        _load_label_free_config(args.config)
        summary = {
            "artifact_type": "dense_level_dynamics_jepa_preflight_v1",
            "status": "preflight_passed",
            "eap_root": args.eap_root.resolve().as_posix(),
            "manifest": args.manifest.resolve().as_posix(),
            "matched_manifest_hash": provenance.matched_manifest_hash,
            "split_hash": provenance.split_hash,
            "sampler_order_hash": provenance.sampler_order_hash,
            "selection_rule": provenance.selection_rule,
            "seed": args.seed,
            "resident_limits": {
                "batch_size": args.batch_size,
                "temporal_steps": args.temporal_steps,
                "horizons": args.horizons,
                "patch_query_chunk_size": args.patch_query_chunk_size,
                "max_updates": args.max_updates,
                "workers": args.workers,
            },
            "manifest_top_level_fields": sorted(manifest),
            "raw_events_opened": False,
            "annotation_files_opened": False,
            "dense_disk_cache_created": False,
        }
        if args.dry_run:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        raise RuntimeError(
            "The signed manifest preflight passed, but raw label-free batch decoding is not yet "
            "wired. Use --dry-run until the dedicated matched-manifest builder supplies the "
            "LabelFreeDataset "
            "adapter; this entry point will not recreate selection or inspect annotation data."
        )
    except Exception as exc:
        print(
            f"Dense Level-Dynamics pretraining request rejected: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
