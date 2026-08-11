#!/usr/bin/env python
"""Freeze hardware-only rescue copies of an existing scientific experiment config.

Only training seed, microbatch size, gradient accumulation, num_workers and
prefetch_factor may change. Model/data/loss/selection/gates remain unchanged.
This is for CUDA/RAM feasibility rescue only and must never be selected by
validation performance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    if not isinstance(value.get("training"), dict):
        raise ValueError(f"Config has no training mapping: {path}")
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--output-config", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--gradient-accumulation-steps", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()

    if args.seed < 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("invalid seed/batch/accumulation")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("invalid DataLoader hardware profile")

    source = args.source_config.resolve()
    out = args.output_config.resolve()
    cfg = copy.deepcopy(_read(source))
    training = cfg["training"]
    original_batch = int(training.get("batch_size", -1))
    original_accum = int(training.get("gradient_accumulation_steps", -1))
    if original_batch <= 0 or original_accum <= 0:
        raise ValueError("source config has invalid batch/accumulation")

    original_effective = original_batch * original_accum
    rescue_effective = args.batch_size * args.gradient_accumulation_steps
    if rescue_effective != original_effective:
        raise ValueError(
            f"hardware rescue must preserve effective batch: source={original_effective}, "
            f"rescue={rescue_effective}"
        )

    training["seed"] = int(args.seed)
    training["batch_size"] = int(args.batch_size)
    training["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    training["num_workers"] = int(args.num_workers)
    training["prefetch_factor"] = int(args.prefetch_factor)

    decision = cfg.setdefault("decision_contract", {})
    if not isinstance(decision, dict):
        raise ValueError("decision_contract must be a mapping")
    decision["hardware_rescue_only"] = True
    decision["hardware_rescue_selection_source"] = "CUDA_feasibility_only_not_validation"
    decision["hardware_rescue_original_effective_batch"] = original_effective
    decision["hardware_rescue_microbatch"] = int(args.batch_size)
    decision["hardware_rescue_gradient_accumulation_steps"] = int(
        args.gradient_accumulation_steps
    )
    decision["private_test_remains_closed"] = True

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "artifact_type": "scientific_recovery_hardware_rescue_config_v1",
        "source_config": str(source),
        "source_sha256": _sha(source),
        "output_config": str(out),
        "output_sha256": _sha(out),
        "seed": args.seed,
        "original_batch_size": original_batch,
        "original_gradient_accumulation_steps": original_accum,
        "effective_batch_size": original_effective,
        "rescue_batch_size": args.batch_size,
        "rescue_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "selection_source": "CUDA_feasibility_only_not_validation",
        "private_test_opened": False,
    }
    manifest_path = args.manifest.resolve() if args.manifest else out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
