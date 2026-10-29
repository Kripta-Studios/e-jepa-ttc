#!/usr/bin/env python
"""Freeze strict model-prefix-causal counterparts after a legacy winner is selected."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAUSAL_MODELS = {
    "a4": "configs/model/e_jepa_causal_scale_event_v8_t015_resize_conv_causal.yaml",
    "a5": "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_causal.yaml",
    "a6": "configs/model/e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_causal.yaml",
    "a7": "configs/model/e_jepa_causal_scale_event_v11_dual_transport_r1_t002_causal.yaml",
}


def _read(path: Path) -> dict[str, Any]:
    x = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        raise ValueError(path)
    return x


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _write(path: Path, x: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(x, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )


def _base_mutate(
    source: dict[str, Any], arm: str, seed: int, num_workers: int, prefetch_factor: int
) -> dict[str, Any]:
    x = copy.deepcopy(source)
    x["model_config"] = CAUSAL_MODELS[arm]
    x["training"]["seed"] = seed
    x["training"]["num_workers"] = int(num_workers)
    x["training"]["prefetch_factor"] = int(prefetch_factor)
    exp = x.setdefault("experiment", {})
    old = str(exp.get("name", arm))
    exp["name"] = f"{old}_causal_left_seed{seed}"
    exp["protocol_version"] = f"{exp.get('protocol_version', arm)}_causal_left_v1"
    exp["causality_hardening"] = "symmetric_legacy_to_causal_left"
    dc = x.setdefault("decision_contract", {})
    dc["temporal_smoothing_mode"] = "causal_left"
    dc["model_prefix_causal_required"] = True
    dc["oracle_roi_preprocessing_remains"] = True
    dc["public_validation_does_not_authorize_sota_claim"] = True
    dc["private_test_remains_closed"] = True
    return x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a4-source-config", type=Path, required=True)
    p.add_argument("--winner-source-config", type=Path, required=True)
    p.add_argument("--winner-stage", choices=("a5", "a6", "a7"), required=True)
    p.add_argument("--causal-a4-checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    args = p.parse_args()
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid DataLoader hardware profile")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    a4_source = _read(args.a4_source_config.resolve())
    files = {}
    for seed in (7, 13, 23):
        a4 = _base_mutate(a4_source, "a4", seed, args.num_workers, args.prefetch_factor)
        a4_path = out / f"a4_s1_lambda8_causal_left_seed{seed}.yaml"
        _write(a4_path, a4)
        files[a4_path.name] = {"sha256": _sha(a4_path)}

    source = _read(args.winner_source_config.resolve())
    for seed in (7, 13, 23):
        w = _base_mutate(source, args.winner_stage, seed, args.num_workers, args.prefetch_factor)
        if args.winner_stage in {"a6", "a7"}:
            if args.causal_a4_checkpoint is None:
                # First invocation may happen before causal A4 training. Keep a deterministic
                # expected path; a second invocation after A4 will bind its SHA.
                cp = ROOT / "artifacts/runs/scientific_recovery_a4_causal_left_seed7/model_best.pt"
                cp_sha = None
            else:
                cp = args.causal_a4_checkpoint.resolve()
                if not cp.is_file():
                    raise FileNotFoundError(cp)
                cp_sha = _sha(cp)
            tr = w["training"]
            tr["initialization_checkpoint"] = cp.relative_to(ROOT).as_posix()
            if cp_sha is not None:
                tr["initialization_checkpoint_sha256"] = cp_sha
            dc = w["decision_contract"]
            dc["replication_parent_policy"] = "fixed_a4_causal_seed7"
            dc["replication_seed_semantics"] = (
                "transport_training_stochasticity_conditional_on_fixed_parent"
            )
            dc["replication_transport_seed"] = seed
            key = "adapter_contract" if args.winner_stage == "a6" else "dual_stream_contract"
            if isinstance(dc.get(key), dict):
                dc[key]["initialization_checkpoint"] = cp.relative_to(ROOT).as_posix()
                if cp_sha is not None:
                    dc[key]["initialization_checkpoint_sha256"] = cp_sha
        path = out / f"{args.winner_stage}_s1_causal_left_seed{seed}.yaml"
        _write(path, w)
        files[path.name] = {"sha256": _sha(path)}
    manifest = {
        "artifact_type": "scientific_recovery_causal_hardening_configs_v1",
        "winner_stage": args.winner_stage,
        "a4_source": str(args.a4_source_config.resolve()),
        "winner_source": str(args.winner_source_config.resolve()),
        "files": files,
        "hardware_profile": {
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "contract": {
            "model_prefix_causal": True,
            "oracle_roi_preprocessing": True,
            "private_test_opened": False,
            "winner_parent_policy": (
                "fixed_a4_causal_seed7" if args.winner_stage in {"a6", "a7"} else "not_applicable"
            ),
            "winner_seed_semantics": (
                "transport_training_stochasticity_conditional_on_fixed_parent"
                if args.winner_stage in {"a6", "a7"}
                else "full_model_training_stochasticity"
            ),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
