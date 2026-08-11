#!/usr/bin/env python
"""Freeze hardware-only/runtime copies for the preregistered A4-S1 controls.

The scientific YAMLs remain immutable evidence. Runtime copies may change only
seed and DataLoader throughput fields (num_workers/prefetch_factor); all model,
data, loss, optimizer and selection fields are preserved byte-for-value after
normalization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
L4 = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda4_control_v1.yaml"
L8 = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml"


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
    path.write_text(yaml.safe_dump(x, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def _runtime(source: dict[str, Any], seed: int, workers: int, prefetch: int) -> dict[str, Any]:
    x = copy.deepcopy(source)
    x["training"]["seed"] = int(seed)
    x["training"]["num_workers"] = int(workers)
    x["training"]["prefetch_factor"] = int(prefetch)
    dc = x.setdefault("decision_contract", {})
    dc["runtime_hardware_override"] = {
        "scientific_fields_changed": False,
        "allowed_fields": ["training.seed", "training.num_workers", "training.prefetch_factor"],
        "num_workers": int(workers),
        "prefetch_factor": int(prefetch),
    }
    return x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    args = p.parse_args()
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid loader settings")
    l4, l8 = _read(L4), _read(L8)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for label, source, seeds in (("a4_s1_lambda4", l4, (7,)), ("a4_s1_lambda8", l8, (7,13,23))):
        for seed in seeds:
            path = out / f"{label}_seed{seed}.yaml"
            _write(path, _runtime(source, seed, args.num_workers, args.prefetch_factor))
            files[path.name] = {"sha256": _sha(path), "seed": seed}
    manifest = {
        "artifact_type": "a4_s1_runtime_configs_v1",
        "sources": {"lambda4": {"path": str(L4.relative_to(ROOT)), "sha256": _sha(L4)}, "lambda8": {"path": str(L8.relative_to(ROOT)), "sha256": _sha(L8)}},
        "hardware_only_override": {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor},
        "files": files,
        "private_test_opened": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
