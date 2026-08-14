#!/usr/bin/env python
"""Build the signed, immutable downstream reference consumed by V8 JEPA attribution.

A simple temporal winner reuses its own representation/model recipe.  If the
prospective router wins, JEPA attribution is performed on the preregistered A5
constituent encoder because a meta-router is not itself a single transferable
encoder.  The primary TTC winner identity remains R in the artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"unsigned/invalid artifact: {path}")
    return value


def _model_recipe(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid model recipe: {path}")
    fields = set(CausalScaleTTCConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in fields}
    if not values:
        raise ValueError(f"model recipe contains no CausalScaleTTCConfig fields: {path}")
    return values


def _write(path: Path, value: dict[str, Any]) -> None:
    sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/results/aggregate_seed7.json")
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/jepa/downstream_reference.json")
    a = p.parse_args()
    try:
        aggregate, protocol = _signed(a.aggregate), _signed(a.protocol)
        candidate = str(aggregate.get("candidate_id", "A5"))
        routes = {
            "B1_TIMEVOL20_3": (
                ROOT / "configs/model/e_jepa_causal_scale_event_v8_timevol20_3.yaml",
                ROOT / "artifacts/scientific_recovery_v8/cache/timevol20_3/manifest.json",
                "B1_TIMEVOL20_3",
                3,
                {},
            ),
            "B2_EXP6_3": (
                ROOT / "configs/model/e_jepa_causal_scale_event_v8_exp6_3.yaml",
                ROOT / "artifacts/scientific_recovery_v8/cache/exp6_3/manifest.json",
                "B2_EXP6_3",
                3,
                {},
            ),
            "B3_PAIR20_2": (
                ROOT / "configs/model/e_jepa_causal_scale_event_v8_pair20_2.yaml",
                ROOT / "artifacts/scientific_recovery_v8/cache/timevol20_3/manifest.json",
                "B3_PAIR20_2",
                2,
                {},
            ),
            "C1_GATED_EXP6_3": (
                ROOT / "configs/model/e_jepa_causal_scale_event_v8_gated_exp6_3.yaml",
                ROOT / "artifacts/scientific_recovery_v8/cache/exp6_3/manifest.json",
                "C1_GATED_EXP6_3",
                3,
                {
                    "temporal_channel_gate_enabled": True,
                    "temporal_channel_gate_patch_grid": 4,
                    "temporal_channel_gate_hidden_dim": 16,
                },
            ),
        }
        if candidate in routes:
            model_path, cache_path, jepa_candidate, downstream_steps, model_overrides = routes[candidate]
            note = "JEPA uses the selected single-encoder downstream architecture."
        else:
            source = protocol.get("sources", {}).get("a5_model_recipe", {})
            model_path = ROOT / str(source.get("path"))
            cache_path = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
            jepa_candidate = "A5"
            downstream_steps = 3
            model_overrides = {}
            note = (
                "A5 is the fallback JEPA encoder reference. If the TTC winner is R, R remains the "
                "primary TTC winner but JEPA attribution is performed on its A5 constituent because "
                "the meta-router has no single transferable encoder."
            )
        if not model_path.is_file() or not cache_path.is_file():
            raise ValueError(f"JEPA reference source missing: model={model_path}, cache={cache_path}")
        downstream_model_config = _model_recipe(model_path)
        downstream_model_config.update(model_overrides)
        result = {
            "artifact_type": "scientific_recovery_v8_jepa_downstream_reference_v1",
            "status": "admissible",
            "candidate_id": candidate,
            "jepa_reference_candidate_id": jepa_candidate,
            "selection_aggregate": {"path": a.aggregate.as_posix(), "artifact_sha256": aggregate["artifact_sha256"], "sha256": _sha(a.aggregate)},
            "protocol": {"path": a.protocol.as_posix(), "artifact_sha256": protocol["artifact_sha256"], "sha256": _sha(a.protocol)},
            "downstream_model_recipe": {"path": model_path.relative_to(ROOT).as_posix(), "sha256": _sha(model_path)},
            "downstream_model_config": downstream_model_config,
            "downstream_steps": downstream_steps,
            "cache_manifest": {"path": cache_path.relative_to(ROOT).as_posix(), "sha256": _sha(cache_path)},
            "interpretation": note,
            "closed_evaluation": protocol.get("closed_evaluation", {}),
        }
        _write(a.output, result)
        print(json.dumps({"status": "completed", "output": str(a.output), "candidate_id": candidate, "jepa_reference_candidate_id": jepa_candidate}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"V8 JEPA reference build failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
