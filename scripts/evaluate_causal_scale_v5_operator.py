"""Run the synthetic-only causal scale operator gate and write compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.causal_scale_v5 import (  # noqa: E402
    evaluate_operator_gates,
    synthetic_operator_metrics,
    validate_thresholds,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import environment_snapshot  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_gate_v5.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/metrics/causal_scale_v5_synthetic_operator_gate_v1.json"


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.pop("model", None) != "e_jepa_causal_scale_event_v5":
        raise ValueError("model config must declare e_jepa_causal_scale_event_v5")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a YAML list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def run(config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    """Execute the gate without reading any dataset or TTC annotation."""

    dirty = bool(_git("status", "--porcelain"))
    if require_clean and dirty:
        raise RuntimeError("--require-clean refuses a dirty worktree")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")
    data_contract = raw.get("data")
    if not isinstance(data_contract, dict) or any(
        data_contract.get(key) is not False
        for key in ("real_data_opened", "ttc_labels_opened", "eap_opened", "evttc_opened")
    ):
        raise ValueError("synthetic gate data contract must keep every real source closed")
    model_path = ROOT / str(raw["model_config"])
    model_config = _model_config(model_path)
    thresholds_raw = raw.get("operator_gates")
    if not isinstance(thresholds_raw, dict):
        raise ValueError("operator_gates must be a mapping")
    thresholds = validate_thresholds(thresholds_raw)
    experiment = raw.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("experiment metadata must be a mapping")
    seed = int(experiment["seed"])
    started = time.perf_counter()
    metrics = synthetic_operator_metrics(model_config, seed=seed)
    gates = evaluate_operator_gates(metrics, thresholds)
    payload: dict[str, Any] = {
        "artifact_type": "causal_scale_v5_synthetic_operator_gate_v1",
        "protocol_version": str(experiment["protocol_version"]),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_passed" if gates["passed"] else "completed_gate_failed",
        "evidence_scope": "synthetic_mechanistic_only",
        "metrics_are_not_real_dataset_results": True,
        "garl_ttc_comparison_performed": False,
        "sota_claim_authorized": False,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "seed": seed,
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "model_config": {
            "path": model_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(model_path),
        },
        "data_access": data_contract,
        "metrics": metrics,
        "thresholds": thresholds,
        "gates": gates,
        "decision": raw["decision_contract"],
        "environment": environment_snapshot(),
        "elapsed_s": time.perf_counter() - started,
    }
    payload["artifact_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    output = args.output.resolve()
    if output.exists() and not args.force:
        parser.error(f"output exists: {output}; pass --force to replace it")
    try:
        payload = run(config, require_clean=args.require_clean)
    except Exception as error:
        parser.exit(2, f"causal-scale gate failed to execute: {error}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(_canonical(payload))
    return 0 if payload["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
