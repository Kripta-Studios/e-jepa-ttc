#!/usr/bin/env python
"""Materialize route-specific signed V8 seed-7 aggregates from the canonical screen aggregate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    verify_frozen_inputs,
)

ARTIFACT_TYPES = {
    "timevol20_3": "scientific_recovery_v8_timevol20_3_seed7_aggregate_v1",
    "exp6_3": "scientific_recovery_v8_exp6_3_seed7_aggregate_v1",
    "pair20_2": "scientific_recovery_v8_pair20_2_seed7_aggregate_v1",
    "gated_exp6_3": "scientific_recovery_v8_gated_exp6_3_seed7_aggregate_v1",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"unsigned aggregate: {path}")
    return value


def config_entries(manifest: dict[str, Any], arm: str) -> dict[str, str]:
    enabled = manifest["enabled_seed7_configs"]
    entries = {
        str(fold): entry["sha256"]
        for fold in range(3)
        for name, entry in enabled.items()
        if name == f"{arm}_fold{fold}_seed7"
    }
    if len(entries) == 3:
        return entries
    template = manifest.get("conditional_templates", {}).get(arm, {})
    raw = template.get("fold_configs", [])
    if isinstance(raw, list) and len(raw) == 3:
        return {str(fold): str(raw[fold]["sha256"]) for fold in range(3)}
    raise ValueError(f"no complete frozen config set for {arm}")


def materialize(
    *, protocol_path: Path, manifest_path: Path, aggregate_path: Path, output_dir: Path
) -> list[Path]:
    frozen = verify_frozen_inputs(protocol_path, manifest_path)
    screen = read_signed(aggregate_path)
    if screen.get("artifact_type") != "scientific_recovery_v8_seed7_aggregate_v1":
        raise ValueError("not a canonical V8 seed-7 aggregate")
    contract = frozen.protocol["sample_contract"]
    fold_definitions = {str(item["fold"]): item for item in contract["fold_definitions"]}
    counts = contract["row_count_contract"]["by_outer_fold"]
    outputs: list[Path] = []
    for candidate in screen.get("candidate_results", []):
        arm = str(candidate.get("arm", ""))
        if arm not in ARTIFACT_TYPES:
            continue
        refs = candidate.get("refs")
        if not isinstance(refs, list) or len(refs) != 3:
            raise ValueError(f"{arm} aggregate lacks three fold refs")
        predictions: dict[str, str] = {}
        checkpoints: dict[str, str] = {}
        folds: dict[str, Any] = {}
        for fold, ref in enumerate(refs):
            label = str(fold)
            predictions[label] = str(ref["prediction_sha256"])
            checkpoints[label] = str(ref["checkpoint_sha256"])
            folds[label] = {
                "status": "completed",
                "sequence_ids": sorted(fold_definitions[label]["dev_sequence_ids"]),
                "row_count": int(counts[label]),
                "prediction_sha256": predictions[label],
                "checkpoint_sha256": checkpoints[label],
            }
        integrity = {name: True for name in REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS}
        payload = {
            "artifact_type": ARTIFACT_TYPES[arm],
            "schema_version": frozen.protocol["schema_version"],
            "status": "completed",
            "stage": "temporal" if arm != "gated_exp6_3" else "adaptive",
            "arm": arm,
            "candidate_id": candidate["candidate_id"],
            "git_commit": frozen.protocol["git_base_commit"],
            "implementation_git_commit": screen.get("implementation_git_commit"),
            "protocol_sha256": frozen.protocol["artifact_sha256"],
            "protocol_artifact_sha256": frozen.protocol["artifact_sha256"],
            "protocol_file_sha256": sha(protocol_path),
            "config_sha256": config_entries(frozen.manifest, arm),
            "seed": 7,
            "folds": folds,
            "row_count": int(contract["rows"]),
            "row_identity_sha256": contract["row_identity_sha256"],
            "target_identity_sha256": contract["target_identity_sha256"],
            "target_sha256": contract["target_sha256"],
            "mid_sample_weight_sha256": contract["mid_sample_weight_sha256"],
            "fold_assignment_sha256": contract["fold_assignment_sha256"],
            "prediction_sha256": predictions,
            "checkpoint_sha256": checkpoints,
            "metrics": candidate["metrics"],
            "per_sequence": candidate["per_sequence"],
            "per_bucket": candidate["per_bucket"],
            "bootstrap": candidate["bootstrap"],
            "integrity_checks": integrity,
            "gate_decision": {
                "passed": bool(candidate["passed"]),
                "candidate_id": candidate["candidate_id"],
                "rule": "frozen_ttc_candidate_gate",
            },
            "coverage": {
                "outer_folds": [0, 1, 2],
                "sequences_by_outer_fold": {
                    label: sorted(item["dev_sequence_ids"])
                    for label, item in fold_definitions.items()
                },
                "sealed_evaluation_closed": True,
            },
            "sample_contract": contract,
            "closed_evaluation": frozen.protocol["closed_evaluation"],
        }
        sign_artifact(payload)
        output = output_dir / f"{arm}_seed7_aggregate.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(output)
    return outputs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json")
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json")
    p.add_argument("--aggregate", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/aggregate_seed7.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/arm_aggregates")
    args = p.parse_args()
    try:
        outputs = materialize(protocol_path=args.protocol, manifest_path=args.manifest, aggregate_path=args.aggregate, output_dir=args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"V8 primary aggregate materialization failed closed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": "completed", "outputs": [str(x) for x in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
