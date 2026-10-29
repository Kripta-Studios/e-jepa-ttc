"""Aggregate real rank evidence and controlled shortcut falsifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.evaluation.semantic_shortcuts import assess_eap_ssl_health  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return payload


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_benchmark(payload: dict[str, Any], path: Path, expected_mode: str) -> None:
    if payload.get("artifact_type") != "jepa_semantic_shortcut_benchmark_v1":
        raise ValueError(f"Wrong benchmark artifact type: {path}.")
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete benchmark artifact: {path}.")
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("shortcut_mode") != expected_mode:
        raise ValueError(f"Expected shortcut_mode={expected_mode!r}: {path}.")
    if payload.get("seeds") != [7, 13, 23]:
        raise ValueError(f"Benchmark must use the signed seeds 7,13,23: {path}.")


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eap-health",
        type=Path,
        default=Path("artifacts/metrics/eap_ssl_smoke_current_seed7_migrated_v4.json"),
    )
    parser.add_argument(
        "--sequence-shortcut",
        type=Path,
        default=Path("artifacts/metrics/jepa_semantic_shortcut_benchmark_v1.json"),
    )
    parser.add_argument(
        "--frame-control",
        type=Path,
        default=Path("artifacts/metrics/jepa_semantic_shortcut_frame_control_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/jepa_semantic_capacity_audit_v1.json"),
    )
    args = parser.parse_args()

    eap_payload = _read(args.eap_health)
    sequence_payload = _read(args.sequence_shortcut)
    frame_payload = _read(args.frame_control)
    _verify_benchmark(sequence_payload, args.sequence_shortcut, "sequence")
    _verify_benchmark(frame_payload, args.frame_control, "frame")
    sequence_decision = sequence_payload["decision"]
    frame_decision = frame_payload["decision"]
    residual_supported_slow = bool(sequence_decision.get("temporal_residual_passes_synthetic_gate"))
    r2_supported_slow = bool(sequence_decision.get("r2_lite_passes_synthetic_gate"))
    frame_exposes_failure = bool(frame_decision.get("benchmark_exposes_semantic_shortcut"))
    payload: dict[str, Any] = {
        "artifact_type": "jepa_semantic_capacity_audit_v1",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "sources": {
            "eap_health": {
                "path": args.eap_health.as_posix(),
                "sha256": _file_hash(args.eap_health),
            },
            "sequence_shortcut": {
                "path": args.sequence_shortcut.as_posix(),
                "sha256": _file_hash(args.sequence_shortcut),
            },
            "frame_control": {
                "path": args.frame_control.as_posix(),
                "sha256": _file_hash(args.frame_control),
            },
        },
        "real_eap_smoke_health": assess_eap_ssl_health(eap_payload),
        "controlled_results": {
            "sequence_shortcut": sequence_payload["aggregate"],
            "sequence_decision": sequence_decision,
            "frame_control": frame_payload["aggregate"],
            "frame_decision": frame_decision,
        },
        "decision": {
            "full_r2_status": "rejected_for_production" if not r2_supported_slow else "pending",
            "temporal_residual_status": (
                "conditional_candidate_for_slow_nuisances"
                if residual_supported_slow and not frame_exposes_failure
                else "rejected_or_inconclusive"
            ),
            "visreg_alone_status": "does_not_fix_semantic_shortcut_in_controlled_test",
            "intact_status": "not_applicable_without_expert_actions",
            "real_eap_semantic_collapse_proven": False,
            "production_change_authorized": False,
            "reason": (
                "R2-lite failed its TTC gate. Temporal residuals fixed the planted slow "
                "shortcut but harmed the frame-varying control. The existing eAP smoke "
                "proves rank deficiency, not which semantic variable occupies the latent."
            ),
            "next_required_experiment": (
                "Implement the missing matched high-resolution JEPA level baseline, then "
                "compare level versus level+temporal-residual on the same raw eAP rows and "
                "three seeds with frozen probes for expansion, event rate, sequence ID, "
                "and TTC; do not add R2/HSIC/CMI before that gate."
            ),
        },
    }
    payload["artifact_sha256"] = _canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
