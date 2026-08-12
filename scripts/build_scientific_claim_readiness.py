#!/usr/bin/env python
"""Build a fail-closed public-only scientific claim/readiness contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import (  # noqa: E402
    sign_artifact,
    verify_artifact_hash,
)


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Scientific artifact must be a JSON object: {path}")
    return value


def _nested(value: dict[str, Any] | None, *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _source_present(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and isinstance(value.get("sha256"), str)
        and bool(value["sha256"])
    )


def _paired_validation(
    candidate: dict[str, Any] | None,
    garl: dict[str, Any] | None,
    paired: dict[str, Any] | None,
) -> dict[str, Any]:
    sources = paired.get("sources") if isinstance(paired, dict) else None
    ejepa_source = sources.get("ejepa_predictions") if isinstance(sources, dict) else None
    garl_source = sources.get("garl_predictions") if isinstance(sources, dict) else None
    cluster_source = sources.get("cluster_metadata") if isinstance(sources, dict) else None
    cluster_definition = paired.get("cluster_definition") if isinstance(paired, dict) else None
    cluster_required = isinstance(cluster_definition, str) and (
        "external_metadata" in cluster_definition
    )
    checks = {
        "artifact_signature": bool(paired and verify_artifact_hash(paired)),
        "artifact_type": _nested(paired, "artifact_type")
        == "paired_cluster_bootstrap_ejepa_vs_garl_v1",
        "status": _nested(paired, "status") == "completed_public_validation_only",
        "comparison_scope": _nested(paired, "comparison_scope")
        == "current_candidate_public_validation",
        "exact_sample_tokens": _nested(paired, "checks", "exact_sample_tokens") is True,
        "target_equality": _nested(paired, "checks", "target_equality_atol_1e_5") is True,
        "paired_evaluation": _nested(paired, "checks", "paired_evaluation") is True,
        "private_test_closed": _nested(paired, "checks", "private_test_opened") is False,
        "sources_mapping": isinstance(sources, dict),
        "ejepa_source_present": _source_present(ejepa_source),
        "garl_source_present": _source_present(garl_source),
        "cluster_metadata_provenance": (
            _source_present(cluster_source)
            if cluster_required
            else cluster_source is None or _source_present(cluster_source)
        ),
        "input_provenance_complete": _nested(paired, "checks", "input_provenance_complete") is True,
        "candidate_predictions_sha_matches": bool(
            _source_present(ejepa_source)
            and _nested(candidate, "predictions", "sha256") == ejepa_source["sha256"]
        ),
        "garl_predictions_sha_matches": bool(
            _source_present(garl_source)
            and _nested(garl, "artifacts", "predictions", "sha256") == garl_source["sha256"]
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "candidate_predictions_sha256": _nested(candidate, "predictions", "sha256"),
        "garl_predictions_sha256": _nested(garl, "artifacts", "predictions", "sha256"),
        "paired_artifact_sha256": _nested(paired, "artifact_sha256"),
    }


def build_readiness(
    contract: dict[str, Any] | None,
    prefix: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    garl: dict[str, Any] | None,
    paired: dict[str, Any] | None,
    *,
    candidate_mode: str,
) -> dict[str, Any]:
    if contract is None or prefix is None:
        raise ValueError("contract and prefix audits are required")
    oracle_privilege_matched = bool(
        _nested(contract, "claim_contract", "matched_oracle_roi_comparison_allowed")
    )
    event_only_forward = bool(
        _nested(contract, "claim_contract", "event_only_neural_forward_claim_allowed")
    )
    prefix_ok = bool(
        candidate_mode != "legacy"
        and _nested(prefix, "results", candidate_mode, "prefix_invariant") is True
    )
    candidate_complete = bool(
        candidate
        and candidate.get("status") == "completed_public_validation_only"
        and candidate.get("official_test_opened") is False
        and verify_artifact_hash(candidate)
    )
    candidate_promotable = bool(candidate_complete and candidate.get("selectable") is True)
    garl_complete = bool(
        garl
        and garl.get("status")
        in {
            "completed_max_epochs",
            "completed_early_stopping",
            "completed_public_validation_only",
        }
        and _nested(garl, "sealed_sources", "private_test_opened") is False
        and verify_artifact_hash(garl)
    )
    paired_validation = _paired_validation(candidate, garl, paired)
    paired_complete = paired_validation["status"] == "PASS"
    private_opened = bool(
        _nested(contract, "sealed_sources", "private_test_opened") is True
        or _nested(prefix, "sealed_sources", "private_test_opened") is True
        or _nested(candidate, "official_test_opened") is True
        or _nested(garl, "sealed_sources", "private_test_opened") is True
        or _nested(paired, "checks", "private_test_opened") is True
    )

    if private_opened:
        readiness = "BLOCKED_PRIVATE_TEST_WAS_OPENED"
    elif not candidate_complete or not candidate_promotable:
        readiness = "NO_PROMOTABLE_CANDIDATE"
    elif not prefix_ok:
        readiness = "ENDPOINT_WINDOW_ORACLE_ROI_CANDIDATE_ONLY__STRICT_CAUSAL_RETRAIN_REQUIRED"
    elif not garl_complete or not paired_complete:
        readiness = "CAUSAL_ORACLE_ROI_CANDIDATE__BUDGET_MATCHED_GARL_COMPARISON_BLOCKED"
    else:
        readiness = "READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST"
    claims_blocked = readiness != "READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST"

    result: dict[str, Any] = {
        "artifact_type": "scientific_claim_readiness_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "readiness": readiness,
        "claims_blocked": claims_blocked,
        "checks": {
            "candidate_public_validation_complete": candidate_complete,
            "candidate_explicitly_promotable": candidate_promotable,
            "model_prefix_causal": prefix_ok,
            "matched_oracle_roi_privilege": oracle_privilege_matched,
            "garl_preprocessing_parity": "PARTIAL",
            "garl_8192_budget_matched_complete": garl_complete,
            "paired_exact_sample_comparison_complete": paired_complete,
            "public_validation_has_been_adaptively_reused": True,
            "private_test_opened": private_opened,
        },
        "paired_provenance_validation": paired_validation,
        "authorized_claims": {
            "event_only_neural_forward_under_oracle_roi": candidate_complete
            and event_only_forward
            and oracle_privilege_matched,
            "model_level_prefix_causal": candidate_complete and prefix_ok,
            "end_to_end_no_oracle_localization": False,
            "strict_end_to_end_streaming_causal": False,
            "public_validation_sota": False,
            "sota": False,
        },
        "required_for_future_sota_claim": [
            "freeze commit/config/checkpoint before sealed evaluation",
            "same sealed samples and targets for E-JEPA and Garl",
            "same oracle-ROI privilege or explicitly scoped claim",
            "same failure/coverage metric implementation",
            "budget-matched Garl comparator",
            "one-shot sealed/private evaluation with no post-result tuning",
        ],
        "honest_wording": {
            "oracle_roi": (
                "event-only neural TTC with exact sample/target/budget/metric parity "
                "and matched oracle-ROI privilege; preprocessing representations differ"
            ),
            "strict_model_causal": (
                "model-prefix-causal event TTC under oracle-ROI preprocessing"
                if prefix_ok
                else None
            ),
            "forbidden": "end-to-end no-oracle SOTA / strict streaming end-to-end causal",
        },
        "sota_claim_authorized": False,
    }
    sign_artifact(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-audit", type=Path, required=True)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument(
        "--candidate-mode",
        choices=("legacy", "causal_left", "none"),
        default="legacy",
    )
    parser.add_argument("--garl-budget-summary", type=Path)
    parser.add_argument("--paired-bootstrap", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_readiness(
            _read(args.contract_audit),
            _read(args.prefix_audit),
            _read(args.candidate_summary),
            _read(args.garl_budget_summary),
            _read(args.paired_bootstrap),
            candidate_mode=args.candidate_mode,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"claim readiness failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
