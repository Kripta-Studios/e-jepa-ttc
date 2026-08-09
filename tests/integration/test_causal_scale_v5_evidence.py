from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/metrics/causal_scale_v5_synthetic_operator_gate_v1.json"


def test_causal_scale_v5_evidence_is_signed_clean_and_synthetic_only() -> None:
    payload = cast(dict[str, Any], json.loads(ARTIFACT.read_text(encoding="utf-8")))
    assert verify_artifact_hash(payload)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "3fd4d2a25b85173cf34bb8738f5b7e80190f31f26acc9ed9a4d3c818d10afb20"
    )
    assert payload["git_commit"] == "7945e9936af7b6fa802ef2cca4172487adf0f2d6"
    assert payload["git_dirty"] is False
    assert payload["worktree"]["tracked_dirty"] is False
    assert payload["worktree"]["untracked_code_paths"] == []
    assert payload["status"] == "completed_passed"
    assert payload["evidence_scope"] == "synthetic_mechanistic_only"
    assert payload["metrics_are_not_real_dataset_results"] is True
    assert payload["garl_ttc_comparison_performed"] is False
    assert payload["sota_claim_authorized"] is False
    assert payload["data_access"] == {
        "eap_opened": False,
        "evttc_opened": False,
        "real_data_opened": False,
        "source": "analytic_foreground_rectangles",
        "ttc_labels_opened": False,
    }
    assert payload["gates"]["passed"] is True
