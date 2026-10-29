from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from operational.rgb_port.accounting import sha256_file
from operational.rgb_port_c2f_graph.contracts import (
    ADMISSION_GATES,
    SCHEMA,
    _repeat_policy_is_exact,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SCRIPT = (
    ROOT
    / "artifacts/rgb_port_20261008/current_bottleneck_20261009"
    / "verify_c2f_graph_admission_numerical_v2_frozen.py"
)
EXTERNAL_POLICY = ROOT / "artifacts/rgb_port_20261008/c2f_graph/C2F_GRAPH_NUMERICAL_POLICY_V2.json"
EXTERNAL_FREEZE = (
    ROOT / "artifacts/rgb_port_20261008/c2f_graph/C2F_GRAPH_ADMISSION_V2_SOURCE_FREEZE.json"
)


def _sample(value: float) -> dict[str, Any]:
    return {
        "total": value,
        "components": {"a": value},
        "gradient": torch.tensor([value, value + 1.0], dtype=torch.float32),
    }


def _pairwise_reference(left: list[dict], right: list[dict], *, same: bool) -> list[dict]:
    """Apply the preregistered pairwise relative-L2 definition to every pair."""
    result = []
    for left_index, a in enumerate(left):
        start = left_index + 1 if same else 0
        for right_index in range(start, len(right)):
            b = right[right_index]
            denominator = max(
                float(torch.linalg.vector_norm(a["gradient"])),
                float(torch.linalg.vector_norm(b["gradient"])),
                1e-30,
            )
            result.append(
                {
                    "left": left_index,
                    "right": right_index,
                    "gradient_relative_l2": float(
                        torch.linalg.vector_norm(a["gradient"] - b["gradient"])
                    )
                    / denominator,
                }
            )
    return result


@pytest.fixture
def exact_policy(tmp_path: Path) -> dict:
    failed = tmp_path / "preserved_failure.json"
    failed.write_text('{"status":"FAILED"}', encoding="utf-8")
    executed = tmp_path / "executed_source.py"
    executed.write_text("# retained executed source\n", encoding="utf-8")
    return {
        "schema": "rgb_port_c2f_graph_numerical_policy_v2",
        "status": "REGISTERED_BEFORE_MEASUREMENT",
        "modes": ["eager", "compiled"],
        "loss_cases": ["full", "foreground"],
        "resets_per_mode_case": 5,
        "gradient_relative_l2": {
            "all_cross_pairwise_max_lte": 0.002,
            "compiled_intra_pairwise_max_lte": 0.002,
            "compiled_intra_lte_eager_intra_plus": 0.002,
        },
        "gradient_relative_l2_definition": "l2(left-right)/max(l2(left),l2(right),1e-30)",
        "loss_component_pairwise_scope": (
            "all_eager_intra_compiled_intra_and_eager_compiled_cross_pairs"
        ),
        "loss_abs_pairwise_max_lte": 0.0005,
        "component_abs_pairwise_max_lte": 0.0005,
        "output_abs_lte": 0.0005,
        "requirements": {
            "actual_cuda_graph_launch": True,
            "every_repeat_gradient_vector_finite": True,
            "full_state_rng_restore_exact": True,
            "optimizer_updates": 0,
            "record_every_repeat": True,
            "localize_discrepant_parameters": True,
        },
        "preserved_exact_repeat_failure": {
            "status": "FAILED",
            "path": str(failed),
            "sha256": sha256_file(failed),
            "executed_source_path": str(executed),
            "executed_source_sha256": sha256_file(executed),
        },
    }


def test_pairwise_v2_measures_all_5x5_cross_and_10_intra_pairs() -> None:
    left = [_sample(float(index)) for index in range(5)]
    right = [_sample(float(index) + 0.01) for index in range(5)]
    assert len(_pairwise_reference(left, left, same=True)) == 10
    cross = _pairwise_reference(left, right, same=False)
    assert len(cross) == 25
    expected = torch.linalg.vector_norm(left[0]["gradient"] - right[0]["gradient"])
    expected /= max(
        torch.linalg.vector_norm(left[0]["gradient"]),
        torch.linalg.vector_norm(right[0]["gradient"]),
    )
    assert cross[0]["gradient_relative_l2"] == pytest.approx(float(expected), rel=1e-7)


def test_production_contract_accepts_only_complete_exact_policy(exact_policy: dict) -> None:
    assert _repeat_policy_is_exact(exact_policy)
    for path, value in (
        (("resets_per_mode_case",), 4),
        (("gradient_relative_l2_definition",), "different"),
        (("requirements", "optimizer_updates"), 1),
        (("preserved_exact_repeat_failure", "sha256"), "0" * 64),
    ):
        changed = copy.deepcopy(exact_policy)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        assert not _repeat_policy_is_exact(changed)


@pytest.mark.skipif(
    not all(path.is_file() for path in (EXTERNAL_SCRIPT, EXTERNAL_POLICY, EXTERNAL_FREEZE)),
    reason="optional historical C2F frozen-evidence bundle is not shipped in a clean clone",
)
def test_external_frozen_source_matches_its_historical_receipts() -> None:
    """Verify retained historical evidence when its external bundle is mounted."""
    policy = json.loads(EXTERNAL_POLICY.read_text(encoding="utf-8"))
    freeze = json.loads(EXTERNAL_FREEZE.read_text(encoding="utf-8"))
    failed = policy["preserved_exact_repeat_failure"]
    assert _repeat_policy_is_exact(policy)
    assert freeze["status"] == "FROZEN_BEFORE_EXECUTION"
    assert freeze["policy_sha256"] == sha256_file(EXTERNAL_POLICY)
    assert freeze["source_sha256"][str(EXTERNAL_SCRIPT)] == sha256_file(EXTERNAL_SCRIPT)
    assert sha256_file(Path(failed["path"])) == failed["sha256"]


def test_v2_contract_has_only_preregistered_numeric_gates() -> None:
    assert SCHEMA == "rgb_port_c2f_graph_freeze_v2"
    for case in ("full", "foreground"):
        assert f"{case}_gradient_cross_max" in ADMISSION_GATES
        assert f"{case}_compiled_intra_max" in ADMISSION_GATES
        assert f"{case}_compiled_vs_eager_jitter" in ADMISSION_GATES
        assert f"{case}_loss_repeat_max" in ADMISSION_GATES
        assert f"{case}_component_repeat_max" in ADMISSION_GATES
    assert "full_repeatability" not in ADMISSION_GATES
