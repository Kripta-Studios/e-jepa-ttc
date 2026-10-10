from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, cast

import torch

from operational.rgb_port.accounting import sha256_file
from operational.rgb_port_c2f_graph.contracts import ADMISSION_GATES, SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "artifacts/rgb_port_20261008/current_bottleneck_20261009"
    / "verify_c2f_graph_admission_numerical_v2_frozen.py"
)
POLICY = ROOT / "artifacts/rgb_port_20261008/c2f_graph/C2F_GRAPH_NUMERICAL_POLICY_V2.json"
SOURCE_FREEZE = (
    ROOT / "artifacts/rgb_port_20261008/c2f_graph/C2F_GRAPH_ADMISSION_V2_SOURCE_FREEZE.json"
)


def _sample(value: float) -> dict[str, Any]:
    return {
        "total": value,
        "components": {"a": value},
        "gradient": torch.tensor([value, value + 1.0], dtype=torch.float32),
    }


def test_pairwise_v2_measures_all_5x5_cross_and_10_intra_pairs() -> None:
    namespace = runpy.run_path(str(SCRIPT), run_name="c2f_graph_numerical_v2_test")
    pairwise = cast(Any, namespace["_pairwise"])
    left = [_sample(float(index)) for index in range(5)]
    right = [_sample(float(index) + 0.01) for index in range(5)]
    assert len(pairwise(left, left, same=True)) == 10
    cross = pairwise(left, right, same=False)
    assert len(cross) == 25
    assert all(item["gradient_relative_l2"] >= 0.0 for item in cross)


def test_v2_policy_and_source_freeze_precede_execution_and_preserve_failure() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    failed = policy["preserved_exact_repeat_failure"]
    assert policy["status"] == "REGISTERED_BEFORE_MEASUREMENT"
    assert policy["resets_per_mode_case"] == 5
    assert sha256_file(ROOT / failed["path"]) == failed["sha256"]
    assert sha256_file(ROOT / failed["executed_source_path"]) == failed[
        "executed_source_sha256"
    ]
    assert freeze["status"] == "FROZEN_BEFORE_EXECUTION"
    assert freeze["policy_sha256"] == sha256_file(POLICY)
    assert freeze["source_sha256"][str(SCRIPT)] == sha256_file(SCRIPT)


def test_v2_contract_has_only_preregistered_numeric_gates() -> None:
    assert SCHEMA == "rgb_port_c2f_graph_freeze_v2"
    for case in ("full", "foreground"):
        assert f"{case}_gradient_cross_max" in ADMISSION_GATES
        assert f"{case}_compiled_intra_max" in ADMISSION_GATES
        assert f"{case}_compiled_vs_eager_jitter" in ADMISSION_GATES
        assert f"{case}_loss_repeat_max" in ADMISSION_GATES
        assert f"{case}_component_repeat_max" in ADMISSION_GATES
    assert "full_repeatability" not in ADMISSION_GATES

