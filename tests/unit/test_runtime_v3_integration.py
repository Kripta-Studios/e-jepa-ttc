from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operational.final_refit.producers import _runtime_optimization_options
from operational.final_refit.recipe import canonical_sha256


def _fixture(root: Path, **overrides: bool) -> Path:
    options = {
        "shared_resource_snapshot": True,
        "host_loss_metadata": True,
        "transport_reuse": True,
        "scalar_transfer": True,
        "merge_parts_by_t": False,
        "cudagraph_local_transport": False,
        **overrides,
    }
    receipt = {
        "schema": "final_refit_rgb_optimization_admission_v1",
        "status": "PASS",
        "scientific_optimizer_updates": 0,
        "checks": dict.fromkeys(
            ("numerical_parity", "state_parity", "resume_parity", "rng_parity"), "PASS"
        ),
        "admitted_options": options,
        "source_sha256_before_after": {
            "example.py": {"before_sha256": "a" * 64, "after_sha256": "b" * 64}
        },
    }
    receipt["admission_sha256"] = canonical_sha256(receipt)
    path = root / "RUNTIME_OPTIMIZATION_ADMISSION.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    plan = root / "PRODUCER_PLAN.json"
    plan.write_text(
        json.dumps(
            {
                "input_sha256": {
                    "runtime_optimization_admission": hashlib.sha256(path.read_bytes()).hexdigest()
                },
                "executed_source_sha256": {"example.py": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_legacy_plan_enables_no_optimizations(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"input_sha256": {}}', encoding="utf-8")
    assert _runtime_optimization_options(tmp_path, path) == {}


def test_complete_admission_selects_only_verified_options(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    options = _runtime_optimization_options(tmp_path, path)
    assert options["shared_resource_snapshot"]
    assert options["transport_reuse"]
    assert not options["merge_parts_by_t"]


@pytest.mark.parametrize("option", ["merge_parts_by_t", "cudagraph_local_transport"])
def test_experimental_paths_cannot_be_enabled_by_receipt(tmp_path: Path, option: str) -> None:
    path = _fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match="Experimental"):
        _runtime_optimization_options(tmp_path, path)


def test_changed_admission_is_rejected(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    (tmp_path / "RUNTIME_OPTIMIZATION_ADMISSION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        _runtime_optimization_options(tmp_path, path)


def test_changed_plan_source_is_rejected(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["executed_source_sha256"]["example.py"] = "c" * 64
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="source closure"):
        _runtime_optimization_options(tmp_path, path)
