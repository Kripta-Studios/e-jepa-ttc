from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "aggregate_scientific_recovery_v8_router.py"
    spec = importlib.util.spec_from_file_location("v8_router_aggregate_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_aggregate_reads_config_hashes_from_signed_plan() -> None:
    module = _load_module()
    frozen = module.verify_frozen_inputs(
        ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
        ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    pointer = frozen.manifest["c1_analysis_plans"]["router_regime"]
    assert "source_aggregate_contract" not in pointer
    hashes = module._frozen_router_config_sha256_by_fold(frozen)
    assert set(hashes) == {"0", "1", "2"}
    assert all(len(value) == 64 for value in hashes.values())


def test_router_aggregate_rejects_manifest_pointer_without_signed_plan() -> None:
    module = _load_module()
    frozen = SimpleNamespace(
        protocol={"c1_analysis_plans": {"router_regime": {"path": "missing.json"}}},
        manifest={"c1_analysis_plans": {"router_regime": {"path": "missing.json"}}},
    )
    with pytest.raises(module.RouterAggregateError, match="router_regime"):
        module._frozen_router_config_sha256_by_fold(frozen)


def test_router_aggregate_records_repo_relative_oof_path_from_relative_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.chdir(ROOT)
    relative = Path(
        "artifacts/scientific_recovery_v8/results/router/aggregate_seed7/router_oof_predictions.csv"
    )
    posix = module._repo_relative(relative)
    assert posix == relative.as_posix()
    assert not Path(posix).is_absolute()
