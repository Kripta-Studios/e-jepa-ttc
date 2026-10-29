from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_c1():
    path = ROOT / "scripts" / "build_scientific_recovery_v8_c1_opening.py"
    spec = importlib.util.spec_from_file_location("v8_c1_opening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_primary():
    path = ROOT / "scripts" / "build_scientific_recovery_v8_primary_aggregates.py"
    spec = importlib.util.spec_from_file_location("v8_primary_aggregates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fold_csv(path: Path, *, fold: int, choose_c2f: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "token_id": [f"tok-{fold}"],
            "sequence_id": [f"seq-{fold}"],
            "outer_fold": [fold],
            "choose_c2f": [choose_c2f],
        }
    ).to_csv(path, index=False, lineterminator="\n")
    return _sha(path)


def test_primary_aggregates_materialize_router_as_router_stage() -> None:
    module = _load_primary()
    assert module.ARTIFACT_TYPES["router"] == "scientific_recovery_v8_router_seed7_aggregate_v1"
    assert module.STAGE_BY_ARM["router"] == "router"


def test_c1_opening_script_binds_router_route_to_arm_aggregate() -> None:
    source = (ROOT / "scripts" / "build_scientific_recovery_v8_c1_opening.py").read_text(
        encoding="utf-8"
    )
    assert 'arm_aggregates" / "router_seed7_aggregate.json' in source
    assert 'results" / "router" / "aggregate_seed7"' not in source


def test_load_router_oof_frame_binds_official_fold_csvs(tmp_path: Path) -> None:
    module = _load_c1()
    predictions: dict[str, str] = {}
    for fold in range(3):
        path = tmp_path / "results" / "runs" / f"router_fold{fold}_seed7" / "dev_predictions.csv"
        predictions[str(fold)] = _write_fold_csv(path, fold=fold, choose_c2f=0.4 + 0.1 * fold)
    frame = module.load_router_oof_frame({"prediction_sha256": predictions}, results_root=tmp_path)
    assert list(frame["outer_fold"]) == [0, 1, 2]
    assert "choose_c2f" in frame.columns


def test_load_router_oof_frame_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _load_c1()
    path = tmp_path / "results" / "runs" / "router_fold0_seed7" / "dev_predictions.csv"
    _write_fold_csv(path, fold=0, choose_c2f=0.5)
    with pytest.raises(ValueError, match="fold 0 prediction CSV binding is invalid"):
        module.load_router_oof_frame(
            {"prediction_sha256": {"0": "0" * 64}},
            results_root=tmp_path,
        )


def test_c1_opening_exits_zero_when_no_route_sources(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_scientific_recovery_v8_c1_opening.py"),
            "--results-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "closed"
    assert payload["artifacts"] == []
