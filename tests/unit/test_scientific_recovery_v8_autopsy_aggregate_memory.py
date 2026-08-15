from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (
    OOF_V8_REQUIRED_COLUMNS,
    REPLAY_MECHANISM_REQUIRED_COLUMNS,
    sha256_file,
)


def _load_aggregate_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "aggregate_scientific_recovery_v8_autopsy.py"
    spec = importlib.util.spec_from_file_location("v8_autopsy_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(token: str) -> dict[str, object]:
    row: dict[str, object] = {
        "token_id": token,
        "sequence_id": "sequence_a",
        "track_id": f"track_{token}",
        "outer_fold": 0,
        "seed": 7,
        "target_ttc": 4.0,
        "sample_weight": 1.0,
        "prediction_ttc": 3.5,
        "prediction_log_variance": 0.0,
        "finite": True,
        "failure_reason": "",
        "event_count": 10,
        "event_rate": 100.0,
        "support_ms": 100.0,
        "model_name": "a5",
        "config_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }
    scalar = {
        "guard_margin": 0.1,
        "analytic_log_height_ratio": 0.01,
        "residual_log_height_ratio": 0.02,
        "occupancy_entropy": 0.5,
        "motion_magnitude": 0.4,
    }
    for column in REPLAY_MECHANISM_REQUIRED_COLUMNS:
        row[column] = scalar.get(column, "[" + ("0," * 5000) + "0]")
    return row


def _manifest(path: Path, csv_path: Path) -> Path:
    value = {
        "artifact_type": "test_replay",
        "status": "completed_replay_without_optimizer_steps",
        "interventions": {
            "baseline": {"path": csv_path.name, "sha256": sha256_file(csv_path)}
        },
    }
    sign_artifact(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_autopsy_frame_prunes_high_dimensional_replay_payloads(tmp_path: Path) -> None:
    module = _load_aggregate_module()
    csv_path = tmp_path / "baseline.csv"
    pd.DataFrame([_row("t0"), _row("t1")]).to_csv(csv_path, index=False)
    manifest = _manifest(tmp_path / "manifest.json", csv_path)

    frame = module._frame(manifest, "baseline", replay=True)

    assert len(frame) == 2
    assert set(frame.columns) == set(module._AUTOPSY_SCALAR_REPLAY_COLUMNS)
    assert set(OOF_V8_REQUIRED_COLUMNS).issubset(frame.columns)
    assert "geometry_tokens" not in frame.columns
    assert "transport_tokens" not in frame.columns


def test_autopsy_frame_still_requires_complete_replay_header(tmp_path: Path) -> None:
    module = _load_aggregate_module()
    csv_path = tmp_path / "baseline.csv"
    frame = pd.DataFrame([_row("t0")]).drop(columns=["geometry_tokens"])
    frame.to_csv(csv_path, index=False)
    manifest = _manifest(tmp_path / "manifest.json", csv_path)

    with pytest.raises(ValueError, match="geometry_tokens"):
        module._frame(manifest, "baseline", replay=True)



def _garl_row(token: str, *, fold: int) -> dict[str, object]:
    return {
        "token_id": token,
        "sequence_id": f"sequence_{token}",
        "track_id": f"track_{token}",
        "outer_fold": fold,
        "seed": 7,
        "target_ttc": 4.0 + fold,
        "prediction_ttc": 3.5 + fold,
        "prediction_log_variance": 0.0,
        "event_rate": 0.0,
    }


def test_external_garl_binding_uses_its_real_narrow_schema(tmp_path: Path) -> None:
    module = _load_aggregate_module()
    csv_path = tmp_path / "baseline.csv"
    pd.DataFrame([_garl_row("t0", fold=0), _garl_row("t1", fold=1)]).to_csv(
        csv_path, index=False
    )
    manifest = _manifest(tmp_path / "manifest.json", csv_path)

    frame = module._garl_frame(manifest)

    assert set(frame.columns) == set(module._GARL_COMPARATOR_REQUIRED_COLUMNS)
    assert "checkpoint_sha256" not in frame.columns
    assert "config_sha256" not in frame.columns
    assert "sample_weight" not in frame.columns


def test_external_garl_alignment_rejects_outer_fold_drift() -> None:
    module = _load_aggregate_module()
    a5 = pd.DataFrame([_row("t0"), _row("t1")])
    a5.loc[:, "sequence_id"] = ["sequence_t0", "sequence_t1"]
    a5.loc[:, "outer_fold"] = [0, 1]
    a5.loc[:, "target_ttc"] = [4.0, 5.0]
    c2f = a5.copy()
    c2f.loc[:, "model_name"] = "c2f"
    garl = pd.DataFrame([_garl_row("t0", fold=1), _garl_row("t1", fold=1)])
    garl.loc[:, "target_ttc"] = [4.0, 5.0]

    with pytest.raises(ValueError, match="outer-fold assignments"):
        module._align_with_external_garl(a5, c2f, garl)


def test_pruned_replay_can_drive_mechanism_cuts_without_vector_payloads(tmp_path: Path) -> None:
    module = _load_aggregate_module()
    csv_path = tmp_path / "baseline.csv"
    pd.DataFrame([_row(f"t{i}") for i in range(8)]).to_csv(csv_path, index=False)
    manifest = _manifest(tmp_path / "manifest.json", csv_path)

    frame = module._frame(manifest, "baseline", replay=True)
    cuts = module.mechanism_cuts_from_pruned_replay(frame)

    assert "ttc_bucket" in cuts
    assert "sequence" in cuts
    assert "geometry_tokens" not in frame.columns
    assert "transport_tokens" not in frame.columns
