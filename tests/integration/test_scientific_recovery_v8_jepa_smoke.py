"""CPU smoke: every D0--D4 arm performs a real update and signs outputs."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheConfig,
    write_scientific_recovery_v8_cache_for_testing,
)

ROOT = Path(__file__).resolve().parents[2]


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(3):
        for offset in range(4):
            token = f"fold{fold}-track{offset}"
            rows.append(
                {
                    "representation": np.full((3, 6, 32, 32), fold + offset, dtype=np.float32),
                    "endpoint_us": [100, 200, 300],
                    "row_identity": [str(fold), token, "target", "split", "0"],
                    "sample_token": token,
                    "sequence_id": f"seq-{fold}",
                    "track_id": f"track-{offset}",
                    "target_ttc": -2.0 if offset == 3 else 1.0 + 0.1 * offset,
                    "sample_weight": 1.0,
                    "outer_fold": fold,
                    "common_roi_xyxy": [0.0, 0.0, 32.0, 32.0],
                    "endpoint_boxes_xyxy": [[4.0, 4.0, 20.0, 20.0]] * 3,
                    "visible_heights_px": [16.0, 16.0],
                }
            )
    return rows


def test_v8_jepa_d0_to_d4_fixture_cpu_smoke(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_scientific_recovery_v8_cache_for_testing(
        records=_records(),
        output_dir=cache_dir,
        config=ScientificRecoveryV8CacheConfig(
            representation="exp6", steps=3, roi_size=32, expected_rows=None, shard_size=64
        ),
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sign_artifact(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    winner = {
        "artifact_type": "scientific_recovery_v8_downstream_winner_fixture_v1",
        "status": "admissible",
        "candidate_id": "fixture_a5",
        "downstream_model_config": {
            "modality": "event",
            "hidden_dim": 8,
            "geometry_dim": 16,
            "residual_depth": 1,
            "dropout": 0.0,
            "foreground_temporal_smoothing": 0.0,
            "foreground_temporal_smoothing_mode": "none",
        },
    }
    sign_artifact(winner)
    winner_path = tmp_path / "winner.json"
    winner_path.write_text(json.dumps(winner), encoding="utf-8")
    config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs/experiment/scientific_recovery_v8_jepa", config_dir)
    for path in config_dir.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("supervised_updates: 400", "supervised_updates: 1")
        text = text.replace("total_updates: 1000", "total_updates: 1")
        text = text.replace("batch_size: 32", "batch_size: 4")
        path.write_text(text, encoding="utf-8")
    runner = runpy.run_path(str(ROOT / "scripts/run_scientific_recovery_v8_jepa_attribution.py"))
    summaries = runner["execute"](
        config_dir=config_dir,
        cache_manifest=manifest_path,
        winner_artifact=winner_path,
        output_root=tmp_path / "runs",
        device="cpu",
        folds=[0],
        allow_fixture_cache=True,
    )
    assert {value["arm"] for value in summaries} == {"D0", "D1", "D2", "D3", "D4"}
    assert all(verify_artifact_hash(value) for value in summaries)
    assert (tmp_path / "runs/d4/fold0/seed7/pretrain/jepa_checkpoint.pt").is_file()
    d0_predictions = json.loads(
        (tmp_path / "runs/d0/fold0/seed7/oof_predictions.json").read_text(encoding="utf-8")
    )
    assert any(row["target_ttc"] < 0.0 for row in d0_predictions["fractions"]["1.0"]), (
        "the V8 downstream runner must not clamp signed negative TTC targets"
    )
    reference_hashes = d0_predictions["oof_contract_hashes"]
    architecture_hash = d0_predictions["downstream_architecture_sha256"]
    assert d0_predictions["optimization_loss_contract"] == "unweighted_signed_log_ratio_huber_v1"
    for arm in ("d1", "d2", "d3", "d4"):
        prediction = json.loads(
            (tmp_path / f"runs/{arm}/fold0/seed7/oof_predictions.json").read_text(encoding="utf-8")
        )
        assert prediction["oof_contract_hashes"] == reference_hashes
        assert prediction["downstream_architecture_sha256"] == architecture_hash
        assert prediction["optimization_loss_contract"] == "unweighted_signed_log_ratio_huber_v1"
