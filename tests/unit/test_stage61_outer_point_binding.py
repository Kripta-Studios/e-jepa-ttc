from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts/run_scientific_recovery_v9_stage61.py"
    spec = importlib.util.spec_from_file_location("stage61_runner_binding", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_expert_uses_finite_trainer_point_prediction(tmp_path) -> None:
    def write_expert(name: str, point: float) -> Path:
        root = tmp_path / name / "outer_dev"
        (root / "train").mkdir(parents=True)
        trainer_path = root / "train/dev_predictions.csv"
        pd.DataFrame(
            {
                "token_id": ["token"],
                "target_ttc": [2.0],
                "prediction_ttc": [np.nan],
            }
        ).to_csv(root / "expert_oof.csv", index=False)
        pd.DataFrame(
            {
                "sample_token": ["token"],
                "target_ttc_s": [2.0],
                "point_prediction_ttc_s": [point],
            }
        ).to_csv(trainer_path, index=False)
        summary = sign_artifact(
            {
                "artifact_type": "fixture",
                "predictions": {
                    "path": trainer_path.name,
                    "sha256": compute_file_hash(str(trainer_path)),
                },
            }
        )
        (root / "train/summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return root / "expert_oof.csv"

    a5, c2f = _module()._aligned(write_expert("a5", 2.1), write_expert("c2f", 1.9))
    assert a5["prediction_ttc"].tolist() == [2.1]
    assert c2f["prediction_ttc"].tolist() == [1.9]
