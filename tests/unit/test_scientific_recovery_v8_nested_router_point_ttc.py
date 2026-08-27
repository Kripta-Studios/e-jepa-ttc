from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "p" * 64


def _load_module():
    path = ROOT / "scripts" / "run_scientific_recovery_v8_nested_router.py"
    spec = importlib.util.spec_from_file_location("v8_nested_router_point_ttc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _signed_expert_tree(
    tmp_path: Path,
    *,
    oof_prediction: float | None,
    point_prediction: float | None,
    include_trainer: bool = True,
) -> Path:
    expert_dir = tmp_path / "artifacts" / "expert"
    oof_path = expert_dir / "expert_oof.csv"
    _write_csv(
        oof_path,
        pd.DataFrame(
            {
                "token_id": ["token-a"],
                "prediction_ttc": [oof_prediction],
                "finite": [True],
            }
        ),
    )
    if include_trainer:
        trainer_path = expert_dir / "train" / "dev_predictions.csv"
        _write_csv(
            trainer_path,
            pd.DataFrame(
                {
                    "sample_token": ["token-a"],
                    "prediction_ttc_s": [oof_prediction],
                    "point_prediction_ttc_s": [point_prediction],
                }
            ),
        )
        summary = sign_artifact(
            {
                "artifact_type": "causal_scale_train_summary_test",
                "git_commit": "train-head",
                "git_dirty": False,
                "predictions": {
                    "path": "dev_predictions.csv",
                    "sha256": _sha256(trainer_path),
                },
            }
        )
        trainer_path.with_name("summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    artifact = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_router_expert_prediction_v1",
            "status": "completed",
            "expert": "A5",
            "role": "inner_oof",
            "protocol_sha256": PROTOCOL,
            "git_commit": "train-head",
            "git_dirty": False,
            "checkpoint": {"sha256": "c" * 64},
            "oof_csv": {
                "path": oof_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(oof_path),
            },
        }
    )
    artifact_path = expert_dir / "expert_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_path


def test_load_expert_artifact_binds_finite_point_ttc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    artifact_path = _signed_expert_tree(tmp_path, oof_prediction=float("nan"), point_prediction=2.5)
    frame, _payload = module._load_expert_artifact(
        artifact_path,
        expert="A5",
        role="inner_oof",
        protocol_sha256=PROTOCOL,
    )
    assert float(frame.loc[0, "prediction_ttc"]) == pytest.approx(2.5)
    assert bool(frame.loc[0, "finite"])


def test_load_expert_artifact_fails_closed_on_nonfinite_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    artifact_path = _signed_expert_tree(
        tmp_path, oof_prediction=float("nan"), point_prediction=float("nan")
    )
    with pytest.raises(module.RouterStageError, match="point TTC"):
        module._load_expert_artifact(
            artifact_path,
            expert="A5",
            role="inner_oof",
            protocol_sha256=PROTOCOL,
        )


def test_merged_inner_oof_without_trainer_rejects_nonfinite_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    artifact_path = _signed_expert_tree(
        tmp_path,
        oof_prediction=float("nan"),
        point_prediction=2.5,
        include_trainer=False,
    )
    with pytest.raises(module.RouterStageError, match="no trainer"):
        module._load_expert_artifact(
            artifact_path,
            expert="A5",
            role="inner_oof",
            protocol_sha256=PROTOCOL,
        )
