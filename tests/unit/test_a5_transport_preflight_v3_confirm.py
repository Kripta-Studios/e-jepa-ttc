from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "a5_preflight_v3", ROOT / "scripts" / "diagnose_a5_transport_preflight_v3_confirm.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sequence_specificity_uses_absolute_post_match_error() -> None:
    rows = [
        {
            "kind": "teacher",
            "sequence_id": "a",
            "real_best_error": 0.02,
            "shuffled_best_error": 0.05,
            "spatial_null_best_error": 0.06,
        },
        {
            "kind": "teacher",
            "sequence_id": "b",
            "real_best_error": 0.03,
            "shuffled_best_error": 0.06,
            "spatial_null_best_error": 0.07,
        },
    ]
    summary, fraction = MODULE._sequence_specificity(
        rows,
        kind="teacher",
        teacher_shuffled_min=0.20,
        teacher_spatial_min=0.20,
    )
    assert fraction == 1.0
    assert summary["a"]["passed"] is True
    assert summary["b"]["passed"] is True


def test_student_sequence_specificity_requires_real_pair_advantage() -> None:
    rows = [
        {
            "kind": "student",
            "model": "A4",
            "sequence_id": "a",
            "real_top1_cosine": 0.95,
            "shuffled_top1_cosine": 0.90,
            "spatial_null_top1_cosine": 0.85,
        },
        {
            "kind": "student",
            "model": "A4",
            "sequence_id": "b",
            "real_top1_cosine": 0.90,
            "shuffled_top1_cosine": 0.91,
            "spatial_null_top1_cosine": 0.80,
        },
    ]
    summary, fraction = MODULE._sequence_specificity(
        rows,
        kind="student",
        teacher_shuffled_min=0.0,
        teacher_spatial_min=0.0,
    )
    assert fraction == 0.5
    assert summary["a"]["passed"] is True
    assert summary["b"]["passed"] is False


def test_v3_candidate_must_match_v2_minimum_soft_epe(tmp_path: Path) -> None:
    artifact = tmp_path / "a5_transport_preflight_v2.json"
    temp = tmp_path / "temps.csv"
    temp.write_text(
        "model,radius,temperature,soft_physical_epe,soft_physical_epe_improvement_over_zero\n"
        "A4,1,0.02,1.00,0.10\n"
        "A4,1,0.04,1.05,0.05\n"
        "A4,2,0.02,1.10,0.00\n",
        encoding="utf-8",
    )
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "a5_transport_preflight_train_only_v2",
                "scope": {
                    "public_train_only": True,
                    "validation_or_test_opened": False,
                    "optimizer_steps": 0,
                    "samples": 512,
                },
                "files": {"temperature_csv": temp.name},
            }
        ),
        encoding="utf-8",
    )
    discovery = MODULE._load_v2_discovery(artifact, 1, 0.02)
    assert discovery["discovery_best_soft_physical_epe"] == 1.0
