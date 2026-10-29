from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json"
CONFIG_ROOT = ROOT / "configs/experiment/scientific_recovery_v9_eclock"

EXPECTED_MID = {
    "official_a5_oof": 158.44857930928274,
    "official_c2f_oof": 158.57314044954794,
    "nested_router_retrained_a5_constituent": 162.19984180136834,
    "nested_router_retrained_c2f_constituent": 158.92456189064018,
    "prospective_router_r": 153.87679951674625,
}


def _reference() -> dict[str, object]:
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert verify_artifact_hash(payload)
    return payload


def test_exactly_five_reference_families_are_signed_and_distinct() -> None:
    reference = _reference()
    registry = reference["reference_family_registry"]
    families = reference["families"]
    assert registry == list(EXPECTED_MID)
    assert set(families) == set(EXPECTED_MID)
    predictions = set()
    for name, expected_mid in EXPECTED_MID.items():
        family = families[name]
        assert family["reference_family"] == name
        assert family["row_count"] == 8192
        assert family["finite"] is True
        assert family["coverage_fraction"] == 1.0
        assert family["folds"] == [0, 1, 2]
        assert family["recomputed_mid"] == pytest.approx(expected_mid, abs=1.0e-12)
        predictions.add(family["prediction_sha256"])
    assert len(predictions) == 5


def test_official_a5_and_nested_a5_cannot_be_substituted() -> None:
    reference = _reference()
    families = reference["families"]
    official = families["official_a5_oof"]
    nested = families["nested_router_retrained_a5_constituent"]
    assert official["prediction_sha256"] != nested["prediction_sha256"]
    assert reference["x0_a5_replay_reference_family"] == "official_a5_oof"
    assert reference["x0_pair_u_checkpoint_family"] == "official_a5_oof"
    assert len(official["official_fold_checkpoints"]) == 3


@pytest.mark.parametrize("name", ["x0_a5_replay.yaml", "x0_pair_u.yaml"])
def test_a5_consuming_configs_name_official_family(name: str) -> None:
    config = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    assert config["checkpoint_reference_family"] == "official_a5_oof"
    assert "source_a5_checkpoint" not in config
    assert "source_a5_checkpoint_sha256" not in config
