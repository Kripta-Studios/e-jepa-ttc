"""A controller must not enable fitting from incomplete or changed data contracts."""

import pytest

from operational.efficient_context.common import atomic_json, digest
from operational.train40_system.controller import complete_fit, publish_fit_freezes


def test_incomplete_teacher_cannot_enable_fit(tmp_path):
    atomic_json(tmp_path / "TRAINING_PROTOCOL.json", {"fixed": True})
    atomic_json(
        tmp_path / "ENGINEERING_FREEZE.json",
        {
            "files": [],
            "protocol_sha256": digest(tmp_path / "TRAINING_PROTOCOL.json"),
            "QA": {},
        },
    )
    atomic_json(
        tmp_path / "INPUT_MANIFEST.json", {"status": "COMPLETE_VERIFIED", "row_count": 88744}
    )
    atomic_json(
        tmp_path / "TEACHER_MANIFEST.json", {"status": "COMPLETE_VERIFIED", "row_count": 8192}
    )
    with pytest.raises(ValueError, match="Full TRAIN40"):
        publish_fit_freezes(tmp_path)
    assert not (tmp_path / "MODELS_FREEZE.json").exists()


def test_recipe_change_cannot_enable_fit(tmp_path):
    atomic_json(tmp_path / "TRAINING_PROTOCOL.json", {"changed": True})
    atomic_json(tmp_path / "ENGINEERING_FREEZE.json", {"files": [], "protocol_sha256": "different"})
    with pytest.raises(ValueError, match="recipe"):
        publish_fit_freezes(tmp_path)


def test_partial_fit_is_pending_not_endpoint(tmp_path):
    receipt = tmp_path / "fits" / "a5_seed7" / "CHECKPOINT_RECEIPT.json"
    atomic_json(receipt, {"status": "PAUSED_RESOURCE", "committed_updates": 3200})
    assert not complete_fit(tmp_path, "a5_seed7")
