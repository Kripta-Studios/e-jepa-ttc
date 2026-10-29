"""Decision rules for the finite unattended RGB queue."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/run_detection_autonomous_queue.py"
SPEC = importlib.util.spec_from_file_location("autonomous_queue", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_r2_and_seed_changes_leave_source_configs_untouched():
    r0 = {
        "seed": 7,
        "experiment_id": "r0",
        "optimization": {"backbone_lr_multiplier": 1.0, "scheduler": "constant"},
    }
    r1 = {
        "seed": 7,
        "experiment_id": "r1",
        "optimization": {"backbone_lr_multiplier": 1.0, "scheduler": "warmup_cosine"},
    }
    result = MODULE.templates(r0, r1)
    assert len(result) == 9
    assert result["r2_seed13"]["optimization"] == {
        "backbone_lr_multiplier": 0.1,
        "scheduler": "warmup_cosine",
    }
    assert result["r2_seed13"]["seed"] == 13
    assert r1["optimization"]["backbone_lr_multiplier"] == 1.0
    assert r0["experiment_id"] == "r0"


def test_screen_ties_preserve_baseline_and_confirmation_avoids_duplicates():
    records = {f"{name}_seed7": {"metric": 0.1} for name in ("r0", "r1", "r2")}
    assert MODULE.choose_screen(records) == "r0"
    assert MODULE.confirmation_jobs("r0") == ["r0_seed13", "r0_seed23"]
    records["r2_seed7"]["metric"] = 0.2
    assert MODULE.choose_screen(records) == "r2"
    assert MODULE.confirmation_jobs("r2") == ["r0_seed13", "r2_seed13", "r0_seed23", "r2_seed23"]
    with pytest.raises(ValueError):
        MODULE.confirmation_jobs("unknown")


def test_source_change_is_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    bound = MODULE.binding(path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Pinned file changed"):
        MODULE.check(bound)
