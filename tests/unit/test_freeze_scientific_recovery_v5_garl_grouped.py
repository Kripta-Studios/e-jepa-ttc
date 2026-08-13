"""Tests for frozen fold-local Garl development runs."""

from __future__ import annotations

import json
import subprocess
import sys

from scripts.freeze_scientific_recovery_v5_garl_grouped import (
    PROTOCOL_PATH,
    build_manifest,
)


def test_garl_fold_runs_are_from_scratch_and_train_only() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    manifest = build_manifest(protocol)

    assert len(manifest["runs"]) == 3
    for run, fold in zip(manifest["runs"], protocol["folds"], strict=True):
        assert run["fold"] == fold["fold"]
        assert run["train_rows"] == fold["train_rows"]
        assert run["dev_rows"] == fold["dev_rows"]
        assert run["from_scratch"] is True
        assert run["pretrained_checkpoint"] is None
        assert "--num-workers 0" in run["command"]
        assert f"--fold {fold['fold']}" in run["command"]
    contracts = manifest["contracts"]
    assert contracts["public_validation_used_for_selection"] is False
    assert contracts["private_test_opened"] is False
    assert contracts["preprocessing_identical_to_ejepa"] is False
    assert (
        contracts["comparison_scope"]
        == "exact-sample, target, budget, metric and oracle-ROI matched"
    )


def test_garl_freezer_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/freeze_scientific_recovery_v5_garl_grouped.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
