"""A complete source manifest is reusable only while its frozen digest still matches."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from operational.efficient_context.common import digest
from operational.train40_system import controller_sealed


def test_complete_frozen_inputs_avoid_a_redundant_fragment_walk(tmp_path: Path) -> None:
    (tmp_path / "INPUT_PREPARATION_PROGRESS.json").write_text(
        json.dumps({"status": "COMPLETE", "completed_rows": 88744}), encoding="utf-8"
    )
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(
        json.dumps({"status": "COMPLETE_VERIFIED", "row_count": 88744, "sequence_count": 40}),
        encoding="utf-8",
    )
    (tmp_path / "MODELS_FREEZE.json").write_text(
        json.dumps({"input_manifest_sha256": digest(manifest)}), encoding="utf-8"
    )
    with patch.object(controller_sealed, "_refresh", side_effect=AssertionError("Must not walk")):
        assert controller_sealed.refresh_complete_inputs(tmp_path)
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sealed complete TRAIN40"):
            controller_sealed.refresh_complete_inputs(tmp_path)


def test_incomplete_preparation_retains_original_admission(tmp_path: Path) -> None:
    with patch.object(controller_sealed, "_refresh", return_value=False) as original:
        assert not controller_sealed.refresh_complete_inputs(tmp_path)
        original.assert_called_once_with(tmp_path)
