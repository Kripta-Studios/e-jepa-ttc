import hashlib
import json
import sys

import pytest

from operational.phase_robust.bundle import load_bundle_fusion
from scripts.predict_phase_robust import main


def test_competition_cli_rejects_native_road_claim_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--bundle",
            str(tmp_path),
            "--sensors",
            str(tmp_path),
            "--sequence",
            "CCRm",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(ValueError, match="native road"):
        main()


def test_bundle_rejects_swapped_valid_freeze_for_another_variant(tmp_path):
    (tmp_path / "BUNDLE.json").write_text("{}")
    (tmp_path / "evaluated").mkdir()
    (tmp_path / "evaluated/FINAL_FREEZE.json").write_text('{"different":"valid model"}')
    binding = {
        "bundle_sha256": hashlib.sha256(b"{}").hexdigest(),
        "fusion_freezes": {"evaluated": hashlib.sha256(b"original freeze").hexdigest()},
    }
    (tmp_path / "FUSION_BINDING.json").write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="fusion variant"):
        load_bundle_fusion(tmp_path, "evaluated")
