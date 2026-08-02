from __future__ import annotations

import pytest

from tests.integration._artifact_helpers import ROOT, artifact_path, read_artifact


def test_garl_cache_smoke_manifest_and_audit_agree() -> None:
    manifest_path = ROOT / "artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json"
    if not manifest_path.is_file():
        preserved = artifact_path(
            "artifacts/audits/recovery_preserved_20260802/"
            "cache_metadata_archive_20260802/"
            "garlttc_lhr_v4_smoke_workers4/manifest.json"
        )
        assert preserved.is_file()
        pytest.skip("Garl cache was intentionally pruned; readiness cache gate is red.")
    manifest = read_artifact("artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json")
    audit = read_artifact("artifacts/audits/garlttc_lhr_v4_smoke_workers4_audit.json")
    assert manifest["discard_count"] == 0
    assert manifest["config"]["preprocessing_device"] == "cpu"
    assert audit["status"] == "PASS"
    assert audit["manifest"].endswith("garlttc_lhr_v4_smoke_workers4\\manifest.json")
    assert "target_ttc" in manifest["forbidden_model_input_fields"]
    assert "jepa_pair_valid" in audit["model_input_fields"]
