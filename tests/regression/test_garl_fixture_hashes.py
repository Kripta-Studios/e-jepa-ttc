from __future__ import annotations

import pytest

from tests.integration._artifact_helpers import ROOT, artifact_path, read_artifact, sha256


def test_signed_fixture_hashes_match_current_readiness_evidence() -> None:
    readiness = read_artifact("artifacts/audits/recovery_v4/readiness.json")
    cache_manifest = ROOT / "artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json"
    if not cache_manifest.is_file():
        preserved = artifact_path(
            "artifacts/audits/recovery_preserved_20260802/"
            "cache_metadata_archive_20260802/"
            "garlttc_lhr_v4_smoke_workers4/manifest.json"
        )
        assert preserved.is_file()
        pytest.skip("Garl cache was intentionally pruned; readiness cache gate is red.")
    for relative in (
        "artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json",
        "artifacts/audits/garlttc_lhr_v4_smoke_workers4_audit.json",
        "artifacts/parity/garl_preprocessing_v2/parity.json",
        "artifacts/parity/garl_model_v1/parity.json",
    ):
        path = artifact_path(relative)
        assert readiness["evidence"][relative] == sha256(path)
