from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration._artifact_helpers import read_artifact


@pytest.mark.external_data
def test_official_garl_release_audit_is_pass_and_roots_are_present() -> None:
    if not Path(r"E:\Garl-TTC").is_dir() or not Path(r"E:\GarlTTC_dataset").is_dir():
        pytest.skip("official Garl release/data roots are not mounted")
    audit = read_artifact("artifacts/audits/garl_release_v1/audit.json")
    assert audit["status"] == "pass"
    assert all(check["status"] == "pass" for check in audit["checks"]["checkpoints"].values())
    assert audit["checks"]["input_roots"]["garlttc"]["status"] == "pass"
