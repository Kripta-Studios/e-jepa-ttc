"""C2F claims must exclude the exact future local fit without affecting A5."""

import pytest

from operational.runpod_migration.ownership_c2f import reserve, verify_claim


@pytest.mark.parametrize("rotation", [2, 3, 4])
def test_c2f_exclusion_preserves_other_fit_and_rejects_progress(tmp_path, rotation):
    run = tmp_path / "run"
    run.mkdir()
    (run / "SCIENTIFIC_CONFIG.json").write_text("authority")
    other = run / f"rotations/{rotation}/fits/E_A5/PROGRESS.json"
    other.parent.mkdir(parents=True)
    other.write_text('{"completed_updates": 5542}')
    before = other.read_bytes()
    receipt = tmp_path / "claim.json"
    value = reserve(run, receipt, rotation)
    assert value["tasks"] == [{"rotation": rotation, "fit": "R_C2F"}]
    assert other.read_bytes() == before
    assert verify_claim(run, receipt, rotation) == value
    (run / f"rotations/{rotation}/fits/R_C2F/PROGRESS.json").write_text('{"completed_updates": 1}')
    with pytest.raises(ValueError, match="Local fit has updates"):
        verify_claim(run, receipt, rotation)


def test_c2f_rejects_unknown_rotation(tmp_path):
    with pytest.raises(ValueError):
        reserve(tmp_path, tmp_path / "receipt.json", 1)
