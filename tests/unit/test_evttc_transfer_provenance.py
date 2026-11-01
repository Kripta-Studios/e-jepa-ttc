from pathlib import Path

import pytest

from operational.evttc_transfer_audit.provenance import (
    assert_sources_unchanged,
    bind_candidate,
    snapshot_sources,
)


def test_diagnosis_rejects_different_candidate_from_protocol() -> None:
    with pytest.raises(ValueError, match="candidate"):
        bind_candidate(
            {"components": {"head": "a" * 64}}, {"candidate_binding": {"head": "b" * 64}}
        )
    with pytest.raises(ValueError, match="candidate"):
        bind_candidate({"components": {"head": "a" * 64}}, {})


def test_preparation_source_changes_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "operational/evttc_competition/prepare.py"
    path.parent.mkdir(parents=True)
    path.write_text("original", encoding="utf-8")
    snapshot = snapshot_sources(tmp_path, ["operational/evttc_competition/prepare.py"])
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        assert_sources_unchanged(tmp_path, snapshot)
