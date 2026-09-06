"""Phase-list completeness tests using an explicitly mocked endpoint validator."""

import json

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.phase_manifest import EndpointReceipt, fit_key, seal_phase
from e_jepa_ttc.simplex_t.registry import registered_graph

AVAILABILITY = dict(
    d1=False,
    density=False,
    t3=False,
    latent=False,
    replicate_scalar=False,
    replicate_latent=False,
)


def arguments(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    receipts = {}
    for index, spec in enumerate(registered_graph(**AVAILABILITY)):
        path = root / f"metadata_fixture_{index}"
        path.write_bytes(b"not a trained checkpoint; validator mocked by unit test")
        receipts[fit_key(spec)] = EndpointReceipt(path, sha256(path), "a" * 64)
    return dict(
        output=tmp_path / "phase.json",
        checkpoint_root=root,
        stage="T2",
        availability=AVAILABILITY,
        freeze_sha256="b" * 64,
        receipts=receipts,
    )


def test_complete_phase_is_checked_before_publication_and_cannot_be_replaced(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    checked = []

    def validate(path, config, **identity):
        assert not args["output"].exists()
        checked.append(path)

    monkeypatch.setattr("e_jepa_ttc.simplex_t.phase_manifest.load_endpoint", validate)
    result = seal_phase(**args)
    assert len(checked) == result["fits"] == 24
    data = json.loads(args["output"].read_text())
    assert data["optimizer_endpoint_updates"] == 60000  # Metadata fixture only.
    assert not data["scores_read_by_sealer"]
    assert result["sha256"] == sha256(args["output"])
    with pytest.raises(FileExistsError, match="already frozen"):
        seal_phase(**args)


def test_missing_fit_refused_before_checkpoint_reads(tmp_path):
    args = arguments(tmp_path)
    args["receipts"].pop(next(iter(args["receipts"])))
    with pytest.raises(ValueError, match="endpoint set"):
        seal_phase(**args)
    assert not args["output"].exists()


def test_validator_failure_publishes_no_phase(tmp_path, monkeypatch):
    args = arguments(tmp_path)

    def invalid(*args, **kwargs):
        raise ValueError("partial checkpoint fixture")

    monkeypatch.setattr("e_jepa_ttc.simplex_t.phase_manifest.load_endpoint", invalid)
    with pytest.raises(ValueError, match="partial checkpoint"):
        seal_phase(**args)
    assert not args["output"].exists()


def test_alias_and_outside_checkpoint_paths_refused(tmp_path):
    args = arguments(tmp_path)
    first, second, *_ = args["receipts"]
    args["receipts"][second] = args["receipts"][first]
    with pytest.raises(ValueError, match="alias"):
        seal_phase(**args)
    outside = tmp_path / "outside_fixture"
    outside.write_bytes(b"out of scope")
    args["receipts"][second] = EndpointReceipt(outside, sha256(outside), "a" * 64)
    with pytest.raises(ValueError, match="outside"):
        seal_phase(**args)
