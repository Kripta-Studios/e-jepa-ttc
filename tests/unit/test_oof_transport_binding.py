"""The next runtime must preserve its exact parent checkpoint and prior binding."""

import pytest

from operational import oof_transport_binding as binding
from operational.final_refit.recipe import sha256
from operational.rgb_port.accounting import atomic_write_json, read_json_shared


def fixture(tmp_path, monkeypatch, bound=True):
    run = tmp_path / "run"
    directory = run / "rotations/0/fits/R_C2F"
    directory.mkdir(parents=True)
    (run / "OOF_TRANSPORT_ADMISSION.json").write_text("{}")
    (directory / "OOF_RUNTIME_BINDING.json").write_text("{}")
    (directory / "checkpoint_last.pt").write_bytes(b"checkpoint")
    digest = sha256(directory / "checkpoint_last.pt")
    prior = sha256(directory / "OOF_RUNTIME_BINDING.json") if bound else None
    receipt = dict(
        completed_updates=101,
        checkpoint_sha256=digest,
        identity_sha256="a" * 64,
        execution_binding_sha256=prior,
    )
    atomic_write_json(directory / "CHECKPOINT_RECEIPT.json", receipt)
    atomic_write_json(directory / "CHECKPOINT_POINTER.json", receipt)
    atomic_write_json(
        directory / "UPDATE_JOURNAL.json",
        dict(
            completed_updates=101,
            durable_updates=101,
            pending_update_upper=0,
            execution_binding_sha256=prior,
        ),
    )
    monkeypatch.setattr(binding, "load_admission", lambda _: {})
    monkeypatch.setattr(binding, "parent_bind", lambda *_: {})
    monkeypatch.setattr(
        binding,
        "parent_validate",
        lambda *_: dict(identity_sha256="a" * 64, origin_checkpoint_sha256=digest),
    )
    return run, directory, prior, digest


@pytest.mark.parametrize("bound", [True, False])
def test_migration_keeps_exact_prior_and_is_repeatable(tmp_path, monkeypatch, bound):
    run, directory, prior, digest = fixture(tmp_path, monkeypatch, bound)
    result = binding.bind_fit(run, directory)
    assert result["execution_binding_origin_prior_sha256"] == prior
    assert result["execution_binding_origin_checkpoint_sha256"] == digest
    (directory / "checkpoint_last.pt").write_bytes(b"new runtime checkpoint")
    assert binding.bind_fit(run, directory) == result
    assert sha256(directory / "transport_origin_v3/checkpoint_last.pt") == digest


def test_pending_update_prevents_migration(tmp_path, monkeypatch):
    run, directory, _, _ = fixture(tmp_path, monkeypatch)
    path = directory / "UPDATE_JOURNAL.json"
    value = read_json_shared(path)
    value["pending_update_upper"] = 1
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="durable origin"):
        binding.bind_fit(run, directory)
    assert not (directory / "OOF_TRANSPORT_BINDING.json").exists()


def test_archived_origin_tamper_is_rejected(tmp_path, monkeypatch):
    run, directory, _, _ = fixture(tmp_path, monkeypatch)
    binding.bind_fit(run, directory)
    (directory / "transport_origin_v3/checkpoint_last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="origin changed"):
        binding.validate_binding(run, directory)


def test_parent_binding_tamper_is_rejected(tmp_path, monkeypatch):
    run, directory, _, _ = fixture(tmp_path, monkeypatch)
    binding.bind_fit(run, directory)
    (directory / "OOF_RUNTIME_BINDING.json").write_text("changed")
    with pytest.raises(ValueError, match="lineage differs"):
        binding.validate_binding(run, directory)
