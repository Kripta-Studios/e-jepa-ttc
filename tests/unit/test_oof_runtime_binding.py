"""The optimization cannot silently reinterpret an old checkpoint or source."""

import pytest

from operational import oof_runtime_binding as binding
from operational.final_refit.recipe import canonical_sha256, sha256
from operational.rgb_port.accounting import atomic_write_json


def setup_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    config = run / "SCIENTIFIC_CONFIG.json"
    config.write_text("{}")
    source = run / "runtime.py"
    source.write_text("unchanged")
    monkeypatch.setattr(binding, "runtime_sources", lambda: (source,))
    proof = run / "proof.json"
    atomic_write_json(
        proof,
        dict(
            schema="oof_prefetch_exact_state_probe_v1",
            status="PASS",
            exact_training_state_parity=True,
            exact_resume_parity=True,
            original_checkpoint_unchanged=True,
            technical_optimizer_updates=81,
            scientific_optimizer_updates=0,
            sources_unchanged=True,
            source_sha256={str(source): sha256(source)},
            scientific_config_sha256=sha256(config),
            cases=[
                dict(name=n, state_sha256={k: "a" * 64 for k in binding._STATE_FIELDS})
                for n in (
                    "baseline_A",
                    "optimized_A",
                    "optimized_B",
                    "baseline_B",
                    "optimized_resume",
                )
            ],
        ),
    )
    admission = dict(
        schema="oof_exact_prefetch_admission_v1",
        status="PASS",
        options=binding.OPTIONS,
        scientific_config_sha256=sha256(config),
        source_sha256={str(source): sha256(source)},
        evidence_sha256={str(proof): sha256(proof)},
        parity_report=str(proof),
    )
    admission["payload_sha256"] = canonical_sha256(admission)
    atomic_write_json(run / "OOF_RUNTIME_ADMISSION.json", admission)
    monkeypatch.setattr(binding, "load_config", lambda _: {"source_sha256": {}})
    directory = run / "rotations/0/fits/R_A5"
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint_last.pt"
    checkpoint.write_bytes(b"original immutable checkpoint bytes")
    receipt = dict(
        completed_updates=100,
        checkpoint_sha256=sha256(checkpoint),
        execution_binding_sha256=None,
        identity_sha256="a" * 64,
    )
    atomic_write_json(directory / "CHECKPOINT_RECEIPT.json", receipt)
    atomic_write_json(directory / "CHECKPOINT_POINTER.json", receipt)
    atomic_write_json(
        directory / "UPDATE_JOURNAL.json",
        dict(
            completed_updates=100,
            durable_updates=100,
            pending_update_upper=0,
            execution_binding_sha256=None,
            identity_sha256="a" * 64,
        ),
    )
    return run, directory, source


def test_origin_is_copied_and_new_checkpoints_do_not_rebind_it(tmp_path, monkeypatch):
    run, directory, _ = setup_run(tmp_path, monkeypatch)
    expected = sha256(directory / "checkpoint_last.pt")
    kwargs = binding.bind_fit(run, directory)
    assert kwargs["execution_binding_origin_checkpoint_sha256"] == expected
    assert kwargs["execution_binding_sha256"] == sha256(directory / "OOF_RUNTIME_BINDING.json")
    (directory / "checkpoint_last.pt").write_bytes(b"new checkpoint")
    assert binding.bind_fit(run, directory) == kwargs
    assert sha256(directory / "runtime_origin_v2/checkpoint_last.pt") == expected


def test_origin_tamper_rejected(tmp_path, monkeypatch):
    run, directory, _ = setup_run(tmp_path, monkeypatch)
    binding.bind_fit(run, directory)
    (directory / "runtime_origin_v2/checkpoint_last.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="origin changed"):
        binding.validate_binding(run, directory)


def test_new_source_bytes_rejected(tmp_path, monkeypatch):
    run, _, source = setup_run(tmp_path, monkeypatch)
    source.write_text("changed")
    with pytest.raises(ValueError, match="binding changed"):
        binding.load_admission(run)


def test_pending_update_cannot_be_migrated(tmp_path, monkeypatch):
    run, directory, _ = setup_run(tmp_path, monkeypatch)
    atomic_write_json(
        directory / "UPDATE_JOURNAL.json",
        dict(
            completed_updates=100,
            durable_updates=100,
            pending_update_upper=1,
            execution_binding_sha256=None,
        ),
    )
    with pytest.raises(ValueError, match="durable unbound"):
        binding.bind_fit(run, directory)
    assert not (directory / "OOF_RUNTIME_BINDING.json").exists()


def test_rgb_endpoint_without_binding_rejected(tmp_path, monkeypatch):
    _, directory, _ = setup_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="RGB endpoint lacks"):
        binding.verify_runtime_endpoint(directory / "ENDPOINT_ADMISSION.json", {"fit_id": "R_A5"})


def test_incomplete_origin_archive_rejected_even_if_resealed(tmp_path, monkeypatch):
    run, directory, _ = setup_run(tmp_path, monkeypatch)
    binding.bind_fit(run, directory)
    path = directory / "OOF_RUNTIME_BINDING.json"
    value = binding._sealed(path)
    value["origin_files_sha256"] = {}
    value.pop("payload_sha256")
    value["payload_sha256"] = canonical_sha256(value)
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="archive is incomplete"):
        binding.validate_binding(run, directory)
