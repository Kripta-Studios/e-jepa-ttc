"""Checkpoint corruption and publication faults, without optimizer updates."""

import pytest
import torch

from e_jepa_ttc.simplex_t.training import atomic_checkpoint, load_checkpoint, state_digest


def payload():
    return dict(
        model={"weight": torch.arange(4, dtype=torch.float32)},
        optimizer={0: {"step": torch.tensor(1.0)}},
        completed_updates=1,
        losses=[0.5],
        sampler_hashes=["a" * 64],
        status="TECHNICAL_PARTIAL",
    )


def test_complete_state_digest_binds_types_and_tensor_bytes():
    assert state_digest({"a": 1, "b": 2}) == state_digest({"b": 2, "a": 1})
    assert state_digest([1, 2]) != state_digest((1, 2))
    assert state_digest(torch.ones(2)) != state_digest(torch.ones(2, dtype=torch.float64))
    changed = payload()
    original = state_digest(changed)
    changed["model"]["weight"][0] += 1
    assert state_digest(changed) != original


def test_tensor_mutation_is_refused_on_resume(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(path, payload())
    assert load_checkpoint(path)["completed_updates"] == 1
    state = torch.load(path, weights_only=True)
    state["model"]["weight"][0] += 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path)


def test_publication_fault_retains_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(path, payload())
    original = path.read_bytes()

    def failure(*args):
        raise OSError("injected full disk/rename failure")

    with monkeypatch.context() as context:
        context.setattr("e_jepa_ttc.simplex_t.training.os.replace", failure)
        with pytest.raises(OSError, match="injected"):
            atomic_checkpoint(path, payload())
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp"))  # Failed bytes remain as evidence, never auto-cleaned.
    atomic_checkpoint(path, payload())  # Stale own temp cannot prevent a new publication.
    assert load_checkpoint(path)["completed_updates"] == 1


def test_resealed_false_endpoint_is_still_rejected(tmp_path):
    path = tmp_path / "checkpoint.pt"
    state = payload()
    state["status"] = "COMPLETED"
    atomic_checkpoint(path, state)
    with pytest.raises(ValueError, match="partial"):
        load_checkpoint(path)


def test_unsealed_legacy_state_requires_explicit_migration(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(payload(), path)
    with pytest.raises(ValueError, match="lacks"):
        load_checkpoint(path)


def test_progress_count_binds_logs(tmp_path):
    path = tmp_path / "checkpoint.pt"
    state = payload()
    state["losses"] = []
    atomic_checkpoint(path, state)
    with pytest.raises(ValueError, match="logs"):
        load_checkpoint(path)
