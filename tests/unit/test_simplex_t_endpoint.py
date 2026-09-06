"""Endpoint refusal and inference tests; no optimizer updates are executed."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.endpoint import load_endpoint, predict_cached
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.training import atomic_checkpoint


class Inputs:
    population = 129
    identity_sha256 = "a" * 64

    def __init__(self):
        self.visited = []

    def gather(self, ids):
        self.visited.extend(ids.tolist())
        b = len(ids)
        experts = ids.float()[:, None].expand(b, 3) / 1000
        return (
            torch.zeros(b, 1, 17),
            torch.zeros(b, 1, 4),
            torch.ones(b, 1, dtype=torch.bool),
            experts,
            torch.zeros(b),
            torch.ones(b) / self.population,
        )


def test_inference_covers_tail_batch_and_preserves_order():
    model = TemporalRefiner(TemporalConfig()).eval()
    source = Inputs()
    result = predict_cached(model, source, resource_ok=lambda: True)
    assert source.visited == list(range(129))
    np.testing.assert_array_equal(result["raw_location"], np.arange(129, dtype=np.float32) / 1000)
    assert result["relative_cost"].shape == (129, 3)
    assert result["point_phase"].dtype == np.float32


def test_resource_pause_cannot_return_partial_predictions():
    source = Inputs()
    checks = iter([True, False])
    with pytest.raises(InterruptedError, match="no partial"):
        predict_cached(
            TemporalRefiner(TemporalConfig()).eval(),
            source,
            resource_ok=lambda: next(checks),
        )
    assert source.visited == list(range(128))


def test_train_mode_refused_before_any_input_is_read():
    source = Inputs()
    with pytest.raises(ValueError, match="evaluation-mode"):
        predict_cached(TemporalRefiner(TemporalConfig()), source, resource_ok=lambda: True)
    assert source.visited == []


def test_partial_checkpoint_refused_even_with_matching_file_digest(tmp_path):
    path = tmp_path / "technical_fixture.pt"
    atomic_checkpoint(
        path,
        {
            "completed_updates": 0,
            "status": "PAUSED_RESOURCE",
            "losses": [],
            "sampler_hashes": [],
        },
    )
    with pytest.raises(ValueError, match="update2500"):
        load_endpoint(
            path,
            TemporalConfig(),
            seed=7,
            freeze_sha256="b" * 64,
            train_source_sha256="a" * 64,
            endpoint_sha256=sha256(path),
        )


def test_wrong_endpoint_file_refused_before_deserialization(tmp_path):
    path = tmp_path / "not_a_checkpoint"
    path.write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="sealed phase endpoint"):
        load_endpoint(
            path,
            TemporalConfig(),
            seed=7,
            freeze_sha256="b" * 64,
            train_source_sha256="a" * 64,
            endpoint_sha256="c" * 64,
        )


def test_training_identity_and_model_loading_after_verified_decoder(tmp_path, monkeypatch):
    # Mock the already-verified decoder boundary, not a claimed trained checkpoint.
    # No fixture with fabricated optimizer history is published.
    path = tmp_path / "decoder_boundary_fixture"
    path.write_bytes(b"unit test boundary only")
    config = TemporalConfig()
    identity = {
        "source": "a" * 64,
        "freeze": "b" * 64,
        "config": asdict(config),
        "seed": 7,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    initialized = TemporalRefiner(config)
    state = {
        "completed_updates": 2500,
        "status": "COMPLETED",
        "identity": identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "model": initialized.state_dict(),
    }
    monkeypatch.setattr("e_jepa_ttc.simplex_t.endpoint.load_checkpoint", lambda _: state)
    arguments = dict(
        seed=7,
        freeze_sha256="b" * 64,
        train_source_sha256="a" * 64,
        endpoint_sha256=sha256(path),
    )
    restored = load_endpoint(path, config, **arguments)
    assert not restored.training
    for name, value in initialized.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    for change in ({"seed": 13}, {"freeze_sha256": "c" * 64}, {"train_source_sha256": "d" * 64}):
        with pytest.raises(ValueError, match="training identity"):
            load_endpoint(path, config, **(arguments | change))
