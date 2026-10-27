from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from operational.final_refit import scientific_fold_producers as runner
from operational.final_refit.recipe import canonical_sha256


class Source:
    population_size = 4
    sampling_groups = ((0, 1), (2, 3))
    identity = {"rotation": 1}


class Monitor:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def guard(self, *args, **kwargs):
        return self.allowed, {
            "host_available_bytes": 8 * 1024**3,
            "windows_commit_headroom_bytes": 8 * 1024**3,
        }


class Batch:
    parts = ()

    def __init__(self, indices):
        self.indices = indices

    def to(self, device):
        return self


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setattr(
        runner, "execution_binding", lambda *args: {"path": "test", "sha256": "0" * 64}
    )
    monkeypatch.setattr(
        runner, "load_config", lambda *args: {"config": "unused", "fold_contract": "unused"}
    )
    recipe = SimpleNamespace(
        seed=7,
        updates=12,
        epochs=6,
        effective_batch_size=2,
        training_config={
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "minimum_learning_rate": 0.001,
            "precision": "float32",
            "grad_clip_norm": 1.0,
        },
    )
    monkeypatch.setattr(runner, "scientific_recipe", lambda *args: recipe)
    identity = {
        "fit_id": "E_A5",
        "technical": False,
        "rotation": 1,
        "schema": "test_only",
        "seed": 7,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    monkeypatch.setattr(runner, "identity_for", lambda *args, **kwargs: identity)
    monkeypatch.setattr(runner, "source_bindings", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "build_producer",
        lambda _: torch.nn.Sequential(
            torch.nn.Linear(1, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1)
        ),
    )
    monkeypatch.setattr(runner, "_effective_batch", lambda source, indices, recipe: Batch(indices))

    def loss(model, batch, recipe, epoch):
        x = torch.tensor(batch.indices, dtype=torch.float32).reshape(-1, 1)
        return ((model(x) - 0.3 * x) ** 2).mean(), {}

    monkeypatch.setattr(runner, "producer_objective", loss)
    monkeypatch.setattr(runner, "write_admission", lambda *args: {})
    return recipe


def run(path, limit=None, monitor=None):
    return runner.train(
        source=Source(),
        fit="E_A5",
        config=Path("unused"),
        contract=Path("unused"),
        run=path,
        monitor=monitor or Monitor(),
        device="cpu",
        max_updates=limit,
    )


def payload(path):
    return torch.load(
        path / "rotations/1/fits/E_A5/checkpoint_last.pt", weights_only=False, map_location="cpu"
    )


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_resume_matches_uninterrupted_optimizer_scheduler_rng_and_cursor(tmp_path, tiny_training):
    whole, split = tmp_path / "whole", tmp_path / "split"
    assert run(whole)["scientific_endpoint"] is True
    first = run(split, 3)
    assert first["scientific_endpoint"] is False
    assert first["completed_updates"] == 3
    assert run(split)["completed_updates"] == 12
    a, b = payload(whole), payload(split)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "cursor"):
        assert_nested_equal(a[key], b[key])


def test_resource_pause_keeps_durable_state_and_resumes(tmp_path, tiny_training):
    first = run(tmp_path, 3)
    before = payload(tmp_path)["model_state_dict"]
    paused = run(tmp_path, monitor=Monitor(False))
    assert paused["status"] == "PAUSED_RESOURCE"
    assert paused["completed_updates"] == first["completed_updates"]
    assert_nested_equal(before, payload(tmp_path)["model_state_dict"])
    assert run(tmp_path)["scientific_endpoint"] is True


def test_manual_stop_is_preserved(tmp_path, tiny_training):
    (tmp_path / "STOP_REQUEST").write_text("stop")
    receipt = run(tmp_path)
    assert receipt["status"] == "PAUSED_RESOURCE"
    assert receipt["completed_updates"] == 0
    assert (tmp_path / "STOP_REQUEST").read_text() == "stop"


def test_incomplete_or_corrupt_producer_endpoint_rejected(tmp_path, tiny_training):
    run(tmp_path, 3)
    fit = tmp_path / "rotations/1/fits/E_A5"
    value = payload(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        runner.verify_endpoint(fit, value["identity"], 12)
    run(tmp_path)
    runner.verify_endpoint(fit, payload(tmp_path)["identity"], 12)
    (fit / "checkpoint_last.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incomplete"):
        runner.verify_endpoint(fit, value["identity"], 12)


def test_real_recipe_keeps_exposure_seed_and_objective():
    source = SimpleNamespace(
        population_size=53,
        identity={
            "schema": "final_refit_fold_source_v1",
            "role": "producer_pair",
            "population_size": 53,
            "inventory_sha256": "a" * 64,
        },
    )
    base = runner.build_fold_recipe(source, "E_A5_MATCHED", Path("configs/rgb_port/producers.json"))
    actual = runner.scientific_recipe(source, "E_A5", Path("configs/rgb_port/producers.json"))
    assert actual.seed == 7 and actual.updates == 36 and actual.epochs == 18
    assert actual.loss_config == base.loss_config
    assert actual.model_config == {**base.model_config, "deterministic_transport_sampling": True}
    assert (
        replace(actual, model_config=base.model_config, training_config=base.training_config)
        == base
    )


@pytest.mark.parametrize(
    "change",
    [{"modality": "rgb"}, {"ordered_native_ordinal_sha256": "bad"}, {"role": "generation"}],
)
def test_scientific_source_rejects_wrong_modality_ordinal_or_role(monkeypatch, change):
    role = SimpleNamespace(
        group_ids=("g",),
        population=1,
        ordered_sample_token_sha256="tokens",
        ordered_native_ordinal_sha256="ordinals",
    )
    monkeypatch.setattr(
        runner, "load_rotation_identity", lambda *args: SimpleNamespace(roles={"producer": role})
    )
    source = SimpleNamespace(
        population_size=1,
        identity={
            "schema": "final_refit_fold_source_v1",
            "role": "producer_pair",
            "modality": "event",
            "rotation": 0,
            "group_ids": ["g"],
            "ordered_sample_token_sha256": "tokens",
            "ordered_native_ordinal_sha256": "ordinals",
            **change,
        },
    )
    with pytest.raises(ValueError, match="escaped"):
        runner.identity_for(
            source, None, fit="E_A5", contract=Path("unused"), bindings={}, device="cpu"
        )
