"""Contract tests for the sealed V8 causal-scale JEPA attribution core."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Hashable

import pytest
import torch

from e_jepa_ttc.models.causal_scale_jepa_v8 import (
    JEPA_VIEW_NAMES,
    CausalScaleJEPAV8,
    CausalScaleJEPAV8Config,
    CausalScaleJEPAV8Output,
    apply_jepa_to_experts,
    d3_partial_finetune_allowlist,
    make_d1_random_frozen_model,
    ordered_state_sha256,
    strict_encoder_transfer,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training.scientific_recovery_v8_jepa import (
    ScientificRecoveryV8JEPATrainer,
    ScientificRecoveryV8JEPATrainerConfig,
    assert_all_online_encoder_gradients,
    deterministic_shuffled_future,
)


def _endpoint() -> CausalScaleTTC:
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=4,
            hidden_dim=8,
            geometry_dim=16,
            residual_depth=1,
            dropout=0.0,
        )
    )


def _inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    return tuple(torch.randn(batch, 4, 32, 32) for _ in range(3))  # type: ignore[return-value]


def test_v8_uses_exact_causal_encoder_copies_and_no_label_signature() -> None:
    source = _endpoint()
    model = CausalScaleJEPAV8(source)

    assert tuple(source.encoder.state_dict()) == tuple(model.online_encoder.state_dict())
    assert ordered_state_sha256(source.encoder) == ordered_state_sha256(model.online_encoder)
    assert ordered_state_sha256(model.online_encoder) == ordered_state_sha256(model.target_encoder)
    assert not model.target_encoder.training
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert tuple(inspect.signature(model.forward).parameters) == ("t0", "t1", "t2")
    assert {"ttc", "bbox", "mask", "category", "metadata"}.isdisjoint(
        inspect.signature(model.forward).parameters
    )


def test_v8_three_view_loss_is_exact_equal_mean_and_all_encoder_params_receive_gradients() -> None:
    model = CausalScaleJEPAV8(_endpoint())
    t0, t1, t2 = _inputs()
    output = model(t0, t1, t2)

    assert set(output.losses) == set(JEPA_VIEW_NAMES)
    assert torch.allclose(
        output.loss,
        sum(output.losses[name] for name in JEPA_VIEW_NAMES) / 3.0,
    )
    assert not output.target_dense_features.requires_grad
    assert not output.target_global_token.requires_grad
    assert not output.target_foreground_logits.requires_grad
    output.loss.backward()
    magnitudes = assert_all_online_encoder_gradients(model)
    assert set(magnitudes) == {name for name, _ in model.online_encoder.named_parameters()}
    assert all(value > 0.0 for value in magnitudes.values())


def test_v8_strict_transfer_and_d1_d3_freeze_contracts() -> None:
    source = _endpoint()
    destination = _endpoint()
    transfer = strict_encoder_transfer(source, destination)
    assert transfer["source_encoder_sha256"] == transfer["destination_encoder_sha256"]

    random_frozen = make_d1_random_frozen_model(source.config)
    assert not any(parameter.requires_grad for parameter in random_frozen.encoder.parameters())

    allowed = d3_partial_finetune_allowlist(destination)
    expected_final_block = max(
        int(name.split(".")[1])
        for name, child in destination.encoder.named_modules()
        if name.startswith("features.") and child.__class__.__name__ == "_ResidualBlock"
    )
    assert any(name.startswith("foreground.") for name in allowed)
    assert any(name.startswith(f"features.{expected_final_block}.") for name in allowed)
    for name, parameter in destination.encoder.named_parameters():
        assert parameter.requires_grad == (name in allowed)


def test_v8_router_helper_transfers_each_expert_without_creating_router() -> None:
    a5_source, c2f_source = _endpoint(), _endpoint()
    a5_jepa, c2f_jepa = CausalScaleJEPAV8(a5_source), CausalScaleJEPAV8(c2f_source)
    a5_destination, c2f_destination = _endpoint(), _endpoint()

    result = apply_jepa_to_experts(
        a5_model=a5_destination,
        c2f_model=c2f_destination,
        a5_jepa=a5_jepa,
        c2f_jepa=c2f_jepa,
        mode="frozen",
    )

    assert result["mode"]["router_is_encoder"] is False
    assert ordered_state_sha256(a5_destination.encoder) == ordered_state_sha256(
        a5_jepa.online_encoder
    )
    assert ordered_state_sha256(c2f_destination.encoder) == ordered_state_sha256(
        c2f_jepa.online_encoder
    )
    assert not any(parameter.requires_grad for parameter in a5_destination.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in c2f_destination.encoder.parameters())


def test_v8_shuffled_future_is_deterministic_and_equal_compute() -> None:
    t0, t1, t2 = _inputs(batch=4)
    track_ids: list[Hashable] = ["a", "a", "b", "b"]
    shuffled_one, permutation_one = deterministic_shuffled_future(
        t2, track_ids=track_ids, seed=7, update_index=3
    )
    shuffled_two, permutation_two = deterministic_shuffled_future(
        t2, track_ids=track_ids, seed=7, update_index=3
    )
    assert torch.equal(permutation_one, permutation_two)
    assert torch.equal(shuffled_one, shuffled_two)
    assert shuffled_one.shape == t2.shape
    assert shuffled_one.dtype == t2.dtype
    assert shuffled_one.device == t2.device
    assert all(index != int(permutation_one[index]) for index in range(t2.shape[0]))
    assert all(
        track_ids[index] != track_ids[int(permutation_one[index])] for index in range(t2.shape[0])
    )

    source = _endpoint()
    regular = CausalScaleJEPAV8(source)
    shuffled = copy.deepcopy(regular)
    regular_trainer = ScientificRecoveryV8JEPATrainer(
        regular, ScientificRecoveryV8JEPATrainerConfig(total_updates=1, seed=7)
    )
    shuffled_trainer = ScientificRecoveryV8JEPATrainer(
        shuffled,
        ScientificRecoveryV8JEPATrainerConfig(total_updates=1, seed=7, shuffled_future=True),
    )
    regular_result = regular_trainer.step(t0, t1, t2)
    shuffled_result = shuffled_trainer.step(t0, t1, t2, track_ids=track_ids)
    assert regular_trainer.update_count == shuffled_trainer.update_count == 1
    assert regular_result["permutation"] is None
    assert len(shuffled_result["permutation"]) == t2.shape[0]
    assert (
        regular_trainer.compute_manifest()["total_updates"]
        == shuffled_trainer.compute_manifest()["total_updates"]
    )
    manifest = shuffled_trainer.compute_manifest()
    assert manifest["d4_future_pairing"]["cross_track"] is True
    assert manifest["d4_future_pairing"]["no_fixed_points"] is True
    assert manifest["only_difference_from_matched"] == "future_pairing"


def test_v8_shuffled_future_rejects_infeasible_or_invalid_track_identities() -> None:
    _, _, t2 = _inputs(batch=4)
    with pytest.raises(ValueError, match="complete cross-track"):
        deterministic_shuffled_future(t2, track_ids=["a", "a", "a", "b"], seed=7, update_index=0)
    with pytest.raises(ValueError, match="row-aligned"):
        deterministic_shuffled_future(t2, track_ids=["a", "b"], seed=7, update_index=0)
    with pytest.raises(ValueError, match="hashable"):
        deterministic_shuffled_future(
            t2, track_ids=[["a"], ["b"], ["c"], ["d"]], seed=7, update_index=0
        )
    with pytest.raises(ValueError, match="stable JSON"):
        deterministic_shuffled_future(
            t2,
            track_ids=[object(), object(), object(), object()],
            seed=7,
            update_index=0,
        )


def test_v8_cross_track_matching_handles_a_case_where_naive_rotation_fails() -> None:
    _, _, t2 = _inputs(batch=5)
    track_ids: list[Hashable] = ["a", "b", "b", "c", "c"]
    _, permutation = deterministic_shuffled_future(t2, track_ids=track_ids, seed=19, update_index=4)
    assert all(index != int(permutation[index]) for index in range(t2.shape[0]))
    assert all(
        track_ids[index] != track_ids[int(permutation[index])] for index in range(t2.shape[0])
    )
    naive = torch.roll(torch.arange(t2.shape[0]), shifts=1)
    assert any(track_ids[index] == track_ids[int(naive[index])] for index in range(t2.shape[0]))


def test_v8_shuffled_trainer_requires_track_ids_but_matched_trainer_does_not() -> None:
    t0, t1, t2 = _inputs(batch=3)
    shuffled = ScientificRecoveryV8JEPATrainer(
        CausalScaleJEPAV8(_endpoint()),
        ScientificRecoveryV8JEPATrainerConfig(total_updates=1, shuffled_future=True),
    )
    with pytest.raises(ValueError, match="track_ids"):
        shuffled.step(t0, t1, t2)

    matched = ScientificRecoveryV8JEPATrainer(
        CausalScaleJEPAV8(_endpoint()), ScientificRecoveryV8JEPATrainerConfig(total_updates=1)
    )
    result = matched.step(t0, t1, t2)
    assert result["permutation"] is None


def test_v8_d4_checkpoint_signs_exact_cross_track_compute_contract() -> None:
    trainer = ScientificRecoveryV8JEPATrainer(
        CausalScaleJEPAV8(_endpoint()),
        ScientificRecoveryV8JEPATrainerConfig(total_updates=2, shuffled_future=True, seed=23),
    )
    state = trainer.checkpoint_state()
    manifest = state["compute_manifest"]
    assert state["compute_manifest_sha256"]
    assert manifest["uses_labels"] is False
    assert manifest["d4_future_pairing"] == {
        "cross_track": True,
        "deterministic_seed_and_update": True,
        "no_fixed_points": True,
        "target_marginal_preserved": True,
    }
    assert manifest["matched_compute"] == {
        "augmentations": "identical",
        "batches": "identical",
        "ema_schedule": "identical",
        "masking": "identical",
        "model": "identical",
        "optimizer": "identical",
        "outer_train_pool": "identical",
        "update_count": "identical",
    }


def test_v8_fails_closed_on_a_single_collapsed_view() -> None:
    model = CausalScaleJEPAV8(
        _endpoint(), CausalScaleJEPAV8Config(collapse_patience=1, collapse_fraction_threshold=0.80)
    )
    trainer = ScientificRecoveryV8JEPATrainer(
        model, ScientificRecoveryV8JEPATrainerConfig(total_updates=1)
    )
    scalar = torch.tensor(0.0)
    output = CausalScaleJEPAV8Output(
        predicted_dense_features=scalar,
        target_dense_features=scalar,
        predicted_global_token=scalar,
        target_global_token=scalar,
        predicted_foreground_logits=scalar,
        target_foreground_logits=scalar,
        losses={name: scalar for name in JEPA_VIEW_NAMES},
        health={
            "dense": {"collapsed_dimension_fraction": 0.0},
            "global": {"collapsed_dimension_fraction": 0.0},
            "foreground": {"collapsed_dimension_fraction": 1.0},
        },
    )
    with pytest.raises(RuntimeError, match="foreground"):
        trainer._enforce_health(output)


def test_v8_checkpoint_restores_target_and_rng_invariants(tmp_path: pytest.TempPathFactory) -> None:
    model = CausalScaleJEPAV8(_endpoint())
    trainer = ScientificRecoveryV8JEPATrainer(
        model, ScientificRecoveryV8JEPATrainerConfig(total_updates=2, seed=11)
    )
    t0, t1, t2 = _inputs()
    trainer.step(t0, t1, t2)
    path = tmp_path / "v8_jepa.pt"
    trainer.save_checkpoint(path)

    resumed = ScientificRecoveryV8JEPATrainer(
        CausalScaleJEPAV8(_endpoint()),
        ScientificRecoveryV8JEPATrainerConfig(total_updates=2, seed=11),
    )
    resumed.load_checkpoint(path)
    assert resumed.update_count == 1
    assert not resumed.model.target_encoder.training
    assert not any(
        parameter.requires_grad for parameter in resumed.model.target_encoder.parameters()
    )
