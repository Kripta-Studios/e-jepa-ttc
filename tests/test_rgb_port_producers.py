from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.reproducibility import seed_everything
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.recipe import assert_matched_pair, resolved_recipe, verify_authority
from operational.rgb_port.train_producers import (
    ProducerCheckpoint,
    _epoch_order,
    _homogeneous_microbatches,
    _pair_intervals,
    _role_input_spans,
    build_producer,
    producer_features,
)

CONFIG = Path("configs/rgb_port/producers.json")
ROLE_SHA = "a" * 64


def _recipe(fit_id: str, population: int = 33):
    return resolved_recipe(
        CONFIG,
        fit_id=fit_id,
        producer_population=population,
        role_manifest_sha256=ROLE_SHA,
    )


def test_matched_recipes_change_only_modality_and_channels() -> None:
    event = _recipe("E_A5_MATCHED")
    rgb = _recipe("R_A5")
    assert_matched_pair(event, rgb)
    assert event.updates == rgb.updates == 36
    assert event.model_config["in_channels"] == 12
    assert rgb.model_config["in_channels"] == 3
    assert event.loss_config == rgb.loss_config
    assert event.teacher_policy == rgb.teacher_policy


def test_expanded_config_matches_audited_historical_authority() -> None:
    authority = Path(
        "../e-jepa-ttc-v12-efficient-context/artifacts/"
        "train40_system_20261005/TRAINING_PROTOCOL.json"
    )
    result = verify_authority(CONFIG, authority)
    assert result["status"] == "MATCHED"
    assert result["producer_ids"] == [
        "E_A5_MATCHED",
        "E_C2F_MATCHED",
        "R_A5",
        "R_C2F",
    ]


def test_rgb_forward_backward_and_label_free_feature_contract() -> None:
    recipe = _recipe("R_A5", population=32)
    model = build_producer(recipe)
    rgb = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
    delta = torch.full((1, 1), 0.1)

    features = producer_features(
        model,
        rgb,
        delta,
        anchor_us=torch.tensor([200_000], dtype=torch.int64),
        available_us=torch.tensor([200_500], dtype=torch.int64),
    )
    features["prediction_ttc"].sum().backward()

    assert features["token128"].shape == (1, 128)
    assert features["support"].shape == (1, 2)
    assert features["flow"].shape == (1,)
    assert features["point_phase"].shape == (1,)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_rgb_rejects_imagenet_normalized_sensor_input() -> None:
    model = build_producer(_recipe("R_C2F", population=32))
    values = torch.full((1, 2, 3, 16, 16), -1.0)
    try:
        producer_features(model, values, torch.full((1, 1), 0.1))
    except ValueError as error:
        assert "raw [0,1]" in str(error)
    else:
        raise AssertionError("ImageNet-normalized sensor input was accepted")


def test_event_builder_is_exact_fresh_base_model() -> None:
    recipe = _recipe("E_C2F_MATCHED", population=32)
    wrapped = build_producer(recipe)
    seed_everything(7, deterministic=True)
    direct = CausalScaleTTC(CausalScaleTTCConfig(**recipe.model_config))
    assert wrapped.state_dict().keys() == direct.state_dict().keys()
    for name, value in wrapped.state_dict().items():
        torch.testing.assert_close(value, direct.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_receipt_is_queue_compatible_without_optimizer_step(tmp_path: Path) -> None:
    recipe = _recipe("E_A5_MATCHED", population=32)
    model = build_producer(recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
    generator = torch.Generator().manual_seed(7)
    identity = {**recipe.identity, "identity_sha256": recipe.identity_sha256}
    state = ProducerCheckpoint(tmp_path / "fits" / recipe.fit_id, identity, recipe.updates)
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(32)}

    state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    restored = state.restore(model, optimizer, scheduler, generator)

    receipt = read_json_shared(state.receipt)
    assert receipt["completed_updates"] == 0
    assert receipt["fit_id"] == recipe.fit_id
    assert receipt["scientific_endpoint"] is False
    assert receipt["checkpoint_path"] == str(state.path.resolve())
    assert receipt["complete_state"] is True
    assert receipt["accumulation_index"] == 0
    assert restored["epoch"] == 1


def test_checkpoint_pointer_repairs_lagging_alias_and_receipt(tmp_path: Path) -> None:
    recipe = _recipe("E_A5_MATCHED", population=32)
    model = build_producer(recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
    generator = torch.Generator().manual_seed(7)
    identity = {**recipe.identity, "identity_sha256": recipe.identity_sha256}
    state = ProducerCheckpoint(tmp_path / "fits" / recipe.fit_id, identity, recipe.updates)
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(32)}
    state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    expected = read_json_shared(state.pointer)["checkpoint_sha256"]
    stale = state.path.with_suffix(".stale.pt")
    stale.write_bytes(b"interrupted stale alias")
    stale.replace(state.path)
    atomic_write_json(state.receipt, {"completed_updates": -1})

    state.restore(model, optimizer, scheduler, generator)

    assert read_json_shared(state.receipt)["checkpoint_sha256"] == expected
    assert state.path.stat().st_size > len(b"interrupted stale alias")


def test_effective_batch_groups_real_t2_t3_without_padding() -> None:
    source = type("Source", (), {"frame_counts": [2, 3, 2, 3, 3]})()
    groups = _homogeneous_microbatches(source, [0, 1, 2, 3, 4], 2)  # type: ignore[arg-type]
    assert groups == [[0, 2], [1, 3], [4]]
    assert sorted(index for group in groups for index in group) == list(range(5))


def test_real_rgb_pair_intervals_are_not_retargeted() -> None:
    measured = torch.tensor([[0.087, 0.113], [0.099, 0.102]], dtype=torch.float32)
    assert _pair_intervals(measured, batch_size=2, steps=3) is measured
    legacy = _pair_intervals(torch.tensor([0.1, 0.2]), batch_size=2, steps=3)
    torch.testing.assert_close(legacy, torch.tensor([[0.1, 0.1], [0.2, 0.2]]))


def test_matched_modalities_share_complete_cache_local_epoch_order() -> None:
    source = type("Source", (), {"population_size": 53890})()
    event = _epoch_order(source, torch.Generator().manual_seed(7))  # type: ignore[arg-type]
    rgb = _epoch_order(source, torch.Generator().manual_seed(7))  # type: ignore[arg-type]
    assert torch.equal(event, rgb)
    assert torch.equal(event.sort().values, torch.arange(53890))


def test_role_input_spans_come_from_original_sensor_windows(tmp_path: Path) -> None:
    index = tmp_path / "TRAIN40_INDEX.npz"
    np.savez(
        index,
        tokens=np.asarray(["a", "b"]),
        windows_us=np.asarray(
            [
                [[100, 200], [200, 300], [300, 401]],
                [[1_000, 1_100], [1_155, 1_255], [1_255, 1_355]],
            ],
            dtype=np.int64,
        ),
    )
    assert _role_input_spans(index, ["b", "a"]) == [355, 301]


def test_role_input_spans_reject_mixed_or_oversized_clock(tmp_path: Path) -> None:
    index = tmp_path / "TRAIN40_INDEX.npz"
    np.savez(
        index,
        tokens=np.asarray(["a"]),
        windows_us=np.asarray([[[0, 100], [90, 200], [200, 300]]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="one monotonic sensor clock"):
        _role_input_spans(index, ["a"])
    np.savez(
        index,
        tokens=np.asarray(["a"]),
        windows_us=np.asarray([[[0, 100], [100, 200], [200, 650_001]]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="input span"):
        _role_input_spans(index, ["a"])
