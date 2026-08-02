from __future__ import annotations

import json

import pytest
import torch

from e_jepa_ttc.losses.level_dynamics_jepa import LevelDynamicsLossConfig, ObjectiveArm
from e_jepa_ttc.models.dense_level_dynamics_jepa import DenseLevelDynamicsConfig
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHRConfig
from e_jepa_ttc.training.eap_highres_jepa import (
    LABEL_FAMILY_PROVENANCE_FIELDS,
    EAPHighResJEPATrainer,
    EAPHighResJEPATrainerConfig,
    LabelFreeBatch,
    LabelFreeManifestProvenance,
    load_signed_label_free_manifest,
)


def _model_config() -> DenseLevelDynamicsConfig:
    return DenseLevelDynamicsConfig(
        encoder=EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=16,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
        ),
        projection_dim=16,
        predictor_dim=16,
        predictor_layers=1,
        predictor_heads=4,
        predictor_mlp_ratio=2,
        patch_query_chunk_size=4,
        max_temporal_steps=3,
        max_patches=16,
        max_horizons=3,
    )


def _batch() -> LabelFreeBatch:
    torch.manual_seed(99)
    return LabelFreeBatch(
        context_events=torch.randn(1, 3, 2, 16, 16),
        future_events=torch.randn(1, 3, 3, 2, 16, 16),
        horizon_delta_t_s=torch.tensor([[0.1, 0.2, 0.3]]),
        sequence_ids=("sequence-0",),
        track_ids=("track-0",),
        reference_timestamps_s=torch.tensor([0.0]),
        future_timestamps_s=torch.tensor([[0.1, 0.2, 0.3]]),
    )


def _provenance() -> LabelFreeManifestProvenance:
    return LabelFreeManifestProvenance(
        matched_manifest_hash="a" * 64,
        dataset_hashes={"events": "b" * 64},
        split_hash="c" * 64,
        sampler_order_hash="d" * 64,
        selection_rule="signed_block_order_v1",
        label_family_provenance={key: False for key in LABEL_FAMILY_PROVENANCE_FIELDS},
    )


def test_label_key_rejection_and_nce_preflight_happen_before_optimizer_step() -> None:
    with pytest.raises(ValueError, match="prohibited label-family key"):
        LabelFreeBatch.from_mapping(
            {
                "context_events": torch.randn(1, 3, 2, 16, 16),
                "future_events": torch.randn(1, 3, 3, 2, 16, 16),
                "horizon_delta_t_s": torch.ones(1, 3),
                "sequence_ids": ["s"],
                "track_ids": ["t"],
                "reference_timestamps_s": torch.zeros(1),
                "future_timestamps_s": torch.ones(1, 3),
                "ttc_seconds": torch.ones(1),
            }
        )

    config = EAPHighResJEPATrainerConfig(
        model=_model_config(),
        loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL_DYNAMICS_NCE),
        total_updates=3,
    )
    trainer = EAPHighResJEPATrainer(config)
    invalid = _batch()
    invalid = LabelFreeBatch(
        **{
            **invalid.__dict__,
            "future_timestamps_s": torch.tensor([[0.1, 0.1, 0.1]]),
        }
    )
    with pytest.raises(RuntimeError, match="NCE preflight failed"):
        trainer.train_step(invalid)
    assert trainer.update_count == 0


def test_label_free_batch_preserves_timestamp_precision_and_rejects_invalid_times() -> None:
    batch = _batch()
    moved = batch.to(torch.device("cpu"))

    assert moved.horizon_delta_t_s.dtype == torch.float64
    assert moved.reference_timestamps_s.dtype == torch.float64
    assert moved.future_timestamps_s.dtype == torch.float64

    with pytest.raises(ValueError, match="finite positive"):
        LabelFreeBatch(
            **{
                **batch.__dict__,
                "horizon_delta_t_s": torch.tensor([[0.1, 0.0, float("nan")]]),
            }
        )
    with pytest.raises(ValueError, match="timestamps must be finite"):
        LabelFreeBatch(
            **{
                **batch.__dict__,
                "reference_timestamps_s": torch.tensor([float("inf")]),
            }
        )


def test_nce_uses_desired_future_times_and_excludes_context_invalid_anchors() -> None:
    trainer = EAPHighResJEPATrainer(
        EAPHighResJEPATrainerConfig(
            model=_model_config(),
            loss=LevelDynamicsLossConfig(
                objective=ObjectiveArm.LEVEL_DYNAMICS_NCE,
                nce_positive_tolerance_s=1e-6,
                nce_exclusion_window_s=0.01,
            ),
            total_updates=3,
        )
    )
    original = _batch()
    timed = LabelFreeBatch(
        **{
            **original.__dict__,
            "horizon_delta_t_s": torch.tensor([[5.0, 5.005, 5.1]]),
            "reference_timestamps_s": torch.tensor([5.0]),
            "future_timestamps_s": torch.tensor([[10.0, 10.005, 10.1]]),
        }
    ).to(torch.device("cpu"))
    output = trainer.model(
        timed.context_events,
        timed.future_events,
        timed.horizon_delta_t_s,
    )
    nce = trainer._nce_output(output, timed)

    # For horizon 0 the second candidate is near the desired future at 10.0 s,
    # whereas the third is a far same-track negative.  The reference is 5.0 s,
    # so this would fail if exclusion were centered on the reference timestamp.
    assert not nce.negative_mask[0, 1]
    assert nce.negative_mask[0, 2]

    context_invalid = LabelFreeBatch(
        **{
            **timed.__dict__,
            "context_valid": torch.zeros(1, 3, dtype=torch.bool),
        }
    )
    invalid_output = trainer.model(
        context_invalid.context_events,
        context_invalid.future_events,
        context_invalid.horizon_delta_t_s,
        context_valid_temporal_mask=context_invalid.context_valid,
    )
    invalid_nce = trainer._nce_output(invalid_output, context_invalid)
    assert not invalid_nce.valid_anchor_mask.any()


def test_nce_candidate_mask_contract_is_explicit_and_shape_checked() -> None:
    batch = _batch()
    valid_mask = torch.ones(1, 3, 1, 3, dtype=torch.bool)
    typed = LabelFreeBatch.from_mapping(
        {
            "context_events": batch.context_events,
            "future_events": batch.future_events,
            "horizon_delta_t_s": batch.horizon_delta_t_s,
            "sequence_ids": batch.sequence_ids,
            "track_ids": batch.track_ids,
            "reference_timestamps_s": batch.reference_timestamps_s,
            "future_timestamps_s": batch.future_timestamps_s,
            "nce_candidate_mask": valid_mask,
        }
    )
    assert typed.nce_candidate_mask is valid_mask
    assert typed.to(torch.device("cpu")).nce_candidate_mask is not None

    controlled_mask = valid_mask.clone()
    controlled_mask[0, 0, 0, 2] = False
    controlled = LabelFreeBatch(**{**batch.__dict__, "nce_candidate_mask": controlled_mask}).to(
        torch.device("cpu")
    )
    trainer = EAPHighResJEPATrainer(
        EAPHighResJEPATrainerConfig(
            model=_model_config(),
            loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL_DYNAMICS_NCE),
            total_updates=3,
        )
    )
    output = trainer.model(
        controlled.context_events,
        controlled.future_events,
        controlled.horizon_delta_t_s,
    )
    nce = trainer._nce_output(output, controlled)
    assert nce.candidate_mask[0, 0]  # Explicit positive remains even if masks change.
    assert not nce.candidate_mask[0, 2]

    with pytest.raises(ValueError, match="nce_candidate_mask"):
        LabelFreeBatch(**{**batch.__dict__, "nce_candidate_mask": torch.ones(1, 3, 3)})
    with pytest.raises(ValueError, match="nce_candidate_mask"):
        LabelFreeBatch(
            **{
                **batch.__dict__,
                "nce_candidate_mask": torch.ones(1, 3, 1, 3, dtype=torch.int64),
            }
        )


def test_nce_two_pass_preflight_is_streaming_and_rejects_one_shot_iterators() -> None:
    config = EAPHighResJEPATrainerConfig(
        model=_model_config(),
        loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL_DYNAMICS_NCE),
        total_updates=2,
    )

    class ReiterableBatches:
        def __init__(self) -> None:
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            yield _batch()

    trainer = EAPHighResJEPATrainer(config)
    source = ReiterableBatches()
    rows = trainer.train_batches(source, max_updates=1)
    assert len(rows) == 1
    assert source.iteration_count == 2

    one_shot = (_batch() for _ in range(1))
    rejected = EAPHighResJEPATrainer(config)
    with pytest.raises(TypeError, match="re-iterable batch source"):
        rejected.train_batches(one_shot, max_updates=1)
    assert rejected.update_count == 0

    level_only = EAPHighResJEPATrainer(
        EAPHighResJEPATrainerConfig(
            model=_model_config(),
            loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL),
            total_updates=1,
        )
    )
    assert len(level_only.train_batches(_batch() for _ in range(1))) == 1


def test_checkpoint_excludes_ssl_only_state_from_transfer_and_restores_core(tmp_path) -> None:
    config = EAPHighResJEPATrainerConfig(
        model=_model_config(),
        loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL),
        total_updates=3,
    )
    trainer = EAPHighResJEPATrainer(config)
    trainer.train_step(_batch())
    state = trainer.checkpoint_state(_provenance())

    transfer = state["downstream_transfer_state"]
    assert set(transfer) == {"online_encoder_state_dict", "online_encoder_config"}
    assert all(
        key.startswith(("patch_embed.", "spatial.", "merge.", "temporal.", "final_norm."))
        for key in transfer["online_encoder_state_dict"]
    )
    assert not any("predictor" in key or "target" in key or "head" in key for key in transfer)
    assert state["target_representation_state_dict"]
    assert state["predictor_state_dict"]

    path = tmp_path / "resume.pt"
    trainer.save_checkpoint(path, _provenance())
    restored = EAPHighResJEPATrainer(config)
    restored.load_checkpoint(path)
    assert restored.update_count == trainer.update_count
    assert not restored.model.target_representation.training
    assert not any(
        parameter.requires_grad for parameter in restored.model.target_representation.parameters()
    )


@pytest.mark.parametrize(
    ("arm", "expects_residual", "expects_nce", "expects_visreg"),
    [
        (ObjectiveArm.LEVEL, False, False, False),
        (ObjectiveArm.LEVEL_TEMPORAL_RESIDUAL, True, False, False),
        (ObjectiveArm.LEVEL_DYNAMICS_NCE, False, True, False),
        (ObjectiveArm.LEVEL_DYNAMICS_NCE_RESIDUAL_VISREG, False, True, True),
    ],
)
def test_all_preregistered_arms_keep_level_and_enable_only_their_approved_terms(
    arm: ObjectiveArm,
    expects_residual: bool,
    expects_nce: bool,
    expects_visreg: bool,
) -> None:
    trainer = EAPHighResJEPATrainer(
        EAPHighResJEPATrainerConfig(
            model=_model_config(),
            loss=LevelDynamicsLossConfig(objective=arm),
            total_updates=3,
        )
    )
    batch = _batch().to(torch.device("cpu"))
    output = trainer.model(
        batch.context_events,
        batch.future_events,
        batch.horizon_delta_t_s,
    )
    objective = trainer.assemble_objective(output, batch)

    assert torch.isfinite(objective.level_loss)
    assert (objective.residual_target is not None) is expects_residual
    assert (objective.nce is not None) is expects_nce
    assert (objective.visreg is not None) is expects_visreg


def test_signed_manifest_reader_is_minimal_and_fail_closed(tmp_path) -> None:
    payload = {
        "matched_manifest_hash": "m" * 64,
        "dataset_hashes": {"events": "e" * 64},
        "split_hash": "s" * 64,
        "sampler_order_hash": "o" * 64,
        "selection_rule": "block_order_v1",
        "label_family_provenance": {key: False for key in LABEL_FAMILY_PROVENANCE_FIELDS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    import hashlib

    payload["signature"] = hashlib.sha256(encoded.encode()).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, provenance = load_signed_label_free_manifest(path)
    assert provenance.matched_manifest_hash == "m" * 64
    payload["selection_rule"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        load_signed_label_free_manifest(path)
