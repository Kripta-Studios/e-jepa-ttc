import json

import pandas as pd
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from e_jepa_ttc.data.garlttc_eap import GarlTTCBatch, GarlTTCEAPIndex
from e_jepa_ttc.models import infer_tubelet_token_geometry, pool_object_embeddings
from e_jepa_ttc.training.checkpoints import validate_external_eap_ttc_checkpoint
from e_jepa_ttc.training.eap_jepa import (
    EAPJEPATrainerConfig,
    build_eap_jepa_models,
    compute_eap_jepa_objective,
    update_eap_jepa_ema,
)
from e_jepa_ttc.training.eap_ttc import (
    EAPSignedTTCHead,
    gather_object_ttc_targets,
    pretrain_eap_jepa_ttc,
)
from e_jepa_ttc.utils.hashing import sha256_file


def module_grad_norm(module: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def test_gather_object_ttc_targets_variable_objects():
    batch = GarlTTCBatch(
        context=torch.empty(2, 21, 90, 160),
        futures=torch.empty(
            2,
            3,
            21,
            90,
            160,
        ),
        future_valid=torch.ones(
            2,
            3,
            dtype=torch.bool,
        ),
        bbox_masks=[],
        target_ttc=[
            torch.tensor(
                [0.1, 0.2],
                dtype=torch.float32,
            ),
            torch.tensor(
                [-0.3],
                dtype=torch.float32,
            ),
        ],
        sequence_ids=["seq_a", "seq_b"],
        track_ids=[
            ["a0", "a1"],
            ["b0"],
        ],
        timestamp_us=torch.tensor(
            [1, 2],
            dtype=torch.int64,
        ),
        events_paths=["a.h5", "b.h5"],
        original_bboxes=[[], []],
        transformed_bboxes=[[], []],
        context_event_counts=torch.zeros(
            2,
            dtype=torch.int64,
        ),
        future_event_counts=torch.zeros(
            2,
            3,
            dtype=torch.int64,
        ),
    )

    targets = gather_object_ttc_targets(
        batch=batch,
        object_indices=[
            (0, 0),
            (0, 1),
            (1, 0),
        ],
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        targets,
        torch.tensor(
            [0.1, 0.2, -0.3],
            dtype=torch.float32,
        ),
    )


def test_ttc_arrives_at_encoder():
    config = EAPJEPATrainerConfig(
        epochs=1,
        batch_size=1,
    )

    models = build_eap_jepa_models(
        config=config,
        device=torch.device("cpu"),
    )

    encoder = models.online_encoder

    geometry = infer_tubelet_token_geometry(
        encoder,
        input_height=90,
        input_width=160,
    )

    head = EAPSignedTTCHead(
        embed_dim=int(encoder.output_dim),
    )

    context = torch.randn(
        1,
        21,
        90,
        160,
    )

    tokens = encoder.forward_tokens(context)

    mask = torch.zeros(
        1,
        5,
        10,
        dtype=torch.bool,
    )

    mask[0, 2, 4] = True

    embeddings, _ = pool_object_embeddings(
        tokens=tokens,
        bbox_masks=[mask],
        geometry=geometry,
    )

    target = torch.tensor(
        [0.2],
        dtype=torch.float32,
    )

    prediction = head(embeddings)

    loss = F.smooth_l1_loss(
        prediction,
        target,
        beta=0.05,
    )

    loss.backward()

    assert module_grad_norm(encoder) > 0.0

    assert module_grad_norm(head) > 0.0


def test_target_without_gradient():
    config = EAPJEPATrainerConfig(
        epochs=1,
        batch_size=1,
    )

    models = build_eap_jepa_models(
        config=config,
        device=torch.device("cpu"),
    )

    context = torch.randn(1, 21, 90, 160)
    futures = torch.randn(1, 3, 21, 90, 160)
    future_valid = torch.ones(1, 3, dtype=torch.bool)

    jepa_out = compute_eap_jepa_objective(
        online_encoder=models.online_encoder,
        target_encoder=models.target_encoder,
        predictor=models.predictor,
        context=context,
        futures=futures,
        future_valid=future_valid,
        config=config,
    )
    jepa_out.loss.backward()

    assert not models.target_encoder.training

    assert all(parameter.grad is None for parameter in models.target_encoder.parameters())


def test_ssl_optimizer_step():
    config = EAPJEPATrainerConfig(
        epochs=1,
        batch_size=1,
    )

    models = build_eap_jepa_models(
        config=config,
        device=torch.device("cpu"),
    )

    optimizer = torch.optim.AdamW(
        list(models.online_encoder.parameters()) + list(models.predictor.parameters()),
        lr=1e-3,
    )

    context = torch.randn(1, 21, 90, 160)
    futures = torch.randn(1, 3, 21, 90, 160)
    future_valid = torch.ones(1, 3, dtype=torch.bool)

    jepa_out = compute_eap_jepa_objective(
        online_encoder=models.online_encoder,
        target_encoder=models.target_encoder,
        predictor=models.predictor,
        context=context,
        futures=futures,
        future_valid=future_valid,
        config=config,
    )

    loss = jepa_out.loss
    assert torch.isfinite(loss)

    loss.backward()

    assert module_grad_norm(models.online_encoder) > 0.0

    optimizer.step()
    optimizer.zero_grad()


def test_ema_direction():
    config = EAPJEPATrainerConfig(
        epochs=1,
        batch_size=1,
    )

    models = build_eap_jepa_models(
        config=config,
        device=torch.device("cpu"),
    )

    online = models.online_encoder
    target = models.target_encoder

    for p in online.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    divergence, momentum = update_eap_jepa_ema(
        target_encoder=target,
        online_encoder=online,
        optimizer_step=1,
        total_optimizer_steps=10,
        config=config,
    )

    assert 0 < momentum < 1
    assert divergence > 0.0
    assert torch.isfinite(torch.tensor(divergence))


def test_checkpoint_validates_itself(tmp_path):
    config = EAPJEPATrainerConfig(
        epochs=1,
        batch_size=1,
    )

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "result": "PASS",
            }
        ),
        encoding="utf-8",
    )

    audit_sha256 = sha256_file(audit_path)

    models = build_eap_jepa_models(
        config=config,
        device=torch.device("cpu"),
    )

    head = EAPSignedTTCHead(embed_dim=int(models.online_encoder.output_dim))

    from e_jepa_ttc.training.eap_ttc import _checkpoint
    from e_jepa_ttc.utils.io import write_structured

    index = GarlTTCEAPIndex(
        sequence_ids=["a", "b"],
        train_sequences=["a"],
        validation_sequences=["b"],
        data_sha256="d",
        annotations_sha256="a",
        join_keys_sha256="j",
        source_data_row_count=100,
        source_annotation_row_count=100,
        source_merged_row_count=100,
        selected_row_count=100,
        merged=pd.DataFrame(),
    )

    split_payload = {
        "artifact_sha256": "a" * 64,
        "inventory_artifact_sha256": "b" * 64,
    }

    split_path = tmp_path / "split.json"
    write_structured(split_path, split_payload)

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}", encoding="utf-8")

    import e_jepa_ttc.training.carla_jepa
    import e_jepa_ttc.training.checkpoints

    orig_git_commit = e_jepa_ttc.training.carla_jepa._git_commit
    orig_source_tree = e_jepa_ttc.training.carla_jepa._source_tree_hash
    orig_file_sha = e_jepa_ttc.training.checkpoints._file_sha256
    orig_artifact_hash = e_jepa_ttc.training.carla_jepa._artifact_hash
    orig_verify_hash = e_jepa_ttc.training.checkpoints.verify_artifact_hash

    try:
        e_jepa_ttc.training.carla_jepa._git_commit = lambda: "commit"
        e_jepa_ttc.training.carla_jepa._source_tree_hash = lambda: "tree"
        e_jepa_ttc.training.carla_jepa._artifact_hash = lambda p: split_payload.get(
            p.name.split("_")[0] + "_artifact_sha256",
            "b" * 64 if "inventory" in p.name else "a" * 64,
        )
        e_jepa_ttc.training.checkpoints._file_sha256 = lambda p: "f" * 64
        e_jepa_ttc.training.checkpoints.verify_artifact_hash = lambda payload: True

        cp = _checkpoint(
            encoder=models.online_encoder,
            target_encoder=models.target_encoder,
            predictor=models.predictor,
            ttc_head=head,
            epoch=1,
            role="best",
            config=config,
            inventory_path=inventory_path,
            split_path=split_path,
            garlttc_index=index,
            audit_json_path=audit_path,
            audit_json_sha256=audit_sha256,
            audit_result="PASS",
            train_context_ids_sha256="1" * 64,
            validation_context_ids_sha256="2" * 64,
            train_context_count=50,
            validation_context_count=50,
            history={"records": []},
            optimizer_state_dict={},
            scheduler_state_dict={},
            scaler_state_dict={},
            optimizer_step=1,
            best_validation_loss=0.1,
        )

        cp_path = tmp_path / "cp.pt"
        torch.save(cp, cp_path)

        validated = validate_external_eap_ttc_checkpoint(
            checkpoint_path=cp_path,
            checkpoint=cp,
            source_split_path=split_path,
        )

        assert validated["pretraining_regime"] == "eap_ttc"
    finally:
        e_jepa_ttc.training.carla_jepa._git_commit = orig_git_commit
        e_jepa_ttc.training.carla_jepa._source_tree_hash = orig_source_tree
        e_jepa_ttc.training.carla_jepa._artifact_hash = orig_artifact_hash
        e_jepa_ttc.training.checkpoints._file_sha256 = orig_file_sha
        e_jepa_ttc.training.checkpoints.verify_artifact_hash = orig_verify_hash


def test_resume_fails_with_not_implemented_error(tmp_path):
    audit_file = tmp_path / "f.json"
    audit_file.write_text("{}", encoding="utf-8")

    split_file = tmp_path / "d.json"
    split_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        NotImplementedError,
        match="Safe resume",
    ):
        pretrain_eap_jepa_ttc(
            eap_root="a",
            garlttc_root="b",
            inventory_path="c",
            split_path=split_file,
            output_dir="e",
            config=EAPJEPATrainerConfig(),
            audit_json_path=audit_file,
            audit_result="PASS",
            resume=True,
        )


def test_limit_dataset_contexts():
    from e_jepa_ttc.training.eap_ttc import limit_dataset_contexts

    class DummyDataset:
        def __init__(self):
            self.samples = [
                ("seq1", 100),
                ("seq1", 200),
                ("seq2", 100),
                ("seq2", 200),
                ("seq3", 100),
            ]
            self.selected_context_ids = [f"{s[0]}_{s[1]}" for s in self.samples]
            self.selected_context_ids_hash = None

        def __len__(self):
            return len(self.samples)

    dataset = DummyDataset()

    limit_dataset_contexts(dataset, maximum=3)

    assert len(dataset.samples) == 3
    assert dataset.samples == [
        ("seq1", 100),
        ("seq2", 100),
        ("seq3", 100),
    ]
    assert isinstance(dataset.selected_context_ids_hash, str)
    assert len(dataset.selected_context_ids_hash) == 64
