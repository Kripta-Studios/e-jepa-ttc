from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from e_jepa_ttc.detection.campaign_training import (
    CONFIG_SCHEMA,
    AugmentationConfig,
    CampaignConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizationConfig,
    ResourceConfig,
    ResumablePermutationSampler,
    UpdateLRScheduler,
    canonical_sha256,
    dataset_cache_kwargs,
    inherit_selected_checkpoint,
    load_campaign_checkpoint,
    load_campaign_config,
    optimizer_parameter_groups,
    run_update_loop,
    save_campaign_checkpoint,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_against_metadata,
    validate_population_audit,
    validate_stop_after_update,
)


class TinyCampaignModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.backbone(value)))


def config() -> CampaignConfig:
    zeros = "0" * 64
    return CampaignConfig(
        schema=CONFIG_SCHEMA,
        experiment_id="toy",
        seed=7,
        data=DataConfig(
            root="data",
            manifest="manifest.json",
            split="split.json",
            train_role="train",
            selection_role="selection",
            expected_dataset_revision="a" * 40,
            expected_manifest_sha256=zeros,
            expected_split_sha256=zeros,
            full_selection_tokens="full.json",
            expected_full_selection_sha256=zeros,
            cache_capacity=4,
            label_cache_sequences=24,
        ),
        model=ModelConfig(
            backbone="resnet50",
            num_classes=8,
            queries=50,
            group_num=11,
            predict_velocity=True,
            velocity_loss_coef=0.1,
            width=640,
            height=384,
            backbone_weights="weights.pt",
            backbone_weights_sha256=zeros,
            source_root="source",
            portable_root="portable",
        ),
        optimization=OptimizationConfig(
            updates=8,
            checkpoint_interval=2,
            batch_size=2,
            precision="fp32",
            matmul_precision="highest",
            learning_rate=0.01,
            backbone_lr_multiplier=0.1,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            scheduler="warmup_cosine",
            warmup_fraction=0.25,
            minimum_lr_ratio=0.05,
        ),
        augmentation=AugmentationConfig(0.0, 0.0, 0.0),
        evaluation=EvaluationConfig(
            "default", (4, 8), (8,), "dev_AP101_supported_classes", "max", True
        ),
        resources=ResourceConfig(3600.0, 1, 1.0, 1.0),
    )


def population_fixture(tmp_path: Path) -> tuple[dict[str, object], CampaignConfig, Path, Path]:
    root = tmp_path / "dataset"
    labels = {
        "train-sequence": root / "data/train/train-sequence/labels.parquet",
        "selection-sequence": root / "data/train/selection-sequence/labels.parquet",
    }
    for sequence, path in labels.items():
        path.parent.mkdir(parents=True)
        path.write_bytes(f"labels:{sequence}".encode())
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    manifest_path = campaign_dir / "manifest.json"
    cfg = replace(
        config(),
        data=replace(
            config().data,
            root=str(root),
            manifest=str(manifest_path),
            expected_split_sha256="d" * 64,
        ),
    )
    manifest: dict[str, object] = {
        "dataset_revision": cfg.data.expected_dataset_revision,
        "source_media_sha256": "b" * 64,
        "candidate_split_sha256": cfg.data.expected_split_sha256,
        "roles": {
            "train": {
                "sample_tokens": ("train-token",),
                "sequence_ids": ("train-sequence",),
            },
            "selection": {
                "sample_tokens": ("selection-token",),
                "sequence_ids": ("selection-sequence",),
            },
        },
        "historical_panels": {"pilot_selection256": {"sample_tokens": ("selection-token",)}},
    }
    inventory = Path(__file__).parents[2] / "data/manifests/eap_detection_public_files_v1.json"
    audit = {
        "schema": "eap_detection_population_audit_v2",
        "dataset_revision": cfg.data.expected_dataset_revision,
        "source_split": "train",
        "source_media_sha256": manifest["source_media_sha256"],
        "candidate_split_sha256": cfg.data.expected_split_sha256,
        "evaluation_protocol_sha256": (
            "10da30cf8f08247d48d975f3650cb02f1b43375fe0e841a5389dfb3b5d7b4a5e"
        ),
        "pinned_inventory_sha256": sha256_file(inventory),
        "label_files": [
            {
                "path": f"data/train/{sequence}/labels.parquet",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "verified",
            }
            for sequence, path in labels.items()
        ],
    }
    audit_path = campaign_dir / "population_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["population_audit_sha256"] = canonical_sha256(audit)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, cfg, audit_path, labels["train-sequence"]


def optimizer_and_scheduler(
    model: TinyCampaignModel, cfg: CampaignConfig
) -> tuple[torch.optim.Optimizer, UpdateLRScheduler]:
    groups, ledger = optimizer_parameter_groups(model, cfg.optimization)
    assert ledger["groups"]["backbone"]["parameter_tensors"] == 2
    optimizer = torch.optim.AdamW(groups)
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=cfg.optimization.updates,
        kind=cfg.optimization.scheduler,
        warmup_fraction=cfg.optimization.warmup_fraction,
        minimum_ratio=cfg.optimization.minimum_lr_ratio,
    )
    return optimizer, scheduler


def test_shipped_configs_are_strict_and_activate_r0_r1_r2() -> None:
    root = Path(__file__).parents[2]
    loaded = {
        name: load_campaign_config(root / f"configs/detection_campaign/{name}.yaml")
        for name in ("rgb_r0_d4096_seed7", "rgb_r1_d4096_seed7", "rgb_r2_d4096_seed7")
    }
    assert loaded["rgb_r0_d4096_seed7"].optimization.scheduler == "constant"
    assert loaded["rgb_r1_d4096_seed7"].optimization.warmup_fraction == pytest.approx(0.05)
    assert loaded["rgb_r1_d4096_seed7"].optimization.backbone_lr_multiplier == 1.0
    assert loaded["rgb_r2_d4096_seed7"].optimization.backbone_lr_multiplier == 0.1
    assert all(item.optimization.updates == 8192 for item in loaded.values())
    assert all(
        dataset_cache_kwargs(item) == {"max_label_sequences": 24} for item in loaded.values()
    )
    assert all(
        item.evaluation.updates == (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
        for item in loaded.values()
    )
    assert all(item.evaluation.full_selection_updates == (4096, 8192) for item in loaded.values())


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["optimization"]["ignored_option"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*ignored_option"):
        load_campaign_config(path)


def test_training_seed_does_not_change_frozen_data_population(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema": "eap_detection_campaign_manifest_v1",
        "dataset_revision": "a" * 40,
        "source_split": "train",
        "source_media_sha256": "b" * 64,
        "candidate_split_sha256": sha256_file(split_path),
        "population_audit_sha256": "c" * 64,
        "sampling_seed": 7,
        "roles": {
            "train": {"sample_tokens": ["t:1"], "sequence_ids": ["t"]},
            "selection": {"sample_tokens": ["s:1"], "sequence_ids": ["s"]},
        },
        "historical_panels": {
            "pilot_selection256": {"sample_tokens": ["s:1"], "source": "fixture"}
        },
        "sampling": {},
        "limitations": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = config()
    cfg = replace(
        cfg,
        seed=13,
        data=replace(
            cfg.data,
            manifest=str(manifest_path),
            split=str(split_path),
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
        ),
    )
    assert validate_campaign_manifest(cfg)["sampling_seed"] == 7


def test_population_audit_rejects_tampered_audit(tmp_path: Path) -> None:
    manifest, cfg, audit_path, _ = population_fixture(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_split"] = "test"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        validate_population_audit(manifest, cfg)


def test_population_audit_rejects_tampered_label_bytes(tmp_path: Path) -> None:
    manifest, cfg, _, label_path = population_fixture(tmp_path)
    label_path.write_bytes(b"changed-label-bytes")
    with pytest.raises(ValueError, match="Label bytes differ"):
        validate_population_audit(manifest, cfg)


def test_full_selection_rejects_token_owned_by_other_sequence(tmp_path: Path) -> None:
    import pandas as pd

    manifest, cfg, audit_path, _ = population_fixture(tmp_path)
    media_path = Path(cfg.data.root) / "data/train.parquet"
    pd.DataFrame(
        [
            {"sample_token": "train-token", "sequence_id": "train-sequence", "split": "train"},
            {
                "sample_token": "selection-token",
                "sequence_id": "selection-sequence",
                "split": "train",
            },
            {"sample_token": "foreign-token", "sequence_id": "train-sequence", "split": "train"},
        ]
    ).to_parquet(media_path, index=False)
    media_sha = sha256_file(media_path)
    split_path = tmp_path / "split.json"
    split = {
        "dataset_revision": cfg.data.expected_dataset_revision,
        "media_sha256": media_sha,
        "assignments": {
            "train": ["train-sequence"],
            "selection": ["selection-sequence"],
        },
    }
    split_path.write_text(json.dumps(split), encoding="utf-8")
    split_sha = sha256_file(split_path)
    full_path = tmp_path / "full.json"
    full = {
        "schema": "eap_detection_full_selection_v1",
        "dataset_revision": cfg.data.expected_dataset_revision,
        "candidate_split_sha256": split_sha,
        "source_media_sha256": media_sha,
        "role": "selection",
        "sequence_ids": ["selection-sequence"],
        "sample_tokens": ["selection-token", "foreign-token"],
    }
    full_path.write_text(json.dumps(full), encoding="utf-8")
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            split=str(split_path),
            expected_split_sha256=split_sha,
            full_selection_tokens=str(full_path),
            expected_full_selection_sha256=sha256_file(full_path),
        ),
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_media_sha256"] = media_sha
    audit["candidate_split_sha256"] = split_sha
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["source_media_sha256"] = media_sha
    manifest["candidate_split_sha256"] = split_sha
    manifest["population_audit_sha256"] = canonical_sha256(audit)
    with pytest.raises(ValueError, match="Full-selection tokens"):
        validate_manifest_against_metadata(manifest, cfg)


def test_sampler_resume_starts_at_exact_next_index() -> None:
    full = ResumablePermutationSampler(5, 7)
    expected = [full.next_batch(3) for _ in range(5)]
    partial = ResumablePermutationSampler(5, 7)
    observed = [partial.next_batch(3) for _ in range(2)]
    state = partial.state_dict()
    resumed = ResumablePermutationSampler(5, 7)
    resumed.load_state_dict(state)
    observed.extend(resumed.next_batch(3) for _ in range(3))
    assert observed == expected


def test_execution_stop_segments_preserve_nominal_update_arithmetic() -> None:
    assert validate_stop_after_update(4, start_update=0, maximum_update=8) == 4
    assert validate_stop_after_update(8, start_update=4, maximum_update=8) == 8
    with pytest.raises(ValueError, match="after resume"):
        validate_stop_after_update(4, start_update=4, maximum_update=8)
    with pytest.raises(ValueError, match="nominal budget"):
        validate_stop_after_update(9, start_update=4, maximum_update=8)


def test_warmup_cosine_is_update_indexed_and_reaches_floor() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([{"params": [parameter], "lr": 2.0}])
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=10,
        kind="warmup_cosine",
        warmup_fraction=0.2,
        minimum_ratio=0.05,
    )
    assert scheduler.apply(1) == pytest.approx([1.0])
    assert scheduler.apply(2) == pytest.approx([2.0])
    assert scheduler.apply(10) == pytest.approx([0.1])


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    cfg = config()
    protocol_sha = "f" * 64
    provenance = {"code_sha256": "e" * 64, "effective_model_sha256": "d" * 64}
    progress = {
        "training_seconds": 3.0,
        "empty_target_exposures": 0,
        "valid_velocity_object_exposures": 4,
    }
    members = {"sequence:frame": "a" * 64}

    def initialize() -> tuple[
        TinyCampaignModel,
        torch.optim.Optimizer,
        UpdateLRScheduler,
        ResumablePermutationSampler,
    ]:
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        model = TinyCampaignModel()
        optimizer, scheduler = optimizer_and_scheduler(model, cfg)
        return model, optimizer, scheduler, ResumablePermutationSampler(7, cfg.seed)

    def runner(
        model: TinyCampaignModel, optimizer: torch.optim.Optimizer
    ) -> Callable[[int, Sequence[int], Sequence[float]], tuple]:
        def step(update: int, indices: Sequence[int], _lrs: Sequence[float]) -> tuple:
            optimizer.zero_grad(set_to_none=True)
            noise = random.random() + float(np.random.random()) + float(torch.rand(()))
            x = torch.tensor(indices, dtype=torch.float32).unsqueeze(1) / 7
            loss = (model(x).mean() - noise / 10) ** 2
            loss.backward()
            optimizer.step()
            return update, tuple(indices), float(loss.detach())

        return step

    full_model, full_optimizer, full_scheduler, full_sampler = initialize()
    full_records = run_update_loop(
        start_update=0,
        maximum_update=8,
        batch_size=2,
        sampler=full_sampler,
        scheduler=full_scheduler,
        step=runner(full_model, full_optimizer),
    )

    first_model, first_optimizer, first_scheduler, first_sampler = initialize()
    first_records = run_update_loop(
        start_update=0,
        maximum_update=3,
        batch_size=2,
        sampler=first_sampler,
        scheduler=first_scheduler,
        step=runner(first_model, first_optimizer),
    )
    checkpoint = tmp_path / "last.pt"
    save_campaign_checkpoint(
        checkpoint,
        model=first_model,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        sampler=first_sampler,
        config=cfg,
        protocol_sha256=protocol_sha,
        runtime_provenance=provenance,
        update=3,
        best_metric=None,
        progress=progress,
        data_member_sha256=members,
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    model, optimizer, scheduler, sampler = initialize()
    update, best, restored_progress, member_hashes = load_campaign_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=cfg,
        protocol_sha256=protocol_sha,
        runtime_provenance=provenance,
    )
    assert (update, best, restored_progress, member_hashes) == (
        3,
        None,
        progress,
        members,
    )
    resumed_records = run_update_loop(
        start_update=update,
        maximum_update=8,
        batch_size=2,
        sampler=sampler,
        scheduler=scheduler,
        step=runner(model, optimizer),
    )
    assert first_records + resumed_records == full_records
    for left, right in zip(model.parameters(), full_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_checkpoint_rejects_changed_effective_config(tmp_path: Path) -> None:
    cfg = config()
    model, optimizer, scheduler, sampler = (
        TinyCampaignModel(),
        None,
        None,
        ResumablePermutationSampler(3, 7),
    )
    optimizer, scheduler = optimizer_and_scheduler(model, cfg)
    path = tmp_path / "last.pt"
    save_campaign_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=cfg,
        protocol_sha256="a" * 64,
        runtime_provenance={"code": "b" * 64},
        update=0,
        best_metric=None,
        progress={
            "training_seconds": 0.0,
            "empty_target_exposures": 0,
            "valid_velocity_object_exposures": 0,
        },
        data_member_sha256={},
    )
    changed = replace(cfg, experiment_id="different")
    with pytest.raises(ValueError, match="configuration differs"):
        load_campaign_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=changed,
            protocol_sha256="a" * 64,
            runtime_provenance={"code": "b" * 64},
        )


def test_resume_inherits_prior_selected_when_later_metric_does_not_improve(
    tmp_path: Path,
) -> None:
    cfg = config()
    provenance = {"runtime": "a" * 64}
    model = TinyCampaignModel()
    optimizer, scheduler = optimizer_and_scheduler(model, cfg)
    sampler = ResumablePermutationSampler(3, cfg.seed)
    progress = {
        "training_seconds": 12.0,
        "empty_target_exposures": 1,
        "valid_velocity_object_exposures": 2,
    }
    ancestor = tmp_path / "ancestor/checkpoints"
    for name in ("selected.pt", "last.pt"):
        save_campaign_checkpoint(
            ancestor / name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=cfg,
            protocol_sha256="f" * 64,
            runtime_provenance=provenance,
            update=4 if name == "selected.pt" else 6,
            best_metric=0.8,
            progress=progress,
            data_member_sha256={},
        )
    inherited = tmp_path / "resumed/checkpoints/selected.pt"
    receipt = inherit_selected_checkpoint(
        ancestor / "last.pt",
        inherited,
        config=cfg,
        protocol_sha256="f" * 64,
        runtime_provenance=provenance,
        best_metric=0.8,
    )
    assert receipt is not None
    assert receipt["selected_update"] == 4
    assert inherited.read_bytes() == (ancestor / "selected.pt").read_bytes()
