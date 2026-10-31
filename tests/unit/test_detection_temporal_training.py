"""CPU tests for isolated S3 training, evaluation, and continuation state."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from e_jepa_ttc.detection.campaign_training import (
    UpdateLRScheduler,
    canonical_sha256,
    load_campaign_config,
    sha256_file,
)
from e_jepa_ttc.detection.temporal_campaign import (
    TEMPORAL_CAMPAIGN_SCHEMA,
    TemporalCampaignBatch,
    TemporalCampaignConfig,
    TemporalCampaignSampler,
)
from e_jepa_ttc.detection.temporal_detector import (
    ProjectedFeatureBackend,
    ProjectedFeatureState,
    TemporalBatchContext,
    TemporalDetectorAdapter,
    TemporalFrameIdentity,
)
from e_jepa_ttc.detection.temporal_training import (
    TEMPORAL_CONTINUATION_SCHEMA,
    TEMPORAL_TRAINING_SCHEMA,
    ParentCheckpointConfig,
    TemporalRuntimeConfig,
    TemporalTrainingConfig,
    immutable_resource_contract,
    load_temporal_checkpoint,
    load_temporal_training_config,
    publish_selected_candidate,
    save_temporal_checkpoint,
    seeded_fusion,
    temporal_eval_forward,
    temporal_optimizer_parameter_groups,
    temporal_train_step,
    validate_evaluation_commit,
    validate_parent_checkpoint_payload,
    validate_temporal_parity_evidence,
)


class Backend(ProjectedFeatureBackend):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 256, 1)
        self.head = nn.Linear(256, 1)

    def extract_projected_features(self, images: torch.Tensor) -> ProjectedFeatureState:
        value = self.backbone(images)
        return ProjectedFeatureState((value, value, value, value), object())

    def decode_projected_features(
        self, state, features, calibs, targets, image_sizes, dn_args=None
    ) -> dict[str, torch.Tensor]:
        del state, calibs, targets, image_sizes, dn_args
        return {"value": self.head(features[0].mean((2, 3)))}


class Criterion(nn.Module):
    weight_dict = {"loss": 1.0}

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, outputs, targets):
        self.calls.add_(1)
        wanted = torch.stack([target["wanted"] for target in targets])
        return {"loss": (outputs["value"].flatten() - wanted).square().mean()}


def temporal_config(variant: str = "correct_past_2") -> TemporalCampaignConfig:
    return TemporalCampaignConfig(
        TEMPORAL_CAMPAIGN_SCHEMA,
        variant,
        "parent-v1",
        width=64,
        height=64,
        include_velocity=True,
    )


def batch(*, targets: bool = True, history_requires_grad: bool = False) -> TemporalCampaignBatch:
    current = TemporalFrameIdentity("s", "c")
    histories = (TemporalFrameIdentity("s", "h0"), TemporalFrameIdentity("s", "h1"))
    history = torch.randn(1, 2, 3, 2, 2, requires_grad=history_requires_grad)
    return TemporalCampaignBatch(
        current_images=torch.randn(1, 3, 2, 2, requires_grad=True),
        calibs=torch.zeros(1, 3, 4),
        targets=({"wanted": torch.tensor(0.5)},) if targets else None,
        image_sizes=torch.tensor([[2.0, 2.0]]),
        temporal_context=TemporalBatchContext(
            "parent-v1", "correct_past", (current,), (histories,)
        ),
        history_images=history,
        history_ages_us=torch.tensor([[200_000.0, 100_000.0]]),
        history_mask=torch.tensor([[True, True]]),
        history_limit=2,
        provenance={},
    )


def baseline_config():
    path = Path("output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r1.yaml")
    return load_campaign_config(path)


def full_config(variant: str = "correct_past_2") -> TemporalTrainingConfig:
    baseline = baseline_config()
    return TemporalTrainingConfig(
        TEMPORAL_TRAINING_SCHEMA,
        "fixture.yaml",
        baseline,
        temporal_config(variant),
        ParentCheckpointConfig("parent.pt", "1" * 64, 8192, "2" * 64),
        TemporalRuntimeConfig(
            "portable",
            "source.json",
            "3" * 64,
            "4" * 64,
            "parity.json",
            "7" * 64,
            "admission.json",
            "5" * 64,
            "history.json",
            "6" * 64,
        ),
    )


class Dataset:
    def __init__(self, value: TemporalCampaignBatch, variant: str = "correct_past_2") -> None:
        self.value = value
        self.current_tokens = ("0", "1", "2", "3")
        self.manifest_sha256 = "fixture"
        self.config = temporal_config(variant)

    def __len__(self) -> int:
        return 4

    def collate(self, indices):
        assert len(indices) == 1
        return self.value


def setup(variant: str = "correct_past_2"):
    torch.manual_seed(4)
    adapter = TemporalDetectorAdapter(
        Backend(), seeded_fusion(7), producer_version="parent-v1", maximum_history_age_us=500_000
    )
    criterion = Criterion()
    config = full_config(variant)
    groups, ledger = temporal_optimizer_parameter_groups(adapter, config.baseline)
    optimizer = torch.optim.AdamW(groups)
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=8192,
        kind="constant",
        warmup_fraction=0,
        minimum_ratio=1,
    )
    return adapter, criterion, optimizer, scheduler, config, ledger


def test_same_seed_initializes_all_controls_identically() -> None:
    left, right = seeded_fusion(13), seeded_fusion(13)
    assert all(
        torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict()
    )


def test_legacy_config_hash_is_unchanged_by_optional_parent_config() -> None:
    config = full_config()
    historical = asdict(config)
    historical.pop("parent_config")
    assert config.sha256 == canonical_sha256(historical)


def test_continuation_parent_is_bound_to_its_own_config_not_child_budget() -> None:
    config = full_config()
    parent_config = config.baseline
    child = replace(
        config.baseline,
        optimization=replace(config.baseline.optimization, updates=1024),
    )
    config = replace(config, baseline=child, parent_config=parent_config)
    payload = {
        "schema": "eap_detection_campaign_checkpoint_v1",
        "producer": "scripts/train_detection_campaign.py",
        "config_sha256": parent_config.sha256,
        "update": config.parent.update,
        "runtime_provenance_sha256": config.parent.runtime_provenance_sha256,
    }
    validate_parent_checkpoint_payload(payload, config)
    payload["config_sha256"] = child.sha256
    with pytest.raises(ValueError, match="configured selected"):
        validate_parent_checkpoint_payload(payload, config)


def test_continuation_config_loads_dfull_parent_with_short_child_budget(tmp_path: Path) -> None:
    source = Path("output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r1.yaml")
    parent = yaml.safe_load(source.read_text(encoding="utf-8"))
    parent["optimization"]["updates"] = 38726
    parent["evaluation"]["updates"] = [38726]
    parent["evaluation"]["full_selection_updates"] = []
    parent_path = tmp_path / "parent.yaml"
    parent_path.write_text(yaml.safe_dump(parent), encoding="utf-8")
    child = copy.deepcopy(parent)
    child["optimization"]["updates"] = 1024
    child["evaluation"]["updates"] = [512, 1024]
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    wrapper = yaml.safe_load(Path("configs/experiment/temporal_campaign_template.yaml").read_text())
    wrapper.update(
        schema=TEMPORAL_CONTINUATION_SCHEMA,
        baseline_config=str(child_path),
        parent_config=str(parent_path),
    )
    wrapper["parent"].update(update=34333, sha256="1" * 64, runtime_provenance_sha256="2" * 64)
    for field in ("source_manifest_sha256", "expected_output_tree_sha256", "parity_report_sha256"):
        wrapper["runtime"][field] = "3" * 64
    path = tmp_path / "wrapper.yaml"
    path.write_text(yaml.safe_dump(wrapper), encoding="utf-8")
    config = load_temporal_training_config(path)
    assert config.parent.update == 34333
    assert config.baseline.optimization.updates == 1024
    assert config.parent_config is not None
    assert config.parent_config.optimization.updates == 38726
    child["model"]["width"] = 320
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    with pytest.raises(ValueError, match="parent model contract"):
        load_temporal_training_config(path)


def test_optimizer_preserves_prefixed_backbone_lr_and_fusion_group() -> None:
    adapter, _, _, _, config, ledger = setup()
    groups, _ = temporal_optimizer_parameter_groups(adapter, config.baseline)
    assert groups[0]["name"] == "backbone"
    assert groups[0]["lr"] == pytest.approx(
        config.baseline.optimization.learning_rate
        * config.baseline.optimization.backbone_lr_multiplier
    )
    assert ledger["detector_and_fusion"]["tensors"] > 0


def test_train_step_updates_fusion_and_detaches_history() -> None:
    adapter, criterion, optimizer, scheduler, config, _ = setup()
    value = batch(history_requires_grad=True)
    before = {key: item.clone() for key, item in adapter.fusion.state_dict().items()}
    result = temporal_train_step(
        adapter,
        criterion,
        value,
        optimizer,
        scheduler,
        1,
        config.baseline.optimization.gradient_clip_norm,
    )
    assert result["loss_total"] >= 0
    assert value.current_images.grad is not None
    assert value.history_images is not None and value.history_images.grad is None
    assert any(
        not torch.equal(before[key], item) for key, item in adapter.fusion.state_dict().items()
    )


def test_eval_forward_is_target_free() -> None:
    adapter, *_ = setup()
    value = replace(batch(), targets=None)
    assert temporal_eval_forward(adapter, value)["value"].shape == (1, 1)
    with pytest.raises(ValueError, match="must not receive targets"):
        temporal_eval_forward(adapter, batch())


def test_checkpoint_resume_restores_all_state_and_next_sample(tmp_path: Path) -> None:
    adapter, criterion, optimizer, scheduler, config, _ = setup()
    dataset = Dataset(batch())
    sampler = TemporalCampaignSampler(dataset, seed=7)  # type: ignore[arg-type]
    sampler.next_batch(1)
    temporal_train_step(adapter, criterion, batch(), optimizer, scheduler, 1, 1.0)
    runtime = {"source": "fixture"}
    path = tmp_path / "last.pt"
    save_temporal_checkpoint(
        path,
        adapter=adapter,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        runtime_provenance=runtime,
        update=1,
        elapsed_seconds=12.5,
    )
    expected_indices, _ = sampler.next_batch(1)
    restored = setup()
    restored_sampler = TemporalCampaignSampler(Dataset(batch()), seed=7)  # type: ignore[arg-type]
    update, best, elapsed, selected = load_temporal_checkpoint(
        path,
        adapter=restored[0],
        criterion=restored[1],
        optimizer=restored[2],
        scheduler=restored[3],
        sampler=restored_sampler,
        config=restored[4],
        runtime_provenance=runtime,
    )
    actual_indices, _ = restored_sampler.next_batch(1)
    assert update == 1 and best is None and elapsed == 12.5 and selected is None
    assert actual_indices == expected_indices
    assert all(
        torch.equal(adapter.state_dict()[key], restored[0].state_dict()[key])
        for key in adapter.state_dict()
    )


def test_resume_rejects_wrong_variant_parent_or_runtime(tmp_path: Path) -> None:
    adapter, criterion, optimizer, scheduler, config, _ = setup()
    sampler = TemporalCampaignSampler(Dataset(batch()), seed=7)  # type: ignore[arg-type]
    path = tmp_path / "last.pt"
    runtime = {"training_code_hashes": {"runner": "original"}}
    save_temporal_checkpoint(
        path,
        adapter=adapter,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        runtime_provenance=runtime,
        update=0,
    )
    wrong = replace(config, temporal=temporal_config("correct_past_1"))
    with pytest.raises(ValueError, match="contract differs"):
        load_temporal_checkpoint(
            path,
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=wrong,
            runtime_provenance=runtime,
        )
    wrong_parent = replace(config, parent=replace(config.parent, sha256="9" * 64))
    with pytest.raises(ValueError, match="contract differs"):
        load_temporal_checkpoint(
            path,
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=wrong_parent,
            runtime_provenance=runtime,
        )
    with pytest.raises(ValueError, match="contract differs"):
        load_temporal_checkpoint(
            path,
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=config,
            runtime_provenance={"training_code_hashes": {"runner": "changed"}},
        )


def test_config_rejects_unknown_key_and_wrong_admission_hash(tmp_path: Path) -> None:
    baseline = Path("output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r1.yaml").resolve()
    value = {
        "schema": TEMPORAL_TRAINING_SCHEMA,
        "baseline_config": str(baseline),
        "temporal": {
            "schema": TEMPORAL_CAMPAIGN_SCHEMA,
            "variant": "current_only",
            "producer_version": "p",
            "maximum_history_age_us": 500000,
            "width": 640,
            "height": 384,
            "include_velocity": True,
            "generation": 0,
            "media_failure_policy": "fail",
        },
        "parent": {
            "path": "selected.pt",
            "sha256": "1" * 64,
            "update": 8192,
            "runtime_provenance_sha256": "2" * 64,
        },
        "runtime": {
            "portable_root": "portable",
            "source_manifest": "source.json",
            "source_manifest_sha256": "3" * 64,
            "expected_output_tree_sha256": "4" * 64,
            "parity_report": "parity.json",
            "parity_report_sha256": "7" * 64,
            "temporal_media_admission": "admission.json",
            "temporal_media_admission_sha256": "0" * 64,
            "history_media_receipt": "history.json",
            "history_media_receipt_sha256": "6" * 64,
        },
    }
    path = tmp_path / "temporal.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="not the preregistered"):
        load_temporal_training_config(path)
    value["unexpected"] = True
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_temporal_training_config(path)


def test_parity_rejects_shallow_pass_and_changed_runtime_sources() -> None:
    root = Path("output/detection_temporal/r0_update512_parity_v2/runtime").resolve()
    report = json.loads(
        Path("output/detection_temporal/r0_update512_parity_v2/report.json").read_text()
    )
    arguments = {
        "parent_sha256": report["checkpoint_sha256"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "repository_root": root,
    }
    validate_temporal_parity_evidence(report, **arguments)
    mutations = []
    missing = copy.deepcopy(report)
    del missing["comparisons"]
    mutations.append(missing)
    state_changed = copy.deepcopy(report)
    state_changed["workers"]["temporal"]["model_state_after_sha256"] = "0" * 64
    mutations.append(state_changed)
    labels_read = copy.deepcopy(report)
    labels_read["workers"]["temporal"]["label_files_read"] = True
    mutations.append(labels_read)
    decoded_missing = copy.deepcopy(report)
    decoded_missing["decoded_predictions_exact"] = False
    mutations.append(decoded_missing)
    runtime_changed = copy.deepcopy(report)
    runtime_changed["runtime_source_hashes"]["src/e_jepa_ttc/detection/temporal_detector.py"] = (
        "0" * 64
    )
    runtime_changed["runtime_source_hashes_sha256"] = canonical_sha256(
        runtime_changed["runtime_source_hashes"]
    )
    mutations.append(runtime_changed)
    for value in mutations:
        with pytest.raises(ValueError, match="parity evidence"):
            validate_temporal_parity_evidence(value, **arguments)


def test_checkpoint_commit_cannot_skip_monitor_or_full_evaluation() -> None:
    config = full_config()
    with pytest.raises(ValueError, match="monitor evaluation is incomplete"):
        validate_evaluation_commit(
            1024,
            monitor_complete=False,
            full_selection_complete=False,
            config=config,
        )
    with pytest.raises(ValueError, match="full-selection evaluation is incomplete"):
        validate_evaluation_commit(
            4096,
            monitor_complete=True,
            full_selection_complete=False,
            config=config,
        )
    assert validate_evaluation_commit(
        4096,
        monitor_complete=True,
        full_selection_complete=True,
        config=config,
    )["full_selection_complete"]


def test_checkpoint_writer_refuses_pre_evaluation_commit(tmp_path: Path) -> None:
    adapter, criterion, optimizer, scheduler, config, _ = setup()
    sampler = TemporalCampaignSampler(Dataset(batch()), seed=7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="monitor evaluation is incomplete"):
        save_temporal_checkpoint(
            tmp_path / "last.pt",
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=config,
            runtime_provenance={"training_code_hashes": {"runner": "a"}},
            update=1024,
        )
    assert not (tmp_path / "last.pt").exists()


def test_dynamic_headroom_is_excluded_from_resume_lineage() -> None:
    config = full_config()
    first = {
        "status": "PASS",
        "available_ram_bytes": 20,
        "available_vram_bytes": 30,
    }
    second = {
        "status": "PASS",
        "available_ram_bytes": 10,
        "available_vram_bytes": 15,
    }
    assert immutable_resource_contract(config, first) == immutable_resource_contract(config, second)


def test_parent_payload_must_match_exact_child_baseline_config() -> None:
    config = full_config()
    payload = {
        "schema": "eap_detection_campaign_checkpoint_v1",
        "producer": "scripts/train_detection_campaign.py",
        "config_sha256": config.baseline.sha256,
        "update": config.parent.update,
        "runtime_provenance_sha256": config.parent.runtime_provenance_sha256,
    }
    validate_parent_checkpoint_payload(payload, config)
    payload["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selected baseline"):
        validate_parent_checkpoint_payload(payload, config)


def test_last_checkpoint_rejects_uncommitted_selected_lineage(tmp_path: Path) -> None:
    adapter, criterion, optimizer, scheduler, config, _ = setup()
    sampler = TemporalCampaignSampler(Dataset(batch()), seed=7)  # type: ignore[arg-type]
    path = tmp_path / "last.pt"
    missing = tmp_path / "selected_update_000001_missing.pt"
    runtime = {"training_code_hashes": {"runner": "stable"}}
    save_temporal_checkpoint(
        path,
        adapter=adapter,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        runtime_provenance=runtime,
        update=1,
        best_metric=0.1,
        selected_lineage={
            "path": str(missing),
            "sha256": "0" * 64,
            "update": 1,
            "best_metric": 0.1,
        },
    )
    with pytest.raises(ValueError, match="selected checkpoint lineage"):
        load_temporal_checkpoint(
            path,
            adapter=adapter,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=config,
            runtime_provenance=runtime,
        )


def test_selected_publication_reuses_orphan_after_crash(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    candidate = checkpoints / ".selected_update_000001.candidate.pt"
    candidate.write_bytes(b"stable selected checkpoint")
    digest = sha256_file(candidate)
    published = publish_selected_candidate(
        candidate,
        checkpoints_dir=checkpoints,
        update=1,
        expected_sha256=digest,
    )
    assert published.name == f"selected_update_000001_{digest}.pt"
    assert not candidate.exists()

    # Replay after a crash before last.pt recorded the already-published artifact.
    candidate.write_bytes(b"stable selected checkpoint")
    replayed = publish_selected_candidate(
        candidate,
        checkpoints_dir=checkpoints,
        update=1,
        expected_sha256=digest,
    )
    assert replayed == published
    assert not candidate.exists()
    assert sha256_file(replayed) == digest
