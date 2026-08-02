import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig
from scripts.pretrain_eap_tubelet_jepa import (
    _load_label_free_config,
)
from scripts.pretrain_eap_tubelet_jepa import (
    main as pretrain_tubelet_main,
)
from scripts.train_e_jepa_tubelet_lhr import (
    _load_pretrained,
    _metrics,
    _predict,
    load_training_spec,
    train_epoch,
)


class _TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        return self.inputs[index], self.targets[index], f"sequence-{index // 2}", str(index)


def _small_model() -> EJEPATubeletLHR:
    return EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=16,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
        )
    )


def test_event_screen_config_resolves_and_rgb_config_is_rejected() -> None:
    model, trainer, provenance = load_training_spec(
        Path("configs/experiment/e_jepa_garl_event_screen_v1.yaml").resolve()
    )
    assert model.in_channels == 21
    assert model.merge_2x2 is True
    assert trainer.max_samples_per_split == 2048
    assert provenance["event_only"] is True

    full_model, full_trainer, _ = load_training_spec(
        Path("configs/experiment/e_jepa_garl_event_full_v1.yaml").resolve()
    )
    assert full_model.embed_dim == 192
    assert full_trainer.max_samples_per_split is None
    assert full_trainer.gradient_accumulation_steps == 6
    assert full_trainer.run_scope == "full_candidate"
    assert full_trainer.require_clean_git is True

    with pytest.raises(NotImplementedError, match="RGB-E fusion is not implemented"):
        load_training_spec(Path("configs/experiment/e_jepa_garl_sota_v1.yaml").resolve())


def test_tubelet_pretraining_exposes_label_free_manifest_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        pretrain_tubelet_main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--manifest" in help_text
    assert "--eap-root" in help_text
    assert "--garlttc-root" not in help_text
    assert "--evttc" not in help_text.lower()


def test_tubelet_pretraining_fails_closed_for_missing_manifest(tmp_path: Path) -> None:
    result = pretrain_tubelet_main(
        [
            "--eap-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "missing.json"),
            "--config",
            "configs/train/eap_jepa_pretrain_v4.yaml",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 1
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("prohibited_key", ["ttc", "depth", "category", "boxes", "mask", "rgb"])
def test_tubelet_pretraining_config_rejects_nested_label_reachability(
    tmp_path: Path,
    prohibited_key: str,
) -> None:
    path = tmp_path / "nested.yaml"
    path.write_text(
        json.dumps({"trainer": {"nested": [{prohibited_key: True}]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prohibited SSL-Pure fields"):
        _load_label_free_config(path)


def test_native_predict_emits_columns_consumed_by_signed_metrics() -> None:
    torch.manual_seed(73)
    model = _small_model()
    inputs = torch.randn(4, 3, 4, 16, 16)
    targets = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    loader = DataLoader(_TinyDataset(inputs, targets), batch_size=2)

    predictions = _predict(model, loader, device=torch.device("cpu"))
    metrics = _metrics(predictions)

    assert list(predictions.columns) == [
        "sample_token",
        "sequence_id",
        "target_ttc_s",
        "prediction_ttc_s",
    ]
    assert torch.isfinite(torch.tensor(metrics["paper_MiD_overall"]))


def test_train_epoch_updates_native_signed_ttc_head() -> None:
    torch.manual_seed(3)
    model = _small_model()
    inputs = torch.randn(2, 3, 4, 16, 16)
    targets = torch.tensor([-1.0, 1.0])
    loader = DataLoader(_TinyDataset(inputs, targets), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = dict(model.named_parameters())["ttc_head.3.weight"].detach().clone()

    loss = train_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        precision="fp32",
        max_grad_norm=1.0,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, dict(model.named_parameters())["ttc_head.3.weight"].detach())


def test_query_pooling_preserves_output_dtype_under_bfloat16_autocast() -> None:
    model = _small_model()
    inputs = torch.randn(2, 3, 4, 16, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(inputs)

    assert output.ttc_mean_seconds.dtype == torch.bfloat16
    assert torch.isfinite(output.ttc_mean_seconds).all()


def test_exact_backbone_only_pretrained_load_rejects_partial_or_task_state(tmp_path: Path) -> None:
    source = _small_model()
    target = _small_model()
    original_ttc = target.ttc_head[3].weight.detach().clone()
    checkpoint = {
        "artifact_type": "dense_level_dynamics_jepa_checkpoint_v1",
        "online_encoder_state_dict": source.backbone_state_dict(),
        "online_encoder_config": source.backbone_structural_config(),
    }
    path = tmp_path / "exact.pt"
    torch.save(checkpoint, path)

    report = _load_pretrained(target, path)

    assert report["used"] is True
    assert report["transferred_keys"] == sorted(source.backbone_state_dict())
    assert torch.equal(target.ttc_head[3].weight, original_ttc)
    for key, value in source.backbone_state_dict().items():
        assert torch.equal(target.backbone_state_dict()[key], value)

    task_state = dict(source.backbone_state_dict())
    task_state["ttc_head.0.weight"] = torch.ones(1)
    checkpoint["online_encoder_state_dict"] = task_state
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Exact backbone transfer rejected"):
        _load_pretrained(target, path)

    partial = source.backbone_state_dict()
    partial.pop(next(iter(partial)))
    checkpoint["online_encoder_state_dict"] = partial
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="missing"):
        _load_pretrained(target, path)

    shape_mismatch = source.backbone_state_dict()
    shape_key = next(iter(shape_mismatch))
    shape_mismatch[shape_key] = shape_mismatch[shape_key][:-1]
    checkpoint["online_encoder_state_dict"] = shape_mismatch
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="shape_mismatches"):
        _load_pretrained(target, path)

    checkpoint["online_encoder_state_dict"] = source.backbone_state_dict()
    mismatched_config = dict(source.backbone_structural_config())
    mismatched_config["patch_size"] = 8
    checkpoint["online_encoder_config"] = mismatched_config
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="structural config mismatch"):
        _load_pretrained(target, path)
