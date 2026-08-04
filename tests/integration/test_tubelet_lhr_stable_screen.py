from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig
from e_jepa_ttc.training.tubelet_finetuning import (
    TubeletOptimizationConfig,
    build_tubelet_optimizer,
    checkpoint_is_eligible,
    prediction_health,
)
from scripts.train_e_jepa_tubelet_lhr import train_epoch


class _SyntheticTTCDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(91)
        self.inputs = torch.randn(2, 2, 2, 8, 8, generator=generator)
        self.targets = torch.tensor([-2.0, 3.0])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        return self.inputs[index], self.targets[index], "synthetic", f"sample-{index}"


def _model() -> EJEPATubeletLHR:
    return EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=8,
            patch_size=4,
            spatial_window=2,
            heads=2,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=False,
            pooling="query",
            query_count=2,
        )
    )


def _config() -> TubeletOptimizationConfig:
    return TubeletOptimizationConfig(
        backbone_learning_rate=1e-2,
        pooling_learning_rate=1e-2,
        head_learning_rate=1e-2,
        warmup_pooling_learning_rate=1e-2,
        warmup_head_learning_rate=1e-2,
        backbone_weight_decay=0.0,
        readout_weight_decay=0.0,
        readout_warmup_optimizer_steps=1,
        min_prediction_std_ratio=0.01,
        collapse_patience=2,
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(value, after[name]) for name, value in before.items())


def test_warmup_then_full_finetune_changes_expected_modules() -> None:
    torch.manual_seed(17)
    model = _model()
    optimizer, _ = build_tubelet_optimizer(model, _config())
    loader = DataLoader(_SyntheticTTCDataset(), batch_size=2, shuffle=False)

    backbone_before = _state(model.patch_embed)
    pooling_before = _state(model.query_attention)
    head_before = _state(model.ttc_head)

    warmup = train_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        precision="fp32",
        max_grad_norm=1.0,
        optimization_config=_config(),
        optimizer_step=0,
        gradient_accumulation_steps=1,
    )
    assert warmup.final_optimizer_step == 1
    assert warmup.optimization_phase == "full_finetune"
    assert not _changed(backbone_before, _state(model.patch_embed))
    assert _changed(pooling_before, _state(model.query_attention))
    assert _changed(head_before, _state(model.ttc_head))

    backbone_after_warmup = copy.deepcopy(_state(model.patch_embed))
    finetune = train_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        precision="fp32",
        max_grad_norm=1.0,
        optimization_config=_config(),
        optimizer_step=warmup.final_optimizer_step,
        gradient_accumulation_steps=1,
    )
    assert finetune.final_optimizer_step == 2
    assert _changed(backbone_after_warmup, _state(model.patch_embed))


def test_checkpoint_gate_rejects_collapse_and_accepts_variation_after_warmup() -> None:
    config = _config()
    collapsed = prediction_health([-2.0, 3.0], [1.0, 1.0])
    variable = prediction_health([-2.0, 3.0], [-1.5, 2.5])
    assert not checkpoint_is_eligible(
        score=10.0,
        health=collapsed,
        optimizer_step=2,
        config=config,
    )
    assert checkpoint_is_eligible(
        score=10.0,
        health=variable,
        optimizer_step=2,
        config=config,
    )
