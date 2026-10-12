"""Frozen R_CTX anchor with an equal-capacity event versus zero-event control."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc

BLOCK = 170
PREDICTIVE = 165


class RGBAnchor(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"EVENT", "ZERO"}:
            raise ValueError("Mode must be EVENT or ZERO")
        self.mode = mode
        self.mlp = nn.Sequential(
            nn.Linear(2 * BLOCK, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
        )
        self.output = nn.Linear(160, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def inputs(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 2 * BLOCK:
            raise ValueError("Expected [B,340] frozen state features")
        if self.mode == "EVENT":
            return features
        result = features.clone()
        result[:, BLOCK : BLOCK + PREDICTIVE] = 0
        return result

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        if rgb_width_logits.shape != (len(features), 2):
            raise ValueError("Expected R_CTX width logits [B,2]")
        correction = self.output(self.mlp(self.inputs(features)))
        location = features[:, 160] + 0.03 * correction[:, 0]
        widths = 0.03 * functional.softplus(rgb_width_logits + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }


def frozen_state(
    model: nn.Module, features: Tensor, timing: Tensor, valid: Tensor, experts: Tensor
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Observe the actual final hidden state without replacing the frozen GRU."""
    captured: dict[str, Tensor] = {}

    def hidden_hook(module: nn.Module, inputs: tuple) -> None:
        captured["hidden"] = inputs[0].detach().clone()

    def width_hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
        captured["width_logits"] = output.detach().clone()

    hooks = [
        model.get_submodule("location").register_forward_pre_hook(hidden_hook),
        model.get_submodule("width").register_forward_hook(width_hook),
    ]
    try:
        with torch.no_grad():
            output = model(features, timing, valid, experts)
    finally:
        for hook in hooks:
            hook.remove()
    block = torch.cat(
        [
            captured["hidden"],
            output["raw_location"][:, None],
            output["point_phase"][:, None],
            experts,
            timing[:, -1],
            valid.sum(1, keepdim=True).to(features.dtype) / 8,
        ],
        -1,
    )
    assert block.shape == (len(features), BLOCK)
    return block, captured["width_logits"], output
