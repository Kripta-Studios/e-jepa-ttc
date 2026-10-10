"""Explicit opt-in inference for the fixed TTC-space candidate checkpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.simplex_t_shared_route.adapter import inputs
from operational.ttc_revision.head import DirectTTCHead
from operational.ttc_revision.runtime import H8Runtime


class DirectRuntime(H8Runtime):
    """Use the same frozen producers with verified candidate heads; no labels needed.

    This opt-in runtime is not automatically promoted over H8. Its head cost is
    not the H8 head cost in the benchmark, even though the producers are shared.
    """

    def __init__(self, frozen: FrozenModels, campaign: Path, *, compact: bool = True) -> None:
        super().__init__(frozen, batch=8 if compact else 16, on_device=compact)
        contract = json.loads((campaign / "TRAINING_FREEZE.json").read_text(encoding="utf-8"))
        if (
            contract["source_sha256"]["head.py"] != digest(Path(__file__).with_name("head.py"))
            or contract["feature_manifest_sha256"]
            != frozen.bindings["normalization"]["manifest_sha256"]
        ):
            raise ValueError("candidate architecture or feature normalization identity changed")
        self.direct_heads: dict[int, DirectTTCHead] = {}
        for seed in (7, 13, 23):
            path = campaign / f"direct_seed{seed}.pt"
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if receipt["status"] != "COMPLETE" or digest(path) != receipt["sha256"]:
                raise ValueError("complete verified candidate checkpoint required")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if (
                checkpoint["contract"] != contract
                or checkpoint["seed"] != seed
                or checkpoint["update"] != contract["recipe"]["updates"]
            ):
                raise ValueError("candidate endpoint or training contract mismatch")
            model = DirectTTCHead().to(frozen.device).eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            self.direct_heads[seed] = model

    @torch.inference_mode()
    def predict_from_features(self, raw: Tensor, *, seeds: Sequence[int] = (7, 13, 23)) -> Tensor:
        """Return signed TTC seconds in requested seed order, without target access."""
        if raw.shape != (8, 17) or not seeds or any(s not in self.direct_heads for s in seeds):
            raise ValueError("eight complete observations and valid seed selection required")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("finite raw features required")
        if self.on_device:
            features = ((raw.double() - self.mean) / self.scale).float()[None]
            timing = raw.new_zeros((1, 8, 4))
            timing[0, :, 0] = (
                torch.arange(7, -1, -1, device=raw.device, dtype=torch.float64) / 20
            ).float()
            timing[0, 1:, 2] = 0.05
            valid = torch.ones((1, 8), dtype=torch.bool, device=raw.device)
            xs = (features, timing, valid)
        else:
            xs = tuple(
                x.to(raw.device)
                for x in inputs(
                    "H8_SEED7",
                    raw.cpu().numpy(),
                    np.ones(8, bool),
                    np.arange(350000, -1, -50000, dtype=np.int64),
                    0,
                    0,
                    self.frozen.mean,
                    self.frozen.scale,
                )[:3]
            )
        result = torch.cat([self.direct_heads[seed](*xs) for seed in seeds])
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite candidate output")
        return result
