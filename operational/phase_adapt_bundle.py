"""Portable adapted-head bundle over the preserved PHASE-TTC upstream graph."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_training import load_head
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor


class HeadExport(nn.Module):
    """Fixed input ABI [B,340],[B,2]; columns TTC, phase, q10 phase, q90 phase."""

    def __init__(self, head: PhaseOnlyFinal) -> None:
        super().__init__()
        self.head = head

    def forward(self, features: Tensor, widths: Tensor) -> Tensor:
        result = self.head(features, widths)
        return torch.stack((result["ttc"], result["point_phase"], result["q10"], result["q90"]), -1)


def verified_paths(root: Path) -> dict[str, Path]:
    """Resolve every bundle member within its root and verify its SHA-256."""
    root = root.resolve(strict=True)
    manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase_adapt_bundle_v1":
        raise ValueError("Unsupported adapted bundle")
    paths = {}
    for name, entry in manifest["files"].items():
        path = (root / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha(path) != entry["sha256"]:
            raise ValueError(f"Bundle component path/hash mismatch: {name}")
        paths[name] = path
    return paths


class AdaptedPredictor(FinalTransferPredictor):
    """Native source-sensor inference with a strict-loaded adapted F1 readout."""

    def __init__(self, root: Path, device: str = "cpu") -> None:
        paths = verified_paths(root)
        super().__init__(paths, device)
        self.models[HEAD] = load_head(paths["head"], file_sha(paths["head"])).to(device)

    def predict(self, row: dict) -> dict:
        result = super().predict(row)
        result["predictions"]["PHASE_ADAPT_FINAL_S7"] = result["predictions"].pop(
            "PHASE_ONLY_TRAIN40_S7"
        )
        result["model_scope"] = "all_compatible_local_refit"
        return result
