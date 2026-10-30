"""Portable adapted-head bundle over the preserved PHASE-TTC upstream graph."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from operational.evttc_transfer_audit.common import canonical_sha, file_sha
from operational.final_refit.recipe import load_recipe
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_training import load_head
from scripts.final_train40_transfer import HEAD, MODEL_IDS, FinalTransferPredictor


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


def validate_head_binding(payload: dict, manifest: dict) -> None:
    """Reject valid checkpoints belonging to a different seed, data or parent graph."""
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("model_class") != "PhaseOnlyFinal"
        or payload.get("binding", {}).get("parent") != manifest.get("parent_binding")
        or payload.get("binding", {}).get("data_receipt_sha256")
        != manifest.get("data_receipt_sha256")
        or manifest.get("new_head_seed") != 7
        or payload.get("recipe", {}).get("seed") != 7
        or payload.get("recipe") != manifest.get("adapted_recipe")
        or payload.get("selection") != "last_endpoint_fixed_budget_no_test_selection"
    ):
        raise ValueError("Adapted head binding differs from declared final seed/data/parent")


def validate_graph(paths: dict[str, Path], manifest: dict) -> None:
    """Bind copied parent bytes to the preserved seal and its original recipe."""
    required = {
        *MODEL_IDS,
        "event_normalizer",
        "rgb_normalizer",
        "MODEL_SEAL.json",
        "OOF_RECIPE.json",
    }
    if set(paths) != required | {"head", "torchscript_head"}:
        raise ValueError("Bundle component set differs from full parent graph plus adapted head")
    parent = manifest["parent_binding"]
    observed = {name: file_sha(paths[name]) for name in required}
    if observed != parent.get("components") or observed["MODEL_SEAL.json"] != parent.get(
        "model_seal_sha256"
    ):
        raise ValueError("Bundle parent component binding differs")
    seal = json.loads(paths["MODEL_SEAL.json"].read_text(encoding="utf-8"))
    if seal["seal_identity_sha256"] != canonical_sha(
        {k: v for k, v in seal.items() if k != "seal_identity_sha256"}
    ):
        raise ValueError("Original model seal changed")
    recipe = load_recipe(paths["OOF_RECIPE.json"])
    if (
        recipe.recipe_sha256 != parent["recipe_sha256"]
        or seal["recipe_sha256"] != recipe.recipe_sha256
    ):
        raise ValueError("Original recipe differs")
    expected = dict(seal["deployment_dependencies"])
    expected[HEAD] = seal["trained_endpoints"][HEAD]["checkpoint_sha256"]
    expected.update({f"{name}_normalizer": value for name, value in seal["normalizers"].items()})
    if any(observed[name] != value for name, value in expected.items()):
        raise ValueError("Parent graph differs from original seal")
    payload = torch.load(paths["head"], map_location="cpu", weights_only=False)
    validate_head_binding(payload, manifest)


class AdaptedPredictor(FinalTransferPredictor):
    """Native source-sensor inference with a strict-loaded adapted F1 readout."""

    def __init__(self, root: Path, device: str = "cpu") -> None:
        paths = verified_paths(root)
        manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
        validate_graph(paths, manifest)
        super().__init__(paths, device)
        self.models[HEAD] = load_head(paths["head"], file_sha(paths["head"])).to(device)

    def predict(self, row: dict) -> dict:
        result = super().predict(row)
        result["predictions"]["PHASE_ADAPT_FINAL_S7"] = result["predictions"].pop(
            "PHASE_ONLY_TRAIN40_S7"
        )
        result["model_scope"] = "all_compatible_local_refit"
        return result
