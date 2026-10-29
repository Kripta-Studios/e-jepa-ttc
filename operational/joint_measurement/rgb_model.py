"""Private RGB parent admission and pair construction, without recipe selection."""

from __future__ import annotations

import torch

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.range_transfer.common import read, sha

from .common import ROOT, RUN, owner


def load_parent() -> tuple[CausalScaleTTC, dict]:
    """Load only the inventoried P-only R_A5 endpoint into a private CPU instance."""
    directory = owner() / "fits/R_A5"
    path = directory / "checkpoint_last.pt"
    inventory = read(RUN / "RGB_INPUT_INVENTORY.json")
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    provenance = read(directory / "RUN_PROVENANCE.json")
    actual = sha(path)
    if (
        actual != inventory["parent_R_A5_sha256"]
        or actual != receipt["checkpoint_sha256"]
        or receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 30330
        or provenance["context"] != "P_only_fixed_epoch18"
        or provenance["source_identity"]["role"] != "P"
        or provenance["source_identity"]["role_manifest_sha256"] != sha(owner() / "P_MANIFEST.json")
    ):
        raise ValueError("RGB parent differs from the admitted P-only endpoint")
    recipe = read(ROOT / "configs/rgb_port/producers.json")["producers"]["R_A5"]
    cfg = CausalScaleTTCConfig(**recipe["model_config"])
    if cfg.in_channels != 3 or cfg.modality != "rgb":
        raise ValueError("RGB must retain its native three-channel encoder")
    model = CausalScaleTTC(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["completed_updates"] != 30330
        or payload["identity_sha256"] != receipt["identity_sha256"]
    ):
        raise ValueError("RGB checkpoint payload and receipt differ")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {
        "parent_sha256": actual,
        "parent_updates": 30330,
        "population": "P",
        "provenance_sha256": sha(directory / "RUN_PROVENANCE.json"),
        "recipe_source_sha256": sha(ROOT / "configs/rgb_port/producers.json"),
    }


def pair_from_parent(
    parent: CausalScaleTTC, *, correlation: bool, frozen: bool, seed: int
) -> PairMeasurement:
    """Create identical new modules for a declared seed without sharing parent storage."""
    if parent.config.in_channels != 3 or parent.config.modality != "rgb":
        raise ValueError("RGB pairs require an actual RGB parent")
    # Do not change the caller's RNG or reserve a CUDA context during CPU admission.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return PairMeasurement.from_native(
            parent, use_correlation=correlation, freeze_encoder=frozen
        )
