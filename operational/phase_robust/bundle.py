"""Relative-path bundle verification for the frozen upstream sensor graph."""

from __future__ import annotations

from pathlib import Path

from operational.evttc_competition.runtime import CompetitionRuntime
from operational.evttc_transfer_audit.common import canonical_sha, file_sha
from operational.final_refit.recipe import load_recipe
from operational.phase_campaign.data import read
from operational.phase_robust.runtime import FusionEnsemble, load_fusion
from scripts.final_train40_transfer import HEAD, MODEL_IDS, load_models


def verified_upstream_paths(root: Path) -> dict[str, Path]:
    """Verify original seal, recipe and every dependency before model loading."""
    root = root.resolve(strict=True)
    manifest = read(root / "BUNDLE.json")
    if manifest.get("schema") != "phase_robust_bundle_v1":
        raise ValueError("Unsupported robust bundle schema")
    required = {
        *MODEL_IDS,
        "event_normalizer",
        "rgb_normalizer",
        "MODEL_SEAL.json",
        "OOF_RECIPE.json",
    }
    if set(manifest["upstream"]) != required:
        raise ValueError("Upstream component inventory differs")
    paths = {}
    for name, item in manifest["upstream"].items():
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha(path) != item["sha256"]:
            raise ValueError("Changed or escaping upstream component")
        paths[name] = path
    parent = manifest["parent"]
    if {k: file_sha(v) for k, v in paths.items()} != parent["components"]:
        raise ValueError("Upstream parent binding differs")
    seal = read(paths["MODEL_SEAL.json"])
    if (
        canonical_sha({k: v for k, v in seal.items() if k != "seal_identity_sha256"})
        != seal["seal_identity_sha256"]
    ):
        raise ValueError("Original seal integrity differs")
    recipe = load_recipe(paths["OOF_RECIPE.json"])
    if recipe.recipe_sha256 != seal["recipe_sha256"]:
        raise ValueError("Original recipe differs")
    expected = {
        **seal["deployment_dependencies"],
        HEAD: seal["trained_endpoints"][HEAD]["checkpoint_sha256"],
        **{f"{name}_normalizer": digest for name, digest in seal["normalizers"].items()},
    }
    if any(file_sha(paths[k]) != digest for k, digest in expected.items()):
        raise ValueError("Weights or normalizers differ from original seal")
    return paths


def load_sensor_runtime(root: Path, device: str = "cpu") -> CompetitionRuntime:
    """Load the competition adapter from verified relative-path components."""
    models, normalizers = load_models(verified_upstream_paths(root), device)
    return CompetitionRuntime(models, normalizers, device=device)


def load_bundle_fusion(root: Path, variant: str = "evaluated") -> FusionEnsemble:
    """Reject a valid model swapped into a differently claimed bundle variant."""
    if variant not in ("evaluated", "all_data_refit"):
        raise ValueError("Unsupported fusion variant")
    binding = read(root / "FUSION_BINDING.json")
    if file_sha(root / "BUNDLE.json") != binding["bundle_sha256"]:
        raise ValueError("Changed upstream bundle binding")
    if file_sha(root / variant / "FINAL_FREEZE.json") != binding["fusion_freezes"][variant]:
        raise ValueError("Changed fusion variant identity")
    if set(binding["artifacts"]) != {
        "evaluated/fusion.ts",
        "all_data_refit/fusion.ts",
        "evaluated/CALIBRATION.json",
        "evaluated/CALIBRATION_SEAL.json",
    }:
        raise ValueError("Fusion artifact inventory differs")
    for relative, digest in binding["artifacts"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError("Changed fusion export or calibration binding")
    return load_fusion(root / variant)
