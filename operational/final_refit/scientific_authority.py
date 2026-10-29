"""Immutable prospective authority for scientific OOF fitting and its consumers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared

from .recipe import canonical_sha256, sha256


def load_config(run: Path) -> dict[str, Any]:
    """Verify the fixed configuration, admitted historical code and prospective code."""
    path = run / "SCIENTIFIC_CONFIG.json"
    value = read_json_shared(path)
    if value.get("schema") != "scientific_oof_execution_v1" or canonical_sha256(
        {k: v for k, v in value.items() if k != "config_sha256"}
    ) != value.get("config_sha256"):
        raise ValueError("Scientific OOF configuration changed")
    for category in ("input_sha256", "source_sha256"):
        bindings = value.get(category)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(f"Missing scientific {category}")
        for bound_path, digest in bindings.items():
            if sha256(Path(bound_path)) != digest:
                raise ValueError(f"Scientific OOF binding changed: {bound_path}")
    for key in ("fold_contract", "config", "rgb_source", "teacher", "recipe"):
        required = Path(value[key]).resolve(strict=True)
        if value["input_sha256"].get(str(required)) != sha256(required):
            raise ValueError(f"Scientific configuration omits a required input binding: {key}")
    old = read_json_shared(Path(value["active_rgb_runtime"]) / "PRODUCER_PLAN.json")
    repository = Path(__file__).resolve().parents[2]
    for name, digest in old["executed_source_sha256"].items():
        if sha256(repository / name) != digest:
            raise ValueError(f"Historical producer source drift: {name}")
    if sha256(Path(value["config"])) != old["input_sha256"]["producer_config"]:
        raise ValueError("OOF producer recipe differs from the frozen family")
    return value


def execution_binding(run: Path) -> dict[str, str]:
    """Return a receipt/checkpoint binding only after validating the entire authority."""
    load_config(run)
    path = (run / "SCIENTIFIC_CONFIG.json").resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def verify_execution_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the actual scientific recipe of a producer or cached-head checkpoint."""
    binding = identity.get("execution_authority")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Scientific checkpoint lacks execution authority")
    path = Path(binding["path"])
    if path.name != "SCIENTIFIC_CONFIG.json" or sha256(path) != binding["sha256"]:
        raise ValueError("Scientific checkpoint authority changed")
    config = load_config(path.parent)
    if "recipe_file_sha256" in identity and identity["recipe_file_sha256"] != sha256(
        Path(config["recipe"])
    ):
        raise ValueError("Scientific head recipe is not the registered recipe")
    if "producer_recipe" in identity:
        recipe = identity["producer_recipe"]
        if recipe.get("seed") != 7 or recipe.get("epochs") != 18:
            raise ValueError("Scientific producer changed seed or exposure")
    return config
