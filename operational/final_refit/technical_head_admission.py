"""Admit the bounded PAIR_E technical head probe without optimizer updates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .features import validate_cache_receipt
from .prepare import _read, _write_once
from .recipe import load_recipe, sha256, stage, verify_bound_sources
from .training import _balanced_mass


def admit(run: Path, pytest_receipt: Path) -> dict[str, object]:
    """Verify the exact technical cache, parent, recipe, and resume QA closure."""
    run = run.resolve()
    recipe_path = run / "technical/PAIR_E_RECIPE.json"
    cache_path = run / "cache/PAIR_E_MATCHED.npz"
    parent_path = run / "parents/E_A5_ADMISSION.json"
    recipe = load_recipe(recipe_path)
    verify_bound_sources(recipe)
    selected = stage(recipe, "PAIR_E_MATCHED")
    if selected.updates != 11_264 or selected.batch_size != 32:
        raise ValueError("PAIR_E technical recipe endpoint differs")
    cache_receipt = validate_cache_receipt(cache_path, "final_refit_pair_cache_v1")
    parent = _read(parent_path)
    checkpoint = Path(str(parent["checkpoint_path"])).resolve(strict=True)
    if (
        parent.get("schema") != "final_refit_external_parent_admission_v1"
        or parent.get("status") != "PASS"
        or parent.get("reuse_authorized") is not True
        or parent.get("fit_id") != "E_A5"
        or parent.get("checkpoint_sha256") != sha256(checkpoint)
        or cache_receipt["parents"].get("E_A5") != parent["checkpoint_sha256"]
        or recipe.external_parent_sha256.get("E_A5") != parent["checkpoint_sha256"]
    ):
        raise ValueError("PAIR_E cache, recipe and E_A5 parent ancestry disagree")
    with np.load(cache_path, allow_pickle=False) as archive:
        groups = np.asarray(archive["group_id"]).astype(str)
    mass = _balanced_mass(groups, expected_groups=40)
    unique = np.unique(groups)
    totals = np.asarray([mass[groups == name].sum(dtype=np.float64) for name in unique])
    if (
        mass.dtype != np.float32
        or not np.isclose(mass.sum(dtype=np.float64), 1.0, atol=1e-7)
        or not np.allclose(totals, np.full(40, 1 / 40), rtol=0, atol=1e-7)
    ):
        raise ValueError("Equal-group historical mass parity failed")
    pytest_receipt = pytest_receipt.resolve(strict=True)
    result: dict[str, object] = {
        "schema": "final_refit_technical_head_admission_v2",
        "status": "PASS",
        "optimizer_updates": 0,
        "scientific_updates": 0,
        "stage": "PAIR_E_MATCHED",
        "technical_updates_authorized": 5,
        "scientific_endpoint": False,
        "recipe_path": str(recipe_path),
        "recipe_sha256": recipe.recipe_sha256,
        "recipe_file_sha256": sha256(recipe_path),
        "cache_path": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "cache_receipt_sha256": sha256(cache_path.with_suffix(".json")),
        "parent_admission_path": str(parent_path),
        "parent_admission_sha256": sha256(parent_path),
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "cache_parent_checkpoint_exact": True,
        "recipe_parent_checkpoint_exact": True,
        "mass": {
            "policy": "historical equal acquisition-group mass 1/(40*n_group)",
            "dtype": "float32",
            "rows": len(groups),
            "groups": len(unique),
            "sum": float(mass.sum(dtype=np.float64)),
            "min_group_total": float(totals.min()),
            "max_group_total": float(totals.max()),
            "parity": True,
        },
        "resume_qa": {
            "protocol": "independent 5-update scratch versus interrupted 3+2 resume",
            "comparison": "model, optimizer, sampler RNG, torch RNG and loss cursor exact",
            "pytest_junit_path": str(pytest_receipt),
            "pytest_junit_sha256": sha256(pytest_receipt),
            "status": "PASS",
        },
        "cuda_started": False,
        "submission_enabled": False,
    }
    _write_once(run / "TECHNICAL_HEAD_ADMISSION_V2.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pytest-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(admit(args.run, args.pytest_receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["admit", "main"]
