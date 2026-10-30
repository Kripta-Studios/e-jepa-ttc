"""Fail-closed migration contracts for the admitted evaluation-only change."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "migration",
    Path(__file__).resolve().parents[2] / "scripts/migrate_runpod_evaluation_checkpoint.py",
)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def pair() -> tuple[dict, dict]:
    old_code = {migration.RUNNER: migration.OLD_RUNNER, "model.py": "1" * 64}
    new_code = {**old_code, migration.RUNNER: migration.NEW_RUNNER, **migration.ADDED}
    rows = [
        {
            "config": {"seed": 7},
            "code_sha256": code,
            "code_sha256_manifest": migration.canonical(code),
        }
        for code in [old_code, new_code]
    ]
    return rows[0], rows[1]


def test_admits_only_the_exact_evaluation_revision() -> None:
    old, new = pair()
    migration.verify_transition(old, new)


@pytest.mark.parametrize("change", ["config", "model", "helper", "extra"])
def test_rejects_other_runtime_changes(change: str) -> None:
    old, new = pair()
    if change == "config":
        new["config"]["seed"] = 13
    else:
        key = {"model": "model.py", "helper": next(iter(migration.ADDED)), "extra": "hidden.py"}[
            change
        ]
        new["code_sha256"][key] = "2" * 64
        new["code_sha256_manifest"] = migration.canonical(new["code_sha256"])
    with pytest.raises(ValueError):
        migration.verify_transition(old, new)


def test_fingerprint_excludes_only_runtime_identity() -> None:
    original = {
        "runtime_provenance": {"x": 1},
        "runtime_provenance_sha256": "old",
        "model": {"weight": torch.arange(4.0)},
        "sampler": {"cursor": 4096},
    }
    migrated = copy.deepcopy(original)
    migrated["runtime_provenance"] = {"x": 2}
    migrated["runtime_provenance_sha256"] = "new"
    assert migration.state_digest(original) == migration.state_digest(migrated)
    migrated["model"]["weight"][0] = 1.0
    assert migration.state_digest(original) != migration.state_digest(migrated)
