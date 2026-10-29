"""Explicit old-to-new source compatibility for existing complete event states.

Historical freezes remain historical evidence. Only the exact, archived before
hash and the exact tested after hash in AUDIT_CODE_MIGRATION.json are accepted.
There is no generic ignore-hash flag and no checkpoint rewriting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "AUDIT_CODE_MIGRATION.json"


def _technical_binding(run: Path) -> str:
    """Retain the pre-existing, explicitly admitted four-update root-QA contract."""
    freeze = read_json_shared(run / "SOURCE_FREEZE.json")
    if (
        os.environ.get("RGB_PORT_TECHNICAL_RESUME_QA") != "1"
        or freeze.get("schema") != "rgb_port_technical_source_freeze_v1"
        or freeze.get("producer_source_identity", {}).get("technical_subset_only") is not True
    ):
        raise ValueError("Scientific continuation requires AUDIT_CODE_MIGRATION.json")
    sources = [
        *sorted(Path(__file__).parent.glob("*.py")),
        ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py",
        ROOT / "operational/rgb_port/train_producers.py",
    ]
    return canonical_sha256(
        {
            "technical_source_freeze_sha256": sha256_file(run / "SOURCE_FREEZE.json"),
            "current_sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        }
    )


def validate(run: Path) -> dict[str, Any]:
    """Verify every changed/added source and the preserved historical manifests."""
    value = read_json_shared(run / MANIFEST)
    if (
        value.get("schema") != "rgb_port_direct_code_migration_v1"
        or value.get("geometry_precision") != "bf16_unchanged"
        or value.get("event_objective_unchanged") is not True
        or value.get("identity_sha256")
        != canonical_sha256({key: item for key, item in value.items() if key != "identity_sha256"})
    ):
        raise ValueError("Invalid direct-code migration contract")
    for relative, entry in value["files"].items():
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256_file(path) != entry["after_sha256"]:
            raise ValueError(f"Unreviewed source change after migration: {relative}")
        if entry["before_sha256"] is not None:
            archived = run / "audit_fixes_20261009/before" / relative
            if sha256_file(archived) != entry["before_sha256"]:
                raise ValueError(f"Historical source archive changed: {relative}")
    for name, digest in value["historical_freezes"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"Historical freeze was rewritten: {name}")
    for source, expected in read_json_shared(run / "SOURCE_FREEZE.json")["files"].items():
        path = Path(source).resolve(strict=True)
        if sha256_file(path) == expected:
            continue
        entry = (
            value["files"].get(path.relative_to(ROOT).as_posix(), {})
            if path.is_relative_to(ROOT)
            else {}
        )
        if entry.get("before_sha256") != expected or entry.get("after_sha256") != sha256_file(path):
            raise ValueError(f"Unreviewed original frozen input changed: {source}")
    return value


def source_matches(path: Path, expected: str, run: Path) -> bool:
    """Accept original bytes or one explicitly authorized before/after pair."""
    path = path.resolve(strict=True)
    if sha256_file(path) == expected:
        return True
    if not (run / MANIFEST).is_file() or not path.is_relative_to(ROOT):
        return False
    value = validate(run)
    entry = value["files"].get(path.relative_to(ROOT).as_posix(), {})
    return entry.get("before_sha256") == expected and entry.get("after_sha256") == sha256_file(path)


def inventory_matches(expected: dict[str, str], actual: dict[str, str], run: Path) -> bool:
    """Keep exact inventory membership as well as each source's identity."""
    if set(expected) != set(actual):
        return False
    changed = [name for name in expected if actual[name] != expected[name]]
    if not changed:
        return True
    if not (run / MANIFEST).is_file():
        return False
    value = validate(run)
    for name in changed:
        path = Path(name).resolve(strict=True)
        if not path.is_relative_to(ROOT):
            return False
        entry = value["files"].get(path.relative_to(ROOT).as_posix(), {})
        if (
            entry.get("before_sha256") != expected[name]
            or entry.get("after_sha256") != actual[name]
        ):
            return False
    return True


def bind_runtime(run: Path, fit_id: str, *, rgb: bool) -> str:
    """Record the amendment beside checkpoints, preserving existing event identity."""
    if not (run / MANIFEST).is_file():
        return _technical_binding(run)
    value = validate(run)
    digest = sha256_file(run / MANIFEST)
    path = run / "fits" / fit_id / "AUDIT_REVISION_RUNTIME.json"
    if path.exists() and read_json_shared(path).get("migration_sha256") != digest:
        raise ValueError("Fit was bound to a different direct-code migration")
    atomic_write_json(
        path,
        {
            "schema": "rgb_port_direct_code_runtime_v1",
            "fit_id": fit_id,
            "migration_sha256": digest,
            "geometry_precision": "bf16_unchanged",
            "event_objective_unchanged": not rgb,
            "event_resume_origin": value["event_origins"].get(fit_id),
            "rgb_effective_batch_reduction": rgb,
            "cache_counter_units": {"reads": "decoded_shards", "hits": "retained_rows"}
            if not rgb
            else None,
        },
    )
    return digest


def validate_rgb_endpoint(directory: Path, payload: dict[str, Any]) -> None:
    """Reject unamended RGB weights under the corrected RGB router/loss code."""
    run = directory.parent.parent
    if (run / MANIFEST).is_file():
        validate(run)
        expected = sha256_file(run / MANIFEST)
    else:
        expected = _technical_binding(run)
    if payload["identity"]["source_identity"].get("audit_revision_sha256") != expected:
        raise ValueError("RGB endpoint was not trained with the current code migration")
