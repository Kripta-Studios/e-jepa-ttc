"""Path-only receipt relocation and non-pickle string serialization for final caches.

The historical scientific sources and receipts remain unchanged.  Repaired caches
are rebuilt from already verified physical OOF stitches and native TRAIN40 rows.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def unicode_supervision(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Preserve every value while storing identifier columns without Python pickle."""
    result = dict(values)
    for name in ("sample_token", "group_id", "sequence_id", "role"):
        array = np.asarray(result[name])
        if array.ndim != 1 or any(not isinstance(item, str) for item in array.tolist()):
            raise ValueError(f"Supervision {name} is not a string vector")
        result[name] = np.asarray(array.tolist(), dtype=str)
        if result[name].tolist() != array.tolist():
            raise ValueError(f"String serialization changed {name}")
    return result


def install(mapping: Any) -> None:
    """Retain all receipt checks, changing only physical path comparison."""
    from operational.final_refit import features, scientific_oof_features, training
    from operational.final_refit import scientific_oof_final as final

    original_native = final._native_supervision
    if not getattr(original_native, "_unicode_compat", False):

        def native(source: Path) -> dict[str, np.ndarray]:
            return unicode_supervision(original_native(source))

        native._unicode_compat = True
        final._native_supervision = native

    def validate(
        cache_path: Path, expected_schema: str, *, expected_role: str = "TRAIN"
    ) -> dict[str, Any]:
        receipt = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        required = {
            "schema",
            "status",
            "role",
            "rows",
            "groups",
            "group_ids",
            "parents",
            "normalizers",
            "output",
            "output_sha256",
            "targets_used_for_feature_construction",
            "receipt_payload_sha256",
        }
        if not required.issubset(receipt):
            raise ValueError("Cache receipt lacks required fields")
        payload = dict(receipt)
        claimed = payload.pop("receipt_payload_sha256")
        if (
            receipt["schema"] != expected_schema
            or receipt["status"] != "COMPLETE"
            or receipt["role"] != expected_role
            or receipt["rows"] <= 0
            or receipt["groups"] != len(receipt["group_ids"])
            or receipt["targets_used_for_feature_construction"] is not False
            or Path(mapping.translate(receipt["output"])).resolve(strict=True)
            != Path(cache_path).resolve(strict=True)
            or receipt["output_sha256"] != features.sha256_file(cache_path)
            or claimed != features._canonical_sha256(payload)
        ):
            raise ValueError("Cache receipt identity, ancestry or payload changed")
        features._validate_parents(receipt["parents"])
        return receipt

    features.validate_cache_receipt = validate
    training.validate_cache_receipt = validate
    scientific_oof_features.validate_cache_receipt = validate


def repair_generated_caches(run: Path) -> dict[str, Any]:
    """Resume-safe reconstruction of three untrained caches, retaining old artifacts."""
    from operational.final_refit import scientific_oof_final as final
    from operational.final_refit.recipe import canonical_sha256, sha256
    from operational.rgb_port.accounting import atomic_write_json

    run = final.Path(str(run)).resolve(strict=True)
    backup = run / "final" / "serialization_repair_v1"
    backup.mkdir(exist_ok=True)
    prep_path = run / "final" / "OOF_PREPARATION.json"
    old_prep = backup / "OOF_PREPARATION.json"
    if not old_prep.exists():
        original = final._read(prep_path)
        if original.get("status") != "COMPLETE":
            raise ValueError("Cannot repair unverified OOF preparation")
        # Removing the publication marker first keeps queue consumers fail-closed.
        os.replace(prep_path, old_prep)
    original = final._read(old_prep)
    if canonical_sha256(
        {k: v for k, v in original.items() if k != "receipt_sha256"}
    ) != original.get("receipt_sha256"):
        raise ValueError("Original OOF preparation identity changed")
    for stage in original["caches"]:
        receipt = run / "final" / "fits" / stage / "CHECKPOINT_RECEIPT.json"
        if receipt.exists() and final._read(receipt).get("completed_updates", 0) > 0:
            raise ValueError("Cannot change a training cache after optimizer updates")
    paths = {name: run / "final" / "oof" / f"{name}.npz" for name in original["stitches"]}
    for name, path in paths.items():
        if sha256(path) != original["stitches"][name]:
            raise ValueError("Verified physical OOF stitch changed")
    for stage, expected in original["caches"].items():
        current = run / "final" / "cache" / (stage + ".npz")
        archived = backup / current.name
        if archived.exists():
            if sha256(archived) != expected:
                raise ValueError("Preserved pre-repair cache changed")
        else:
            if sha256(current) != expected:
                raise ValueError("Untrained cache differs from verified original")
            os.replace(current, archived)
        sidecar = current.with_suffix(".json")
        old_sidecar = archived.with_suffix(".json")
        if not old_sidecar.exists():
            os.replace(sidecar, old_sidecar)
        if final._read(old_sidecar)["output_sha256"] != expected:
            raise ValueError("Preserved pre-repair sidecar differs")
    # The frozen body rebuilds normalizers/caches identically.  Only the expensive
    # recursive stitch phase is replaced by the complete, hash-verified publication
    # above; no input array, normalization coefficient or target is recomputed differently.
    tree = ast.parse(inspect.getsource(final.prepare_oof_training))
    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "paths" for target in node.targets)
    ]
    if len(targets) != 1 or not isinstance(targets[0].value, ast.DictComp):
        raise ValueError("Frozen preparation stitch assignment changed")
    targets[0].value = ast.Name(id="_verified_existing_stitches", ctx=ast.Load())
    tree.body[0].name = "_rebuild_from_verified_stitches"
    ast.fix_missing_locations(tree)
    pending_prep = backup / "REPAIRED_PREPARATION.pending.json"
    original_write = final._write_once

    def staged_write(path: Path, value: dict[str, Any]) -> None:
        if Path(path).resolve() == Path(prep_path).resolve():
            return original_write(pending_prep, value)
        return original_write(path, value)

    namespace = {**vars(final), "_verified_existing_stitches": paths, "_write_once": staged_write}
    exec(compile(tree, "<final-cache-serialization-repair>", "exec"), namespace)
    result = namespace["_rebuild_from_verified_stitches"](run)
    for stage in result["caches"]:
        new = run / "final" / "cache" / (stage + ".npz")
        old = backup / new.name
        with np.load(new, allow_pickle=False) as n, np.load(old, allow_pickle=False) as o:
            for key in n.files:
                if key in {"sample_token", "group_id"}:
                    continue  # Old object arrays are never unpickled.
                if not np.array_equal(n[key], o[key]) or n[key].dtype != o[key].dtype:
                    raise ValueError(f"Numeric cache payload changed: {stage}/{key}")
            new_receipt = final._read(new.with_suffix(".json"))
            old_receipt = final._read(old.with_suffix(".json"))
            for key in (
                "ordered_row_identity_sha256",
                "parents",
                "normalizers",
                "group_ids",
                "rows",
            ):
                if new_receipt[key] != old_receipt[key]:
                    raise ValueError(f"Cache ancestry or row identity changed: {key}")
    os.replace(pending_prep, prep_path)
    proof = {
        "status": "PASS",
        "original_preparation_sha256": sha256(old_prep),
        "repaired_preparation_sha256": sha256(prep_path),
        "numeric_arrays_identical": True,
        "ordered_token_group_identity_identical": True,
        "old_object_arrays_unpickled": False,
        "originals_preserved": str(backup),
        "optimizer_updates": 0,
        "source_sha256": sha256(final.Path(__file__)),
    }
    atomic_write_json(backup / "REPAIR_RECEIPT.json", proof)
    return proof
