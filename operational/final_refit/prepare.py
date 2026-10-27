"""Prepare immutable event-parent admissions, PAIR_E cache and recipe metadata."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_pair_cache_from_features, validate_cache_receipt
from .preflight import build_preflight
from .recipe import (
    build_recipe,
    canonical_sha256,
    ordered_row_identity_sha256,
    sha256,
    write_recipe,
)

PAIR_SHA256 = "1ba3eee9b09900739c5d44f141274b72dccbaf82cce330ef672f8d24dccd5f1e"
PAIR_MANIFEST_SHA256 = "631f23013ddb92f8ad7f95bf23d1e57f52193a79b4b8c28836de3e9c11b50bae"
EVENT_ENDPOINTS = {
    "E_A5": ("a5_seed7", "a82e5b65ec9c3d302981438527f2cfccabbbf25c75f574946bd1e9dc97795758"),
    "E_C2F": (
        "c2f_seed7",
        "2b337dd76bd405c63ac9f96023a82196ee32e8180b86fe0015bfd67f0e2aeb57",
    ),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace changed immutable metadata: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _preflight(workspace: Path, run: Path) -> tuple[dict[str, Any], Path]:
    path = run / "PREFLIGHT.json"
    observed = build_preflight(workspace)
    if path.exists():
        frozen = _read(path)
        stable_fields = (
            "schema",
            "status",
            "population",
            "availability",
            "historical_event_producers",
            "parent_ancestry",
            "sources",
        )
        if any(frozen.get(name) != observed.get(name) for name in stable_fields):
            raise ValueError("Existing preflight differs in a scientific identity field")
        # A later wording clarification split historical target finiteness from a
        # current payload recheck. Both records bind the same immutable teacher.
        for name in ("manifest_sha256", "freeze_sha256", "weights_sha256", "model_revision"):
            if frozen["teacher"].get(name) != observed["teacher"].get(name):
                raise ValueError("Existing preflight differs in teacher ancestry")
        observed = frozen
    else:
        _write_once(path, observed)
    if observed.get("status") != "PASS" or observed.get("optimizer_updates") != 0:
        raise ValueError("Final-refit preflight is not a zero-update PASS")
    return observed, path


def _admit_event_parents(
    workspace: Path,
    run: Path,
    preflight: Mapping[str, Any],
    preflight_path: Path,
    parity_path: Path,
) -> dict[str, Path]:
    parity = _read(parity_path)
    if parity.get("status") != "PASSED_WITH_EXPLICIT_ADAPTER":
        raise ValueError("Event reuse parity is not admitted")
    by_fit = {str(item["fit"]): item for item in preflight["historical_event_producers"]}
    result: dict[str, Path] = {}
    for final_name, (fit_name, expected_sha) in EVENT_ENDPOINTS.items():
        source = by_fit[fit_name]
        checkpoint = workspace / str(source["checkpoint"]["path"])
        original_receipt = checkpoint.with_name("CHECKPOINT_RECEIPT.json")
        if (
            source.get("status") != "COMPLETE_HASH_VERIFIED"
            or source["checkpoint"].get("sha256") != expected_sha
            or sha256(checkpoint) != expected_sha
            or sha256(original_receipt) != source.get("checkpoint_receipt_sha256")
            or int(source.get("scientific_updates", -1)) != 49_932
        ):
            raise ValueError(f"Historical event endpoint changed: {fit_name}")
        admission = {
            "schema": "final_refit_external_parent_admission_v1",
            "status": "PASS",
            "fit_id": final_name,
            "source_fit_id": fit_name,
            "reuse_authorized": True,
            "reuse_scope": "all-TRAIN40 final chain only; not OOF",
            "scientific_updates": 49_932,
            "split_version": source["split_version"],
            "input_manifest_sha256": source["input_manifest_sha256"],
            "teacher_manifest_sha256": source["teacher_manifest_sha256"],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": expected_sha,
            "original_receipt_path": str(original_receipt.resolve()),
            "original_receipt_sha256": sha256(original_receipt),
            "preflight_sha256": sha256(preflight_path),
            "event_reuse_parity_sha256": sha256(parity_path),
            "strict_state_load_and_projection_parity": True,
            "optimizer_updates": 0,
        }
        admission["checkpoint_identity_sha256"] = canonical_sha256(admission)
        target = run / "parents" / f"{final_name}_ADMISSION.json"
        _write_once(target, admission)
        result[final_name] = target
    return result


def _pair_e_cache(
    workspace: Path, run: Path, parents: Mapping[str, Path]
) -> tuple[Path, Path, str]:
    native = workspace / "e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005"
    source_path = native / "PAIR_FEATURES.npz"
    manifest_path = native / "PAIR_FEATURE_MANIFEST.json"
    rows_path = native / "TRAIN40_ROWS.parquet"
    if sha256(source_path) != PAIR_SHA256 or sha256(manifest_path) != PAIR_MANIFEST_SHA256:
        raise ValueError("Historical sealed PAIR_E cache or manifest changed")
    output = run / "cache" / "PAIR_E_MATCHED.npz"
    expected_parents = {
        "E_A5": EVENT_ENDPOINTS["E_A5"][1],
        "E_A5_ADMISSION": sha256(parents["E_A5"]),
        "PAIR_FEATURE_MANIFEST": PAIR_MANIFEST_SHA256,
        "PAIR_FEATURES": PAIR_SHA256,
    }
    rows = pd.read_parquet(rows_path, columns=["sample_token", "sequence_id"])
    if (
        rows.sample_token.duplicated().any()
        or len(rows) != 88_744
        or bool(rows["sample_token"].isna().any())
        or bool(rows["sequence_id"].isna().any())
    ):
        raise ValueError("Native TRAIN40 identity changed")
    native_order_sha256 = ordered_row_identity_sha256(
        rows["sample_token"].astype(str).tolist(),
        rows["sequence_id"].astype(str).tolist(),
    )
    native_rows_sha256 = sha256(rows_path)
    if output.exists():
        receipt = validate_cache_receipt(output, "final_refit_pair_cache_v1")
        bound = receipt.get("parents", {})
        required_lineage = {
            "E_A5": expected_parents["E_A5"],
            "PAIR_FEATURE_MANIFEST": expected_parents["PAIR_FEATURE_MANIFEST"],
            "PAIR_FEATURES": expected_parents["PAIR_FEATURES"],
        }
        if (
            receipt.get("modality") != "event"
            or receipt.get("native_train40_rows_sha256") != native_rows_sha256
            or receipt.get("ordered_row_identity_sha256") != native_order_sha256
            or any(bound.get(name) != value for name, value in required_lineage.items())
        ):
            raise ValueError("Existing PAIR_E cache has different ancestry")
        return output, rows_path, native_order_sha256
    with np.load(source_path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "target_phase", "sequences", "tokens"}:
            raise ValueError("Historical PAIR_FEATURES schema changed")
        values = {name: archive[name] for name in archive.files}
    mapping = rows.set_index("sample_token")["sequence_id"].astype(str)
    tokens = np.asarray(values["tokens"]).astype(str)
    if len(set(tokens.tolist())) != len(tokens) or set(tokens) != set(mapping.index.astype(str)):
        raise ValueError("PAIR_E tokens differ from native TRAIN40")
    groups = mapping.loc[tokens].to_numpy(dtype=str)
    sequences = np.asarray(values["sequences"]).astype(str)
    if not np.array_equal(sequences, groups):
        raise ValueError("PAIR_E sequence identity differs from native TRAIN40 groups")
    build_pair_cache_from_features(
        np.asarray(values["features"], dtype=np.float32),
        {
            "target_phase": np.asarray(values["target_phase"], dtype=np.float32),
            "group_id": groups,
            "sample_token": tokens,
            "sequence_id": sequences,
            "role": np.full(len(tokens), "TRAIN"),
        },
        output,
        modality="event",
        parents=expected_parents,
        native_train40_rows_sha256=native_rows_sha256,
        compressed=True,
    )
    return output, rows_path, native_order_sha256


def prepare(workspace: Path, run: Path, *, mode: str) -> dict[str, Any]:
    """Build zero-update immutable inputs for technical or final execution."""
    if mode not in {"technical", "final"}:
        raise ValueError("mode must be technical or final")
    workspace, run = workspace.resolve(), run.resolve()
    repository = Path(__file__).resolve().parents[2]
    preflight, preflight_path = _preflight(workspace, run)
    parity_path = run / "EVENT_REUSE_PARITY.json"
    parents = _admit_event_parents(
        workspace, run, preflight, preflight_path, parity_path.resolve(strict=True)
    )
    pair_cache, native_rows, native_order_sha256 = _pair_e_cache(workspace, run, parents)
    core_code = [
        repository / "operational/final_refit/recipe.py",
        repository / "operational/final_refit/training.py",
    ]
    full_code = [
        *core_code,
        repository / "operational/final_refit/features.py",
        repository / "operational/final_refit/preflight.py",
        repository / "operational/final_refit/prepare.py",
        repository / "operational/final_refit/producers.py",
        repository / "operational/final_refit/rgb.py",
    ]
    required = [
        *(core_code if mode == "technical" else full_code),
        preflight_path,
        parity_path,
        *parents.values(),
        native_rows,
        pair_cache,
        pair_cache.with_suffix(".json"),
    ]
    source_files = {f"file:{path.resolve()}": path.resolve(strict=True) for path in required}
    recipe = build_recipe(
        source_files=source_files,
        external_parent_sha256={name: EVENT_ENDPOINTS[name][1] for name in ("E_A5", "E_C2F")},
        native_train40_rows_sha256=sha256(native_rows),
        ordered_train40_identity_sha256=native_order_sha256,
    )
    recipe_path = run / ("technical/PAIR_E_RECIPE.json" if mode == "technical" else "RECIPE.json")
    write_recipe(recipe_path, recipe)
    receipt = {
        "schema": "final_refit_prepare_receipt_v1",
        "status": "PASS",
        "mode": mode,
        "optimizer_updates": 0,
        "training_started": False,
        "preflight_sha256": sha256(preflight_path),
        "event_reuse_parity_sha256": sha256(parity_path),
        "parent_admission_sha256": {name: sha256(path) for name, path in parents.items()},
        "pair_e_cache_path": str(pair_cache),
        "pair_e_cache_sha256": sha256(pair_cache),
        "recipe_path": str(recipe_path),
        "recipe_file_sha256": sha256(recipe_path),
        "recipe_sha256": recipe.recipe_sha256,
        "scientific_updates_declared": sum(item.updates for item in recipe.stages),
    }
    receipt_name = "PAIR_E_PREPARE_RECEIPT.json" if mode == "technical" else "PREPARE_RECEIPT.json"
    _write_once(recipe_path.with_name(receipt_name), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("technical", "final"), default="technical")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.workspace, args.run, mode=args.mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "prepare"]
