"""Run three pinned public Garl checkpoints on the preserved EvTTC query population."""

# ruff: noqa: E402 -- bind this checkout before project imports
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch

from e_jepa_ttc.data import garl_official_preprocessing
from operational.evttc_competition import geometry, readers
from operational.garl_evttc_competition import model, prepare
from scripts.run_evttc_competition import (
    EXPECTED_ROWS,
    SEQUENCES,
    append_row,
    atomic_json,
    calibration_record,
    canonical_sha256,
    deterministic_zip,
    expected_queries,
    freeze_json,
    git_identity,
    identity_key,
    preflight_inputs,
    query_identity,
    sha256_file,
    synchronize,
    write_partial_results,
    write_signed_results,
)

VARIANTS = prepare.VARIANTS
ROW_SCHEMA = "garl_evttc_competition_row_v1"
REFERENCE_FILES = (
    "RUN_FREEZE.json",
    "all_rows.jsonl",
    "summary.json",
    "independent_audit.json",
    "results.zip",
)


def read_json(path: Path) -> Any:
    """Read a local JSON receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def reference_record(path: Path) -> dict[str, Any]:
    """Bind the completed PHASE run without reading its predictions for selection."""
    root = path.resolve(strict=True)
    summary = read_json(root / "summary.json")
    audit = read_json(root / "independent_audit.json")
    if summary["status"] != "COMPLETE" or summary["ok_rows"] != EXPECTED_ROWS:
        raise ValueError("PHASE reference must contain all 434 finite predictions")
    if audit.get("audit_status") != "PASS" or audit["rows_verified"] != EXPECTED_ROWS:
        raise ValueError("PHASE reference requires a passing independent audit")
    hashes = {name: sha256_file(root / name) for name in REFERENCE_FILES}
    for name, digest in audit["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"PHASE audited artifact changed: {name}")
    return {"path_resolved": str(root), "artifact_sha256": hashes}


def verify_inventory(inventory: Mapping[str, Any]) -> None:
    """Rehash all checkpoint, config and native graph bytes from the HF inventory."""
    if set(inventory["variants"]) != set(VARIANTS):
        raise ValueError("Inventory must contain exactly full, event and visual variants")
    records = list(inventory["upstream"]["native_import_graph"])
    for variant in VARIANTS:
        records.extend(inventory["variants"][variant][key] for key in ("checkpoint", "config"))
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Pinned Garl asset changed: {path}")


def ground_truth_record(path: Path) -> dict[str, str]:
    """Bind the exact-query identity audit, without reading target values."""
    resolved = path.resolve(strict=True)
    audit = read_json(resolved)
    if (
        audit.get("schema") != "evttc_competition_gt_identity_inventory_v1"
        or audit.get("status") != "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH"
        or audit.get("query_count") != EXPECTED_ROWS
    ):
        raise ValueError("This descriptive-only protocol requires its exact-query GT audit")
    return {
        "path_resolved": str(resolved),
        "sha256": sha256_file(resolved),
        "status": audit["status"],
        "schema": audit["schema"],
    }


def source_hashes() -> dict[str, str]:
    """Bind the new adapter and its unchanged repository helper sources."""
    modules = (
        sys.modules[__name__],
        sys.modules[preflight_inputs.__module__],
        model,
        prepare,
        geometry,
        readers,
        garl_official_preprocessing,
    )
    paths = {Path(inspect.getfile(module)).resolve() for module in modules}
    paths.update(Path(model.__file__).parent.glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def load_rows(path: Path, freeze_sha: str, *, resume: bool) -> dict[tuple[str, str], dict]:
    """Resume only intact canonical rows bound to this exact inference freeze."""
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError("Existing row ledger requires --resume")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["schema"] != ROW_SCHEMA or value["freeze_sha256"] != freeze_sha:
            raise ValueError("Resumed row freeze/schema mismatch")
        if value["variant"] not in VARIANTS or value["status"] not in {"ok", "unavailable"}:
            raise ValueError("Invalid resumed row status or model identity")
        key = (identity_key(value["identity"]), value["variant"])
        if key in rows:
            raise ValueError("Duplicate resumed row")
        rows[key] = value
    return rows


def package_variant(
    output: Path, variant: str, records: list[dict], expected: Mapping
) -> dict[str, Any]:
    """Package complete finite results or explicitly named partial predictions."""
    if len(records) != EXPECTED_ROWS:
        raise ValueError("All query identities must be accounted for before packaging")
    expected_keys = {
        (sequence, ordinal, index, timestamp)
        for sequence, identities in expected.items()
        for ordinal, (index, timestamp) in enumerate(identities)
    }
    actual_keys = {
        (
            row["identity"]["sequence_id"],
            row["identity"]["row_index"],
            row["identity"]["index"],
            row["identity"]["timestamp_text"],
        )
        for row in records
    }
    if actual_keys != expected_keys:
        raise ValueError("Package query population differs from original CSV")
    for row in records:
        if row["variant"] != variant:
            raise ValueError("Mixed checkpoints in one model package")
        value = row["prediction_ttc_seconds"]
        if row["status"] == "ok":
            if value is None or not math.isfinite(value):
                raise ValueError("An ok row requires a finite signed TTC")
        elif row["status"] != "unavailable" or value is not None:
            raise ValueError("Unavailable predictions must remain null")
    root = output / variant
    root.mkdir(parents=True, exist_ok=True)
    available = sum(row["status"] == "ok" for row in records)
    result: dict[str, Any] = {
        "status": "COMPLETE" if available == EXPECTED_ROWS else "PARTIAL",
        "recorded_rows": len(records),
        "ok_rows": available,
        "null_rows": len(records) - available,
    }
    if available == EXPECTED_ROWS:
        validation = write_signed_results(root / "results", records, expected)
        archive = root / "results.zip"
        deterministic_zip(archive, root / "results", validation["files"])
        result.update(
            results_zip=archive.relative_to(output).as_posix(),
            results_zip_sha256=sha256_file(archive),
            results_validation=validation,
        )
    else:
        if (root / "results.zip").exists() or (root / "results").exists():
            raise ValueError("Partial run cannot retain a complete results package")
        partial = write_partial_results(root / "partial_results", records, expected)
        archive = root / "partial_results.zip"
        deterministic_zip(archive, root / "partial_results", partial["files"])
        result.update(
            partial_results_zip=archive.relative_to(output).as_posix(),
            partial_results_zip_sha256=sha256_file(archive),
        )
    atomic_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ground-truth-audit", type=Path, required=True)
    parser.add_argument("--phase-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    error_path = output / "errors.jsonl"
    if error_path.exists() and error_path.stat().st_size:
        raise ValueError("Preserve the failed run and diagnose it before starting a new attempt")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    inventory_path = args.inventory.resolve(strict=True)
    inventory = read_json(inventory_path)
    verify_inventory(inventory)
    gt_identity_audit = ground_truth_record(args.ground_truth_audit)
    reference = reference_record(args.phase_run)
    reference_freeze = read_json(args.phase_run / "RUN_FREEZE.json")
    data_root = args.data_root.resolve(strict=True)
    rows_by_sequence, raw_sources, calibrations = preflight_inputs(data_root)
    queries = {
        sequence: [query_identity(sequence, row) for row in rows_by_sequence[sequence]]
        for sequence in SEQUENCES
    }
    calibration_manifest = {
        sequence: calibration_record(calibrations[sequence]) for sequence in SEQUENCES
    }
    for key, value in (
        ("query_manifest", queries),
        ("raw_sources", raw_sources),
        ("calibrations", calibration_manifest),
    ):
        if value != reference_freeze[key]:
            raise ValueError(f"PHASE/Garl comparison input identity mismatch: {key}")
    runtime = model.OfficialGarlRuntime(
        code_root=Path(inventory["upstream"]["path"]), inventory=inventory, device=args.device
    )
    frozen_sources = source_hashes()
    freeze = {
        "schema": "garl_evttc_competition_freeze_v1",
        "variants": list(VARIANTS),
        "query_manifest": queries,
        "query_manifest_sha256": canonical_sha256(queries),
        "raw_sources": raw_sources,
        "calibrations": calibration_manifest,
        "inventory": inventory,
        "ground_truth_identity_audit": gt_identity_audit,
        "inventory_sha256": sha256_file(inventory_path),
        "phase_reference": reference,
        "executed_source_sha256": frozen_sources,
        "model_bindings": runtime.bindings,
        "policy": prepare.POLICY,
        "data_root_resolved": str(data_root),
        "seed": 7,
        "device": args.device,
        "precision": "float32",
        "targets_read": False,
        "training_updates": 0,
        "checkpoint_selection_from_test": False,
        "timing": {
            "per_variant": "independent_preparation_plus_synchronized_model_to_host_wall_time",
            "no_explicit_tensor_or_preprocessing_reuse": True,
            "fixed_variant_order": list(VARIANTS),
            "shared_state": "OS_and_HDF5_page_cache_and_GPU_warm_state_can_be_shared",
            "cross_variant_latency_ranking_supported": False,
            "excluded": ["model_loading", "indexing", "calibration", "source_hashing"],
            "not_optimized_streaming_latency": True,
        },
    }
    freeze_sha = freeze_json(output / "RUN_FREEZE.json", freeze, resume=args.resume)
    ledger_path = output / "all_rows.jsonl"
    records = load_rows(ledger_path, freeze_sha, resume=args.resume)
    allowed = {
        (identity_key(q), variant) for qs in queries.values() for q in qs for variant in VARIANTS
    }
    if not set(records).issubset(allowed):
        raise ValueError("Resumed ledger includes an unexpected query")
    query_count = 0
    for sequence in SEQUENCES:
        with readers.SensorSequence(data_root, sequence) as reader:
            sequence_rows = rows_by_sequence[sequence]
            for row in sequence_rows:
                identity = query_identity(sequence, row)
                for variant in VARIANTS:
                    key = (identity_key(identity), variant)
                    if key in records:
                        continue
                    before = time.perf_counter()
                    try:
                        prepared = prepare.prepare_query(
                            reader, sequence_rows, row, calibrations[sequence], variant
                        )
                        prep_seconds = time.perf_counter() - before
                        prediction = None
                        model_seconds = 0.0
                        if prepared.sensor is None:
                            status, ttc, reason = "unavailable", None, prepared.reason
                        else:
                            if prepared.delta_t_s is None:
                                raise ValueError("A prepared sensor requires its measured delta")
                            synchronize(args.device)
                            model_start = time.perf_counter()
                            prediction = runtime.predict(
                                variant, prepared.sensor, prepared.delta_t_s
                            )
                            synchronize(args.device)
                            model_seconds = time.perf_counter() - model_start
                            status, ttc, reason = (
                                prediction.status,
                                prediction.model_ttc,
                                prediction.reason,
                            )
                        record = {
                            "schema": ROW_SCHEMA,
                            "freeze_sha256": freeze_sha,
                            "variant": variant,
                            "identity": identity,
                            "status": status,
                            "prediction_ttc_seconds": ttc,
                            "reason": reason,
                            "input_details": prepared.details,
                            "preprocess_seconds": prep_seconds,
                            "model_wall_seconds": model_seconds,
                            "cost_time_seconds": prep_seconds + model_seconds,
                            "prediction": asdict(prediction) if prediction is not None else None,
                        }
                        if status == "ok" and (ttc is None or not math.isfinite(ttc)):
                            raise ValueError("Finite status requires a finite signed TTC")
                        append_row(ledger_path, record)
                        records[key] = record
                    except Exception as error:
                        append_row(
                            error_path,
                            {
                                "identity": identity,
                                "variant": variant,
                                "freeze_sha256": freeze_sha,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                        raise
                query_count += 1
                print(
                    json.dumps(
                        {
                            "queries_completed": query_count,
                            "query": identity_key(identity),
                            "status": {
                                v: records[(identity_key(identity), v)]["status"] for v in VARIANTS
                            },
                        }
                    ),
                    flush=True,
                )
    if (
        source_hashes() != frozen_sources
        or sha256_file(inventory_path) != freeze["inventory_sha256"]
    ):
        raise ValueError("Executed code or pinned inventory changed during inference")
    verify_inventory(inventory)
    if ground_truth_record(args.ground_truth_audit) != gt_identity_audit:
        raise ValueError("Ground-truth identity audit changed during inference")
    if reference_record(args.phase_run) != reference:
        raise ValueError("Preserved PHASE reference changed during inference")
    _, final_raw, final_calibrations = preflight_inputs(data_root)
    if (
        final_raw != raw_sources
        or {s: calibration_record(final_calibrations[s]) for s in SEQUENCES} != calibration_manifest
    ):
        raise ValueError("Raw sensors or calibration changed during inference")
    ordered = [
        records[(identity_key(q), v)] for qs in queries.values() for q in qs for v in VARIANTS
    ]
    expected = expected_queries(rows_by_sequence)
    variants = {
        variant: package_variant(
            output, variant, [r for r in ordered if r["variant"] == variant], expected
        )
        for variant in VARIANTS
    }
    summary = {
        "schema": "garl_evttc_competition_summary_v1",
        "status": "COMPLETE"
        if all(v["status"] == "COMPLETE" for v in variants.values())
        else "PARTIAL",
        "terminal_population_accounted": True,
        "expected_records": EXPECTED_ROWS * len(VARIANTS),
        "recorded_records": len(ordered),
        "freeze_sha256": freeze_sha,
        "ledger_sha256": sha256_file(ledger_path),
        "variants": variants,
        "official_accuracy_evaluated": False,
        "comparison_scope": "local_adapted_frozen_checkpoint_predictions_not_accuracy",
    }
    commit, dirty = git_identity()
    atomic_json(
        output / "environment.json",
        {
            "start_utc": started,
            "end_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": 7,
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "h5py", "pillow", "torchvision", "pyyaml")
            },
        },
    )
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
