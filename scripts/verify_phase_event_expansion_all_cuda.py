"""Label-blind all-77 CUDA parity against sealed CPU predictions; never read targets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.phase_adapt_cache import atomic_receipt
from operational.phase_event_expansion.accelerated import CudaObjectiveCache, estimate_accelerated
from operational.phase_event_expansion.model import ExpansionResult, SolverConfig
from operational.phase_event_expansion.validation import (
    _relative_source,
    require_confirmation_binding,
    validate_cache_manifest,
    validate_cached_window,
)
from scripts.benchmark_phase_event_expansion_accelerated import compare_results
from scripts.evaluate_phase_event_expansion_development import undistort_inputs

# Reviewed exact legacy -> guard-only update. Reject any further changes, including
# changes outside the function imported here; do not infer that arbitrary edits are safe.
LEGACY_EVALUATOR_SHA = "69005cf717fb860a4dcd72c6b7752149a173f148229a8f39ab56f365ff75d939"
GUARDED_EVALUATOR_SHA = "b9a709320ac28213c560ecb171a73e929ddb766e9db3d34ec5935024afaeecf2"
EVALUATOR = Path("scripts/evaluate_phase_event_expansion_development.py")


def sha(path: Path) -> str:
    """Hash exact evidence bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read one explicit JSON receipt, never a population or target report."""
    return json.loads(path.read_text(encoding="utf8"))


def validate_result(result: ExpansionResult) -> None:
    """Require finite physical fields and unambiguous emitted TTC missingness."""
    if not isinstance(result.valid, bool) or result.valid != (result.reason == "available"):
        raise ValueError("Inconsistent validity/reason")
    if (result.q is None) != (result.translation is None):
        raise ValueError("Partial q/translation estimate")
    if result.q is not None:
        if np.asarray(result.q).shape != () or not np.isfinite(result.q):
            raise ValueError("Nonfinite or nonscalar q")
        b = np.asarray(result.translation)
        if b.shape != (2,) or not np.isfinite(b).all():
            raise ValueError("Nonfinite or malformed translation")
    if result.valid:
        if (
            result.q is None
            or result.q == 0
            or result.ttc_seconds is None
            or np.asarray(result.ttc_seconds).shape != ()
            or not np.isfinite(result.ttc_seconds)
            or not np.isclose(result.q * result.ttc_seconds, 1, rtol=1e-12, atol=0)
        ):
            raise ValueError("Valid result requires finite q,b,TTC and q*TTC=1")
    elif result.ttc_seconds is not None:
        raise ValueError("Invalid result must emit TTC=None")


def strict_compare_results(cpu: ExpansionResult, candidate: ExpansionResult) -> dict[str, Any]:
    """Keep q/b tolerance and require emitted TTC agreement, including missingness."""
    errors = {}
    for name, result in (("cpu", cpu), ("candidate", candidate)):
        try:
            validate_result(result)
        except (TypeError, ValueError) as exc:
            errors[name] = str(exc)
    if errors:
        return {"passed": False, "result_contract_pass": False, "contract_errors": errors}
    comparison = compare_results(cpu, candidate)
    missingness = (cpu.ttc_seconds is None) == (candidate.ttc_seconds is None)
    error = (
        abs(cpu.ttc_seconds - candidate.ttc_seconds)
        if cpu.ttc_seconds is not None and candidate.ttc_seconds is not None
        else None
    )
    bound = None
    if cpu.q is not None and candidate.q is not None and error is not None:
        # Inverse sensitivity follows directly from the two reciprocal predictions.
        # Include the accepted reciprocal rounding tolerance, not an independent
        # seconds threshold that changes the original q/b acceptance protocol.
        bound = abs(cpu.q - candidate.q) / abs(cpu.q * candidate.q)
        bound += 2e-12 * (abs(1 / cpu.q) + abs(1 / candidate.q))
    ttc_pass = missingness and (error is None or (bound is not None and error <= bound))
    return {
        **comparison,
        "passed": bool(comparison["passed"] and ttc_pass),
        "result_contract_pass": True,
        "ttc_missingness_match": missingness,
        "ttc_numeric_pass": ttc_pass,
        "ttc_derived_absolute_bound_seconds": bound,
    }


def verify_guarded_evaluator(archived: Path, current: Path) -> None:
    """Allow the reviewed guard addition only; independently compare calibration AST."""
    if sha(archived) != LEGACY_EVALUATOR_SHA or sha(current) not in {
        LEGACY_EVALUATOR_SHA,
        GUARDED_EVALUATOR_SHA,
    }:
        raise ValueError("Unreviewed evaluator change beyond approved guards")

    def function(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf8"))
        matches = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "undistort_inputs"
        ]
        if len(matches) != 1:
            raise ValueError("Missing or duplicated calibration function")
        return ast.dump(matches[0], include_attributes=False)

    if function(archived) != function(current):
        raise ValueError("Calibration function differs from executed source")


def validate_cpu_metadata(config: dict, seal: dict, expected: dict) -> None:
    """Bind CPU recipe, synthetic confirmation, sensor cache and prediction seal."""
    if config.get("schema") != "phase_event_expansion_real_development_v1":
        raise ValueError("Unexpected CPU run schema")
    for key in (
        "solver",
        "cache_manifest_sha256",
        "synthetic_config_sha256",
        "synthetic_summary_sha256",
    ):
        if config.get(key) != expected[key]:
            raise ValueError(f"CPU configuration identity differs: {key}")
    if (
        seal.get("rows") != 77
        or seal.get("targets_read") is not False
        or seal.get("config_sha256") != expected["config_sha256"]
        or seal.get("sha256") != expected["predictions_sha256"]
    ):
        raise ValueError("CPU prediction seal or hash differs")


def validate_cpu_query(record: dict, row: dict, token: str, prediction: float) -> ExpansionResult:
    """Check clock identity and equality with the historically sealed prediction array."""
    if (
        record.get("token") != row["query_id"]
        or token != row["query_id"]
        or record.get("index") != row["frame_index"]
        or record.get("query_us") != row["query_us"]
    ):
        raise ValueError("CPU query token/frame/clock identity differs")
    fields = set(ExpansionResult.__dataclass_fields__)
    if set(record) != fields | {"token", "index", "query_us"}:
        raise ValueError("Unexpected CPU query fields")
    result = ExpansionResult(**{k: record[k] for k in fields})
    validate_result(result)
    if not isinstance(result.valid, bool) or result.valid != (result.reason == "available"):
        raise ValueError("Inconsistent CPU validity/reason")
    if result.q is not None and not np.isfinite(result.q):
        raise ValueError("Nonfinite CPU q")
    if result.translation is not None:
        b = np.asarray(result.translation)
        if b.shape != (2,) or not np.isfinite(b).all():
            raise ValueError("Invalid CPU translation")
    if result.valid:
        if (
            result.q is None
            or result.q == 0
            or result.translation is None
            or result.ttc_seconds is None
            or not np.isfinite(prediction)
            or result.ttc_seconds != prediction
            or not np.isclose(result.q * prediction, 1, rtol=1e-12, atol=0)
        ):
            raise ValueError("CPU result differs from sealed prediction or endpoint physics")
    elif not np.isnan(prediction) or result.ttc_seconds is not None:
        raise ValueError("CPU abstention differs from sealed prediction")
    return result


def verify_accelerator_sources(profile: Path) -> dict[str, str]:
    """Require executed-source archives and current float64 implementation identity."""
    config = read(profile / "CONFIG.json")
    if (
        config.get("schema") != "phase_event_accelerated_parity_v1"
        or config.get("precision") != "float64"
        or config.get("labels_opened") is not False
        or config.get("q_b_absolute_tolerance") != 1e-5
    ):
        raise ValueError("Wrong accelerator parity profile")
    sources: dict[str, str] = {}
    for name, expected in config.get("source_sha256", {}).items():
        path = _relative_source(name)
        if path.as_posix() in sources:
            raise ValueError("Duplicate accelerator source identity")
        archive = profile / "executed_source" / path
        if sha(archive) != expected:
            raise ValueError("Accelerator executed-source archive changed")
        if path == EVALUATOR:
            verify_guarded_evaluator(archive, path)
        elif sha(path) != expected:
            raise ValueError(f"Accelerator source changed: {path}")
        sources[path.as_posix()] = expected
    required = {
        "scripts/benchmark_phase_event_expansion_accelerated.py",
        "operational/phase_event_expansion/accelerated.py",
        "operational/phase_event_expansion/torch_objective.py",
        "operational/phase_event_expansion/model.py",
        "operational/phase_event_expansion/synthetic.py",
        EVALUATOR.as_posix(),
        "tests/unit/test_phase_event_expansion_accelerated.py",
    }
    if set(sources) != required:
        raise ValueError("Missing or unexpected accelerator executed sources")
    summary = read(profile / "SUMMARY.json")
    expected_names = {
        f"synthetic_{seed}_{kind}"
        for seed in (7, 31, 47)
        for kind in ("expansion", "receding", "translation", "stationary", "noise", "edge")
    }
    expected_names |= {f"slider500_{i:02d}" for i in (0, 10, 30, 50, 76)}
    rows = summary.get("rows", [])
    if (
        summary.get("all_parity_pass") is not True
        or summary.get("all_memory_pass") is not True
        or len(rows) != 23
        or {r["name"] for r in rows} != expected_names
    ):
        raise ValueError("Accelerator pilot parity evidence is incomplete")
    for row in rows:
        if read(profile / f"{row['name']}.json") != row:
            raise ValueError("Accelerator pilot summary differs from case receipt")
        for backend in ("cuda_cold", "cuda_warm"):
            if not strict_compare_results(
                ExpansionResult(**row["cpu"]), ExpansionResult(**row[backend])
            )["passed"]:
                raise ValueError("Accelerator pilot fails recomputed parity")
    return sources


def preflight(cache: Path, synthetic: Path, cpu: Path, profile: Path) -> dict[str, Any]:
    """Perform every source/data/provenance guard without allocating GPU tensors."""
    require_confirmation_binding(synthetic)
    sources = verify_accelerator_sources(profile)
    manifest, freeze = read(cache / "MANIFEST.json"), read(cache / "CACHE_FREEZE.json")
    if sha(cache / "CACHE_FREEZE.json") != manifest.get("cache_freeze_sha256"):
        raise ValueError("Sensor cache freeze hash changed")
    validate_cache_manifest(manifest, freeze)
    recipe = read(synthetic / "CONFIG.json")["solver"]
    profile_config = read(profile / "CONFIG.json")
    if profile_config["solver"] != recipe or profile_config["cache_manifest_sha256"] != sha(
        cache / "MANIFEST.json"
    ):
        raise ValueError("Accelerator pilot recipe/cache differs")
    cpu_config, seal = read(cpu / "CONFIG.json"), read(cpu / "PREDICTIONS_SEALED.json")
    validate_cpu_metadata(
        cpu_config,
        seal,
        {
            "solver": recipe,
            "cache_manifest_sha256": sha(cache / "MANIFEST.json"),
            "synthetic_config_sha256": sha(synthetic / "CONFIG.json"),
            "synthetic_summary_sha256": sha(synthetic / "SUMMARY.json"),
            "config_sha256": sha(cpu / "CONFIG.json"),
            "predictions_sha256": sha(cpu / "PREDICTIONS.npz"),
        },
    )
    cpu_sources = {
        _relative_source(k).as_posix(): v for k, v in cpu_config["source_hashes"].items()
    }
    if set(cpu_sources) != {
        EVALUATOR.as_posix(),
        "operational/evttc_competition/geometry.py",
        "operational/phase_event_expansion/model.py",
    }:
        raise ValueError("Unexpected legacy CPU source identities")
    for name, expected in cpu_sources.items():
        if name == EVALUATOR.as_posix():
            if expected != sources[name]:
                raise ValueError("CPU evaluator differs from accelerator archived evaluator")
        elif sha(Path(name)) != expected:
            raise ValueError("CPU numerical source changed")
    expected_files = {f"{i:05d}.json" for i in range(77)}
    if {p.name for p in (cpu / "queries").iterdir()} != expected_files:
        raise ValueError("Missing or extra CPU query receipts")
    records, hashes = [], {}
    with np.load(cpu / "PREDICTIONS.npz", allow_pickle=False) as data:
        if set(data.files) != {"prediction", "token", "cost_seconds"}:
            raise ValueError("Unexpected fields in sealed CPU predictions")
        predictions, tokens = data["prediction"], data["token"]
        if predictions.shape != (77,) or tokens.shape != (77,):
            raise ValueError("Wrong sealed CPU population shape")
        for i, row in enumerate(manifest["rows"]):
            path = cpu / "queries" / f"{i:05d}.json"
            record = read(path)
            validate_cpu_query(record, row, str(tokens[i]), float(predictions[i]))
            records.append(record)
            hashes[path.name] = sha(path)
            window = cache / row["file"]
            if sha(window) != row["sha256"]:
                raise ValueError("Sensor window hash changed")
            with np.load(window, allow_pickle=False) as sensor:
                if set(sensor.files) != {"x", "y", "t_us", "p", "query_us", "roi_xyxy"}:
                    raise ValueError("Unexpected sensor fields")
                validate_cached_window({k: sensor[k] for k in sensor.files}, row)
    return {
        "manifest": manifest,
        "recipe": recipe,
        "cpu_records": records,
        "cpu_query_sha256": hashes,
        "accelerator_archived_sources": sources,
    }


def postvalidate_previous(directory: Path, output: Path, evidence: dict[str, Any]) -> None:
    """Reassess original sealed receipts on CPU without altering or rerunning them."""
    config, summary = read(directory / "CONFIG.json"), read(directory / "SUMMARY.json")
    if sha(directory / "CONFIG.json") != summary["config_sha256"]:
        raise ValueError("Original run configuration seal changed")
    for name, expected in config["source_sha256"].items():
        if sha(directory / "executed_source" / _relative_source(name)) != expected:
            raise ValueError("Original executed source archive changed")
    if config["cpu_query_sha256"] != evidence["cpu_query_sha256"]:
        raise ValueError("Original comparison CPU receipts changed")
    if config["solver"] != evidence["recipe"] or len(summary["rows"]) != 77:
        raise ValueError("Original comparison population/recipe differs")
    expected_files = {f"{i:05d}.json" for i in range(77)}
    if set(summary["query_receipt_sha256"]) != expected_files:
        raise ValueError("Original query seal incomplete")
    rows = []
    for i, row in enumerate(evidence["manifest"]["rows"]):
        filename = f"{i:05d}.json"
        path = directory / "queries" / filename
        if sha(path) != summary["query_receipt_sha256"][filename]:
            raise ValueError("Original CUDA query receipt changed")
        original = read(path)
        if (
            original != summary["rows"][i]
            or original["token"] != row["query_id"]
            or original["index"] != row["frame_index"]
            or original["query_us"] != row["query_us"]
        ):
            raise ValueError("Original CUDA identity differs")
        cpu = ExpansionResult(
            **{k: evidence["cpu_records"][i][k] for k in ExpansionResult.__dataclass_fields__}
        )
        rows.append(
            {
                "token": row["query_id"],
                "original_query_sha256": sha(path),
                "strict_parity": strict_compare_results(cpu, ExpansionResult(**original["result"])),
            }
        )
    sources = [Path(__file__), Path("tests/unit/test_phase_event_expansion_all_cuda.py")]
    receipt = {
        "schema": "phase_event_strict_postvalidation_v1",
        "rows": rows,
        "source_sha256": {str(p): sha(p) for p in sources},
        "original_config_sha256": sha(directory / "CONFIG.json"),
        "original_summary_sha256": sha(directory / "SUMMARY.json"),
        "all_parity_pass": all(r["strict_parity"]["passed"] for r in rows),
        "original_run_modified": False,
        "inference_rerun": False,
        "labels_opened": False,
        "q_b_absolute_tolerance": 1e-5,
        "ttc_difference_bound": "abs(dq)/abs(q_cpu*q_candidate) plus reciprocal rounding allowance",
        "reciprocal_relative_tolerance": 1e-12,
    }
    atomic_receipt(output / "POSTVALIDATION.json", receipt)
    for path in sources:
        destination = output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(f"POSTVALIDATION all_parity_pass={receipt['all_parity_pass']}; no inference rerun")


def main() -> None:
    """Freeze all inputs before one shared-cache CUDA pass and record every parity result."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "synthetic", "cpu", "profile", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--preflight-only", action="store_true", help="Validate without CUDA or output"
    )
    parser.add_argument(
        "--postvalidate", type=Path, help="CPU-only reassessment of existing CUDA receipts"
    )
    args = parser.parse_args()
    complete_started = time.perf_counter()
    if args.output.exists():
        raise ValueError("Use a new immutable output directory")
    evidence = preflight(args.cache, args.synthetic, args.cpu, args.profile)
    preflight_seconds = time.perf_counter() - complete_started
    if args.postvalidate:
        postvalidate_previous(args.postvalidate, args.output, evidence)
        return
    if args.preflight_only:
        print("PREFLIGHT PASS: all77 sensor/CPU identities and source bindings; no CUDA allocation")
        return
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cuda_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required; no fallback")
    torch.cuda.init()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - cuda_started
    paths = {
        Path(__file__),
        Path("tests/unit/test_phase_event_expansion_all_cuda.py"),
        Path("operational/phase_event_expansion/validation.py"),
        Path("operational/evttc_competition/geometry.py"),
    }
    paths |= {Path(p) for p in evidence["accelerator_archived_sources"]}
    config = {
        "schema": "phase_event_all77_cuda_parity_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "solver": evidence["recipe"],
        "precision": "float64",
        "queries": 77,
        "q_b_absolute_tolerance": 1e-5,
        "ttc_difference_bound": "abs(dq)/abs(q_cpu*q_candidate) plus reciprocal rounding allowance",
        "reciprocal_relative_tolerance": 1e-12,
        "validity_reason": "exact equality",
        "optimizer_count_changes": "recorded, not asserted identical",
        "labels_opened": False,
        "cpu_rerun": False,
        "production_integrated": False,
        "cache_policy": "one shared CudaObjectiveCache; one resident event shape; B1/B81 graphs",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "threads": 2,
        "source_sha256": {str(p): sha(p) for p in sorted(paths)},
        "cpu_query_sha256": evidence["cpu_query_sha256"],
        "evidence_sha256": {
            str(p): sha(p)
            for p in (
                args.cpu / "CONFIG.json",
                args.cpu / "PREDICTIONS_SEALED.json",
                args.cpu / "PREDICTIONS.npz",
                args.cache / "MANIFEST.json",
                args.cache / "CACHE_FREEZE.json",
                args.synthetic / "CONFIG.json",
                args.synthetic / "SUMMARY.json",
                args.synthetic / "CONFIRMATION_BINDING.json",
                args.profile / "CONFIG.json",
                args.profile / "SUMMARY.json",
            )
        },
        "historical_seal_limit": "CPU q/b receipts were not individually sealed; token/TTC checked "
        "against historical sealed NPZ; all query receipt hashes frozen here before inference",
        "guarded_evaluator": "exact reviewed guard-only hash; calibration AST equals archive",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for path in paths:
        relative = path.resolve().relative_to(Path.cwd())
        destination = args.output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    active = CudaObjectiveCache()
    rows = []
    loop_started = time.perf_counter()
    try:
        for i, row in enumerate(evidence["manifest"]["rows"]):
            query_started = time.perf_counter()
            path = args.cache / row["file"]
            if sha(path) != row["sha256"]:
                raise ValueError("Sensor window changed after preflight")
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            validate_cached_window(arrays, row)
            read_seconds = time.perf_counter() - query_started
            undistort_started = time.perf_counter()
            try:
                events, roi = undistort_inputs(arrays, evidence["manifest"]["calibration"])
            except ValueError as exc:
                result = ExpansionResult(
                    None,
                    None,
                    None,
                    False,
                    "calibration_inversion_failure",
                    time.perf_counter() - undistort_started,
                    {"error": str(exc)},
                )
                undistort_seconds = time.perf_counter() - undistort_started
                solver_seconds = 0.0
            else:
                undistort_seconds = time.perf_counter() - undistort_started
                solver_started = time.perf_counter()
                result = estimate_accelerated(
                    events, roi, int(row["query_us"]), SolverConfig(**evidence["recipe"]), active
                )
                torch.cuda.synchronize()
                solver_seconds = time.perf_counter() - solver_started
            cpu = ExpansionResult(
                **{k: evidence["cpu_records"][i][k] for k in ExpansionResult.__dataclass_fields__}
            )
            record = {
                "token": row["query_id"],
                "index": row["frame_index"],
                "query_us": row["query_us"],
                "result": asdict(result),
                "parity": strict_compare_results(cpu, result),
                "cpu_query_sha256": evidence["cpu_query_sha256"][f"{i:05d}.json"],
                "read_validate_seconds": read_seconds,
                "undistort_seconds": undistort_seconds,
                "solver_wall_seconds": solver_seconds,
                "query_wall_before_receipt_seconds": time.perf_counter() - query_started,
                "captures_cumulative": active.capture_count,
                "replays_cumulative": active.replay_count,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_receipt(args.output / "queries" / f"{i:05d}.json", record)
            rows.append(record)
            print(
                f"QUERY {i + 1}/77 parity={record['parity']['passed']} reason={result.reason} "
                f"wall={record['query_wall_before_receipt_seconds']:.4f}s",
                flush=True,
            )
        captures, replays = active.capture_count, active.replay_count
        loop_seconds = time.perf_counter() - loop_started
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    finally:
        close_started = time.perf_counter()
        active.close()
        close_seconds = time.perf_counter() - close_started
    summary = {
        "schema": "phase_event_all77_cuda_parity_v1",
        "rows": rows,
        "all_parity_pass": all(r["parity"]["passed"] for r in rows) and len(rows) == 77,
        "all_memory_pass": max(allocated, reserved) < 1_000_000_000,
        "optimizer_count_divergences": [
            r["token"] for r in rows if not r["parity"]["objective_count_matches"]
        ],
        "capture_count": captures,
        "replay_count": replays,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "timing": {
            "preflight_seconds": preflight_seconds,
            "cuda_initialization_seconds": initialization_seconds,
            "all77_loop_with_receipts_seconds": loop_seconds,
            "cache_close_seconds": close_seconds,
            "complete_wall_before_summary_seconds": time.perf_counter() - complete_started,
        },
        "labels_opened": False,
        "cpu_rerun": False,
        "config_sha256": sha(args.output / "CONFIG.json"),
        "query_receipt_sha256": {
            p.name: sha(p) for p in sorted((args.output / "queries").glob("*.json"))
        },
    }
    atomic_receipt(args.output / "SUMMARY.json", summary)
    print(
        f"COMPLETE parity={summary['all_parity_pass']} memory={summary['all_memory_pass']}",
        flush=True,
    )
    if not summary["all_parity_pass"] or not summary["all_memory_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
