"""Parity-gated offline TRAIN40 replay; not a physical 50 Hz certification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402

from operational.train40_runtime.measurement import (  # noqa: E402
    coverage_report,
    latency_distribution,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.predictor import PersistentTrain40Predictor  # noqa: E402
from scripts.final_train40_transfer import (  # noqa: E402
    FinalTransferPredictor,
    load_models,
    read,
    reject_targets,
    sha,
    verify_candidate,
    verify_frozen_runtime,
)

SCOPE = "OFFLINE_DISK_INCLUSIVE_PRIMARY_OBJECT_QUERY_REPLAY"
MODES = ("oracle", "eager", "graphs", "full-graphs", "batched-rgb", "block-graphs", "reuse-graphs")


def _write(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_output(output: Path, protected: Sequence[Path]) -> None:
    output = output.resolve()
    if any(
        output == p.resolve()
        or output.is_relative_to(p.resolve())
        or p.resolve().is_relative_to(output)
        for p in protected
    ):
        raise ValueError("Output cannot equal, contain, or be inside a frozen source directory")


def _candidate_inventory(candidate: Path) -> list[Path]:
    candidate = candidate.resolve(strict=True)
    inventory = candidate / "PORTABLE_PATHS.json"
    files = [inventory]
    for name, entry in read(inventory)["components"].items():
        path = Path(entry["local_path"]).resolve(strict=True)
        if not path.is_relative_to(candidate) or sha(path) != entry["sha256"]:
            raise ValueError(f"Portable component location/bytes differ: {name}")
        files.append(path)
    return files


def _validate_reference(
    reference: Path, manifest: Path, rows: list[dict[str, Any]], binding: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    reference = reference.resolve(strict=True)
    parent = reference.parent
    seal_path, freeze_path = parent / "PREDICTIONS_SEALED.json", parent / "INFERENCE_FREEZE.json"
    seal, freeze = read(seal_path), read(freeze_path)
    manifest_sha, freeze_sha = sha(manifest), sha(freeze_path)
    if (
        seal.get("schema") != "final_train40_transfer_predictions_v1"
        or seal.get("status") != "COMPLETE"
        or seal.get("queries") != len(rows)
        or seal.get("manifest_sha256") != manifest_sha
        or seal.get("binding_sha256") != freeze_sha
        or freeze.get("manifest_sha256") != manifest_sha
        or freeze.get("model_seal_sha256") != binding["model_seal_sha256"]
        or freeze.get("model_hashes") != binding["components"]
        or any(item.get("targets_read") is not False for item in (seal, freeze))
        or any(item.get("optimizer_updates") != 0 for item in (seal, freeze))
    ):
        raise ValueError("Reference seal does not bind this complete population and candidate")
    fragments = {}
    for name, digest in seal.get("fragments", {}).items():
        path = (parent / name.replace("\\", "/")).resolve(strict=True)
        if not path.is_relative_to(reference) or path in fragments:
            raise ValueError("Reference seal has an escaped or duplicate fragment path")
        fragments[path] = digest
    expected = [reference / f"query_{index:05d}.json" for index in range(len(rows))]
    if set(fragments) != set(expected):
        raise ValueError("Reference seal fragment population differs from the full manifest")
    results = []
    for path, row in zip(expected, rows, strict=True):
        if sha(path) != fragments[path]:
            raise ValueError(f"Reference prediction bytes differ: {path.name}")
        result = read(path)
        if (
            result.get("query_id") != row["query_id"]
            or result.get("manifest_sha256") != manifest_sha
            or result.get("binding_sha256") != freeze_sha
            or result.get("targets_read") is not False
            or result.get("optimizer_updates") != 0
        ):
            raise ValueError(f"Reference prediction identity differs: {path.name}")
        results.append(result)
    return results, [seal_path, freeze_path, *expected], freeze


def _source_inventory(frozen: dict[str, Any], reference_binding: dict[str, Any]) -> list[Path]:
    files = {Path(__file__).resolve(), ROOT / "scripts/final_train40_transfer.py"}
    files.update((ROOT / "operational/train40_runtime").glob("*.py"))
    sources = {**frozen["source_sha256"], **reference_binding["sources"]}
    for name, expected in sources.items():
        path = Path(name)
        path = (
            path.resolve(strict=True) if path.is_absolute() else (ROOT / path).resolve(strict=True)
        )
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"Frozen oracle source location/bytes differ: {name}")
        files.add(path)
    return sorted(files)


def _route_summary(
    records: list[list[dict[str, Any]]], reference: list[dict[str, Any]], expected: list[str]
) -> list[dict[str, Any]]:
    result = []
    for repetition, current in enumerate(records):
        summary = {
            "repetition": repetition,
            "parity": prediction_parity(reference, current, expected),
        }
        if current:
            summary["latency_ms"] = latency_distribution(
                [item["complete_wall_ms"] for item in current]
            )
            keys = set(current[0]["seconds"])
            if any(set(item["seconds"]) != keys for item in current):
                raise ValueError("Stage timing fields changed inside a route/repetition")
            summary["stages_ms"] = {
                key: latency_distribution([item["seconds"][key] * 1000 for item in current])
                for key in sorted(keys)
            }
        result.append(summary)
    return result


def run_benchmark(args: argparse.Namespace) -> None:
    """Preserve partial evidence and verify original bytes after every run failure."""
    _validate_output(args.output, [args.candidate, args.reference, args.manifest.parent])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = {mode: [[] for _ in range(args.repeats)] for mode in args.modes}
    predictors: dict[str, Any] = {}
    original: dict[str, dict[str, Any]] = {}
    setup: dict[str, float] = {}
    reference: list[dict[str, Any]] = []
    expected: list[str] = []
    summaries: dict[str, Any] = {}
    errors: list[str] = []
    caught: BaseException | None = None
    loop_complete = False
    intact: dict[str, Any] = {"passed": False, "reason": "inventory_not_established"}
    peak_memory = None
    try:
        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        all_rows = read(args.manifest)["rows"]
        reject_targets(all_rows)
        coverage_report([row["query_id"] for row in all_rows], all_rows)
        indices = (
            args.indices if args.indices is not None else list(range(0, len(all_rows), args.stride))
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(not 0 <= index < len(all_rows) for index in indices)
        ):
            raise ValueError("Query indices must be unique, nonempty and inside the manifest")
        full_reference, reference_files, ref_binding = _validate_reference(
            args.reference, args.manifest, all_rows, binding
        )
        sources = _source_inventory(frozen, ref_binding)
        original = snapshot_files(
            [*_candidate_inventory(args.candidate), args.manifest, *reference_files, *sources]
        )
        _write(output / "original_snapshot.json", original)
        rows = [all_rows[index] for index in indices]
        expected = [row["query_id"] for row in rows]
        reference = [full_reference[index] for index in indices]
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        binding.update(
            manifest_sha256=sha(args.manifest),
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
            ).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            reference_seal_sha256=sha(args.reference.parent / "PREDICTIONS_SEALED.json"),
            reference_binding_sha256=sha(args.reference.parent / "INFERENCE_FREEZE.json"),
            device=torch.cuda.get_device_name(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            precision="FP32_TF32_OFF",
            seed=7,
            cpu_threads=2,
            cudnn_benchmark=False,
            frozen_runtime=frozen,
            indices=indices,
            repeats=args.repeats,
            modes=args.modes,
            warmup=args.warmup,
            synchronized_stage_profiling=args.synchronize_stages,
            scope=SCOPE,
            target_50hz_demonstrated=False,
            acceptance_tolerance={"atol": 1e-5, "rtol": 1e-5},
            cache_policy=(
                "Clear service raw caches after warmup; OS cache uncontrolled; "
                "retain history across queries"
            ),
            ordering="Alternate route order by query and repetition; repeats reported separately",
        )
        _write(output / "binding.json", binding)
        start = time.perf_counter()
        models, normalizers = load_models(paths, "cuda")
        setup["model_load_seconds"] = time.perf_counter() - start
        for mode in args.modes:
            if mode == "oracle":
                oracle = FinalTransferPredictor.__new__(FinalTransferPredictor)
                oracle.device, oracle.models, oracle.normalizers = (
                    torch.device("cuda"),
                    models,
                    normalizers,
                )
                predictors[mode] = oracle
            else:
                predictors[mode] = PersistentTrain40Predictor(
                    models,
                    normalizers,
                    graphs=mode != "eager",
                    context_graphs=mode
                    in {"full-graphs", "batched-rgb", "block-graphs", "reuse-graphs"},
                    batch_rgb=mode in {"batched-rgb", "block-graphs", "reuse-graphs"},
                    block_graphs=mode in {"block-graphs", "reuse-graphs"},
                    reuse_encodings=mode == "reuse-graphs",
                    synchronize_stages=args.synchronize_stages,
                )
        for mode in args.modes:
            start = time.perf_counter()
            for _ in range(args.warmup):
                predictors[mode].predict(rows[0])
            if mode != "oracle":
                predictors[mode].inputs.close()
            setup[mode] = time.perf_counter() - start
            print(json.dumps({"warmup": mode, "seconds": setup[mode]}), flush=True)
        _write(output / "setup.json", setup)
        torch.cuda.reset_peak_memory_stats()
        for position, (index, row) in enumerate(zip(indices, rows, strict=True)):
            for repetition in range(args.repeats):
                modes = (
                    args.modes if (position + repetition) % 2 == 0 else list(reversed(args.modes))
                )
                for mode in modes:
                    torch.cuda.synchronize()
                    start_ns = time.perf_counter_ns()
                    result = predictors[mode].predict(row)
                    torch.cuda.synchronize()
                    completed_ns = time.perf_counter_ns()
                    result.update(
                        query_id=row["query_id"],
                        index=index,
                        mode=mode,
                        repetition=repetition,
                        started_ns=start_ns,
                        completed_ns=completed_ns,
                        complete_wall_ms=(completed_ns - start_ns) / 1e6,
                        execution_order=list(modes),
                    )
                    records[mode][repetition].append(result)
                    with (output / "queries.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                json.dumps({"completed": position + 1, "queries": len(rows), "index": index}),
                flush=True,
            )
        peak_memory = torch.cuda.max_memory_allocated()
        loop_complete = True
    except BaseException as exc:
        caught = exc
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        for mode, predictor in predictors.items():
            if mode != "oracle":
                try:
                    predictor.close()
                except BaseException as exc:
                    errors.append(f"cleanup {mode}: {type(exc).__name__}: {exc}")
        if original:
            try:
                intact = verify_snapshot(original)
            except BaseException as exc:
                errors.append(f"immutable verification: {type(exc).__name__}: {exc}")
        if expected:
            for mode in args.modes:
                try:
                    summaries[mode] = _route_summary(records[mode], reference, expected)
                except BaseException as exc:
                    errors.append(f"summary {mode}: {type(exc).__name__}: {exc}")
        all_parity = len(summaries) == len(args.modes) and all(
            item["parity"]["passed"] for route in summaries.values() for item in route
        )
        passed = loop_complete and all_parity and intact["passed"] and not errors
        status = (
            "COMPLETE"
            if passed
            else "INTERRUPTED"
            if isinstance(caught, (KeyboardInterrupt, SystemExit))
            else "FAILED"
        )
        _write(
            output / "summary.json",
            {
                "status": status,
                "passed": passed,
                "scope": SCOPE,
                "setup_seconds": setup,
                "routes": summaries,
                "originals": intact,
                "errors": errors,
                "error": errors[0] if errors else None,
                "loop_complete": loop_complete,
                "peak_allocated_cuda_bytes": peak_memory,
                "target_50hz_demonstrated": False,
                "reason": (
                    "Offline primary-object replay has no paced physical multiobject arrivals"
                ),
            },
        )
    if caught is not None:
        raise caught.with_traceback(caught.__traceback__)
    if not passed:
        raise RuntimeError(
            "Benchmark failed: inspect summary.json for parity or preservation failures"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Validate options before creating a fresh experiment directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "manifest", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["oracle", "full-graphs"])
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--synchronize-stages", action="store_true")
    args = parser.parse_args(argv)
    if min(args.stride, args.repeats) < 1 or args.warmup < 0:
        parser.error("stride/repeats must be positive and warmup nonnegative")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Runtime modes must be unique")
    run_benchmark(args)


if __name__ == "__main__":
    main()
