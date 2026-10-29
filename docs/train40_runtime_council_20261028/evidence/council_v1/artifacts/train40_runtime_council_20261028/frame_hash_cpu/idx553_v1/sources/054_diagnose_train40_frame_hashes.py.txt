"""Compare SHA-256 and single-thread BLAKE3 on unchanged real RGB frame bytes.

The two scopes include or exclude a fresh immutable payload copy. This is an
isolated CPU diagnostic; it changes no runtime token, normalizer, model or gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import zipfile
from importlib.metadata import Distribution
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
SCOPE = "CPU_ONLY_SHA256_VS_BLAKE3_256_SINGLE_THREAD_NO_RUNTIME_CHANGE"


def load_dependency(directory: Path) -> tuple[Any, dict[str, Any], list[Path]]:
    """Verify the pinned wheel and installed bytes before private process-local import."""
    directory = directory.resolve()
    receipt_path = directory / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["package"] != "blake3" or receipt["version"] != "1.0.11":
        raise ValueError("This diagnostic requires the separately preserved blake3 1.0.11 wheel")
    wheel = directory / "wheels" / receipt["filename"]
    payload = wheel.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["wheel_sha256"]:
        raise ValueError("Pinned BLAKE3 wheel SHA-256 differs")
    metadata = directory / "pypi_release_metadata.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != receipt["release_metadata_sha256"]:
        raise ValueError("Pinned PyPI release metadata differs")
    site = directory / "site"
    files = [wheel, receipt_path, metadata]
    installed = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            path = (site / name).resolve()
            if not path.is_relative_to(site) or not path.is_file():
                raise ValueError("Installed dependency file missing or outside private target")
            original = archive.read(name)
            if path.read_bytes() != original:
                raise ValueError(f"Installed dependency bytes differ from wheel: {name}")
            files.append(path)
            installed.append({"path": name, "sha256": hashlib.sha256(original).hexdigest()})
    sys.path.insert(0, str(site))
    module = importlib.import_module("blake3")
    filename = module.__file__
    if filename is None or not Path(filename).resolve().is_relative_to(site):
        raise ValueError("BLAKE3 must load from the verified private target")
    extension = importlib.import_module("blake3.blake3")
    extension_file = extension.__file__
    if extension_file is None or Path(extension_file).resolve() not in files:
        raise ValueError("BLAKE3 native extension must be one of the verified wheel files")
    installed_version = Distribution.at(site / "blake3-1.0.11.dist-info").version
    if installed_version != receipt["version"]:
        raise ValueError("Installed BLAKE3 version differs from the pinned receipt")
    return (
        module,
        {
            **receipt,
            "installed_files": installed,
            "imported_module": str(Path(filename).resolve()),
            "imported_extension": str(Path(extension_file).resolve()),
            "installed_version": installed_version,
            "api_reference": "https://github.com/oconnor663/blake3-py",
            "max_threads": 1,
            "digest_bytes": 32,
            "shared_environment_modified": False,
        },
        files,
    )


def digest(payload: bytes, algorithm: str, blake3: Any) -> str:
    """Compute each requested digest independently with explicit BLAKE3 thread count."""
    if algorithm == "sha256":
        return hashlib.sha256(payload).hexdigest()
    if algorithm == "blake3":
        return blake3.blake3(payload, max_threads=1).hexdigest(length=32)
    raise ValueError("Use sha256 or blake3")


def blake3_incremental_reference(payload: bytes, blake3: Any) -> str:
    """Cross-check the one-shot path with separately chunked single-thread hashing."""
    hasher = blake3.blake3(max_threads=1)
    view = memoryview(payload)
    for start in range(0, len(view), 1024 * 1024):
        hasher.update(view[start : start + 1024 * 1024])
    return hasher.hexdigest(length=32)


def measure_route(
    raw: Any,
    originals: dict[int, Any],
    blake3_reference: dict[int, str],
    blake3: Any,
    *,
    algorithm: str,
    copy_inside: bool,
) -> dict[str, Any]:
    """Time a history with per-frame intervals; validate every byte after the clock."""
    import numpy as np

    values = []
    started = time.perf_counter_ns()
    for index in sorted(raw.frames):
        before = time.perf_counter_ns()
        payload = raw.frames[index].tobytes(order="C") if copy_inside else originals[index].payload
        copied = time.perf_counter_ns()
        result = digest(payload, algorithm, blake3)
        finished = time.perf_counter_ns()
        values.append(
            (index, payload, result, copied - before, finished - copied, finished - before)
        )
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    checks = []
    for index, payload, result, copy_ns, hash_ns, total_ns in values:
        reference = originals[index]
        expected = reference.content_sha256 if algorithm == "sha256" else blake3_reference[index]
        pixels = np.frombuffer(payload, dtype=np.uint8).reshape(raw.frames[index].shape)
        checks.append(
            {
                "frame_id": index,
                "timestamp_us": raw.timestamps_us[index],
                "shape": list(raw.frames[index].shape),
                "strides": list(raw.frames[index].strides),
                "dtype": str(raw.frames[index].dtype),
                "bytes": len(payload),
                "algorithm": algorithm,
                "digest": result,
                "expected_digest": expected,
                "original_sha256": reference.content_sha256,
                "blake3_reference": blake3_reference[index],
                "digest_matches_same_algorithm_reference": result == expected,
                "payload_bytes_exact": payload == reference.payload,
                "pixels_exact": bool(np.array_equal(pixels, raw.frames[index])),
                "copy_ms": copy_ns / 1e6 if copy_inside else 0.0,
                "payload_access_ms_excluded_copy_scope": copy_ns / 1e6 if not copy_inside else 0.0,
                "hash_ms": hash_ns / 1e6,
                "frame_total_ms": total_ns / 1e6,
            }
        )
    passed = all(
        row["digest_matches_same_algorithm_reference"]
        and row["payload_bytes_exact"]
        and row["pixels_exact"]
        for row in checks
    )
    return {
        "algorithm": algorithm,
        "scope": "copy_plus_hash" if copy_inside else "hash_only_preexisting_bytes",
        "max_threads": 1 if algorithm == "blake3" else None,
        "digest_bytes": 32,
        "wall_ms": wall_ms,
        "copy_ms": sum(row["copy_ms"] for row in checks),
        "hash_ms": sum(row["hash_ms"] for row in checks),
        "frame_checks": checks,
        "passed": passed,
        "validation_outside_clock": True,
    }


def run(args: argparse.Namespace) -> None:
    """Preserve every warmup/measured paired route with source and wheel bindings."""
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("Nonnegative warmups and positive repeats required")
    output, manifest = args.output.resolve(), args.manifest.resolve()
    protected = (manifest.parent, args.dependencies.resolve())
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError("Evidence must be separate from frozen inputs and dependency artifacts")
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": "train40_frame_hash_cpu_diagnostic_v1",
        "scope": SCOPE,
        "status": "FAILED",
        "gpu_used": False,
        "runtime_modified": False,
        "token_v2_implemented": False,
        "optimizer_updates": 0,
        "targets_read": False,
        "weights_or_normalizers_loaded": False,
        "records": [],
    }
    original: dict[str, Any] = {}
    raw_path = None
    raw_stat = None
    caught = None
    try:
        blake3, dependency, dependency_files = load_dependency(args.dependencies)
        import numpy as np
        import torch

        from operational.train40_runtime.inputs import PersistentSensorInputs
        from operational.train40_runtime.measurement import snapshot_files, verify_snapshot
        from operational.train40_runtime.multiobject import read_raw_rgb_history
        from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
        from scripts.benchmark_train40_rgb_pil_exact import project_source_paths
        from scripts.diagnose_train40_frame_sealing import distribution
        from scripts.final_train40_transfer import reject_targets

        torch.set_num_threads(2)
        rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
        reject_targets(rows)
        if not 0 <= args.index < len(rows):
            raise ValueError("Index lies outside preserved query manifest")
        row = rows[args.index]
        raw_path = Path(row["raw_path"]).resolve()
        stat = raw_path.stat()
        raw_stat = (stat.st_size, stat.st_mtime_ns)
        source = RGBSourceIdentity(
            str(raw_path), hashlib.sha256(json.dumps(raw_stat).encode()).hexdigest()
        )
        sources = {Path(__file__).resolve(), *project_source_paths(sys.modules, ROOT)}
        original = snapshot_files([manifest, *sources, *dependency_files])
        (output / "original_snapshot.json").write_text(
            json.dumps(original, indent=2), encoding="utf-8"
        )
        (output / "sources").mkdir()
        saved_sources = []
        for index, path in enumerate(sorted(sources)):
            saved = Path("sources") / f"{index:03d}_{path.name}.txt"
            payload = path.read_bytes()
            (output / saved).write_bytes(payload)
            saved_sources.append(
                {
                    "source": str(path.relative_to(ROOT)),
                    "snapshot": saved.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (output / "sources.json").write_text(json.dumps(saved_sources, indent=2), encoding="utf-8")
        summary["dependency"] = dependency
        summary["binding"] = {
            "command_argv": sys.argv,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "torch_threads": torch.get_num_threads(),
            "clock": vars(time.get_clock_info("perf_counter")),
            "index": args.index,
            "query_id": row["query_id"],
            "query_us": int(row["anchor_us"]),
            "raw_path": str(raw_path),
            "raw_size_bytes": raw_stat[0],
            "raw_mtime_ns": raw_stat[1],
            "warmup_pairs_per_scope": args.warmups,
            "measured_pairs_per_scope": args.repeats,
            "paired_order": "Algorithms and copy scopes each alternate AB/BA per repeat",
            "reference_setup": "Original freeze SHA-256 plus incremental BLAKE3 outside all timing",
            "future_frames_prehashed": False,
            "sustained_or_causal_claim": False,
            "clock_excludes": "HDF5, reference setup, verification, source inventory and reporting",
        }
        with PersistentSensorInputs() as inputs:
            raw = read_raw_rgb_history(inputs, row)
            if len(raw.frames) != 7 or any(
                value > raw.query_us for value in raw.timestamps_us.values()
            ):
                raise ValueError("Expected exactly seven already-available original RAW frames")
            setup = time.perf_counter()
            originals = {
                index: freeze_rgb_frame(raw.frames[index], index, raw.timestamps_us[index], source)
                for index in sorted(raw.frames)
            }
            expected_b3 = {
                index: blake3_incremental_reference(token.payload, blake3)
                for index, token in originals.items()
            }
            summary["reference_setup_seconds_excluded"] = time.perf_counter() - setup
            summary["references"] = [
                {
                    "frame_id": index,
                    "original_sha256": token.content_sha256,
                    "blake3_256": expected_b3[index],
                    "bytes": token.hashed_bytes,
                }
                for index, token in originals.items()
            ]
            for phase, count in (("warmup", args.warmups), ("measured", args.repeats)):
                for repeat in range(count):
                    algorithms = ["sha256", "blake3"] if repeat % 2 == 0 else ["blake3", "sha256"]
                    copies = [False, True] if repeat % 2 == 0 else [True, False]
                    for copy_inside in copies:
                        pair = {
                            "phase": phase,
                            "repeat": repeat,
                            "copy_inside": copy_inside,
                            "order": algorithms,
                            "routes": [],
                        }
                        for algorithm in algorithms:
                            pair["routes"].append(
                                measure_route(
                                    raw,
                                    originals,
                                    expected_b3,
                                    blake3,
                                    algorithm=algorithm,
                                    copy_inside=copy_inside,
                                )
                            )
                        pair["passed"] = all(route["passed"] for route in pair["routes"])
                        summary["records"].append(pair)
                        with (output / "pairs.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(pair, allow_nan=False) + "\n")
            distributions = []
            for phase in ("warmup", "measured"):
                for copy_inside in (False, True):
                    for algorithm in ("sha256", "blake3"):
                        routes = [
                            route
                            for pair in summary["records"]
                            if pair["phase"] == phase and pair["copy_inside"] == copy_inside
                            for route in pair["routes"]
                            if route["algorithm"] == algorithm
                        ]
                        if routes:
                            frames = [frame for route in routes for frame in route["frame_checks"]]
                            distributions.append(
                                {
                                    "phase": phase,
                                    "copy_inside": copy_inside,
                                    "algorithm": algorithm,
                                    "history_wall_ms": distribution(
                                        [route["wall_ms"] for route in routes]
                                    ),
                                    "per_frame_hash_ms": distribution(
                                        [frame["hash_ms"] for frame in frames]
                                    ),
                                    "per_frame_copy_ms": distribution(
                                        [frame["copy_ms"] for frame in frames]
                                    ),
                                    "per_frame_total_ms": distribution(
                                        [frame["frame_total_ms"] for frame in frames]
                                    ),
                                }
                            )
            summary["distributions"] = distributions
        expected = 2 * (args.warmups + args.repeats)
        summary["coverage"] = {
            "expected_scope_pairs": expected,
            "recorded_scope_pairs": len(summary["records"]),
            "passed": expected == len(summary["records"]),
        }
        summary["all_parity_passed"] = all(pair["passed"] for pair in summary["records"])
        summary["status"] = (
            "COMPLETE"
            if summary["coverage"]["passed"] and summary["all_parity_passed"]
            else "PARITY_FAILED"
        )
    except BaseException as error:
        caught = error
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if original:
            from operational.train40_runtime.measurement import verify_snapshot

            summary["preservation"] = verify_snapshot(original)
            if not summary["preservation"]["passed"]:
                summary["status"] = "FAILED"
        if raw_path is not None and raw_stat is not None:
            try:
                after = raw_path.stat()
                summary["raw_stat_preserved"] = (after.st_size, after.st_mtime_ns) == raw_stat
            except OSError as error:
                summary["raw_stat_preserved"] = False
                summary["raw_stat_error"] = f"{type(error).__name__}: {error}"
            if not summary["raw_stat_preserved"]:
                summary["status"] = "FAILED"
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    if caught is not None:
        raise caught
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    """Build an opt-in CPU CLI with no implicit package install or GPU use."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--dependencies", "--dependency-path", dest="dependencies", type=Path, required=True
    )
    result.add_argument("--index", type=int, default=553)
    result.add_argument("--warmups", type=int, default=2)
    result.add_argument("--repeats", type=int, default=20)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
