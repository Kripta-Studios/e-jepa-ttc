#!/usr/bin/env python
"""Profile the isolated exact AP101 candidate on retained campaign payloads."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import ast
import cProfile
import ctypes
import hashlib
import json
import pstats
import random
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection import evaluation_v2  # noqa: E402
from e_jepa_ttc.detection.evaluation_ap_fast import (  # noqa: E402
    ap101_value,
    ap101_with_curve,
)
from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationBox,
    EvaluationGroundTruth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402

DEFAULT_INPUT = ROOT / (
    "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
    "r0_1024/evaluation/selection_update_001024.json"
)
DEFAULT_OUTPUT = ROOT / "output/evaluation_ap_fast/ap101_integration_v4"
MAX_RSS_BYTES = 1024**3
FROZEN_ROOT = ROOT / (
    "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
    "r0_1024/code/src/e_jepa_ttc/detection"
)
FROZEN_V2 = FROZEN_ROOT / "evaluation_v2.py"
FROZEN_LEGACY = FROZEN_ROOT / "development_metrics.py"
FROZEN_SHA256 = {
    FROZEN_V2: "0264fb73f46ad887cd1048ea2c8cc1e417d52ff7ed369203f6c0d143a61783bc",
    FROZEN_LEGACY: "1f53f8fc042d1840d95c9a438d334bbd6ab82e7f56fb9767cefae6b7bcb582c6",
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_bytes() -> tuple[int, int]:
    """Return current and peak Windows working sets without extra dependencies."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def rss_bytes() -> int:
    """Return the current Windows process working set."""
    return memory_bytes()[0]


def require_rss(stage: str) -> int:
    observed, peak = memory_bytes()
    if observed >= MAX_RSS_BYTES or peak >= MAX_RSS_BYTES:
        raise MemoryError(
            f"RSS gate failed at {stage}: current={observed}, peak={peak}, "
            f"limit={MAX_RSS_BYTES}"
        )
    return observed


def box_from_dict(value: dict[str, Any]) -> EvaluationBox:
    return EvaluationBox(
        class_name=value["class_name"],
        translation_m=tuple(value["translation_m"]),
        size_lwh_m=tuple(value["size_lwh_m"]),
        yaw_rad=value["yaw_rad"],
        source_index=value["source_index"],
        ignore_reason=value["ignore_reason"],
    )


def ground_truth_from_dict(value: dict[str, Any]) -> EvaluationGroundTruth:
    ignored_by_reason = value["ignored_by_reason"]
    return EvaluationGroundTruth(
        protocol_sha256=value["protocol_sha256"],
        eligible=tuple(box_from_dict(row) for row in value["eligible"]),
        ignored=tuple(box_from_dict(row) for row in value["ignored"]),
        raw_count=value["raw_count"],
        ignored_by_reason=tuple(sorted(ignored_by_reason.items())),
        intrinsic=tuple(tuple(row) for row in value["intrinsic"]),
        event_from_ego=tuple(tuple(row) for row in value["event_from_ego"]),
        image_wh=tuple(value["image_wh"]),
    )


def frozen_ap101(path: Path) -> Callable[[Sequence[int], int], object]:
    """Compile only `_ap101` from an exact hash-pinned frozen source snapshot."""
    observed = sha256_file(path)
    if observed != FROZEN_SHA256[path]:
        raise ValueError(f"Frozen AP source SHA-256 mismatch: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_ap101"
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ValueError(f"Expected exactly one synchronous frozen _ap101: {path}")
    module = ast.fix_missing_locations(ast.Module(body=[functions[0]], type_ignores=[]))
    namespace: dict[str, object] = {"Sequence": Sequence}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - hash-pinned source
    function = namespace["_ap101"]
    if not callable(function):
        raise TypeError("Frozen _ap101 did not compile to a callable")
    return function


def direct_parity(
    original_v2: Callable[[Sequence[int], int], object],
    original_legacy: Callable[[Sequence[int], int], object],
) -> dict[str, object]:
    cases: list[tuple[list[int], int]] = [
        ([], 0),
        ([], 7),
        ([0], 1),
        ([1], 1),
        ([0] * 100, 1),
        ([1] * 100, 100),
        ([0, 1] * 100, 100),
        ([1, 0] * 100, 100),
        ([0] * 99 + [1], 1),
        ([1] * 100 + [0] * 100, 100),
    ]
    generator = random.Random(7)
    for _ in range(1000):
        length = generator.randrange(0, 1025)
        cases.append(
            ([generator.randrange(2) for _ in range(length)], generator.randrange(0, 1025))
        )
    for index, (flags, count) in enumerate(cases):
        if ap101_with_curve(flags, count) != original_v2(flags, count):
            raise AssertionError(f"v2 AP mismatch at direct case {index}")
        if ap101_value(flags, count) != original_legacy(flags, count):
            raise AssertionError(f"legacy AP mismatch at direct case {index}")
    return {"status": "PASS", "case_count": len(cases), "seed": 7}


def profile_call(
    predictions: dict[str, Any],
    ground_truth: dict[str, EvaluationGroundTruth],
    protocol: EvaluationProtocol,
    *,
    helper: object,
    profile_path: Path,
) -> tuple[dict[str, object], float, int]:
    previous = evaluation_v2._ap101
    evaluation_v2._ap101 = helper  # type: ignore[assignment]
    profiler = cProfile.Profile()
    try:
        started = time.perf_counter()
        profiler.enable()
        result = evaluation_v2.evaluate_detections_v2(
            predictions, ground_truth, protocol=protocol
        )
        profiler.disable()
        elapsed = time.perf_counter() - started
    finally:
        evaluation_v2._ap101 = previous
    profiler.dump_stats(str(profile_path))
    stats_path = profile_path.with_suffix(".txt")
    with stats_path.open("w", encoding="utf-8") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
    return result, elapsed, require_rss(profile_path.stem)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256, 1024])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        print("direct parity", flush=True)
        original_v2 = frozen_ap101(FROZEN_V2)
        original_legacy = frozen_ap101(FROZEN_LEGACY)
        direct = direct_parity(original_v2, original_legacy)
        print("load retained payload", flush=True)
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        protocol = EvaluationProtocol.default()
        if payload["metrics_v2"]["protocol_sha256"] != protocol.sha256:
            raise ValueError("Retained payload protocol differs from default")
        tokens = sorted(payload["predictions"])
        ground_truth_all = {
            token: ground_truth_from_dict(payload["ground_truth_v2"][token]) for token in tokens
        }
        expected_gt_hashes = payload["metrics_v2"]["population"][
            "ground_truth_sha256_by_token"
        ]
        if any(
            ground_truth_all[token].sha256 != expected_gt_hashes[token] for token in tokens
        ):
            raise ValueError("Reconstructed retained ground truth hash mismatch")
        require_rss("payload_load")

        subsets = []
        for size in args.sizes:
            selected = tokens[:size]
            predictions = {token: payload["predictions"][token] for token in selected}
            ground_truth = {token: ground_truth_all[token] for token in selected}
            token_sha = canonical_sha256(selected)
            print(f"profile original {size}", flush=True)
            baseline, baseline_seconds, baseline_rss = profile_call(
                predictions,
                ground_truth,
                protocol,
                helper=original_v2,
                profile_path=output / f"original_{size}.prof",
            )
            baseline_sha = canonical_sha256(baseline)
            print(f"profile fast {size}", flush=True)
            candidate, candidate_seconds, candidate_rss = profile_call(
                predictions,
                ground_truth,
                protocol,
                helper=ap101_with_curve,
                profile_path=output / f"fast_{size}.prof",
            )
            if candidate != baseline:
                raise AssertionError(f"Deep result parity failed for {size} frames")
            candidate_sha = canonical_sha256(candidate)
            if candidate_sha != baseline_sha:
                raise AssertionError(f"Canonical result hash parity failed for {size} frames")
            subsets.append(
                {
                    "frame_count": size,
                    "token_list_sha256": token_sha,
                    "result_sha256": baseline_sha,
                    "deep_exact_equal": True,
                    "original_seconds": baseline_seconds,
                    "fast_seconds": candidate_seconds,
                    "speedup": baseline_seconds / candidate_seconds,
                    "rss_after_original_bytes": baseline_rss,
                    "rss_after_fast_bytes": candidate_rss,
                    "process_peak_rss_bytes": memory_bytes()[1],
                    "original_profile": f"original_{size}.prof",
                    "fast_profile": f"fast_{size}.prof",
                }
            )
            del baseline, candidate, predictions, ground_truth
            print(f"complete {size}", flush=True)

        receipt = {
            "schema": "phase_ttc_detection_ap101_exact_profile_v1",
            "status": "PASS",
            "cpu_threads": 1,
            "gpu_used": False,
            "input": {
                "path": args.input.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.input),
                "frame_count": len(tokens),
            },
            "sources": {
                "fast_helper_sha256": sha256_file(
                    ROOT / "src/e_jepa_ttc/detection/evaluation_ap_fast.py"
                ),
                "frozen_v2_sha256": sha256_file(
                    FROZEN_V2
                ),
                "frozen_legacy_sha256": sha256_file(
                    FROZEN_LEGACY
                ),
                "integrated_v2_sha256": sha256_file(
                    ROOT / "src/e_jepa_ttc/detection/evaluation_v2.py"
                ),
                "integrated_legacy_sha256": sha256_file(
                    ROOT / "src/e_jepa_ttc/detection/development_metrics.py"
                ),
                "profiler_sha256": sha256_file(Path(__file__)),
            },
            "protocol_sha256": protocol.sha256,
            "direct_helper_parity": direct,
            "subsets": subsets,
            "rss_gate_bytes": MAX_RSS_BYTES,
            "process_peak_rss_bytes": memory_bytes()[1],
            "limitations": [
                f"Profiles cover retained subsets through {max(args.sizes):,} frames, not "
                "the 23,958-frame panel.",
                "Only AP101 is monkeypatched in-process; frozen evaluator source files "
                "are unchanged.",
                "Elapsed times include the complete v2 evaluator and cProfile overhead.",
                "Legacy helper parity is direct; retained payload does not serialize legacy "
                "ground truth for a legacy end-to-end rerun.",
            ],
        }
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("PASS", flush=True)
        return 0
    except BaseException as error:
        failure = {
            "schema": "phase_ttc_detection_ap101_exact_profile_failure_v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (output / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
