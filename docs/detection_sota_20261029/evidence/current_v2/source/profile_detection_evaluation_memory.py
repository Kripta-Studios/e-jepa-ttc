"""Bounded, resource-only memory scaling probe for detection evaluation v2.

The probe clones the historical 256-frame pilot prediction/GT payload and gives
each clone a synthetic token.  It does not read media or HDF5 and its scores are
deliberately not persisted: this is a resource measurement, not an evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationGroundTruth,
    build_evaluation_ground_truth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2  # noqa: E402

DEFAULT_SOURCE = (
    ROOT.parent
    / "phase-ttc-detection-local"
    / "output"
    / "portable_experiments"
    / "pilot256_seed7_velocity_v1"
    / "selection.json"
)
K = ((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0))
EVENT_FROM_EGO = (
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _memory() -> dict[str, int]:
    info = psutil.Process().memory_info()
    return {"rss_bytes": int(info.rss), "private_bytes": int(getattr(info, "private", 0))}


def _measure(label: str, operation: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    samples: list[dict[str, int]] = []
    stopped = threading.Event()

    def sample() -> None:
        while not stopped.wait(0.005):
            samples.append(_memory())

    before = _memory()
    thread = threading.Thread(target=sample, daemon=True)
    started = time.perf_counter()
    thread.start()
    try:
        value = operation()
    finally:
        stopped.set()
        thread.join()
    after = _memory()
    samples.extend((before, after))
    return value, {
        "label": label,
        "seconds": time.perf_counter() - started,
        "before": before,
        "after": after,
        "sample_count": len(samples),
        "peak_rss_bytes": max(item["rss_bytes"] for item in samples),
        "peak_private_bytes": max(item["private_bytes"] for item in samples),
    }


def _annotation(row: dict[str, object]) -> dict[str, object]:
    length, width, height = row["size_lwh_m"]  # type: ignore[misc]
    x, y, z = row["translation_m"]  # type: ignore[misc]
    return {
        "category": row["class_name"],
        "bbox_3d_ego": [x, y, z, length, height, width, row["yaw_rad"]],
    }


def _clone_population(
    source: dict[str, object], frames: int, protocol: EvaluationProtocol
) -> tuple[dict[str, list[dict[str, object]]], dict[str, EvaluationGroundTruth]]:
    source_predictions = source["predictions"]
    source_truth = source["ground_truth"]
    if not isinstance(source_predictions, dict) or not isinstance(source_truth, dict):
        raise ValueError("Pilot artifact lacks prediction/ground-truth mappings")
    tokens = sorted(source_predictions)
    if len(tokens) != 256 or set(tokens) != set(source_truth):
        raise ValueError("Expected the exact 256-frame pilot population")
    predictions: dict[str, list[dict[str, object]]] = {}
    truth: dict[str, EvaluationGroundTruth] = {}
    for index in range(frames):
        template = tokens[index % len(tokens)]
        token = f"resource_only_synthetic_{index:06d}"
        predictions[token] = copy.deepcopy(source_predictions[template])
        rows = copy.deepcopy(source_truth[template])
        truth[token] = build_evaluation_ground_truth(
            [_annotation(row) for row in rows],
            intrinsic=K,
            event_from_ego=EVENT_FROM_EGO,
            image_wh=(100, 100),
            protocol=protocol,
        )
    return predictions, truth


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _stream_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.serialization_mode == "dumps_write" else "_stream"
    receipt_path = args.output / f"worker_{args.frames}{suffix}.json"
    protocol = EvaluationProtocol.default()
    recipe = {
        "frame_count": args.frames,
        "serialization_mode": args.serialization_mode,
        "source_sha256": _sha256(args.source),
        "protocol_sha256": protocol.sha256,
        "synthetic_token_policy": "deep clones of pilot pred/GT with unique resource_only tokens",
        "max_predictions_per_frame": protocol.max_predictions_per_frame,
    }
    receipt: dict[str, object] = {
        "schema": "phase_ttc_detection_evaluation_memory_worker_v1",
        "status": "FAILED",
        "resource_only": True,
        "scientific_metric_claims": False,
        "synthetic_token_policy": "deep clones of pilot pred/GT with unique resource_only tokens",
        "frame_count": args.frames,
        "serialization_mode": args.serialization_mode,
        "source": str(args.source.resolve()),
        "source_sha256": recipe["source_sha256"],
        "executed_script_sha256": _sha256(Path(__file__).resolve()),
        "recipe": recipe,
        "recipe_sha256": _canonical_sha256(recipe),
        "process_id": os.getpid(),
    }
    try:
        source, load_phase = _measure(
            "load_source_json", lambda: json.loads(args.source.read_text(encoding="utf-8"))
        )
        population, clone_phase = _measure(
            "clone_population", lambda: _clone_population(source, args.frames, protocol)
        )
        predictions, truth = population
        metrics, score_phase = _measure(
            "evaluate_detections_v2",
            lambda: evaluate_detections_v2(predictions, truth, protocol=protocol),
        )

        def materialize_payload() -> dict[str, object]:
            return {
                "metrics_v2": metrics,
                "predictions": predictions,
                "ground_truth_v2": {token: item.to_dict() for token, item in truth.items()},
                "decode_counts": {token: {"synthetic_resource_only": True} for token in truth},
            }

        payload, payload_phase = _measure("materialize_payload", materialize_payload)
        serialized_path = args.output / f"synthetic_payload_{args.frames}{suffix}.json"
        serialize = _write_json if args.serialization_mode == "dumps_write" else _stream_json
        _, serialize_phase = _measure(
            f"json_{args.serialization_mode}", lambda: serialize(serialized_path, payload)
        )
        phases = [load_phase, clone_phase, score_phase, payload_phase, serialize_phase]
        max_rss = max(int(phase["peak_rss_bytes"]) for phase in phases)
        max_private = max(int(phase["peak_private_bytes"]) for phase in phases)
        if max_rss > args.memory_limit_bytes:
            raise MemoryError("Measured RSS exceeded the configured bounded-probe limit")
        serialized_sha256 = _sha256(serialized_path)
        reference_match = None
        if args.reference_payload is not None:
            reference_match = serialized_sha256 == _sha256(args.reference_payload)
            if not reference_match:
                raise ValueError("Streaming JSON bytes differ from the reference payload")
        receipt.update(
            {
                "status": "PASS",
                "protocol_sha256": protocol.sha256,
                "prediction_count": sum(len(rows) for rows in predictions.values()),
                "ground_truth_raw_count": sum(item.raw_count for item in truth.values()),
                "phases": phases,
                "process_peak_rss_bytes": max_rss,
                "process_peak_private_bytes": max_private,
                "serialized_payload": str(serialized_path.resolve()),
                "serialized_bytes": serialized_path.stat().st_size,
                "serialized_sha256": serialized_sha256,
                "reference_payload": (
                    str(args.reference_payload.resolve()) if args.reference_payload else None
                ),
                "reference_payload_exact_sha256_match": reference_match,
                "note": "Scores were exercised for resource measurement and intentionally omitted.",
            }
        )
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        _write_json(receipt_path, receipt)
        raise
    _write_json(receipt_path, receipt)
    return 0


def _training_residents() -> list[dict[str, object]]:
    residents = []
    own_pid = os.getpid()
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        if process.pid == own_pid:
            continue
        command = process.info.get("cmdline") or []
        if not any("train_detection_campaign.py" in token for token in command):
            continue
        info = process.info.get("memory_info")
        residents.append(
            {
                "pid": process.pid,
                "rss_bytes": int(info.rss) if info is not None else None,
                "command": command,
            }
        )
    return residents


def watchdog(args: argparse.Namespace) -> int:
    """Run exactly one owned worker and terminate only it when a safety gate trips."""
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.serialization_mode == "dumps_write" else "_stream"
    receipt_path = args.output / f"watchdog_{args.frames}{suffix}.json"
    stdout_path = args.output / f"watchdog_{args.frames}{suffix}.stdout.log"
    stderr_path = args.output / f"watchdog_{args.frames}{suffix}.stderr.log"
    available = int(psutil.virtual_memory().available)
    receipt: dict[str, object] = {
        "schema": "phase_ttc_detection_evaluation_memory_watchdog_v1",
        "status": "FAILED",
        "resource_only": True,
        "scientific_metric_claims": False,
        "frame_count": args.frames,
        "serialization_mode": args.serialization_mode,
        "source_sha256": _sha256(args.source),
        "executed_script_sha256": _sha256(Path(__file__).resolve()),
        "prestart_host_available_bytes": available,
        "prestart_required_bytes": args.prestart_available_bytes,
        "child_rss_limit_bytes": args.watchdog_child_rss_bytes,
        "host_available_floor_bytes": args.watchdog_host_available_bytes,
        "co_resident_training_processes": _training_residents(),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    if available < args.prestart_available_bytes:
        receipt["error"] = "Prestart host available memory is below the admission threshold"
        _write_json(receipt_path, receipt)
        return 2
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--frames",
        str(args.frames),
        "--serialization-mode",
        args.serialization_mode,
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--memory-limit-bytes",
        str(args.watchdog_child_rss_bytes),
    ]
    receipt["command"] = command
    maximum_child_rss = 0
    minimum_host_available = available
    sample_count = 0
    abort_reason = None
    observed_owned_pids: set[int] = set()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = psutil.Popen(command, stdout=stdout, stderr=stderr)
        receipt["child_pid"] = process.pid
        while process.poll() is None:
            try:
                owned = [process, *process.children(recursive=True)]
                child_rss = sum(int(item.memory_info().rss) for item in owned)
                observed_owned_pids.update(item.pid for item in owned)
            except psutil.NoSuchProcess:
                time.sleep(0.1)
                continue
            host_available = int(psutil.virtual_memory().available)
            maximum_child_rss = max(maximum_child_rss, child_rss)
            minimum_host_available = min(minimum_host_available, host_available)
            sample_count += 1
            if child_rss > args.watchdog_child_rss_bytes:
                abort_reason = "child_rss_limit_exceeded"
            elif host_available < args.watchdog_host_available_bytes:
                abort_reason = "host_available_floor_crossed"
            if abort_reason is not None:
                owned = [*process.children(recursive=True), process]
                for item in reversed(owned):
                    try:
                        item.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(owned, timeout=10)
                for item in alive:
                    try:
                        item.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=10)
                break
            time.sleep(0.1)
        returncode = process.wait()
    receipt.update(
        {
            "returncode": returncode,
            "watchdog_sample_count": sample_count,
            "maximum_child_rss_bytes": maximum_child_rss,
            "minimum_host_available_bytes": minimum_host_available,
            "abort_reason": abort_reason,
            "observed_owned_pids": sorted(observed_owned_pids),
            "status": "PASS" if returncode == 0 and abort_reason is None else "FAILED",
        }
    )
    worker_receipt = args.output / f"worker_{args.frames}{suffix}.json"
    if worker_receipt.exists():
        receipt["worker_receipt"] = str(worker_receipt.resolve())
        receipt["worker_receipt_sha256"] = _sha256(worker_receipt)
    _write_json(receipt_path, receipt)
    return 0 if receipt["status"] == "PASS" else 3


def _conservative_projection(rows: list[dict[str, Any]], target: int) -> dict[str, object]:
    projections: dict[str, object] = {}
    for field in ("process_peak_rss_bytes", "process_peak_private_bytes", "serialized_bytes"):
        pairs = [(int(row["frame_count"]), int(row[field])) for row in rows]
        slopes = [
            (right_value - left_value) / (right_n - left_n)
            for (left_n, left_value), (right_n, right_value) in zip(
                pairs[:-1], pairs[1:], strict=True
            )
        ]
        marginal_slope = max(slopes)
        envelope_slope = max(value / count for count, value in pairs)
        largest_n, largest_value = pairs[-1]
        marginal_linear = largest_value + max(0.0, marginal_slope) * (target - largest_n)
        envelope_linear = envelope_slope * target
        conservative_base = max(marginal_linear, envelope_linear)
        projections[field] = {
            "max_observed_marginal_bytes_per_frame": marginal_slope,
            "marginal_linear_bytes": int(marginal_linear),
            "max_observed_total_bytes_per_frame": envelope_slope,
            "total_per_frame_envelope_bytes": int(envelope_linear),
            "conservative_base_bytes": int(conservative_base),
            "conservative_1_5x_bytes": int(conservative_base * 1.5),
        }
    return projections


def orchestrate(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for frames in args.sizes:
        receipt_path = args.output / f"worker_{frames}.json"
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "PASS"
                and existing.get("source_sha256") == _sha256(args.source)
            ):
                rows.append(existing)
                continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--frames",
            str(frames),
            "--source",
            str(args.source),
            "--output",
            str(args.output),
            "--memory-limit-bytes",
            str(args.memory_limit_bytes),
        ]
        completed = subprocess.run(command, check=False)
        if receipt_path.exists():
            row = json.loads(receipt_path.read_text(encoding="utf-8"))
            rows.append(row)
        if (
            completed.returncode
            or not receipt_path.exists()
            or not rows
            or rows[-1].get("status") != "PASS"
        ):
            failures.append({"frame_count": frames, "returncode": completed.returncode})
            break
    aggregate = {
        "schema": "phase_ttc_detection_evaluation_memory_scaling_v1",
        "status": "FAILED" if failures else "PASS",
        "resource_only": True,
        "scientific_metric_claims": False,
        "source": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "sizes": args.sizes,
        "target_frame_count": args.target_frames,
        "memory_limit_bytes_per_worker": args.memory_limit_bytes,
        "workers": rows,
        "failures": failures,
        "projection": (
            _conservative_projection(rows, args.target_frames) if len(rows) >= 2 else None
        ),
        "projection_warning": (
            "Linear resource projection from cloned pilot structures; includes 1.5x safety factor, "
            "is not a scientific metric, and cannot capture all allocator/nonlinear effects."
        ),
    }
    output = args.output / "evaluation_memory_scaling_v1.json"
    _write_json(output, aggregate)
    return int(bool(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "output/evaluation_memory")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--target-frames", type=int, default=23_958)
    parser.add_argument("--memory-limit-bytes", type=int, default=1_000_000_000)
    parser.add_argument(
        "--serialization-mode", choices=("dumps_write", "stream_dump"), default="dumps_write"
    )
    parser.add_argument("--reference-payload", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--watchdog-run", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--prestart-available-bytes", type=int, default=5 * 2**30)
    parser.add_argument("--watchdog-child-rss-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--watchdog-host-available-bytes", type=int, default=int(1.5 * 2**30))
    args = parser.parse_args()
    if (args.worker or args.watchdog_run) and (args.frames is None or args.frames < 1):
        parser.error("--worker/--watchdog-run requires a positive --frames")
    if args.worker and args.watchdog_run:
        parser.error("--worker and --watchdog-run are mutually exclusive")
    if any(size < 1 or size > 1024 for size in args.sizes):
        parser.error("orchestrated sizes must be within 1..1024")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        worker(arguments)
        if arguments.worker
        else watchdog(arguments)
        if arguments.watchdog_run
        else orchestrate(arguments)
    )
