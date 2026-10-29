"""Isolated B1/B2/B4 forward-parity profiler for a frozen campaign checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TENSOR_ATOL = 1e-6
TENSOR_RTOL = 1e-5
DECODE_ATOL = 1e-6
BATCH_SIZES = (1, 2, 4)
REPEATS = 3
WARMUPS = 1
_PROGRESS: dict[str, object] = {
    "completed_modes": [],
    "checkpoint_sha256": None,
    "expected_checkpoint_sha256": None,
}


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected numeric comparison evidence")
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Expected integer comparison evidence")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panel(tokens: Sequence[str], selection_tokens: set[str]) -> tuple[str, ...]:
    """Admit only an explicit unique 16/32-token subset of frozen selection."""
    result = tuple(tokens)
    if len(result) not in (16, 32) or len(set(result)) != len(result):
        raise ValueError("Profile panel must contain exactly 16 or 32 unique tokens")
    if any(not token or token not in selection_tokens for token in result):
        raise ValueError("Profile panel must be a subset of frozen selection tokens")
    return result


def validate_runtime_settings(seed: object, num_cpu_threads: object) -> tuple[int, int]:
    """Require concrete campaign seed/thread settings before CUDA initialization."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Campaign seed must be a nonnegative integer")
    if (
        isinstance(num_cpu_threads, bool)
        or not isinstance(num_cpu_threads, int)
        or num_cpu_threads < 1
    ):
        raise ValueError("Campaign CPU thread count must be a positive integer")
    return seed, num_cpu_threads


def batch_ranges(count: int, batch_size: int) -> tuple[range, ...]:
    """Partition a divisible panel without padding or dropping samples."""
    if batch_size not in BATCH_SIZES or count % batch_size:
        raise ValueError("Panel must divide exactly by preregistered batch size")
    return tuple(range(start, start + batch_size) for start in range(0, count, batch_size))


def slice_batch(value: object, index: int, batch_size: int) -> object:
    """Copy one sample from every tensor leaf, preserving nested output structure."""
    if isinstance(value, Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError("Every profiled output tensor must have the leading batch dimension")
        return value[index : index + 1].detach().cpu().clone()
    if isinstance(value, dict):
        return {key: slice_batch(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        return [slice_batch(item, index, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(slice_batch(item, index, batch_size) for item in value)
    return value


def compare_nested(reference: object, candidate: object, path: str = "output") -> dict[str, object]:
    """Compare every output leaf without assuming batching is bit-exact."""
    if isinstance(reference, Tensor):
        if not isinstance(candidate, Tensor):
            return {"structure_equal": False, "error": f"{path}: candidate is not Tensor"}
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            return {
                "structure_equal": False,
                "error": f"{path}: tensor shape/dtype differs",
            }
        exact = torch.equal(reference, candidate)
        if reference.numel() == 0:
            return {
                "structure_equal": True,
                "tensor_count": 1,
                "exact_tensor_count": 1,
                "within_tolerance": True,
                "max_abs": 0.0,
                "max_rel": 0.0,
            }
        left = reference.to(torch.float64)
        right = candidate.to(torch.float64)
        difference = (left - right).abs()
        maximum = torch.maximum(left.abs(), right.abs())
        tolerance = TENSOR_ATOL + TENSOR_RTOL * maximum
        relative = difference / maximum.clamp_min(torch.finfo(torch.float64).tiny)
        return {
            "structure_equal": True,
            "tensor_count": 1,
            "exact_tensor_count": int(exact),
            "within_tolerance": bool((difference <= tolerance).all()),
            "max_abs": float(difference.max()),
            "max_rel": float(relative.max()),
        }
    if type(reference) is not type(candidate):
        return {"structure_equal": False, "error": f"{path}: container type differs"}
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():  # type: ignore[union-attr]
            return {"structure_equal": False, "error": f"{path}: mapping keys differ"}
        parts = [
            compare_nested(reference[key], candidate[key], f"{path}.{key}")  # type: ignore[index]
            for key in reference
        ]
        return merge_comparisons(parts)
    if isinstance(reference, (list, tuple)):
        if len(reference) != len(candidate):  # type: ignore[arg-type]
            return {"structure_equal": False, "error": f"{path}: sequence length differs"}
        parts = [
            compare_nested(left, right, f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(reference, candidate, strict=True))  # type: ignore[arg-type]
        ]
        return merge_comparisons(parts)
    equal = reference == candidate
    return {
        "structure_equal": bool(equal),
        "tensor_count": 0,
        "exact_tensor_count": 0,
        "within_tolerance": bool(equal),
        "max_abs": 0.0,
        "max_rel": 0.0,
    }


def merge_comparisons(parts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Merge per-leaf comparison evidence fail-closed."""
    errors = [str(part["error"]) for part in parts if "error" in part]
    return {
        "structure_equal": not errors and all(bool(part["structure_equal"]) for part in parts),
        "tensor_count": sum(_as_int(part.get("tensor_count", 0)) for part in parts),
        "exact_tensor_count": sum(_as_int(part.get("exact_tensor_count", 0)) for part in parts),
        "within_tolerance": not errors
        and all(bool(part.get("within_tolerance", False)) for part in parts),
        "max_abs": max((_as_float(part.get("max_abs", 0.0)) for part in parts), default=0.0),
        "max_rel": max((_as_float(part.get("max_rel", 0.0)) for part in parts), default=0.0),
        "errors": errors,
    }


def compare_decoded(
    reference: object, candidate: object, path: str = "decoded"
) -> dict[str, object]:
    """Compare decoded categorical structure exactly and finite numbers by fixed tolerance."""
    if isinstance(reference, bool) or isinstance(candidate, bool):
        equal = type(reference) is type(candidate) and reference == candidate
        return {"structure_equal": equal, "within_tolerance": equal, "max_abs": 0.0}
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        left, right = float(reference), float(candidate)
        difference = abs(left - right)
        passed = math.isfinite(left) and math.isfinite(right) and difference <= DECODE_ATOL
        return {"structure_equal": True, "within_tolerance": passed, "max_abs": difference}
    if type(reference) is not type(candidate):
        return {"structure_equal": False, "within_tolerance": False, "error": path}
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():  # type: ignore[union-attr]
            return {"structure_equal": False, "within_tolerance": False, "error": path}
        return merge_decoded(
            [compare_decoded(reference[key], candidate[key], f"{path}.{key}") for key in reference]  # type: ignore[index]
        )
    if isinstance(reference, (list, tuple)):
        if len(reference) != len(candidate):  # type: ignore[arg-type]
            return {"structure_equal": False, "within_tolerance": False, "error": path}
        return merge_decoded(
            [
                compare_decoded(left, right, f"{path}[{index}]")
                for index, (left, right) in enumerate(zip(reference, candidate, strict=True))  # type: ignore[arg-type]
            ]
        )
    equal = reference == candidate
    return {"structure_equal": bool(equal), "within_tolerance": bool(equal), "max_abs": 0.0}


def merge_decoded(parts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    errors = [str(part["error"]) for part in parts if "error" in part]
    return {
        "structure_equal": not errors and all(bool(part["structure_equal"]) for part in parts),
        "within_tolerance": not errors and all(bool(part["within_tolerance"]) for part in parts),
        "max_abs": max((_as_float(part.get("max_abs", 0.0)) for part in parts), default=0.0),
        "errors": errors,
    }


def begin_cuda_mode_boundary() -> dict[str, int | float]:
    """Clear unused allocator cache and establish uncontaminated per-mode peaks."""
    started = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "cleanup_seconds_excluded": time.perf_counter() - started,
        "baseline_allocated_bytes": torch.cuda.memory_allocated(),
        "baseline_reserved_bytes": torch.cuda.memory_reserved(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Profile output already exists: {args.output}")
    expected_checkpoint_sha256 = str(args.expected_checkpoint_sha256)
    _PROGRESS["expected_checkpoint_sha256"] = expected_checkpoint_sha256
    if len(expected_checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_checkpoint_sha256
    ):
        raise ValueError("Expected checkpoint SHA-256 must be 64 lowercase hexadecimal characters")
    observed_checkpoint_sha256 = sha256_file(args.checkpoint)
    _PROGRESS["checkpoint_sha256"] = observed_checkpoint_sha256
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 differs from required command-line identity")

    from infer_detection_campaign import (
        assert_fresh_runtime,
        read_tokens,
        verify_checkpoint_runtime,
        verify_input_member,
    )

    checkpoint, prereg, run_root = verify_checkpoint_runtime(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Batched evaluation profile requires CUDA")
    assert_fresh_runtime()
    config = prereg["config"]
    if config["optimization"]["precision"] != "fp32" or prereg["matmul_precision"] != "highest":
        raise ValueError("Profile is preregistered only for FP32/highest")
    seed, num_cpu_threads = validate_runtime_settings(
        config["seed"], config["resources"]["num_cpu_threads"]
    )
    torch.set_num_threads(num_cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    manifest_path = ROOT / config["data"]["manifest"]
    if sha256_file(manifest_path) != config["data"]["expected_manifest_sha256"]:
        raise ValueError("Frozen campaign manifest differs from config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_tokens = set(manifest["roles"][config["data"]["selection_role"]]["sample_tokens"])
    tokens = validate_panel(read_tokens(args.tokens), selection_tokens)
    data_root = Path(config["data"]["root"])
    if sha256_file(data_root / "data/train.parquet") != prereg["dataset_media_sha256"]:
        raise ValueError("Input media/calibration manifest differs from training provenance")

    sys.path[:0] = [str(run_root / "code/src"), str(run_root / "code/scripts")]
    from smoke_monodgp_portable import build_detector, require_finite

    from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
    from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

    class LabelFreeInputs(EAPRGBDataset):
        def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
            return []

    torch.set_float32_matmul_precision("highest")
    model, _, effective = build_detector(
        run_root / "code/portable_source",
        config["model"]["group_num"],
        predict_velocity=config["model"]["predict_velocity"],
        velocity_loss_coef=config["model"]["velocity_loss_coef"],
    )
    if effective != prereg["effective_model"]:
        raise ValueError("Restored model construction differs from training")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    checkpoint_ledger = checkpoint.get("data_member_sha256", {})
    if not isinstance(checkpoint_ledger, Mapping):
        raise ValueError("Checkpoint RGB member hash ledger is malformed")
    observed: dict[str, str] = {}
    prepared_rows = []
    raw_rows = []
    with LabelFreeInputs(data_root, tokens) as dataset:
        for index in range(len(dataset)):
            raw = dataset[index]
            if raw.annotations:
                raise AssertionError("Label-free profile loader returned annotations")
            verify_input_member(raw.sample_token, raw.member_sha256, checkpoint_ledger, observed)
            prepared_rows.append(
                prepare_sample(
                    raw,
                    width=config["model"]["width"],
                    height=config["model"]["height"],
                    include_velocity=config["model"]["predict_velocity"],
                )
            )
            raw_rows.append(raw)

    def run_mode(batch_size: int, retain: bool) -> tuple[dict[str, object], dict[str, object]]:
        forward_times, h2d_times, decode_times = [], [], []
        sliced: dict[str, object] = {}
        decoded: dict[str, object] = {}
        boundary = begin_cuda_mode_boundary()
        for indices in batch_ranges(len(tokens), batch_size):
            started = time.perf_counter()
            images = torch.stack([prepared_rows[i].image for i in indices]).contiguous().cuda()
            calibrations = torch.stack([prepared_rows[i].calibration for i in indices]).cuda()
            sizes = torch.stack([prepared_rows[i].image_size for i in indices]).cuda()
            torch.cuda.synchronize()
            h2d_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            outputs = model(images, calibrations, None, sizes)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - started)
            require_finite(outputs)
            if "pred_velocity" not in outputs:
                raise ValueError("Velocity head is absent from profiled output")
            started = time.perf_counter()
            for local, global_index in enumerate(indices):
                token = tokens[global_index]
                if retain:
                    sliced[token] = slice_batch(outputs, local, batch_size)
                decoded[token] = decode_eap(
                    outputs,
                    local,
                    prepared_rows[global_index].calibration,
                    raw_rows[global_index].event_from_ego,
                    prepared_rows[global_index].image_size,
                )[0]
            decode_times.append(time.perf_counter() - started)
            del images, calibrations, sizes, outputs
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        return (
            {
                "batches": len(forward_times),
                "samples": len(tokens),
                "h2d_seconds_total": sum(h2d_times),
                "forward_seconds_total": sum(forward_times),
                "forward_seconds_per_sample": sum(forward_times) / len(tokens),
                "decode_host_seconds_total": sum(decode_times),
                "mode_boundary_cleanup_seconds_excluded": boundary["cleanup_seconds_excluded"],
                "baseline_allocated_bytes": boundary["baseline_allocated_bytes"],
                "baseline_reserved_bytes": boundary["baseline_reserved_bytes"],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": (
                    peak_allocated - int(boundary["baseline_allocated_bytes"])
                ),
                "incremental_peak_reserved_bytes": (
                    peak_reserved - int(boundary["baseline_reserved_bytes"])
                ),
            },
            {"outputs": sliced, "decoded": decoded},
        )

    with torch.no_grad():
        for batch_size in BATCH_SIZES:
            for _ in range(WARMUPS):
                run_mode(batch_size, retain=False)
        references: dict[str, object] | None = None
        trials = []
        aggregate: dict[int, list[dict[str, object]]] = {size: [] for size in BATCH_SIZES}
        comparisons: dict[int, list[Mapping[str, object]]] = {size: [] for size in BATCH_SIZES}
        for repeat in range(REPEATS):
            order = BATCH_SIZES if repeat % 2 == 0 else tuple(reversed(BATCH_SIZES))
            trial = {"repeat": repeat + 1, "order": list(order), "modes": {}}
            for batch_size in order:
                metrics, evidence = run_mode(batch_size, retain=True)
                if references is None:
                    if batch_size != 1:
                        raise AssertionError("First retained mode must establish B1 reference")
                    references = evidence
                output_parts, decode_parts = [], []
                for token in tokens:
                    output_parts.append(
                        compare_nested(
                            references["outputs"][token],  # type: ignore[index]
                            evidence["outputs"][token],  # type: ignore[index]
                        )
                    )
                    decode_parts.append(
                        compare_decoded(
                            references["decoded"][token],  # type: ignore[index]
                            evidence["decoded"][token],  # type: ignore[index]
                        )
                    )
                comparison = {
                    "outputs": merge_comparisons(output_parts),
                    "decoded": merge_decoded(decode_parts),
                }
                aggregate[batch_size].append(metrics)
                comparisons[batch_size].append(comparison)
                completed = _PROGRESS["completed_modes"]
                assert isinstance(completed, list)
                completed.append(
                    {
                        "repeat": repeat + 1,
                        "batch_size": batch_size,
                        "metrics": metrics,
                        "comparison_to_first_b1": comparison,
                    }
                )
                trial["modes"][str(batch_size)] = {**metrics, "comparison_to_first_b1": comparison}  # type: ignore[index]
            trials.append(trial)

    summary = {}
    recommendation = "ELIGIBLE_FOR_NUMERICAL_REVIEW"
    for batch_size in BATCH_SIZES:
        mode_comparisons = comparisons[batch_size]
        passed = all(
            bool(item["outputs"]["structure_equal"])  # type: ignore[index]
            and bool(item["outputs"]["within_tolerance"])  # type: ignore[index]
            and bool(item["decoded"]["structure_equal"])  # type: ignore[index]
            and bool(item["decoded"]["within_tolerance"])  # type: ignore[index]
            for item in mode_comparisons
        )
        if not passed:
            recommendation = "REJECT_BATCHED_EVALUATION"
        summary[str(batch_size)] = {
            "forward_seconds_per_sample_median": statistics.median(
                _as_float(item["forward_seconds_per_sample"]) for item in aggregate[batch_size]
            ),
            "peak_allocated_bytes_max": max(
                _as_int(item["peak_allocated_bytes"]) for item in aggregate[batch_size]
            ),
            "peak_reserved_bytes_max": max(
                _as_int(item["peak_reserved_bytes"]) for item in aggregate[batch_size]
            ),
            "incremental_peak_allocated_bytes_max": max(
                _as_int(item["incremental_peak_allocated_bytes"])
                for item in aggregate[batch_size]
            ),
            "incremental_peak_reserved_bytes_max": max(
                _as_int(item["incremental_peak_reserved_bytes"])
                for item in aggregate[batch_size]
            ),
            "max_abs": max(
                _as_float(item["outputs"]["max_abs"])  # type: ignore[index]
                for item in mode_comparisons
            ),
            "max_rel": max(
                _as_float(item["outputs"]["max_rel"])  # type: ignore[index]
                for item in mode_comparisons
            ),
            "decoded_max_abs": max(
                _as_float(item["decoded"]["max_abs"])  # type: ignore[index]
                for item in mode_comparisons
            ),
            "passes_preregistered_tolerance": passed,
        }
    receipt = {
        "schema": "eap_detection_batched_evaluation_profile_v1",
        "status": "PASS",
        "scope": "forward/H2D/decode profile only; no scorer, AP, training or protocol change",
        "checkpoint_sha256": observed_checkpoint_sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "runtime_provenance_sha256": checkpoint["runtime_provenance_sha256"],
        "tokens_sha256": sha256_file(args.tokens),
        "sample_count": len(tokens),
        "batch_sizes": list(BATCH_SIZES),
        "repeats": REPEATS,
        "warmups_per_mode": WARMUPS,
        "precision": "fp32",
        "matmul_precision": "highest",
        "environment": {
            "seed": seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "num_cpu_threads_configured": num_cpu_threads,
            "num_cpu_threads_effective": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "source_hashes": {
            "profiler": sha256_file(Path(__file__)),
            "strict_loader_current": sha256_file(ROOT / "scripts/infer_detection_campaign.py"),
            "strict_loader_frozen": sha256_file(
                run_root / "code/scripts/infer_detection_campaign.py"
            ),
        },
        "targets_argument": None,
        "annotations_returned": 0,
        "velocity_head_required": True,
        "criteria_preregistered_before_result": {
            "tensor_atol": TENSOR_ATOL,
            "tensor_rtol": TENSOR_RTOL,
            "decoded_atol": DECODE_ATOL,
            "structure_and_categorical_fields": "exact",
        },
        "input_member_sha256": observed,
        "trials": trials,
        "summary": summary,
        "recommendation": recommendation,
        "limitations": [
            "Forward timing excludes reader, preparation, H2D, decode and scoring.",
            "This profile cannot establish AP equivalence or campaign speedup.",
            "Integration remains blocked pending numerical and VRAM review.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _argument_from_argv(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError):
        return None


def _failure_output_from_argv(argv: Sequence[str]) -> Path | None:
    value = _argument_from_argv(argv, "--output")
    return Path(value) if value is not None else None


def _existing_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _write_failure_receipt(error: BaseException, argv: Sequence[str]) -> None:
    output = _failure_output_from_argv(argv)
    if output is None or output.exists():
        return
    cuda_initialized = torch.cuda.is_initialized()
    resources: dict[str, object] = {"cuda_initialized": cuda_initialized}
    # is_initialized() is deliberately used instead of is_available(): a checkpoint identity
    # failure must not initialize CUDA merely to describe the failure.
    if cuda_initialized:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            resources.update(
                {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "free_bytes": free_bytes,
                    "total_bytes": total_bytes,
                }
            )
        except Exception as resource_error:  # noqa: BLE001 - failure evidence must survive.
            resources["resource_error"] = f"{type(resource_error).__name__}: {resource_error}"
    checkpoint_argument = _argument_from_argv(argv, "--checkpoint")
    checkpoint_path = Path(checkpoint_argument) if checkpoint_argument is not None else None
    tokens_argument = _argument_from_argv(argv, "--tokens")
    tokens_path = Path(tokens_argument) if tokens_argument is not None else None
    frozen_loader = (
        checkpoint_path.parent.parent / "code/scripts/infer_detection_campaign.py"
        if checkpoint_path is not None
        else None
    )
    receipt = {
        "schema": "eap_detection_batched_evaluation_profile_failure_v1",
        "status": "FAIL",
        "retry_or_batch_downshift_performed": False,
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": traceback.format_exc(),
        "completed_modes": _PROGRESS["completed_modes"],
        "resources_at_failure": resources,
        "source_hashes": {
            "profiler": sha256_file(Path(__file__)),
            "strict_loader_current": _existing_hash(ROOT / "scripts/infer_detection_campaign.py"),
            "strict_loader_frozen": (
                _existing_hash(frozen_loader) if frozen_loader is not None else None
            ),
            "tokens": _existing_hash(tokens_path) if tokens_path is not None else None,
        },
        "checkpoint_sha256": _PROGRESS["checkpoint_sha256"],
        "expected_checkpoint_sha256": _PROGRESS["expected_checkpoint_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    try:
        _main()
    except BaseException as error:
        _write_failure_receipt(error, sys.argv)
        raise
