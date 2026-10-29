"""Run a saved portable MonoDGP checkpoint without opening annotation files.

This process reads only public train input metadata and RGB members. It outputs
predictions and timings, never ground truth or official benchmark scores.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree.

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from e_jepa_ttc.detection.diagnostic_export import (
    build_sidecar,
    capture_exposed_outputs,
    validate_json_finite,
)
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset


class EAPRGBInputs(EAPRGBDataset):
    """Read full RGB/calibration while making annotation loading unreachable."""

    def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Deliberately do not access row['labels_path'] or the parent's loader.
        return []


def sha256(path: Path) -> str:
    """Hash one local checkpoint or code artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_media_source(path: Path, expected_hash: str | None) -> dict[str, str | None]:
    """Record the full media/calibration manifest and reject an expected-hash mismatch."""
    observed = sha256(path)
    if expected_hash is not None and observed != expected_hash:
        raise ValueError("Input media/calibration manifest differs from the expected SHA-256")
    return {
        "source_media_path": str(path),
        "source_media_sha256": observed,
        "expected_source_media_sha256": expected_hash,
        "source_media_verification": "VERIFIED_EXPECTED_HASH"
        if expected_hash
        else "OBSERVED_HASH_ONLY",
    }


RUNTIME_MODULES = (
    ("e_jepa_ttc.detection.development_metrics", "src/e_jepa_ttc/detection/development_metrics.py"),
    ("e_jepa_ttc.detection.eap_rgb_data", "src/e_jepa_ttc/detection/eap_rgb_data.py"),
    ("e_jepa_ttc.detection.portable_attention", "src/e_jepa_ttc/detection/portable_attention.py"),
    ("e_jepa_ttc.detection.monodgp_eap_adapter", "src/e_jepa_ttc/detection/monodgp_eap_adapter.py"),
    ("smoke_monodgp_portable", "scripts/smoke_monodgp_portable.py"),
)
AP_RUNTIME_MODULE = (
    ("e_jepa_ttc.detection.evaluation_ap_fast", "src/e_jepa_ttc/detection/evaluation_ap_fast.py"),
)
AP_RUNTIME_RELATIVE = AP_RUNTIME_MODULE[0][1]
OPTIONAL_RUNTIME_MODULES = tuple(
    (f"e_jepa_ttc.detection.{name}", f"src/e_jepa_ttc/detection/{name}.py")
    for name in (
        "geometry_contract",
        "velocity_diagnostics",
        "velocity_extension",
        "backbone_initialization",
    )
)
RUNTIME_DEPENDENCY_ORDER = (
    AP_RUNTIME_MODULE + RUNTIME_MODULES[:3] + OPTIONAL_RUNTIME_MODULES + RUNTIME_MODULES[3:]
)


def _imports_ap_runtime(path: Path) -> bool:
    """Detect the exact helper dependency from already hash-verified source bytes."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = "e_jepa_ttc.detection.evaluation_ap_fast"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in {module, "evaluation_ap_fast"} or (
                node.module in {"e_jepa_ttc.detection", None}
                and any(alias.name == "evaluation_ap_fast" for alias in node.names)
            ):
                return True
        if isinstance(node, ast.Import) and any(alias.name == module for alias in node.names):
            return True
    return False


def select_runtime_sources(
    recorded_hashes: Mapping[str, str],
    checkpoint_dir: Path,
    current_root: Path = ROOT,
    *,
    require_velocity: bool = False,
) -> list[dict[str, str]]:
    """Resolve every runtime module to exact registered bytes before executing any.

    Prefer the checkpoint's frozen code when it exists; otherwise require the
    current source to match. A corrupt frozen file fails rather than silently
    falling back. Both Windows and POSIX preregistration paths are accepted.
    """
    recorded = {key.replace("\\", "/"): value for key, value in recorded_hashes.items()}
    required = {relative for _, relative in RUNTIME_MODULES}
    if require_velocity:
        required.add("src/e_jepa_ttc/detection/velocity_extension.py")
    missing = required - set(recorded)
    if missing:
        raise ValueError(f"Missing registered runtime hashes: {sorted(missing)}")
    development_relative = RUNTIME_MODULES[0][1]
    development_frozen = checkpoint_dir / "code" / development_relative
    development_path = (
        development_frozen
        if development_frozen.is_file()
        else current_root / development_relative
    )
    if (
        not development_path.is_file()
        or sha256(development_path) != recorded[development_relative]
    ):
        raise ValueError(
            f"Runtime source does not match preregistration: {development_path}"
        )
    if _imports_ap_runtime(development_path) and AP_RUNTIME_RELATIVE not in recorded:
        raise ValueError(f"Missing registered runtime hashes: ['{AP_RUNTIME_RELATIVE}']")
    result = []
    for name, relative in RUNTIME_DEPENDENCY_ORDER:
        if relative not in recorded:
            continue
        frozen = checkpoint_dir / "code" / relative
        path = frozen if frozen.is_file() else current_root / relative
        if not path.is_file() or sha256(path) != recorded[relative]:
            raise ValueError(f"Runtime source does not match preregistration: {path}")
        result.append(
            {
                "module": name,
                "relative_path": relative,
                "path": str(path),
                "sha256": recorded[relative],
                "source": "checkpoint_snapshot" if path == frozen else "matching_current",
            }
        )
    return result


def load_runtime_modules(sources: Sequence[Mapping[str, str]]) -> dict[str, ModuleType]:
    """Load verified modules under canonical names in dependency order."""
    names = tuple(item["module"] for item in sources)
    expected = tuple(name for name, _ in RUNTIME_DEPENDENCY_ORDER if name in names)
    if not {name for name, _ in RUNTIME_MODULES}.issubset(names) or names != expected:
        raise ValueError("Runtime module order or population differs from the declared contract")
    # Recheck all bytes before the first exec, including a potential source change
    # between planning and loading. Existing parent-package attributes are updated
    # so both 'from package.module import ...' and package.module see this version.
    for item in sources:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Runtime source changed before loading: {item['path']}")
    loaded = {}
    for item in sources:
        name = item["module"]
        spec = importlib.util.spec_from_file_location(name, item["path"])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load registered runtime module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(importlib.import_module(parent), leaf, module)
        loaded[name] = module
    return loaded


def compare_saved_predictions(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare ordered decoded predictions without consulting reference GT."""
    mismatches = []
    maximum = 0.0
    if set(predictions) != set(reference):
        raise ValueError("Reference prediction token population differs")
    for token, boxes in predictions.items():
        expected = reference[token]
        if len(boxes) != len(expected):
            mismatches.append({"sample_token": token, "reason": "box_count"})
            continue
        for index, (actual, other) in enumerate(zip(boxes, expected, strict=True)):
            if actual["class_name"] != other["class_name"]:
                mismatches.append({"sample_token": token, "box_index": index, "reason": "class"})
            fields = ["translation_m", "size_lwh_m", "yaw_rad", "score"]
            if "velocity_mps" in actual or "velocity_mps" in other:
                if "velocity_mps" not in actual or "velocity_mps" not in other:
                    mismatches.append(
                        {"sample_token": token, "box_index": index, "reason": "velocity_presence"}
                    )
                else:
                    fields.append("velocity_mps")
            for key in fields:
                left, right = (
                    np.asarray(actual[key], dtype=float),
                    np.asarray(other[key], dtype=float),
                )
                if (
                    left.shape != right.shape
                    or not np.isfinite(left).all()
                    or not np.isfinite(right).all()
                ):
                    mismatches.append({"sample_token": token, "box_index": index, "reason": key})
                    continue
                error = float(np.max(np.abs(left - right)))
                maximum = max(maximum, error)
                if error > tolerance:
                    mismatches.append(
                        {
                            "sample_token": token,
                            "box_index": index,
                            "reason": key,
                            "maximum_abs_error": error,
                        }
                    )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum,
        "mismatches": mismatches,
        "scope": "Predictions only; reference ground_truth is not accessed",
    }


def infer(args: argparse.Namespace) -> dict[str, Any]:
    """Load strict checkpoint in this process and infer using targets=None."""
    if args.output.exists():
        raise FileExistsError("Choose a new inference output file; previous evidence is preserved")
    diagnostic_path = getattr(args, "diagnostic_sidecar", None)
    if diagnostic_path is not None and diagnostic_path.exists():
        raise FileExistsError("Choose a new diagnostic sidecar; previous evidence is preserved")
    if diagnostic_path is not None and diagnostic_path.resolve() == args.output.resolve():
        raise ValueError("Diagnostic sidecar and prediction output must be different paths")
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not args.role:
            raise ValueError("--role is required with --manifest")
        tokens = payload[args.role]["sample_tokens"]
        expected_media_hash = str(payload["source_media_sha256"])
        token_source = args.manifest
    else:
        payload = json.loads(args.tokens_json.read_text(encoding="utf-8"))
        tokens = payload if isinstance(payload, list) else payload["sample_tokens"]
        expected_media_hash = (
            None if isinstance(payload, list) else payload.get("source_media_sha256")
        )
        token_source = args.tokens_json
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Need a nonempty unique token population")
    media_provenance = verify_media_source(
        args.data_root / "data/train.parquet", expected_media_hash
    )
    prereg_path = args.checkpoint.parent / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    variant = str(checkpoint["variant"])
    if variant != prereg["variant"]:
        raise ValueError("Checkpoint belongs to another detector variant")
    if checkpoint["preregistration_sha256"] != sha256(prereg_path):
        raise ValueError("Checkpoint preregistration hash differs")
    runtime_sources = select_runtime_sources(
        prereg["code_sha256"],
        args.checkpoint.parent,
        require_velocity=bool(checkpoint["config"].get("predict_velocity", False)),
    )
    modules = load_runtime_modules(runtime_sources)
    helper = modules["smoke_monodgp_portable"]
    adapter = modules["e_jepa_ttc.detection.monodgp_eap_adapter"]
    data_module = modules["e_jepa_ttc.detection.eap_rgb_data"]
    input_loader = type(
        "CheckpointRGBInputs",
        (data_module.EAPRGBDataset,),
        {"_frame_labels": EAPRGBInputs._frame_labels},
    )
    build_detector, prepare_source, require_finite = (
        helper.build_detector,
        helper.prepare_source,
        helper.require_finite,
    )
    decode_eap, prepare_sample = adapter.decode_eap, adapter.prepare_sample
    code_hashes = {item["relative_path"]: item["sha256"] for item in runtime_sources}
    if [args.height, args.width] != prereg["image_hw"]:
        raise ValueError("Image size differs from the checkpoint experiment")
    torch.set_num_threads(4)
    if checkpoint.get("inference_precision", "fp32") != "fp32":
        raise ValueError("This independent runner currently requires FP32 checkpoint inference")
    matmul_precision = str(checkpoint.get("matmul_precision", "highest"))
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(int(checkpoint["seed"]))
    np.random.seed(int(checkpoint["seed"]))
    checkpoint_hash = sha256(args.checkpoint)
    destination = args.portable_root or ROOT / "output/detector_sources" / (
        "MonoDGP-inference-" + checkpoint_hash[:12]
    )
    patch_path = args.output.with_suffix(".upstream.patch")
    build_kwargs = {}
    for key in ("predict_velocity", "velocity_loss_coef"):
        if key in checkpoint["config"]:
            if key not in inspect.signature(build_detector).parameters:
                raise ValueError(f"Registered builder cannot restore checkpoint option: {key}")
            build_kwargs[key] = checkpoint["config"][key]
    prepare_kwargs = {}
    if "predict_velocity" in checkpoint["config"]:
        if "predict_velocity" not in inspect.signature(prepare_source).parameters:
            raise ValueError("Registered source recipe cannot restore the velocity variant")
        prepare_kwargs["predict_velocity"] = checkpoint["config"]["predict_velocity"]
    source = prepare_source(args.source_root, destination, patch_path, **prepare_kwargs)
    if source["variant"] != variant:
        raise ValueError("Prepared source variant differs from checkpoint")
    if source["patch_sha256"] != prereg["source"]["patch_sha256"]:
        raise ValueError("Inference source patch differs from the training patch")
    model, criterion, config = build_detector(
        destination, checkpoint["config"]["group_num"], **build_kwargs
    )
    if config != checkpoint["config"]:
        raise ValueError("Built model config differs from checkpoint config")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint, criterion
    model.eval()
    predictions, decode_counts, data_receipts, timings = {}, {}, {}, []
    diagnostic_frames: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with input_loader(args.data_root, tokens) as dataset, torch.inference_mode():
        for index, token in enumerate(tokens):
            begin = time.perf_counter()
            sample = dataset[index]
            if sample.annotations:
                raise AssertionError("Label-free input loader returned annotations")
            prepared = prepare_sample(sample, width=args.width, height=args.height)
            if prepared.target["labels"].numel() or prepared.native_ground_truth:
                raise AssertionError("Input preparation produced unexpected targets")
            preparation_seconds = time.perf_counter() - begin
            images = prepared.image.unsqueeze(0).cuda()
            calibration = prepared.calibration.unsqueeze(0).cuda()
            sizes = prepared.image_size.unsqueeze(0).cuda()
            if index == 0:
                for _ in range(args.warmup):
                    model(images, calibration, None, sizes)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            outputs = model(images, calibration, None, sizes)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - tick
            require_finite(outputs)
            if diagnostic_path is not None:
                diagnostic_frames.append(
                    capture_exposed_outputs(outputs, sample_token=token, batch_index=0)
                )
            tick = time.perf_counter()
            predictions[token], decode_counts[token] = decode_eap(
                outputs, 0, prepared.calibration, sample.event_from_ego, prepared.image_size
            )
            decoding_seconds = time.perf_counter() - tick
            data_receipts[token] = {
                "rgb_member_sha256": sample.member_sha256,
                "K_event_sha256_float64_c_order": hashlib.sha256(
                    sample.intrinsic.tobytes()
                ).hexdigest(),
                "T_event_ego_sha256_float64_c_order": hashlib.sha256(
                    sample.event_from_ego.tobytes()
                ).hexdigest(),
                "original_image_wh": [sample.image.shape[1], sample.image.shape[0]],
                "image_wh": [args.width, args.height],
                "exposure_start_us": sample.exposure_start_us,
                "exposure_end_us": sample.exposure_end_us,
                "annotations_returned": 0,
            }
            timings.append(
                {
                    "sample_token": token,
                    "load_and_prepare_seconds": preparation_seconds,
                    "forward_cuda_synchronized_seconds": forward_seconds,
                    "decode_seconds": decoding_seconds,
                    "frame_wall_seconds_including_first_warmup": time.perf_counter() - begin,
                }
            )
            print(
                json.dumps(
                    {"frame": index + 1, "total": len(tokens), "forward_seconds": forward_seconds}
                ),
                flush=True,
            )
    report = {
        "schema": "monodgp_eap_label_free_inference_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "variant": variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load": "PASS_STRICT_WEIGHTS_ONLY",
        "preregistration_sha256": sha256(prereg_path),
        "source": source,
        "inference_source_sha256": sha256(Path(__file__)),
        "code_sha256": code_hashes,
        "token_source_sha256": sha256(token_source),
        **media_provenance,
        "sample_tokens": tokens,
        "model_targets_argument": None,
        "annotation_loader": "override_returns_empty_without_path_access",
        "runtime_sources": runtime_sources,
        "ground_truth_in_output": False,
        "predictions": predictions,
        "decode_counts": decode_counts,
        "data_receipts": data_receipts,
        "timings": timings,
        "forward_seconds_median": statistics.median(
            r["forward_cuda_synchronized_seconds"] for r in timings
        ),
        "inference_loop_seconds": time.perf_counter() - started,
        "warmup_forwards": args.warmup,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "cuda_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cuda_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": matmul_precision,
        "velocity_head": "PRESENT" if config.get("predict_velocity", False) else "ABSENT",
        "official_submission_compatible": False,
    }
    if args.reference_predictions:
        reference = json.loads(args.reference_predictions.read_text(encoding="utf-8"))[
            "predictions"
        ]
        report["prediction_parity"] = compare_saved_predictions(predictions, reference)
        report["reference_sha256"] = sha256(args.reference_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if diagnostic_path is not None:
        sidecar = build_sidecar(
            diagnostic_frames,
            checkpoint_sha256=checkpoint_hash,
            prediction_output_sha256=sha256(args.output),
            observer_path=ROOT / "src/e_jepa_ttc/detection/diagnostic_export.py",
        )
        validate_json_finite(sidecar)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(
            json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    """Run checkpoint inference separately from training and optional prediction-only parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--manifest", type=Path)
    selection.add_argument("--tokens-json", type=Path)
    parser.add_argument("--role", choices=("overfit16", "train", "selection"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument(
        "--diagnostic-sidecar",
        type=Path,
        help="New JSON path for an external query-tensor observer; predictions remain unchanged",
    )
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP"
    )
    parser.add_argument("--portable-root", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("Warmup count cannot be negative")
    report = infer(args)
    print(
        json.dumps(
            {
                "samples": len(report["sample_tokens"]),
                "forward_seconds_median": report["forward_seconds_median"],
                "prediction_parity": report.get("prediction_parity"),
            }
        )
    )
    return 0 if report.get("prediction_parity", {}).get("status", "PASS") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
