"""Verify exact engineering parity of the controlled temporal MonoDGP split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash one artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible evidence canonically."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write one finite JSON receipt without overwriting evidence."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor names, types, shapes and bytes without serialization metadata."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State item is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_nested(value: Any) -> Any:
    """Detach a nested model output for cross-process comparison."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested(item) for item in value)
    return value


def _compare_nested(
    left: Any, right: Any, path: str = "output"
) -> tuple[list[dict[str, object]], float, int]:
    """Compare every nested tensor and scalar, retaining tensor path evidence."""
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return ([{"path": path, "reason": "type"}], float("inf"), 0)
        if left.shape != right.shape or left.dtype != right.dtype:
            return ([{"path": path, "reason": "shape_or_dtype"}], float("inf"), 1)
        exact = torch.equal(left, right)
        maximum = (
            float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
        )
        return ([] if exact else [{"path": path, "reason": "value", "max": maximum}], maximum, 1)
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return ([{"path": path, "reason": "mapping_keys"}], float("inf"), 0)
        mismatches, maximum, tensors = [], 0.0, 0
        for key in left:
            found, observed, count = _compare_nested(left[key], right[key], f"{path}.{key}")
            mismatches.extend(found)
            maximum = max(maximum, observed)
            tensors += count
        return mismatches, maximum, tensors
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return ([{"path": path, "reason": "sequence"}], float("inf"), 0)
        mismatches, maximum, tensors = [], 0.0, 0
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            found, observed, count = _compare_nested(first, second, f"{path}[{index}]")
            mismatches.extend(found)
            maximum = max(maximum, observed)
            tensors += count
        return mismatches, maximum, tensors
    if left != right:
        return ([{"path": path, "reason": "scalar"}], float("inf"), 0)
    return [], 0.0, 0


def _worker(args: argparse.Namespace) -> int:
    """Run one source variant in an isolated process."""
    runtime_root = args.runtime_root.resolve()
    sys.path[:0] = [str(runtime_root / "src"), str(runtime_root / "scripts")]
    import torch
    from smoke_monodgp_portable import build_detector, require_finite

    from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
    from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

    class LabelFreeInputs(EAPRGBDataset):
        label_files_read = False

        def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
            return []

    if not torch.cuda.is_available():
        raise RuntimeError("Temporal parity worker requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prereg = json.loads(args.runtime_preregistration.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != "eap_detection_campaign_checkpoint_v1":
        raise ValueError("Unexpected checkpoint schema")
    if checkpoint.get("update") != 512:
        raise ValueError("Engineering parity is pinned to the stable update-512 checkpoint")
    config = prereg["config"]
    model_config = config["model"]
    torch.set_float32_matmul_precision(config["optimization"]["matmul_precision"])
    model, _, effective = build_detector(
        args.source_root,
        model_config["group_num"],
        predict_velocity=model_config["predict_velocity"],
        velocity_loss_coef=model_config["velocity_loss_coef"],
    )
    if effective != prereg["effective_model"]:
        raise ValueError("Effective model differs from the frozen R0 producer")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model_before = _state_sha256(model.state_dict())
    checkpoint_model_sha = _state_sha256(checkpoint["model"])
    if model_before != checkpoint_model_sha:
        raise ValueError("Strictly loaded model bytes differ from checkpoint model state")

    adapter = None
    fusion_before = None
    if args.arm == "temporal":
        from e_jepa_ttc.detection.temporal_detector import (
            TemporalBatchContext,
            TemporalDetectorAdapter,
            TemporalFrameIdentity,
        )
        from e_jepa_ttc.detection.temporal_fusion import CausalTemporalFeatureFusion

        fusion = CausalTemporalFeatureFusion().cuda().eval()
        adapter = TemporalDetectorAdapter(
            model,
            fusion,
            producer_version="r0_update512_temporal_parity_v1",
            maximum_history_age_us=500_000,
        ).cuda().eval()
        fusion_before = _state_sha256(fusion.state_dict())

    tokens = json.loads(args.tokens.read_text(encoding="utf-8"))["sample_tokens"]
    if len(tokens) != 4 or len(set(tokens)) != 4:
        raise ValueError("Temporal parity fixture must contain exactly four unique frames")
    outputs_by_arm: dict[str, dict[str, Any]] = {"direct": {}}
    decoded_by_arm: dict[str, dict[str, Any]] = {"direct": {}}
    if adapter is not None:
        outputs_by_arm.update(current_only={}, current_repeated={})
        decoded_by_arm.update(current_only={}, current_repeated={})
    input_hashes: dict[str, str] = {}
    timings: dict[str, list[float]] = {key: [] for key in outputs_by_arm}
    torch.cuda.reset_peak_memory_stats()
    started_wall = time.perf_counter()
    with LabelFreeInputs(args.data_root, tokens) as dataset, torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            if raw.annotations:
                raise AssertionError("Label-free reader returned annotations")
            input_hashes[raw.sample_token] = raw.member_sha256
            prepared = prepare_sample(
                raw,
                width=model_config["width"],
                height=model_config["height"],
                include_velocity=model_config["predict_velocity"],
            )
            image = prepared.image.unsqueeze(0).cuda()
            calibration = prepared.calibration.unsqueeze(0).cuda()
            size = prepared.image_size.unsqueeze(0).cuda()

            def execute(
                name: str,
                function: Any,
                token: str,
                calibration_cpu: Any,
                event_from_ego: Any,
                image_size_cpu: Any,
            ) -> None:
                torch.cuda.synchronize()
                before = time.perf_counter()
                value = function()
                torch.cuda.synchronize()
                timings[name].append(time.perf_counter() - before)
                require_finite(value)
                required_outputs = {
                    "pred_logits",
                    "pred_boxes",
                    "pred_3d_dim",
                    "pred_depth",
                    "pred_angle",
                    "pred_velocity",
                    "pred_depth_map_logits",
                    "pred_region_prob",
                    "inter_outputs",
                    "aux_outputs",
                }
                if not isinstance(value, dict) or not required_outputs.issubset(value):
                    raise ValueError("Detector output lacks required geometry or velocity tensors")
                outputs_by_arm[name][token] = _cpu_nested(value)
                decoded_by_arm[name][token] = decode_eap(
                    value,
                    0,
                    calibration_cpu,
                    event_from_ego,
                    image_size_cpu,
                )[0]

            execute(
                "direct",
                lambda image=image, calibration=calibration, size=size: model(
                    image, calibration, None, size
                ),
                raw.sample_token,
                prepared.calibration,
                raw.event_from_ego,
                prepared.image_size,
            )
            if adapter is not None:
                identity = TemporalFrameIdentity(raw.sequence_id, raw.sample_token, 0)
                empty_context = TemporalBatchContext(
                    producer_version="r0_update512_temporal_parity_v1",
                    control="current_only",
                    current=(identity,),
                    history=((None, None),),
                )
                execute(
                    "current_only",
                    lambda image=image,
                    calibration=calibration,
                    size=size,
                    empty_context=empty_context: adapter(
                        image,
                        calibration,
                        None,
                        size,
                        temporal_context=empty_context,
                    ),
                    raw.sample_token,
                    prepared.calibration,
                    raw.event_from_ego,
                    prepared.image_size,
                )
                repeated_context = TemporalBatchContext(
                    producer_version="r0_update512_temporal_parity_v1",
                    control="current_repeated",
                    current=(identity,),
                    history=((identity, identity),),
                )
                histories = image.unsqueeze(1).repeat(1, 2, 1, 1, 1)
                execute(
                    "current_repeated",
                    lambda image=image,
                    calibration=calibration,
                    size=size,
                    repeated_context=repeated_context,
                    histories=histories: adapter(
                        image,
                        calibration,
                        None,
                        size,
                        temporal_context=repeated_context,
                        history_images=histories,
                        history_ages_us=torch.zeros(1, 2, device=image.device),
                        history_mask=torch.ones(1, 2, dtype=torch.bool, device=image.device),
                        history_limit=2,
                    ),
                    raw.sample_token,
                    prepared.calibration,
                    raw.event_from_ego,
                    prepared.image_size,
                )
    model_after = _state_sha256(model.state_dict())
    fusion_after = _state_sha256(adapter.fusion.state_dict()) if adapter is not None else None
    if model_after != model_before or fusion_after != fusion_before:
        raise ValueError("Model or fusion state changed during inference-only parity")
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs_by_arm, args.tensor_output)
    worker_receipt = {
        "schema": "temporal_detector_parity_worker_v1",
        "arm": args.arm,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": checkpoint["update"],
        "source_root": str(args.source_root.resolve()),
        "sample_tokens": tokens,
        "targets_argument": None,
        "annotations_returned": 0,
        "label_files_read": LabelFreeInputs.label_files_read,
        "label_free_reader_override": True,
        "optimizer_updates": 0,
        "input_member_sha256": input_hashes,
        "model_state_before_sha256": model_before,
        "model_state_after_sha256": model_after,
        "fusion_state_before_sha256": fusion_before,
        "fusion_state_after_sha256": fusion_after,
        "tensor_output_sha256": sha256_file(args.tensor_output),
        "decoded": decoded_by_arm,
        "timing_seconds_median": {
            key: statistics.median(values) for key, values in timings.items()
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": time.perf_counter() - started_wall,
        "device": torch.cuda.get_device_name(),
    }
    write_json(args.worker_output, worker_receipt)
    return 0


def _snapshot_runtime(output: Path, frozen_code: Path) -> tuple[Path, dict[str, str]]:
    """Create the only runtime used by isolated workers and hash every source."""
    runtime = output / "runtime"
    if runtime.exists():
        raise FileExistsError(runtime)
    shutil.copytree(frozen_code / "src", runtime / "src")
    (runtime / "scripts").mkdir(parents=True)
    shutil.copy2(frozen_code / "scripts/smoke_monodgp_portable.py", runtime / "scripts")
    for name in ("temporal_detector.py", "temporal_fusion.py"):
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "src/e_jepa_ttc/detection" / name,
            runtime / "src/e_jepa_ttc/detection" / name,
        )
    shutil.copy2(Path(__file__), runtime / "scripts" / Path(__file__).name)
    hashes = {
        path.relative_to(runtime).as_posix(): sha256_file(path)
        for path in sorted(runtime.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    write_json(runtime / "source_hashes.json", hashes)
    return runtime, hashes


def _orchestrate(args: argparse.Namespace) -> int:
    """Freeze fixtures, run isolated source arms, and compare every output exactly."""
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    failure_path = args.checkpoint.parent.parent / "failure.json"
    if not failure_path.is_file():
        raise FileNotFoundError("Failed R0 run receipt is required for scope provenance")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("Temporal source manifest checkpoint differs")
    tokens = json.loads(args.tokens.read_text(encoding="utf-8"))["sample_tokens"][:4]
    fixture_path = args.output / "tokens_first4.json"
    write_json(fixture_path, {"sample_tokens": tokens})
    frozen_code = args.checkpoint.parent.parent / "code"
    runtime, runtime_hashes = _snapshot_runtime(args.output, frozen_code)
    worker_script = runtime / "scripts" / Path(__file__).name
    workers = {}
    sources = (("original", args.original_source), ("temporal", args.temporal_source))
    for arm, source_root in sources:
        tensor_output = args.output / f"{arm}_outputs.pt"
        worker_output = args.output / f"{arm}_worker.json"
        command = [
            sys.executable,
            str(worker_script),
            "--worker",
            "--arm",
            arm,
            "--runtime-root",
            str(runtime),
            "--source-root",
            str(source_root.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--runtime-preregistration",
            str(args.runtime_preregistration.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
            "--tokens",
            str(fixture_path.resolve()),
            "--tensor-output",
            str(tensor_output.resolve()),
            "--worker-output",
            str(worker_output.resolve()),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            write_json(
                args.output / f"{arm}_failure.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            raise RuntimeError(f"Temporal parity {arm} worker failed")
        workers[arm] = json.loads(worker_output.read_text(encoding="utf-8"))
    original = torch.load(
        args.output / "original_outputs.pt", map_location="cpu", weights_only=True
    )
    temporal = torch.load(
        args.output / "temporal_outputs.pt", map_location="cpu", weights_only=True
    )
    comparisons = {}
    for name, left, right in (
        ("original_vs_split", original["direct"], temporal["direct"]),
        ("split_vs_current_only", temporal["direct"], temporal["current_only"]),
        ("split_vs_current_repeated", temporal["direct"], temporal["current_repeated"]),
    ):
        mismatches, maximum, tensor_count = _compare_nested(left, right)
        comparisons[name] = {
            "exact": not mismatches,
            "maximum_absolute_error": maximum,
            "tensor_count": tensor_count,
            "mismatches": mismatches,
        }
    decoded_equal = (
        workers["original"]["decoded"]["direct"]
        == workers["temporal"]["decoded"]["direct"]
        == workers["temporal"]["decoded"]["current_only"]
        == workers["temporal"]["decoded"]["current_repeated"]
    )
    status = (
        "PASS"
        if all(value["exact"] for value in comparisons.values()) and decoded_equal
        else "FAIL"
    )
    report = {
        "schema": "temporal_detector_engineering_parity_v1",
        "status": status,
        "scope": "engineering parity only; failed R0 update-512 fixture; no AP or accuracy claim",
        "r0_failure": json.loads(failure_path.read_text(encoding="utf-8")),
        "r0_failure_sha256": sha256_file(failure_path),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": 512,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "runtime_source_hashes": runtime_hashes,
        "runtime_source_hashes_sha256": canonical_sha256(runtime_hashes),
        "fixture_tokens_sha256": sha256_file(fixture_path),
        "sample_count": 4,
        "targets_read": False,
        "optimizer_updates": 0,
        "comparisons": comparisons,
        "decoded_predictions_exact": decoded_equal,
        "workers": workers,
        "host": platform.node(),
        "python": platform.python_version(),
    }
    write_json(args.output / "report.json", report)
    if status != "PASS":
        raise ValueError("Temporal detector parity differs")
    print(json.dumps({"status": status, "report": str(args.output / "report.json")}))
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the public and isolated-worker CLI."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--arm", choices=("original", "temporal"))
    value.add_argument("--runtime-root", type=Path)
    value.add_argument("--source-root", type=Path)
    value.add_argument("--original-source", type=Path)
    value.add_argument("--temporal-source", type=Path)
    value.add_argument("--source-manifest", type=Path)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--runtime-preregistration", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--tokens", type=Path, required=True)
    value.add_argument("--tensor-output", type=Path)
    value.add_argument("--worker-output", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main() -> int:
    """Dispatch an isolated worker or the complete parity comparison."""
    args = parser().parse_args()
    if args.worker:
        required = (
            args.arm,
            args.runtime_root,
            args.source_root,
            args.tensor_output,
            args.worker_output,
        )
        if any(item is None for item in required):
            raise ValueError("Worker arguments are incomplete")
        return _worker(args)
    required = (args.original_source, args.temporal_source, args.source_manifest, args.output)
    if any(item is None for item in required):
        raise ValueError("Parity orchestration arguments are incomplete")
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
