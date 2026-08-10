"""Run a signed, train-only SAM bbox-prompt smoke from local assets on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import torch
import yaml
from PIL import Image

from e_jepa_ttc.artifacts.hashing import sign_artifact

T = TypeVar("T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_public_train(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.name.lower() != "train.parquet" or "test" in {
        part.lower() for part in resolved.parts
    }:
        raise ValueError(f"only public train.parquet is allowed: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _as_list(value: object) -> list[object]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected list-like value, got {type(value).__name__}")
    return list(value)


def select_train_row(
    frame: pd.DataFrame,
    *,
    sequence_id: str,
    expected_sample_token: str,
) -> pd.Series:
    """Select a deterministic public-train row and verify its preregistered token."""

    selected = frame.loc[frame["sequence_id"].astype(str).eq(sequence_id)].sort_values(
        "sample_token"
    )
    if selected.empty:
        raise ValueError(f"sequence is absent from public train: {sequence_id}")
    row = selected.iloc[0]
    observed = str(row["sample_token"])
    if observed != expected_sample_token:
        raise ValueError(f"preregistered sample mismatch: {observed} != {expected_sample_token}")
    return row


def endpoint_box(value: object, *, endpoint_index: int, box_index: int) -> list[float]:
    """Return one xyxy prompt from the nested public Garl box column."""

    endpoints = _as_list(value)
    if not 0 <= endpoint_index < len(endpoints):
        raise IndexError("endpoint_index is outside boxes_xyxy")
    boxes = _as_list(endpoints[endpoint_index])
    if len(boxes) == 4 and all(isinstance(item, (int, float)) for item in boxes):
        if box_index != 0:
            raise IndexError("single-box endpoint only admits box_index=0")
        box = boxes
    else:
        if not 0 <= box_index < len(boxes):
            raise IndexError("box_index is outside endpoint boxes")
        box = _as_list(boxes[box_index])
    if len(box) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {len(box)}")
    if not all(isinstance(item, (int, float)) for item in box):
        raise TypeError("bbox coordinates must be numeric")
    result = [float(cast(int | float, item)) for item in box]
    x1, y1, x2, y2 = result
    if not (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0):
        raise ValueError(f"invalid xyxy bbox: {result}")
    return result


def _read_rgb_member(eap_root: Path, shard_reference: str, member: str) -> tuple[Image.Image, str]:
    shard_path = (eap_root / shard_reference).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    with tarfile.open(shard_path, mode="r:*") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"member {member!r} absent from {shard_path}")
        payload = extracted.read()
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image.load()
    return image, hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synchronize() -> None:
    torch.cuda.synchronize()


def _elapsed_cuda(action: Callable[[], T]) -> tuple[T, float]:
    _synchronize()
    start = time.perf_counter()
    result = action()
    _synchronize()
    return result, time.perf_counter() - start


def run_smoke(
    *,
    config_path: Path,
    data_parquet: Path,
    eap_root: Path,
    model_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute the preregistered smoke and return a signed result."""

    config_path = config_path.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("protocol config must be a mapping")
    data = cast(dict[str, Any], config["dataset"])
    teacher = cast(dict[str, Any], config["teacher"])
    runtime = cast(dict[str, Any], config["runtime"])
    gate = cast(dict[str, Any], config["gate"])
    if runtime["device"] != "cuda" or runtime["precision"] != "bf16":
        raise ValueError("v1 protocol requires CUDA BF16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the preregistered protocol")
    data_parquet = _require_public_train(data_parquet)
    observed_parquet_hash = sha256_file(data_parquet)
    if observed_parquet_hash != data["expected_parquet_sha256"]:
        raise ValueError("public train parquet hash does not match preregistration")
    model_path = model_path.resolve(strict=True)
    weights_path = model_path / "model.safetensors"
    if model_path.name != teacher["revision"] or sha256_file(weights_path) != teacher[
        "expected_weights_sha256"
    ]:
        raise ValueError("SAM snapshot revision or weights hash mismatch")

    columns = [
        "sequence_id",
        "sample_token",
        "rgb_shard_paths",
        "rgb_member_paths",
        "boxes_xyxy",
    ]
    read_start = time.perf_counter()
    frame = pd.read_parquet(data_parquet, columns=columns)
    row = select_train_row(
        frame,
        sequence_id=str(data["sequence_id"]),
        expected_sample_token=str(data["expected_sample_token"]),
    )
    endpoint_index = int(data["endpoint_index"])
    box_index = int(data["box_index"])
    shards = [str(item) for item in _as_list(row["rgb_shard_paths"])]
    members = [str(item) for item in _as_list(row["rgb_member_paths"])]
    if len(shards) != len(members) or endpoint_index >= len(shards):
        raise ValueError("RGB endpoint lists do not match preregistration")
    bbox = endpoint_box(row["boxes_xyxy"], endpoint_index=endpoint_index, box_index=box_index)
    image, image_sha256 = _read_rgb_member(
        eap_root, shards[endpoint_index], members[endpoint_index]
    )
    data_read_seconds = time.perf_counter() - read_start
    width, height = image.size
    if bbox[2] > width or bbox[3] > height:
        raise ValueError(f"bbox {bbox} is outside image size {(width, height)}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import SamModel, SamProcessor  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("run with `uv run --extra multimodal`") from exc

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor_start = time.perf_counter()
    processor = SamProcessor.from_pretrained(model_path, local_files_only=True)
    processor_load_seconds = time.perf_counter() - processor_start
    model_start = time.perf_counter()
    model = cast(Any, SamModel.from_pretrained(model_path, local_files_only=True))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    _synchronize()
    model_load_seconds = time.perf_counter() - model_start

    preprocess_start = time.perf_counter()
    inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt")
    model_inputs = {
        key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()
    }
    _synchronize()
    preprocessing_seconds = time.perf_counter() - preprocess_start

    def infer() -> object:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(**model_inputs, multimask_output=True)

    raw_outputs, inference_seconds = _elapsed_cuda(infer)
    outputs = cast(Any, raw_outputs)
    post_start = time.perf_counter()
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.float().cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.float().cpu()[0, 0]
    best_index = int(torch.argmax(scores).item())
    best_logits = masks[0, best_index]
    binary_mask = best_logits > 0.0
    postprocessing_seconds = time.perf_counter() - post_start
    mask_fraction = float(binary_mask.float().mean().item())
    predicted_iou = float(scores[best_index].item())
    finite = bool(
        torch.isfinite(outputs.pred_masks).all().item()
        and torch.isfinite(outputs.iou_scores).all().item()
    )
    peak_vram_mib = torch.cuda.max_memory_allocated() / 1024**2
    require_finite = bool(gate["require_finite"])
    gate_checks = {
        "finite": finite if require_finite else True,
        "predicted_iou": predicted_iou >= float(gate["minimum_predicted_iou"]),
        "mask_fraction": (
            float(gate["minimum_mask_fraction"])
            <= mask_fraction
            <= float(gate["maximum_mask_fraction"])
        ),
        "peak_vram": peak_vram_mib <= float(runtime["maximum_peak_vram_mib"]),
        "inference_time": inference_seconds <= float(runtime["maximum_inference_seconds"]),
    }
    result: dict[str, Any] = {
        "artifact_type": "sam_train_bbox_prompt_smoke_v1",
        "status": "passed" if all(gate_checks.values()) else "failed",
        "artifact_claim_eligible": False,
        "code_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_clean_before_run": not bool(_git(repo_root, "status", "--porcelain")),
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source": {
            "public_train_only": True,
            "private_test_opened": False,
            "data_parquet_sha256": observed_parquet_hash,
            "sequence_id": str(row["sequence_id"]),
            "sample_token": str(row["sample_token"]),
            "endpoint_index": endpoint_index,
            "box_index": box_index,
            "bbox_xyxy": bbox,
            "rgb_shard": shards[endpoint_index],
            "rgb_member": members[endpoint_index],
            "rgb_member_sha256": image_sha256,
            "image_size_wh": [width, height],
        },
        "teacher": {
            "repo_id": teacher["repo_id"],
            "revision": teacher["revision"],
            "weights_sha256": teacher["expected_weights_sha256"],
            "local_files_only": True,
            "network_downloads": False,
            "bbox_prompt_training_only": True,
        },
        "runtime": {
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "precision": "bf16",
            "data_read_seconds": data_read_seconds,
            "processor_load_seconds": processor_load_seconds,
            "model_load_seconds": model_load_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "peak_vram_mib": peak_vram_mib,
        },
        "output": {
            "pred_masks_shape": list(outputs.pred_masks.shape),
            "iou_scores_shape": list(outputs.iou_scores.shape),
            "iou_scores": [float(value) for value in scores.tolist()],
            "selected_mask_index": best_index,
            "selected_predicted_iou": predicted_iou,
            "selected_mask_fraction": mask_fraction,
            "outputs_are_finite": finite,
        },
        "gate": {"checks": gate_checks, "all_pass": all(gate_checks.values())},
        "claim_boundary": config["claim_boundary"],
    }
    return sign_artifact(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _git(args.repo_root, "status", "--porcelain"):
        raise RuntimeError("smoke requires a clean worktree before execution")
    report = run_smoke(
        config_path=args.config,
        data_parquet=args.data_parquet,
        eap_root=args.eap_root,
        model_path=args.model_path,
        repo_root=args.repo_root,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
