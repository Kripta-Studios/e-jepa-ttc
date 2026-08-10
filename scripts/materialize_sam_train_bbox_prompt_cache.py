"""Materialize resumable SAM bbox-prompt targets for the exact matched train rows."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.foreground_masks import square_crop_resize_mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tokens_hash(tokens: list[str]) -> str:
    payload = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_list(value: object) -> list[object]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected list-like value, got {type(value).__name__}")
    return list(value)


def endpoint_box(value: object, endpoint_index: int, box_index: int) -> list[float]:
    endpoints = _as_list(value)
    boxes = _as_list(endpoints[endpoint_index])
    if len(boxes) == 4 and all(isinstance(item, (int, float)) for item in boxes):
        if box_index != 0:
            raise IndexError("single-box endpoint only admits box_index=0")
        box = boxes
    else:
        box = _as_list(boxes[box_index])
    if len(box) != 4 or not all(isinstance(item, (int, float)) for item in box):
        raise ValueError("bbox must contain four numeric coordinates")
    result = [float(cast(int | float, item)) for item in box]
    if not (result[2] > result[0] >= 0 and result[3] > result[1] >= 0):
        raise ValueError(f"invalid bbox: {result}")
    return result


def _read_rgb(eap_root: Path, shard: str, member: str) -> tuple[Image.Image, str]:
    shard_path = (eap_root / shard).resolve()
    with tarfile.open(shard_path, mode="r:*") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is absent from {shard_path}")
        payload = extracted.read()
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image.load()
    return image, hashlib.sha256(payload).hexdigest()


def full_mask_geometry(mask: torch.Tensor, bbox: list[float]) -> dict[str, float]:
    binary = mask.to(dtype=torch.bool, device="cpu")
    height, width = binary.shape
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2, iy2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    mask_area = int(binary.sum().item())
    bbox_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inside = int(binary[iy1:iy2, ix1:ix2].sum().item()) if bbox_area else 0
    union = mask_area + bbox_area - inside
    return {
        "mask_area_pixels": float(mask_area),
        "bbox_area_pixels": float(bbox_area),
        "mask_fraction": mask_area / (height * width),
        "bbox_mask_iou": inside / union if union else 0.0,
        "mask_inside_bbox_fraction": inside / mask_area if mask_area else 0.0,
    }


def endpoint_filter_reasons(
    metrics: dict[str, float], filter_config: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if metrics["predicted_iou"] < float(filter_config["minimum_predicted_iou"]):
        reasons.append("low_predicted_iou")
    if not (
        float(filter_config["minimum_mask_fraction"])
        <= metrics["mask_fraction"]
        <= float(filter_config["maximum_mask_fraction"])
    ):
        reasons.append("degenerate_mask_fraction")
    if metrics["bbox_mask_iou"] < float(filter_config["minimum_bbox_mask_iou"]):
        reasons.append("low_bbox_mask_iou")
    if metrics["mask_inside_bbox_fraction"] < float(
        filter_config["minimum_mask_inside_bbox_fraction"]
    ):
        reasons.append("low_inside_bbox_fraction")
    return reasons


def temporal_sign_consistent(mask_ratio: float, bbox_ratio: float, epsilon: float) -> bool:
    if abs(bbox_ratio) <= epsilon:
        return True
    return math.copysign(1.0, mask_ratio) == math.copysign(1.0, bbox_ratio)


def pack_binary_mask(
    mask: np.ndarray, *, bitorder: Literal["big", "little"]
) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.shape != (128, 128):
        raise ValueError(f"expected a 128x128 mask, got {binary.shape}")
    return np.packbits(binary.reshape(-1), bitorder=bitorder)


def unpack_binary_mask(
    packed: np.ndarray, *, bitorder: Literal["big", "little"]
) -> np.ndarray:
    values = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder=bitorder)
    if values.size != 128 * 128:
        raise ValueError("packed mask has an unexpected size")
    return values.reshape(128, 128).astype(bool)


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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
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


def _validate_source(
    *,
    config: dict[str, Any],
    subset_manifest: Path,
    train_data: Path,
    screen_cache_manifest: Path,
) -> pd.DataFrame:
    source = cast(dict[str, Any], config["source"])
    if train_data.name != "train_data.parquet" or "validation" in {
        part.lower() for part in train_data.parts
    }:
        raise ValueError("only exact matched train_data.parquet is allowed")
    for path, expected in (
        (subset_manifest, source["expected_subset_manifest_sha256"]),
        (train_data, source["expected_train_data_sha256"]),
        (screen_cache_manifest, source["expected_screen_cache_manifest_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    manifest = json.loads(subset_manifest.read_text(encoding="utf-8"))
    if not verify_artifact_hash(manifest) or manifest["roles"]["train"]["rows"] != int(
        source["expected_rows"]
    ):
        raise ValueError("matched subset manifest is invalid")
    frame = pd.read_parquet(
        train_data,
        columns=[
            "sequence_id",
            "sample_token",
            "rgb_shard_paths",
            "rgb_member_paths",
            "boxes_xyxy",
        ],
    )
    if len(frame) != int(source["expected_rows"]) or frame["sample_token"].duplicated().any():
        raise ValueError("train row count or token uniqueness mismatch")
    if frame["sequence_id"].nunique() != int(source["expected_sequences"]):
        raise ValueError("train sequence count mismatch")
    return frame


def _load_valid_sidecar(
    sidecar: Path, npz_path: Path, expected_tokens: list[str]
) -> dict[str, Any] | None:
    if not sidecar.is_file() or not npz_path.is_file():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not verify_artifact_hash(payload):
        return None
    if payload.get("tokens_sha256") != canonical_tokens_hash(expected_tokens):
        return None
    if payload.get("npz_sha256") != sha256_file(npz_path):
        return None
    return payload


def _materialize_shard(
    *,
    shard_frame: pd.DataFrame,
    processor: object,
    model: object,
    eap_root: Path,
    config: dict[str, Any],
    npz_path: Path,
    sidecar_path: Path,
    shard_index: int,
) -> dict[str, Any]:
    source = cast(dict[str, Any], config["source"])
    teacher = cast(dict[str, Any], config["teacher"])
    filter_config = cast(dict[str, Any], config["filter"])
    output = cast(dict[str, Any], config["output"])
    row_count = len(shard_frame)
    packed_masks = np.zeros((row_count, 2, 2048), dtype=np.uint8)
    endpoint_valid = np.zeros((row_count, 2), dtype=np.bool_)
    pair_valid = np.zeros(row_count, dtype=np.bool_)
    training_valid = np.zeros((row_count, 2), dtype=np.bool_)
    predicted_iou = np.full((row_count, 2), np.nan, dtype=np.float32)
    bbox_mask_iou = np.full((row_count, 2), np.nan, dtype=np.float32)
    mask_inside = np.full((row_count, 2), np.nan, dtype=np.float32)
    mask_fraction = np.full((row_count, 2), np.nan, dtype=np.float32)
    roi_mask_fraction = np.full((row_count, 2), np.nan, dtype=np.float32)
    common_squares = np.zeros((row_count, 4), dtype=np.float32)
    boxes_array = np.zeros((row_count, 2, 4), dtype=np.float32)
    records: list[dict[str, Any]] = []
    inference_times: list[float] = []
    read_times: list[float] = []
    preprocess_times: list[float] = []
    reason_counts: Counter[str] = Counter()
    processor_runtime = cast(Any, processor)
    model_runtime = cast(Any, model)
    for row_offset, row in enumerate(shard_frame.to_dict(orient="records")):
        shards = [str(value) for value in _as_list(row["rgb_shard_paths"])]
        members = [str(value) for value in _as_list(row["rgb_member_paths"])]
        boxes = [
            endpoint_box(row["boxes_xyxy"], endpoint, int(source["box_index"]))
            for endpoint in (0, 1)
        ]
        square = common_square_from_boxes(
            boxes,
            (0, 1),
            margin_fraction=float(output["common_roi_margin_fraction"]),
            minimum_edge=float(output["common_roi_minimum_edge"]),
        )
        common_squares[row_offset] = np.asarray(square, dtype=np.float32)
        boxes_array[row_offset] = np.asarray(boxes, dtype=np.float32)
        mask_areas: list[float] = []
        bbox_areas: list[float] = []
        row_records: list[dict[str, Any]] = []
        for endpoint in (0, 1):
            read_start = time.perf_counter()
            image, image_sha = _read_rgb(eap_root, shards[endpoint], members[endpoint])
            read_times.append(time.perf_counter() - read_start)
            preprocess_start = time.perf_counter()
            inputs = processor_runtime(
                images=image, input_boxes=[[boxes[endpoint]]], return_tensors="pt"
            )
            model_inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            torch.cuda.synchronize()
            preprocess_times.append(time.perf_counter() - preprocess_start)
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_output = model_runtime(**model_inputs, multimask_output=True)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - inference_start)
            if not (
                torch.isfinite(model_output.pred_masks).all().item()
                and torch.isfinite(model_output.iou_scores).all().item()
            ):
                raise FloatingPointError(f"non-finite SAM output for {row['sample_token']}")
            masks = processor_runtime.image_processor.post_process_masks(
                model_output.pred_masks.float().cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores = model_output.iou_scores.float().cpu()[0, 0]
            selected_index = int(torch.argmax(scores).item())
            full_mask = masks[0, selected_index] > float(teacher["full_frame_logit_threshold"])
            geometry = full_mask_geometry(full_mask, boxes[endpoint])
            metrics = {**geometry, "predicted_iou": float(scores[selected_index].item())}
            reasons = endpoint_filter_reasons(metrics, filter_config)
            endpoint_valid[row_offset, endpoint] = not reasons
            for reason in reasons:
                reason_counts[reason] += 1
            roi_mask = square_crop_resize_mask(
                full_mask.numpy(),
                square,
                output_size=(int(output["roi_size"]), int(output["roi_size"])),
                quantization="threshold",
                threshold=float(teacher["roi_binary_threshold"]),
            )
            bitorder = cast(Literal["big", "little"], output["packbits_bitorder"])
            packed_masks[row_offset, endpoint] = pack_binary_mask(roi_mask, bitorder=bitorder)
            predicted_iou[row_offset, endpoint] = metrics["predicted_iou"]
            bbox_mask_iou[row_offset, endpoint] = metrics["bbox_mask_iou"]
            mask_inside[row_offset, endpoint] = metrics["mask_inside_bbox_fraction"]
            mask_fraction[row_offset, endpoint] = metrics["mask_fraction"]
            roi_mask_fraction[row_offset, endpoint] = float(roi_mask.mean())
            mask_areas.append(metrics["mask_area_pixels"])
            bbox_areas.append(metrics["bbox_area_pixels"])
            row_records.append(
                {
                    "endpoint": endpoint,
                    "rgb_member_sha256": image_sha,
                    "selected_mask_index": selected_index,
                    "filter_reasons": reasons,
                }
            )
        mask_ratio = math.log(max(mask_areas[1], 1e-12) / max(mask_areas[0], 1e-12))
        bbox_ratio = math.log(max(bbox_areas[1], 1e-12) / max(bbox_areas[0], 1e-12))
        pair_consistent = temporal_sign_consistent(
            mask_ratio,
            bbox_ratio,
            float(filter_config["temporal_bbox_area_log_ratio_epsilon"]),
        )
        if not pair_consistent:
            reason_counts["temporal_area_sign_mismatch"] += 1
        pair_valid[row_offset] = bool(endpoint_valid[row_offset].all() and pair_consistent)
        training_valid[row_offset] = endpoint_valid[row_offset] & pair_valid[row_offset]
        records.append(
            {
                "sample_token": str(row["sample_token"]),
                "sequence_id": str(row["sequence_id"]),
                "mask_area_log_ratio": mask_ratio,
                "bbox_area_log_ratio": bbox_ratio,
                "pair_sign_consistent": pair_consistent,
                "pair_valid": bool(pair_valid[row_offset]),
                "endpoints": row_records,
            }
        )
    tokens = [str(value) for value in shard_frame["sample_token"].tolist()]
    arrays = {
        "sample_tokens": np.asarray(tokens),
        "sequence_ids": np.asarray([str(value) for value in shard_frame["sequence_id"].tolist()]),
        "masks_packbits": packed_masks,
        "endpoint_valid": endpoint_valid,
        "pair_valid": pair_valid,
        "training_mask_valid": training_valid,
        "predicted_iou": predicted_iou,
        "bbox_mask_iou": bbox_mask_iou,
        "mask_inside_bbox_fraction": mask_inside,
        "mask_fraction": mask_fraction,
        "roi_mask_fraction": roi_mask_fraction,
        "common_square_xyxy": common_squares,
        "source_boxes_xyxy": boxes_array,
    }
    _atomic_npz(npz_path, arrays)
    sidecar: dict[str, Any] = {
        "artifact_type": "sam_train_bbox_prompt_cache_shard_v1",
        "shard_index": shard_index,
        "row_count": row_count,
        "tokens_sha256": canonical_tokens_hash(tokens),
        "npz_path": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "endpoint_valid_count": int(endpoint_valid.sum()),
        "pair_valid_count": int(pair_valid.sum()),
        "training_mask_valid_count": int(training_valid.sum()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "mean_read_seconds": statistics.fmean(read_times),
        "mean_preprocessing_seconds": statistics.fmean(preprocess_times),
        "mean_inference_seconds": statistics.fmean(inference_times),
        "records": records,
    }
    sign_artifact(sidecar)
    _atomic_json(sidecar_path, sidecar)
    return sidecar


def run_materialization(
    *,
    config_path: Path,
    subset_manifest: Path,
    train_data: Path,
    screen_cache_manifest: Path,
    eap_root: Path,
    model_path: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    git_clean_before_run = not bool(_git(repo_root, "status", "--porcelain"))
    frame = _validate_source(
        config=config,
        subset_manifest=subset_manifest.resolve(strict=True),
        train_data=train_data.resolve(strict=True),
        screen_cache_manifest=screen_cache_manifest.resolve(strict=True),
    )
    teacher = cast(dict[str, Any], config["teacher"])
    output = cast(dict[str, Any], config["output"])
    runtime = cast(dict[str, Any], config["runtime"])
    gate = cast(dict[str, Any], config["gate"])
    weights = model_path.resolve(strict=True) / "model.safetensors"
    if model_path.name != teacher["revision"] or sha256_file(weights) != teacher[
        "expected_weights_sha256"
    ]:
        raise ValueError("SAM snapshot mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = int(output["shard_rows"])
    sidecars: list[dict[str, Any]] = []
    pending: list[tuple[int, pd.DataFrame, Path, Path]] = []
    for shard_index, start in enumerate(range(0, len(frame), shard_rows)):
        shard_frame = frame.iloc[start : start + shard_rows].reset_index(drop=True)
        npz_path = output_dir / f"shard-{shard_index:05d}.npz"
        sidecar_path = output_dir / f"shard-{shard_index:05d}.json"
        tokens = [str(value) for value in shard_frame["sample_token"].tolist()]
        existing = _load_valid_sidecar(sidecar_path, npz_path, tokens)
        if existing is None:
            pending.append((shard_index, shard_frame, npz_path, sidecar_path))
        else:
            sidecars.append(existing)
    model_load_seconds = 0.0
    if pending:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import SamModel, SamProcessor  # pyright: ignore[reportMissingImports]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        processor = SamProcessor.from_pretrained(model_path, local_files_only=True)
        model_start = time.perf_counter()
        model = cast(Any, SamModel.from_pretrained(model_path, local_files_only=True))
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_start
        materialization_start = time.perf_counter()
        for shard_index, shard_frame, npz_path, sidecar_path in pending:
            sidecars.append(
                _materialize_shard(
                    shard_frame=shard_frame,
                    processor=processor,
                    model=model,
                    eap_root=eap_root,
                    config=config,
                    npz_path=npz_path,
                    sidecar_path=sidecar_path,
                    shard_index=shard_index,
                )
            )
        materialization_seconds = time.perf_counter() - materialization_start
    else:
        materialization_seconds = 0.0
    sidecars.sort(key=lambda item: int(item["shard_index"]))
    endpoint_valid_count = sum(int(item["endpoint_valid_count"]) for item in sidecars)
    pair_valid_count = sum(int(item["pair_valid_count"]) for item in sidecars)
    training_valid_count = sum(int(item["training_mask_valid_count"]) for item in sidecars)
    reason_counts: Counter[str] = Counter()
    all_records: list[dict[str, Any]] = []
    for item in sidecars:
        reason_counts.update(cast(dict[str, int], item["reason_counts"]))
        all_records.extend(cast(list[dict[str, Any]], item["records"]))
    expected_rows = int(cast(dict[str, Any], config["source"])["expected_rows"])
    endpoint_fraction = endpoint_valid_count / (expected_rows * 2)
    pair_fraction = pair_valid_count / expected_rows
    mean_inference = statistics.fmean(float(item["mean_inference_seconds"]) for item in sidecars)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if pending else 0.0
    checks = {
        "exact_rows": sum(int(item["row_count"]) for item in sidecars) == expected_rows,
        "endpoint_valid_fraction": endpoint_fraction
        >= float(gate["minimum_endpoint_valid_fraction"]),
        "pair_valid_fraction": pair_fraction >= float(gate["minimum_pair_valid_fraction"]),
        "mean_inference_seconds": mean_inference
        <= float(runtime["maximum_mean_inference_seconds"]),
        "peak_vram": peak_vram <= float(runtime["maximum_peak_vram_mib"]),
        "runtime": materialization_seconds <= float(runtime["maximum_total_seconds"]),
    }
    shard_manifest = [
        {
            "shard_index": int(item["shard_index"]),
            "row_count": int(item["row_count"]),
            "tokens_sha256": item["tokens_sha256"],
            "npz_path": item["npz_path"],
            "npz_sha256": item["npz_sha256"],
            "sidecar_sha256": item["artifact_sha256"],
        }
        for item in sidecars
    ]
    result: dict[str, Any] = {
        "artifact_type": "sam_train_bbox_prompt_cache_manifest_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "artifact_claim_eligible": False,
        "code_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_clean_before_run": git_clean_before_run,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "scope": {
            "public_train_only": True,
            "row_count": expected_rows,
            "endpoint_count": expected_rows * 2,
            "ttc_labels_read": False,
            "validation_or_test_opened": False,
            "network_downloads": False,
        },
        "source": {
            "subset_manifest_sha256": sha256_file(subset_manifest),
            "train_data_sha256": sha256_file(train_data),
            "screen_cache_manifest_sha256": sha256_file(screen_cache_manifest),
            "tokens_sha256": canonical_tokens_hash(
                [str(value) for value in frame["sample_token"].tolist()]
            ),
            "sequence_ids": sorted(str(value) for value in frame["sequence_id"].unique()),
        },
        "teacher": {
            "repo_id": teacher["repo_id"],
            "revision": teacher["revision"],
            "weights_sha256": teacher["expected_weights_sha256"],
            "bbox_prompt_training_only": True,
        },
        "filter": config["filter"],
        "coverage": {
            "endpoint_valid_count": endpoint_valid_count,
            "endpoint_valid_fraction": endpoint_fraction,
            "pair_valid_count": pair_valid_count,
            "pair_valid_fraction": pair_fraction,
            "training_mask_valid_count": training_valid_count,
            "training_mask_valid_fraction": training_valid_count / (expected_rows * 2),
            "reason_counts": dict(sorted(reason_counts.items())),
            "per_sequence_pair_valid": {
                sequence: {
                    "count": sum(
                        bool(record["pair_valid"])
                        for record in all_records
                        if record["sequence_id"] == sequence
                    ),
                    "total": sum(record["sequence_id"] == sequence for record in all_records),
                }
                for sequence in sorted({str(record["sequence_id"]) for record in all_records})
            },
        },
        "runtime": {
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "precision": "bf16",
            "model_load_seconds": model_load_seconds,
            "materialization_seconds_this_invocation": materialization_seconds,
            "mean_inference_seconds": mean_inference,
            "peak_vram_mib_this_invocation": peak_vram,
            "resumed_shard_count": len(sidecars) - len(pending),
            "materialized_shard_count": len(pending),
        },
        "cache": {
            "output_dir": output_dir.resolve().as_posix(),
            "roi_size": output["roi_size"],
            "packbits_bitorder": output["packbits_bitorder"],
            "shards": shard_manifest,
        },
        "gate": {"checks": checks, "all_pass": all(checks.values())},
        "claim_boundary": config["claim_boundary"],
    }
    sign_artifact(result)
    _atomic_json(output_dir / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--screen-cache-manifest", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if _git(args.repo_root, "status", "--porcelain"):
        raise RuntimeError("materialization requires a clean worktree")
    result = run_materialization(
        config_path=args.config,
        subset_manifest=args.subset_manifest,
        train_data=args.train_data,
        screen_cache_manifest=args.screen_cache_manifest,
        eap_root=args.eap_root,
        model_path=args.model_path,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    )
    _atomic_json(args.summary, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
