"""Audit SAM bbox-prompt geometry on deterministic public-train rows only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact
from scripts.smoke_sam_hf_bbox_prompt import (
    _as_list,
    _read_rgb_member,
    endpoint_box,
    sha256_file,
)


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select exact deterministic indices including both sorted endpoints."""

    if count < 2:
        raise ValueError("count must be at least two")
    ordered = frame.sort_values("sample_token").reset_index(drop=True)
    if len(ordered) < count:
        raise ValueError(f"sequence has only {len(ordered)} rows, fewer than {count}")
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64).tolist()
    if len(set(indices)) != count:
        raise ValueError("evenly spaced selector produced duplicate indices")
    return ordered.iloc[indices].reset_index(drop=True)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _pearson(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) < 2 or np.std(left_array) <= 1e-12 or np.std(right_array) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left_array, right_array)[0, 1])


def mask_geometry(mask: torch.Tensor, bbox: list[float]) -> dict[str, float | bool]:
    """Measure binary-mask geometry against its bbox prompt without GT masks."""

    binary = mask.to(dtype=torch.bool, device="cpu")
    height, width = binary.shape
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2, iy2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    mask_area = int(binary.sum().item())
    bbox_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inside = int(binary[iy1:iy2, ix1:ix2].sum().item()) if bbox_area else 0
    union = mask_area + bbox_area - inside
    coordinates = torch.nonzero(binary, as_tuple=False)
    if coordinates.numel() == 0:
        mask_x1 = mask_y1 = mask_x2 = mask_y2 = 0
    else:
        mask_y1 = int(coordinates[:, 0].min().item())
        mask_y2 = int(coordinates[:, 0].max().item()) + 1
        mask_x1 = int(coordinates[:, 1].min().item())
        mask_x2 = int(coordinates[:, 1].max().item()) + 1
    return {
        "mask_fraction": mask_area / (height * width),
        "mask_area_pixels": float(mask_area),
        "bbox_area_pixels": float(bbox_area),
        "bbox_mask_iou": inside / union if union else 0.0,
        "bbox_coverage": inside / bbox_area if bbox_area else 0.0,
        "mask_inside_bbox_fraction": inside / mask_area if mask_area else 0.0,
        "mask_height_fraction": (mask_y2 - mask_y1) / height,
        "mask_width_fraction": (mask_x2 - mask_x1) / width,
        "bbox_height_fraction": (iy2 - iy1) / height,
        "bbox_width_fraction": (ix2 - ix1) / width,
        "touches_image_border": bool(
            mask_area and (mask_x1 == 0 or mask_y1 == 0 or mask_x2 == width or mask_y2 == height)
        ),
    }


def _log_record_ratio(first: dict[str, Any], second: dict[str, Any], field: str) -> float:
    left = float(first[field])
    right = float(second[field])
    return math.log(max(right, 1e-12) / max(left, 1e-12))


def pair_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive t0->t1 ratios from exactly two endpoint records per sample."""

    by_token: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_token.setdefault(str(record["sample_token"]), []).append(record)
    pairs: list[dict[str, Any]] = []
    for token, endpoints in sorted(by_token.items()):
        endpoints.sort(key=lambda item: int(item["endpoint_index"]))
        if [int(item["endpoint_index"]) for item in endpoints] != [0, 1]:
            raise ValueError(f"sample {token} does not have endpoints [0, 1]")
        first, second = endpoints

        pairs.append(
            {
                "sample_token": token,
                "sequence_id": first["sequence_id"],
                "mask_area_log_ratio": _log_record_ratio(
                    first, second, "mask_area_pixels"
                ),
                "bbox_area_log_ratio": _log_record_ratio(
                    first, second, "bbox_area_pixels"
                ),
                "mask_height_log_ratio": _log_record_ratio(
                    first, second, "mask_height_fraction"
                ),
                "bbox_height_log_ratio": _log_record_ratio(
                    first, second, "bbox_height_fraction"
                ),
                "mask_width_log_ratio": _log_record_ratio(
                    first, second, "mask_width_fraction"
                ),
                "bbox_width_log_ratio": _log_record_ratio(
                    first, second, "bbox_width_fraction"
                ),
            }
        )
    return pairs


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("cannot write an empty endpoint audit")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_audit(
    *,
    config_path: Path,
    data_parquet: Path,
    eap_root: Path,
    model_path: Path,
    repo_root: Path,
    output_csv: Path,
) -> dict[str, Any]:
    """Run the frozen train-only audit and return a signed summary."""

    config_path = config_path.resolve(strict=True)
    git_clean_before_run = not bool(_git(repo_root, "status", "--porcelain"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    data = cast(dict[str, Any], config["dataset"])
    teacher = cast(dict[str, Any], config["teacher"])
    runtime = cast(dict[str, Any], config["runtime"])
    gate = cast(dict[str, Any], config["gate"])
    data_parquet = data_parquet.resolve(strict=True)
    if data_parquet.name.lower() != "train.parquet" or "test" in {
        part.lower() for part in data_parquet.parts
    }:
        raise ValueError("only public train.parquet is allowed")
    if sha256_file(data_parquet) != data["expected_parquet_sha256"]:
        raise ValueError("train parquet hash mismatch")
    model_path = model_path.resolve(strict=True)
    weights_path = model_path / "model.safetensors"
    if model_path.name != teacher["revision"] or sha256_file(weights_path) != teacher[
        "expected_weights_sha256"
    ]:
        raise ValueError("teacher revision or weights hash mismatch")
    if runtime["device"] != "cuda" or runtime["precision"] != "bf16":
        raise ValueError("v1 requires CUDA BF16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    frame = pd.read_parquet(
        data_parquet,
        columns=[
            "sequence_id",
            "sample_token",
            "rgb_shard_paths",
            "rgb_member_paths",
            "boxes_xyxy",
        ],
    )
    selected_frames = []
    for sequence_id in cast(list[str], data["sequence_ids"]):
        sequence_frame = frame.loc[frame["sequence_id"].astype(str).eq(sequence_id)]
        selected_frames.append(
            evenly_spaced_rows(sequence_frame, int(data["samples_per_sequence"]))
        )
    selected = pd.concat(selected_frames, ignore_index=True)
    endpoint_indices = [int(value) for value in cast(list[int], data["endpoint_indices"])]
    if endpoint_indices != [0, 1]:
        raise ValueError("v1 requires endpoint indices [0, 1]")

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
    records: list[dict[str, Any]] = []
    inference_times: list[float] = []
    total_start = time.perf_counter()
    for row in selected.to_dict(orient="records"):
        shards = [str(value) for value in _as_list(row["rgb_shard_paths"])]
        members = [str(value) for value in _as_list(row["rgb_member_paths"])]
        for endpoint_index in endpoint_indices:
            bbox = endpoint_box(
                row["boxes_xyxy"],
                endpoint_index=endpoint_index,
                box_index=int(data["box_index"]),
            )
            read_start = time.perf_counter()
            image, image_hash = _read_rgb_member(
                eap_root, shards[endpoint_index], members[endpoint_index]
            )
            read_seconds = time.perf_counter() - read_start
            preprocess_start = time.perf_counter()
            inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt")
            model_inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            torch.cuda.synchronize()
            preprocessing_seconds = time.perf_counter() - preprocess_start
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                outputs = model(**model_inputs, multimask_output=True)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            inference_times.append(inference_seconds)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.float().cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores = outputs.iou_scores.float().cpu()[0, 0]
            selected_index = int(torch.argmax(scores).item())
            selected_mask = masks[0, selected_index] > float(teacher["threshold_logit"])
            geometry = mask_geometry(selected_mask, bbox)
            records.append(
                {
                    "sequence_id": str(row["sequence_id"]),
                    "sample_token": str(row["sample_token"]),
                    "endpoint_index": endpoint_index,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "rgb_member_sha256": image_hash,
                    "selected_mask_index": selected_index,
                    "predicted_iou": float(scores[selected_index].item()),
                    "outputs_finite": bool(
                        torch.isfinite(outputs.pred_masks).all().item()
                        and torch.isfinite(outputs.iou_scores).all().item()
                    ),
                    "read_seconds": read_seconds,
                    "preprocessing_seconds": preprocessing_seconds,
                    "inference_seconds": inference_seconds,
                    **geometry,
                }
            )
    total_seconds = time.perf_counter() - total_start
    pairs = pair_diagnostics(records)
    predicted_ious = [float(record["predicted_iou"]) for record in records]
    bbox_mask_ious = [float(record["bbox_mask_iou"]) for record in records]
    mask_fractions = [float(record["mask_fraction"]) for record in records]
    minimum_fraction = float(gate["degenerate_mask_fraction_minimum"])
    maximum_fraction = float(gate["degenerate_mask_fraction_maximum"])
    degenerate_count = sum(
        value < minimum_fraction or value > maximum_fraction for value in mask_fractions
    )
    area_mask = [float(pair["mask_area_log_ratio"]) for pair in pairs]
    area_bbox = [float(pair["bbox_area_log_ratio"]) for pair in pairs]
    area_pearson = _pearson(area_mask, area_bbox)
    sign_accuracy = statistics.fmean(
        float(math.copysign(1.0, predicted) == math.copysign(1.0, target))
        for predicted, target in zip(area_mask, area_bbox, strict=True)
        if abs(target) > 1e-9
    )
    peak_vram_mib = torch.cuda.max_memory_allocated() / 1024**2
    checks = {
        "pair_count": len(pairs) == int(gate["expected_pair_count"]),
        "endpoint_count": len(records) == int(gate["expected_endpoint_count"]),
        "all_finite": all(bool(record["outputs_finite"]) for record in records),
        "iou_score_p10": _percentile(predicted_ious, 10)
        >= float(gate["minimum_iou_score_p10"]),
        "bbox_mask_iou_median": statistics.median(bbox_mask_ious)
        >= float(gate["minimum_bbox_mask_iou_median"]),
        "degenerate_endpoint_fraction": degenerate_count / len(records)
        <= float(gate["maximum_degenerate_endpoint_fraction"]),
        "temporal_area_ratio_pearson": math.isfinite(area_pearson)
        and area_pearson >= float(gate["minimum_temporal_area_ratio_pearson"]),
        "temporal_area_ratio_sign_accuracy": sign_accuracy
        >= float(gate["minimum_temporal_area_ratio_sign_accuracy"]),
        "peak_vram": peak_vram_mib <= float(runtime["maximum_peak_vram_mib"]),
        "mean_inference_time": statistics.fmean(inference_times)
        <= float(runtime["maximum_mean_inference_seconds"]),
    }
    _write_csv(output_csv, records)
    summary: dict[str, Any] = {
        "artifact_type": "sam_train_bbox_prompt_multisequence_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "artifact_claim_eligible": False,
        "code_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_clean_before_run": git_clean_before_run,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "scope": {
            "public_train_only": True,
            "ttc_columns_read": False,
            "private_test_opened": False,
            "validation_teacher_generation": False,
            "network_downloads": False,
        },
        "selection": {
            "sequence_ids": data["sequence_ids"],
            "samples_per_sequence": data["samples_per_sequence"],
            "sample_tokens": sorted(str(value) for value in selected["sample_token"].tolist()),
            "pair_count": len(pairs),
            "endpoint_count": len(records),
        },
        "teacher": {
            "repo_id": teacher["repo_id"],
            "revision": teacher["revision"],
            "weights_sha256": teacher["expected_weights_sha256"],
        },
        "metrics": {
            "predicted_iou_median": statistics.median(predicted_ious),
            "predicted_iou_p10": _percentile(predicted_ious, 10),
            "bbox_mask_iou_median": statistics.median(bbox_mask_ious),
            "bbox_coverage_median": statistics.median(
                float(record["bbox_coverage"]) for record in records
            ),
            "mask_inside_bbox_fraction_median": statistics.median(
                float(record["mask_inside_bbox_fraction"]) for record in records
            ),
            "degenerate_endpoint_count": degenerate_count,
            "degenerate_endpoint_fraction": degenerate_count / len(records),
            "temporal_area_ratio_pearson": area_pearson,
            "temporal_area_ratio_sign_accuracy": sign_accuracy,
            "temporal_height_ratio_pearson": _pearson(
                [float(pair["mask_height_log_ratio"]) for pair in pairs],
                [float(pair["bbox_height_log_ratio"]) for pair in pairs],
            ),
            "temporal_width_ratio_pearson": _pearson(
                [float(pair["mask_width_log_ratio"]) for pair in pairs],
                [float(pair["bbox_width_log_ratio"]) for pair in pairs],
            ),
            "border_touch_fraction": statistics.fmean(
                float(bool(record["touches_image_border"])) for record in records
            ),
        },
        "runtime": {
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "precision": "bf16",
            "model_load_seconds": model_load_seconds,
            "total_endpoint_loop_seconds": total_seconds,
            "mean_inference_seconds": statistics.fmean(inference_times),
            "p95_inference_seconds": _percentile(inference_times, 95),
            "peak_vram_mib": peak_vram_mib,
        },
        "endpoint_csv": {
            "path": output_csv.resolve().as_posix(),
            "sha256": sha256_file(output_csv),
            "row_count": len(records),
        },
        "gate": {"checks": checks, "all_pass": all(checks.values())},
        "claim_boundary": config["claim_boundary"],
    }
    return sign_artifact(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if _git(args.repo_root, "status", "--porcelain"):
        raise RuntimeError("audit requires a clean worktree before execution")
    report = run_audit(
        config_path=args.config,
        data_parquet=args.data_parquet,
        eap_root=args.eap_root,
        model_path=args.model_path,
        repo_root=args.repo_root,
        output_csv=args.output_csv,
    )
    _atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
