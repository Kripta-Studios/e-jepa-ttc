"""eAP RGB targets and label-free decoding for the declared MonoDGP variant.

HF/original equivalence is documented in GEOMETRY_RESOLUTION.md. This adapter
uses the original pinhole visualization model; released RGB remains an
approximate remap to the distorted event plane. It is not a submission contract.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from e_jepa_ttc.detection.development_metrics import CLASSES
from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners, validate_calibration


@dataclass
class PreparedSample:
    """Full RGB input and separately supervised targets, all on CPU."""

    image: torch.Tensor
    calibration: torch.Tensor
    image_size: torch.Tensor
    target: dict[str, torch.Tensor]
    native_ground_truth: list[dict[str, object]]
    report: dict[str, object]


def wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ego_yaw_to_camera(yaw: float, rotation: np.ndarray) -> float:
    """Project the original length axis to KITTI's camera XZ heading plane."""
    direction = rotation @ np.array([math.sin(yaw), math.cos(yaw), 0.0])
    if math.hypot(direction[0], direction[2]) < 1e-8:
        raise ValueError("Object direction has degenerate camera XZ projection")
    return math.atan2(-direction[2], direction[0])


def camera_yaw_to_ego(yaw: float, rotation: np.ndarray) -> float:
    """Invert the projected ego-ground-plane direction, retaining roll/pitch."""
    mapping = np.array([rotation[0, :2], -rotation[2, :2]])
    if abs(np.linalg.det(mapping)) < 1e-8:
        raise ValueError("Camera XZ projection cannot determine ego ground heading")
    direction = np.linalg.solve(mapping, [math.cos(yaw), math.sin(yaw)])
    return math.atan2(direction[0], direction[1])


def heading_encode(alpha: float) -> tuple[int, float]:
    """Encode twelve circular bins with signed residual around bin centers."""
    width = 2 * math.pi / 12
    shifted = (alpha + width / 2) % (2 * math.pi)
    index = int(shifted / width)
    return index, shifted - (index * width + width / 2)


def resized_intrinsic(
    intrinsic: np.ndarray, original_wh: tuple[int, int], resized_wh: tuple[int, int]
) -> np.ndarray:
    """Apply Pillow's pixel-center resize mapping, including anisotropic scales."""
    if min(*original_wh, *resized_wh) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = np.asarray(resized_wh) / np.asarray(original_wh)
    affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return affine @ intrinsic


def prepare_sample(
    sample: RawRGBSample,
    *,
    width: int = 640,
    height: int = 384,
    max_objects: int = 50,
    min_depth: float = 0.1,
    max_depth: float = 60.0,
    include_velocity: bool = False,
) -> PreparedSample:
    """Build complete-image inputs and counted, visibility-filtered training targets.

    Targets require all corners in front, center inside the image, and depth in
    [min_depth,max_depth). Query-capacity overflow raises instead of truncating.
    Neither GT boxes nor object IDs are part of ``image`` or ``calibration``.
    Optional velocity supervision uses released components directly; missing or
    invalid labels have a false validity mask, never a fabricated valid target.
    """
    if width % 64 or height % 64 or min(width, height, max_objects) < 1:
        raise ValueError("Positive image sizes divisible by 64 and query capacity required")
    if not 0 < min_depth < max_depth:
        raise ValueError("Depth bounds must satisfy 0 < min < max")
    validate_calibration(sample.intrinsic, sample.event_from_ego)
    original_h, original_w = sample.image.shape[:2]
    k = resized_intrinsic(sample.intrinsic, (original_w, original_h), (width, height))
    transform = sample.event_from_ego
    rotation = transform[:3, :3]
    image = np.asarray(
        Image.fromarray(sample.image).resize((width, height), Image.Resampling.BILINEAR)
    )
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    image_tensor = (
        image_tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    region = np.zeros((height, width), dtype=np.float32)
    records: dict[str, list] = {
        key: []
        for key in ("labels", "boxes", "boxes_3d", "depth", "size_3d", "heading_bin", "heading_res")
    }
    if include_velocity:
        records.update(velocity=[], velocity_valid=[])
    ignored: Counter[str] = Counter()
    native = []
    for annotation in sample.annotations:
        category = str(annotation["category"])
        if category not in CLASSES:
            raise ValueError(f"Unknown eAP category {category}")
        box = np.asarray(annotation["bbox_3d_ego"], dtype=np.float64)
        corners = original_bundle_corners(box.tolist())
        camera = corners @ rotation.T + transform[:3, 3]
        center = rotation @ box[:3] + transform[:3, 3]
        if not min_depth <= center[2] < max_depth:
            ignored["center_depth_outside_profile"] += 1
            continue
        if (camera[:, 2] <= min_depth).any():
            ignored["corner_behind_or_near_camera"] += 1
            continue
        projected = camera @ k.T
        pixels = projected[:, :2] / projected[:, 2:3]
        uv_h = k @ center
        uv = uv_h[:2] / uv_h[2]
        if not (0 <= uv[0] < width and 0 <= uv[1] < height):
            ignored["projected_center_outside_image"] += 1
            continue
        minimum = np.maximum(pixels.min(0), [0, 0])
        maximum = np.minimum(pixels.max(0), [width - 1, height - 1])
        extent = maximum - minimum
        if min(extent) <= 1:
            ignored["projected_box_too_small"] += 1
            continue
        distances = np.array(
            [uv[0] - minimum[0], maximum[0] - uv[0], uv[1] - minimum[1], maximum[1] - uv[1]]
        )
        if (distances < 0).any():
            ignored["projected_center_outside_clipped_box"] += 1
            continue
        xywh = np.r_[(minimum + maximum) / 2, extent] / [width, height, width, height]
        lrtb = distances / [width, width, height, height]
        yaw_camera = ego_yaw_to_camera(float(box[6]), rotation)
        alpha = wrap_angle(yaw_camera - math.atan2(center[0], center[2]))
        heading_bin, heading_res = heading_encode(alpha)
        records["labels"].append(CLASSES.index(category))
        records["boxes"].append(xywh)
        records["boxes_3d"].append(np.r_[uv / [width, height], lrtb])
        records["depth"].append([center[2]])
        records["size_3d"].append(box[[4, 5, 3]])
        records["heading_bin"].append([heading_bin])
        records["heading_res"].append([heading_res])
        region[int(minimum[1]) : math.ceil(maximum[1]), int(minimum[0]) : math.ceil(maximum[0])] = 1
        native.append(
            {
                "class_name": category,
                "translation_m": box[:3].tolist(),
                "size_lwh_m": box[[3, 5, 4]].tolist(),
                "yaw_rad": float(box[6]),
            }
        )
        if include_velocity:
            raw_velocity = annotation.get("velocity_3d", annotation.get("velocity"))
            try:
                velocity = np.asarray(raw_velocity, dtype=np.float64)
                valid_velocity = velocity.shape in ((2,), (3,)) and np.isfinite(velocity).all()
            except (TypeError, ValueError):
                valid_velocity = False
            records["velocity_valid"].append(bool(valid_velocity))
            records["velocity"].append(velocity[:2] if valid_velocity else np.zeros(2))
            if valid_velocity:
                native[-1]["velocity_mps"] = velocity[:2].tolist()
    if len(native) > max_objects:
        raise ValueError(
            f"{sample.sample_token}: {len(native)} targets exceed {max_objects} queries"
        )
    dimensions = {
        "labels": (),
        "boxes": (4,),
        "boxes_3d": (6,),
        "depth": (1,),
        "size_3d": (3,),
        "heading_bin": (1,),
        "heading_res": (1,),
    }
    if include_velocity:
        dimensions.update(velocity=(2,), velocity_valid=())
    target = {
        key: torch.as_tensor(
            np.asarray(values).reshape((len(native), *dimensions[key])),
            dtype=(
                torch.bool
                if key == "velocity_valid"
                else torch.long
                if key in ("labels", "heading_bin")
                else torch.float32
            ),
        )
        for key, values in records.items()
    }
    target["obj_region"] = torch.from_numpy(region)
    calibration = torch.zeros(3, 4, dtype=torch.float32)
    calibration[:, :3] = torch.as_tensor(k, dtype=torch.float32)
    return PreparedSample(
        image_tensor,
        calibration,
        torch.tensor([width, height], dtype=torch.float32),
        target,
        native,
        {
            "sample_token": sample.sample_token,
            "original_annotations": len(sample.annotations),
            "admitted_annotations": len(native),
            "ignored_by_reason": dict(ignored),
            "class_counts": dict(Counter(str(row["class_name"]) for row in native)),
            "image_wh": [width, height],
            "original_image_wh": [original_w, original_h],
            "min_depth_m": min_depth,
            "max_depth_m_exclusive": max_depth,
            "velocity_head": "NATIVE_COMPONENT_SUPERVISION" if include_velocity else "ABSENT",
            "geometry": "official_original_pinhole_approximation",
            "heading_reference": "camera alpha relative to projected 3D center ray",
            "dimension_target": "absolute h,w,l; zero class mean",
            **(
                {
                    "velocity_valid_targets": sum(records["velocity_valid"]),
                    "velocity_invalid_or_missing_targets": len(native)
                    - sum(records["velocity_valid"]),
                    "velocity_semantics": "Released first two components; not a derived TTC input",
                }
                if include_velocity
                else {}
            ),
        },
    )


def decode_eap(
    outputs: dict[str, torch.Tensor],
    index: int,
    calibration: torch.Tensor | np.ndarray,
    event_from_ego: np.ndarray,
    image_size: torch.Tensor | np.ndarray,
    *,
    topk: int = 50,
    score_threshold: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Decode predictions using calibration only; no targets, ROIs or IDs are read.

    One class per query. Class sigmoid is the confidence; depth uncertainty is
    separate and never multiplies it. Invalid physical boxes are counted and
    omitted. Velocity is emitted only when its learned head is present; no
    velocity is invented and no benchmark ZIP is produced.
    """
    if topk < 1 or not 0 <= score_threshold <= 1:
        raise ValueError("Positive topk and score threshold in [0,1] required")
    k = calibration.detach().cpu().numpy() if isinstance(calibration, torch.Tensor) else calibration
    k = np.asarray(k)[:, :3]
    size = image_size.detach().cpu().numpy() if isinstance(image_size, torch.Tensor) else image_size
    validate_calibration(k, event_from_ego)
    inverse = np.linalg.inv(event_from_ego)
    values = {
        name: outputs[name][index].detach().cpu()
        for name in ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle")
    }
    if "pred_velocity" in outputs:
        values["pred_velocity"] = outputs["pred_velocity"][index].detach().cpu()
        if values["pred_velocity"].shape != (len(values["pred_logits"]), 2):
            raise ValueError("Velocity head must emit [B,Q,2]")
    if not all(torch.isfinite(tensor).all().item() for tensor in values.values()):
        raise ValueError("Nonfinite detector output; refusing silent sanitization")
    scores, labels = values["pred_logits"].sigmoid().max(dim=-1)
    ranking = sorted(range(len(scores)), key=lambda q: (-float(scores[q]), q))[:topk]
    predictions = []
    counts: Counter[str] = Counter({"queries": len(scores), "considered": len(ranking)})
    for query in ranking:
        score = float(scores[query])
        if score < score_threshold:
            counts["below_score_threshold"] += 1
            continue
        hwl = values["pred_3d_dim"][query].numpy().astype(float)
        depth = float(values["pred_depth"][query, 0])
        if (hwl <= 0).any() or depth <= 0:
            counts["invalid_size_or_depth"] += 1
            continue
        uv = values["pred_boxes"][query, :2].numpy() * np.asarray(size)
        ray = np.linalg.solve(k, [uv[0], uv[1], 1.0])
        camera_center = ray * (depth / ray[2])
        ego_center = inverse[:3, :3] @ camera_center + inverse[:3, 3]
        heading = values["pred_angle"][query]
        heading_bin = int(heading[:12].argmax())
        alpha = heading_bin * 2 * math.pi / 12 + float(heading[12 + heading_bin])
        camera_yaw = alpha + math.atan2(camera_center[0], camera_center[2])
        ego_yaw = camera_yaw_to_ego(camera_yaw, event_from_ego[:3, :3])
        predictions.append(
            {
                "class_name": CLASSES[int(labels[query])],
                "translation_m": ego_center.tolist(),
                "size_lwh_m": hwl[[2, 1, 0]].tolist(),
                "yaw_rad": ego_yaw,
                "score": score,
            }
        )
        if "pred_velocity" in values:
            predictions[-1]["velocity_mps"] = values["pred_velocity"][query].tolist()
    counts["emitted"] = len(predictions)
    return predictions, dict(counts)
