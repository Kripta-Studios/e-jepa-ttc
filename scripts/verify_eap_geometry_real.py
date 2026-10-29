"""Verify train HF boxes against original annotations and render real-image overlays.

This certifies a bounded public-train contract, not official scorer compatibility.
Raw annotations and image overlays stay outside tracked documentation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import pickle
import tarfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

OFFICIAL_COMMIT = "2c5f3a74ba5b693b746820a0a355684e3597dc2d"
OFFICIAL_SHA256 = "e975941c5866af465dcf3897efcdae36d71306808a38a63b580311a35f5ed5a7"


def sha256(path: Path) -> str:
    """Hash a bounded metadata or rendered-image artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnnotationUnpickler(pickle.Unpickler):
    """Refuse executable globals; the inspected release consists of builtin values."""

    def find_class(self, module: str, name: str) -> Any:
        raise ValueError(f"Executable pickle global refused: {module}.{name}")


def corners_ego(box: Any) -> np.ndarray:
    """Use verified xyzlhwyaw: width on X, length on Y, clockwise yaw, center origin."""
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (7,) or not np.isfinite(b).all() or np.any(b[3:6] <= 0):
        raise ValueError("Expected finite xyzlhwyaw with positive dimensions")
    signs = np.asarray(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
        ],
        dtype=float,
    )
    c, s = np.cos(b[6]), np.sin(b[6])
    rotation = np.asarray([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return (signs * b[[5, 3, 4]] / 2) @ rotation.T + b[:3]


def matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    """Convert nested Parquet lists and reject malformed calibration."""
    result = np.stack([np.asarray(row, dtype=float) for row in value])
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {shape} matrix")
    return result


def project_box(box: Any, intrinsic: Any, event_from_ego: Any) -> np.ndarray:
    """Project using the released visualizer pinhole approximation; reject rear corners."""
    k = matrix(intrinsic, (3, 3))
    transform = matrix(event_from_ego, (4, 4))
    camera = (np.column_stack((corners_ego(box), np.ones(8))) @ transform.T)[:, :3]
    if np.any(camera[:, 2] <= 1e-4):
        raise ValueError("Box has corners at or behind camera plane")
    projected = camera @ k.T
    return projected[:, :2] / projected[:, 2:3]


def native_dimensions(box: Any) -> list[float]:
    """Return semantic [length,width,height]; this alone does not certify the scorer."""
    corners_ego(box)  # Same validation as the geometric contract.
    return np.asarray(box, dtype=float)[[3, 5, 4]].tolist()


def verify_pairs(original: list[dict], labels: pd.DataFrame, media: pd.DataFrame) -> dict:
    """Compare exact fields under original/HF frame and instance identity."""
    index = {}
    for row in labels.to_dict("records"):
        key = (row["sample_token"], row["instance_id"])
        if key in index:
            raise ValueError("Ambiguous HF frame/instance identity")
        index[key] = row
    media_index = {row["sample_token"]: row for row in media.to_dict("records")}
    if len(media_index) != len(media):
        raise ValueError("Duplicate media token")
    fields = ["population", "box", "category", "velocity", "calibration", "exposure", "size"]
    failures = dict.fromkeys(fields, 0)
    seen = set()
    for source in original:
        token = f"{source['sequence_id']}:{Path(source['file_name']).stem}"
        key = (token, source["instance_id"])
        if key in seen:
            raise ValueError("Ambiguous original frame/instance identity")
        seen.add(key)
        if key not in index or token not in media_index:
            failures["population"] += 1
            continue
        target, frame = index[key], media_index[token]
        failures["box"] += not np.array_equal(source["bbox_3d"], target["bbox_3d_ego"])
        failures["category"] += source["category"] != target["category"]
        failures["velocity"] += not np.array_equal(source["velocity"], target["velocity_3d"])
        failures["size"] += not np.array_equal(
            target["size"], np.asarray(source["bbox_3d"])[[5, 3, 4]]
        )
        failures["calibration"] += any(
            not np.array_equal(matrix(source[key], shape), matrix(frame[key], shape))
            for key, shape in [("K_event", (3, 3)), ("T_event_ego", (4, 4))]
        )
        failures["exposure"] += any(
            int(source[key]) != int(frame[key])
            for key in ["rgb_exposure_start_timestamp_us", "rgb_exposure_end_timestamp_us"]
        )
    failures["population"] += len(set(index) - seen)
    return {
        "original_annotation_rows": len(original),
        "hf_annotation_rows": len(labels),
        "field_mismatches": failures,
        "all_pairs_exact": not any(failures.values()),
    }


def official_function(path: Path) -> Callable[..., np.ndarray]:
    """Extract only the hash-verified official corner function, without its CLI/imports."""
    if sha256(path) != OFFICIAL_SHA256:
        raise ValueError("Official visualizer differs from pinned inspected source")
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "bbox_to_corner_ego"
    )
    scope = {"np": np, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["bbox_to_corner_ego"]


def verify(args: argparse.Namespace) -> dict:
    """Verify one sequence and produce bounded deterministic real-image overlays."""
    with args.original_annotations.open("rb") as source:
        original = AnnotationUnpickler(source).load()
    if not isinstance(original, list) or not original:
        raise ValueError("Original annotations must be a nonempty list")
    if {r["sequence_id"] for r in original} != {args.sequence}:
        raise ValueError("Original annotations contain another sequence")
    manifest_path = args.eap_root / "data/train.parquet"
    label_path = args.eap_root / "data/train" / args.sequence / "labels.parquet"
    media = pd.read_parquet(manifest_path)
    media = media[media.sequence_id.eq(args.sequence)].sort_values("sample_token")
    labels = pd.read_parquet(label_path)
    if not (media["split"].eq("train").all() and labels["split"].eq("train").all()):
        raise ValueError("Only public train data are allowed")
    pairs = verify_pairs(original, labels, media)
    if not pairs["all_pairs_exact"]:
        raise ValueError(f"Original/HF mismatch: {pairs}")
    reference = official_function(args.official_visualizer)
    positions = np.linspace(0, len(media) - 1, min(args.frames, len(media)), dtype=int)
    args.overlays_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in media.iloc[positions].to_dict("records"):
        with tarfile.open(args.eap_root / row["rgb_shard_path"], "r:") as archive:
            member = archive.getmember(row["rgb_member_path"])
            if not member.isfile() or member.size > 20_000_000:
                raise ValueError("Invalid or oversized RGB member")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("RGB member unavailable")
            raw = extracted.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        draw = ImageDraw.Draw(image)
        objects = [r for r in original if r["file_name"] == row["rgb_member_path"]]
        diagnostics = []
        for obj in objects:
            official = reference(*obj["bbox_3d"])
            corner_error = float(np.max(np.abs(official - corners_ego(obj["bbox_3d"]))))
            diagnostic = {
                "instance_id": obj["instance_id"],
                "official_corner_max_abs_error_m": corner_error,
            }
            try:
                uv = project_box(obj["bbox_3d"], row["K_event"], row["T_event_ego"])
                bbox = np.concatenate((uv.min(0), uv.max(0)))
                limits = np.asarray([image.width - 1, image.height - 1] * 2)
                bbox = np.clip(bbox, 0, limits)
                diagnostic["clipped_bbox_residual_vs_original_px"] = float(
                    np.max(np.abs(bbox - np.asarray(obj["bbox"])))
                )
                diagnostic["status"] = "PROJECTED"
                edges = [(i, (i + 1) % 4) for i in range(4)]
                edges += [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
                edges += [(i, i + 4) for i in range(4)]
                for a, b in edges:
                    draw.line([tuple(uv[a]), tuple(uv[b])], fill=(30, 255, 70), width=2)
                draw.rectangle(obj["bbox"], outline=(255, 170, 20), width=1)
            except ValueError as error:
                diagnostic.update(status="REJECTED", reason=str(error))
            diagnostics.append(diagnostic)
        draw.text(
            (8, 8), f"HF=original | {row['sample_token']} | green3D orangepublished2D", fill="white"
        )
        output = args.overlays_dir / f"{row['sample_token'].replace(':', '_')}.jpg"
        image.save(output, quality=92)
        frames.append(
            {
                "sample_token": row["sample_token"],
                "rgb_member_path": row["rgb_member_path"],
                "rgb_sha256": hashlib.sha256(raw).hexdigest(),
                "image_size": list(image.size),
                "overlay_path": str(output),
                "overlay_sha256": sha256(output),
                "objects": diagnostics,
            }
        )
    projected = [o for f in frames for o in f["objects"] if o["status"] == "PROJECTED"]
    if not projected:
        raise ValueError("No front-camera boxes were checked on the real frames")
    if any(o["official_corner_max_abs_error_m"] > 1e-4 for o in projected):
        raise ValueError("Corner implementation differs from the pinned official function")
    if any(o["clipped_bbox_residual_vs_original_px"] >= 1.01 for o in projected):
        raise ValueError("Published 2D boxes are inconsistent with the projection hypothesis")
    return {
        "schema": "eap_original_hf_real_geometry_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_ORIGINAL_HF_TRAIN_GEOMETRY",
        "script_sha256": sha256(Path(__file__)),
        "sequence_id": args.sequence,
        "sources": {
            "original_annotations": {
                "path": str(args.original_annotations),
                "sha256": sha256(args.original_annotations),
            },
            "hf_labels": {"path": str(label_path), "sha256": sha256(label_path)},
            "hf_train_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "official_visualizer": {
                "commit": OFFICIAL_COMMIT,
                "sha256": OFFICIAL_SHA256,
                "url": f"https://github.com/NAIL-HNU/eAP_dataset/blob/{OFFICIAL_COMMIT}/release_visualizer/visualize_release_sequence.py",
            },
        },
        "pair_checks": pairs,
        "frame_selection": "evenly spaced sorted train tokens; no outcome filtering",
        "frames": frames,
        "summary": {
            "frames_read": len(frames),
            "boxes_projected": len(projected),
            "projection_residual_max_px": max(
                (o["clipped_bbox_residual_vs_original_px"] for o in projected), default=None
            ),
            "official_corner_max_abs_error_m": max(
                (o["official_corner_max_abs_error_m"] for o in projected), default=None
            ),
        },
        "contract": {
            "bbox_3d_ego": "x,y,z,length,height,width,yaw",
            "center": "geometric center, ego frame",
            "ego_axes": "front X, left Y, up Z",
            "local_axes": "width X, length Y, height Z",
            "rotation": "Rz(-yaw)",
            "camera_transform": "T_event_ego applied directly",
            "semantic_size_lwh_indices": [3, 5, 4],
            "hf_size_indices": [5, 3, 4],
        },
        "limits": [
            "Exact original/HF identity verified for one train sequence; not every sequence",
            "Pinhole projection follows official visualizer; paper describes distorted event plane",
            "Actual overlay inspection is qualitative, not pixel-level calibration certification",
            "No official native scorer obtained; submission yaw compatibility remains unverified",
            "Velocity is an unchanged released target, not a certified physical TTC input",
            "Garl object IDs are outside this detector-only contract; full G0 remains open",
        ],
    }


def main() -> int:
    """Write the receipt; keep real images in the caller's untracked artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--original-annotations", type=Path, required=True)
    parser.add_argument("--official-visualizer", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--overlays-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    report = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "pairs": report["pair_checks"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
