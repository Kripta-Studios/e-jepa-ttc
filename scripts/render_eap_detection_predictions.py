"""Render saved eAP detector predictions on fixed real train/selection images.

This script performs no model inference. All image artifacts remain below output/.
The display confidence threshold never changes the prediction JSON or AP inputs.
"""

# ruff: noqa: E402 -- direct script execution from an uninstalled worktree.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset, RawRGBSample
from scripts.verify_eap_geometry_real import project_box

GT_COLOR = (60, 230, 120)
PRED_COLOR = (245, 85, 220)
EDGES = (
    tuple((i, (i + 1) % 4) for i in range(4))
    + tuple((i + 4, (i + 1) % 4 + 4) for i in range(4))
    + tuple((i, i + 4) for i in range(4))
)


def sha256(path: Path) -> str:
    """Hash a finite saved output artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_tokens(tokens: Sequence[str], count: int) -> list[str]:
    """Choose equally spaced positions in manifest order without inspecting results."""
    if not tokens or count < 1:
        raise ValueError("Need a nonempty token population and a positive display count")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Manifest token population is ambiguous")
    positions = np.linspace(0, len(tokens) - 1, min(count, len(tokens)), dtype=int)
    return [tokens[index] for index in positions]


def native_to_original_box(box: Mapping[str, Any]) -> np.ndarray:
    """Convert semantic l,w,h to the verified original x,y,z,l,h,w,yaw layout."""
    center = np.asarray(box["translation_m"], dtype=np.float64)
    size = np.asarray(box["size_lwh_m"], dtype=np.float64)
    yaw = float(box["yaw_rad"])
    if center.shape != (3,) or size.shape != (3,):
        raise ValueError("Center and l,w,h must each have three entries")
    if not np.isfinite(center).all() or not np.isfinite(size).all() or not math.isfinite(yaw):
        raise ValueError("Box geometry must be finite")
    if np.any(size <= 0):
        raise ValueError("Box dimensions must be positive")
    return np.concatenate((center, size[[0, 2, 1]], [yaw]))


def clipped_segment(
    start: Sequence[float], end: Sequence[float], width: int, height: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip projected edges to the image before handing coordinates to Pillow."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    x, y = float(start[0]), float(start[1])
    dx, dy = float(end[0]) - x, float(end[1]) - y
    if not all(math.isfinite(value) for value in (x, y, dx, dy)):
        raise ValueError("Line coordinates must be finite")
    low, high = 0.0, 1.0
    for p, q in zip((-dx, dx, -dy, dy), (x, width - 1 - x, y, height - 1 - y), strict=True):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None
    return (x + low * dx, y + low * dy), (x + high * dx, y + high * dy)


def overlay(
    sample: RawRGBSample,
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    label: str,
    score_threshold: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Draw accepted GT and thresholded predictions; record all omitted projections."""
    if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("Display threshold must be finite in [0,1]")
    image = Image.fromarray(sample.image.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    width, height = image.size
    counts: dict[str, Any] = {
        "gt_total": len(ground_truth),
        "pred_total": len(predictions),
        "gt_drawn": 0,
        "pred_drawn": 0,
        "pred_below_display_threshold": 0,
        "projection_omissions": [],
    }
    for kind, boxes, color in (("gt", ground_truth, GT_COLOR), ("pred", predictions, PRED_COLOR)):
        for index, box in enumerate(boxes):
            try:
                score = float(box["score"]) if kind == "pred" else None
                if score is not None and not math.isfinite(score):
                    raise ValueError("Nonfinite prediction confidence")
                if score is not None and score < score_threshold:
                    counts["pred_below_display_threshold"] += 1
                    continue
                pixels = project_box(
                    native_to_original_box(box), sample.intrinsic, sample.event_from_ego
                )
                visible = False
                for first, second in EDGES:
                    segment = clipped_segment(pixels[first], pixels[second], width, height)
                    if segment is not None:
                        draw.line(segment, fill=color, width=2)
                        visible = True
                if not visible:
                    raise ValueError("All projected edges lie outside the image")
                counts[kind + "_drawn"] += 1
                text = f"{kind.upper()} {box['class_name']}"
                if score is not None:
                    text += f" {score:.2f}"
                anchor = np.clip(pixels.min(axis=0), [0, 0], [width - 1, height - 1])
                x, y = float(anchor[0]), float(anchor[1])
                bounds = draw.textbbox((x, y), text, font=font)
                draw.rectangle(bounds, fill=(0, 0, 0))
                draw.text((x, y), text, fill=color, font=font)
            except (KeyError, ValueError, TypeError) as error:
                counts["projection_omissions"].append(
                    {"kind": kind, "box_index": index, "reason": f"{type(error).__name__}: {error}"}
                )
    header_height = 90
    tile = Image.new("RGB", (width, height + header_height), (20, 23, 29))
    tile.paste(image, (0, header_height))
    title = ImageDraw.Draw(tile)
    title.text((12, 8), f"{label} | {sample.sample_token}", fill="white", font=font)
    title.text(
        (12, 30),
        f"GT retained by adapter: {counts['gt_drawn']}/{counts['gt_total']} drawn",
        fill=GT_COLOR,
        font=font,
    )
    title.text(
        (12, 51),
        f"Pred >= {score_threshold:.2f}: {counts['pred_drawn']}/{counts['pred_total']} drawn"
        " | threshold is display-only; AP uses all saved predictions",
        fill=PRED_COLOR,
        font=font,
    )
    title.text(
        (12, 71),
        "Pinhole overlay; RGB/event pixel rectification is not certified.",
        fill=(205, 205, 205),
        font=font,
    )
    return tile, counts


def training_curve(path: Path, output: Path) -> dict[str, Any]:
    """Plot measured losses and step time without smoothing away raw observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Training log is empty")
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, key, label in zip(
        axes,
        ("loss", "loss_center", "step_seconds"),
        ("Weighted train loss", "Center train loss", "Measured step time (s)"),
        strict=True,
    ):
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Training log contains nonfinite {key}")
        axis.plot(steps, values, linewidth=1, color="#314a89")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Optimizer step (zero based)")
    fig.suptitle("Measured training trace; optimizer updates, no smoothing")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"path": str(output), "sha256": sha256(output), "training_rows": len(rows)}


def render(args: argparse.Namespace) -> dict[str, Any]:
    """Render deterministic comparisons from saved predictions and full RGB frames."""
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("Image outputs must remain under this repository's output/ directory")
    if output.exists():
        raise FileExistsError(
            "Choose a new visual output directory; existing artifacts are preserved"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.eap_root / "data/train.parquet") != manifest["source_media_sha256"]:
        raise ValueError("Current RGB/calibration manifest differs from the recorded pilot")
    tokens = fixed_tokens(manifest[args.role]["sample_tokens"], args.count)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.predictions]
    labels = args.labels or [path.stem for path in args.predictions]
    if len(labels) != len(payloads):
        raise ValueError("Provide one label per prediction JSON")
    for payload in payloads:
        if any(
            token not in payload[key] for key in ("predictions", "ground_truth") for token in tokens
        ):
            raise ValueError("A selected manifest token is missing from a prediction artifact")
    if any(
        payload["ground_truth"][token] != payloads[0]["ground_truth"][token]
        for payload in payloads[1:]
        for token in tokens
    ):
        raise ValueError(
            "Before/after artifacts contain different ground truth for a selected frame"
        )
    output.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "schema": "eap_detection_visuals_v1",
        "selected_tokens": tokens,
        "selection": "linspace positions in declared manifest order; independent of predictions",
        "display_score_threshold": args.score_threshold,
        "prediction_sources": {str(path): sha256(path) for path in args.predictions},
        "manifest_sha256": sha256(args.manifest),
        "renderer_sha256": sha256(Path(__file__)),
        "geometry_source_sha256": sha256(ROOT / "scripts/verify_eap_geometry_real.py"),
        "model_inference_executed": False,
        "metrics_recomputed": False,
        "frames": [],
        "artifacts": [],
    }
    rows = []
    with EAPRGBDataset(args.eap_root, tokens) as dataset:
        for index, token in enumerate(tokens):
            sample = dataset[index]
            panels = []
            for payload, label in zip(payloads, labels, strict=True):
                recorded_hash = payload["data_receipts"][token]["rgb_member_sha256"]
                if recorded_hash != sample.member_sha256:
                    raise ValueError(f"RGB bytes differ from the prediction artifact: {token}")
                tile, counts = overlay(
                    sample,
                    payload["predictions"][token],
                    payload["ground_truth"][token],
                    label=label,
                    score_threshold=args.score_threshold,
                )
                panels.append(tile)
                receipt["frames"].append(
                    {
                        "sample_token": token,
                        "label": label,
                        "rgb_member_sha256": sample.member_sha256,
                        **counts,
                    }
                )
            row = Image.new(
                "RGB", (sum(panel.width for panel in panels), max(p.height for p in panels))
            )
            cursor = 0
            for panel in panels:
                row.paste(panel, (cursor, 0))
                cursor += panel.width
            path = output / f"frame_{index:02d}.png"
            row.save(path)
            receipt["artifacts"].append({"path": str(path), "sha256": sha256(path)})
            rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)))
    cursor = 0
    for row in rows:
        sheet.paste(row, (0, cursor))
        cursor += row.height
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path)
    receipt["artifacts"].append({"path": str(sheet_path), "sha256": sha256(sheet_path)})
    if args.training_log:
        receipt["artifacts"].append(
            training_curve(args.training_log, output / "training_curve.png")
        )
        receipt["training_log_sha256"] = sha256(args.training_log)
    (output / "visual_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    """CLI to render one prediction set or a fixed before/after comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--role", choices=("overfit16", "train", "selection"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--training-log", type=Path)
    args = parser.parse_args()
    receipt = render(args)
    print(
        json.dumps(
            {"selected_tokens": receipt["selected_tokens"], "artifacts": receipt["artifacts"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
