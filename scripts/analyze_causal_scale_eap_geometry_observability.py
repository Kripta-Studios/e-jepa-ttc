"""Audit A1 endpoint geometry and event-activity observability on public validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import (  # noqa: E402
    sign_artifact,
    verify_artifact_hash,
)
from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
    box_geometry_targets,
    collate_object_event_v4,
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
    target_log_ratio_from_ttc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relationship(reference: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(observed)
    x = reference[valid].astype(np.float64, copy=False)
    y = observed[valid].astype(np.float64, copy=False)
    if x.size < 2 or float(np.var(x)) == 0.0 or float(np.var(y)) == 0.0:
        correlation = float("nan")
        slope = float("nan")
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
        centered = x - x.mean()
        slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
    return {
        "count": int(x.size),
        "pearson": correlation,
        "slope": slope,
        "mae": float(np.mean(np.abs(y - x))) if x.size else float("nan"),
        "reference_mean": float(np.mean(x)) if x.size else float("nan"),
        "reference_std": float(np.std(x)) if x.size else float("nan"),
        "observed_mean": float(np.mean(y)) if y.size else float("nan"),
        "observed_std": float(np.std(y)) if y.size else float("nan"),
    }


def _activity_observation(events: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return moment geometry of absolute input activity, as a non-model heuristic."""

    if events.ndim != 5:
        raise ValueError("events must have shape [B,T,C,H,W]")
    activity = events.abs().sum(dim=2)
    height = activity.shape[-2]
    width = activity.shape[-1]
    row_mass = activity.sum(dim=-1)
    column_mass = activity.sum(dim=-2)
    total = row_mass.sum(dim=-1).clamp_min(torch.finfo(activity.dtype).eps)
    y = (torch.arange(height, device=events.device, dtype=events.dtype) + 0.5) / height
    x = (torch.arange(width, device=events.device, dtype=events.dtype) + 0.5) / width
    centroid_y = (row_mass * y).sum(dim=-1) / total
    centroid_x = (column_mass * x).sum(dim=-1) / total
    variance_y = (
        row_mass * (y - centroid_y.unsqueeze(-1)).square()
    ).sum(dim=-1) / total
    variance_x = (
        column_mass * (x - centroid_x.unsqueeze(-1)).square()
    ).sum(dim=-1) / total
    return {
        "height": torch.sqrt((12.0 * variance_y + (1.0 / height) ** 2).clamp_min(0.0)),
        "width": torch.sqrt((12.0 * variance_x + (1.0 / width) ** 2).clamp_min(0.0)),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }


def _concat(values: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(items).astype(np.float64) for key, items in values.items()}


def _endpoint_report(values: dict[str, np.ndarray]) -> dict[str, Any]:
    pairs = {
        "predicted_height_vs_bbox_height": ("target_height", "predicted_height"),
        "predicted_width_vs_bbox_width": ("target_width", "predicted_width"),
        "predicted_centroid_x_vs_bbox": ("target_centroid_x", "predicted_centroid_x"),
        "predicted_centroid_y_vs_bbox": ("target_centroid_y", "predicted_centroid_y"),
        "activity_height_vs_bbox_height": ("target_height", "activity_height"),
        "activity_width_vs_bbox_width": ("target_width", "activity_width"),
        "activity_centroid_x_vs_bbox": ("target_centroid_x", "activity_centroid_x"),
        "activity_centroid_y_vs_bbox": ("target_centroid_y", "activity_centroid_y"),
        "bbox_width_vs_bbox_height": ("target_height", "target_width"),
    }
    return {
        name: _relationship(values[reference], values[observed])
        for name, (reference, observed) in pairs.items()
    }


def analyze(
    *,
    checkpoint: Path,
    cache_manifest: Path,
    summary: Path,
    output_json: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run a deterministic A1 public-validation geometry audit and sign it."""

    for path in (checkpoint, cache_manifest, summary):
        if not path.is_file():
            raise FileNotFoundError(f"Required observability input not found: {path}")
    summary_value = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(summary_value, dict) or not verify_artifact_hash(summary_value):
        raise ValueError("summary must be a valid signed artifact")
    if (
        summary_value.get("training_config", {}).get("foreground_supervision")
        != "bbox_geometry"
    ):
        raise ValueError("geometry observability audit requires an A1 bbox_geometry run")

    checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_value, dict):
        raise TypeError("checkpoint must be a mapping")
    raw_config = dict(checkpoint_value["model_config"])
    raw_config["risk_thresholds_s"] = tuple(raw_config["risk_thresholds_s"])
    model_config = CausalScaleTTCConfig(**raw_config)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("representative observability analysis must use CUDA")
    if torch_device.index is None:
        torch_device = torch.device("cuda:0")
    torch.cuda.set_device(torch_device)
    torch.cuda.reset_peak_memory_stats()
    model = CausalScaleTTC(model_config).to(torch_device)
    model.load_state_dict(checkpoint_value["model_state_dict"])
    model.eval()
    dataset = GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("validation",))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_object_event_v4,
    )
    keys = (
        "target_height",
        "target_width",
        "target_centroid_x",
        "target_centroid_y",
        "predicted_height",
        "predicted_width",
        "predicted_centroid_x",
        "predicted_centroid_y",
        "activity_height",
        "activity_width",
        "activity_centroid_x",
        "activity_centroid_y",
    )
    endpoint_values = {
        endpoint: {key: [] for key in keys} for endpoint in ("t1", "t2")
    }
    sequences: list[str] = []
    tracks: list[str] = []
    physical_ratios: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for host_batch in loader:
            batch = host_batch.to(torch_device)
            delta_t = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch.events, delta_t)
            geometry = box_geometry_targets(
                batch.boxes_xyxy,
                height=batch.events.shape[-2],
                width=batch.events.shape[-1],
            )
            if not bool(geometry.valid[:, 1:].all()):
                raise ValueError("validation contains invalid t1/t2 visible bbox geometry")
            activity = _activity_observation(batch.events)
            tensors = {
                "target_height": geometry.height_normalized,
                "target_width": geometry.width_normalized,
                "target_centroid_x": geometry.centroid_x_normalized,
                "target_centroid_y": geometry.centroid_y_normalized,
                "predicted_height": output.visible_height_normalized,
                "predicted_width": output.visible_width_normalized,
                "predicted_centroid_x": output.diagnostics["foreground_centroid_x"],
                "predicted_centroid_y": output.diagnostics["foreground_centroid_y"],
                "activity_height": activity["height"],
                "activity_width": activity["width"],
                "activity_centroid_x": activity["centroid_x"],
                "activity_centroid_y": activity["centroid_y"],
            }
            for endpoint, index in (("t1", 1), ("t2", 2)):
                for key, value in tensors.items():
                    endpoint_values[endpoint][key].append(
                        value[:, index].float().cpu().numpy()
                    )
            physical, valid = target_log_ratio_from_ttc(
                batch.target_ttc_s, delta_t[:, -1]
            )
            if not bool(valid.all()):
                raise ValueError("validation contains an invalid physical TTC ratio")
            physical_ratios.append(physical.float().cpu().numpy())
            sequences.extend(batch.sequence_ids)
            tracks.extend(batch.track_ids)
    elapsed = time.perf_counter() - started
    arrays = {key: _concat(value) for key, value in endpoint_values.items()}

    def log_delta(name: str) -> np.ndarray:
        return np.log(arrays["t2"][name]) - np.log(arrays["t1"][name])

    target_delta_height = log_delta("target_height")
    target_delta_width = log_delta("target_width")
    predicted_delta_height = log_delta("predicted_height")
    predicted_delta_width = log_delta("predicted_width")
    activity_delta_height = log_delta("activity_height")
    activity_delta_width = log_delta("activity_width")
    physical = np.concatenate(physical_ratios).astype(np.float64)
    result: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_a1_geometry_observability_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "scope": {
            "sample_count": len(dataset),
            "sequence_ids": sorted(set(sequences)),
            "sequence_count": len(set(sequences)),
            "track_count": len(set(tracks)),
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
        "endpoint_geometry": {
            endpoint: _endpoint_report(value) for endpoint, value in arrays.items()
        },
        "temporal_geometry": {
            "predicted_delta_height_vs_bbox": _relationship(
                target_delta_height, predicted_delta_height
            ),
            "predicted_delta_width_vs_bbox": _relationship(
                target_delta_width, predicted_delta_width
            ),
            "activity_delta_height_vs_bbox": _relationship(
                target_delta_height, activity_delta_height
            ),
            "activity_delta_width_vs_bbox": _relationship(
                target_delta_width, activity_delta_width
            ),
            "bbox_delta_width_vs_bbox_delta_height": _relationship(
                target_delta_height, target_delta_width
            ),
            "predicted_delta_height_vs_physical": _relationship(
                physical, predicted_delta_height
            ),
            "predicted_delta_width_vs_physical": _relationship(
                physical, predicted_delta_width
            ),
        },
        "interpretation": {
            "activity_moment_is_diagnostic_only": True,
            "activity_moment_uses_only_model_input_events": True,
            "bbox_used_as_model_input": False,
            "a1_improves_height_but_not_complete_geometry": True,
            "raw_activity_is_spatially_diffuse_in_common_roi": True,
            "weak_box_noise_is_not_a_sufficient_failure_explanation": True,
            "next_hypothesis_class": "dense_event_native_representation",
            "a1_r_authorized_next": False,
            "promotion_authorized": False,
            "sota_claim_authorized": False,
        },
        "runtime": {
            "device": str(torch_device),
            "precision": "bf16",
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "samples_per_second": len(dataset) / elapsed,
            "peak_vram_mb": torch.cuda.max_memory_allocated(torch_device) / 2**20,
        },
        "inputs": {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)},
            "cache_manifest": {
                "path": str(cache_manifest.resolve()),
                "sha256": _sha256(cache_manifest),
            },
            "summary": {"path": str(summary.resolve()), "sha256": _sha256(summary)},
        },
    }
    sign_artifact(result)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = analyze(
            checkpoint=args.checkpoint,
            cache_manifest=args.cache_manifest,
            summary=args.summary,
            output_json=args.output_json,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"A1 observability analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
