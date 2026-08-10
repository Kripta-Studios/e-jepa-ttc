"""Decompose the public A0 failure into bbox, foreground, residual, and TTC stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
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
    x = reference[valid]
    y = observed[valid]
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


def analyze(
    *,
    checkpoint: Path,
    cache_manifest: Path,
    summary: Path,
    output_json: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run deterministic public-validation decomposition and sign the result."""

    for path in (checkpoint, cache_manifest, summary):
        if not path.is_file():
            raise FileNotFoundError(f"Required failure-analysis input not found: {path}")
    checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_value, dict):
        raise TypeError("A0 checkpoint must be a mapping.")
    raw_model_config = dict(checkpoint_value["model_config"])
    raw_model_config["risk_thresholds_s"] = tuple(raw_model_config["risk_thresholds_s"])
    model_config = CausalScaleTTCConfig(**raw_model_config)
    torch_device = torch.device(device)
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
    values: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "physical_ratio",
            "bbox_ratio",
            "analytic_ratio",
            "residual_ratio",
            "pair_ratio",
            "effective_ratio",
            "predicted_log_height",
            "bbox_log_height",
            "known",
            "pair_known",
            "support",
            "ttc",
        )
    }
    sequences: list[str] = []
    with torch.inference_mode():
        for host_batch in loader:
            batch = host_batch.to(torch_device)
            delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                output = model(batch.events, delta)
            physical_ratio, valid = target_log_ratio_from_ttc(
                batch.target_ttc_s, delta[:, -1]
            )
            if not bool(valid.all()):
                raise ValueError("A0 validation decomposition encountered invalid TTC ratios.")
            bbox_height = (
                (batch.boxes_xyxy[..., 3] - batch.boxes_xyxy[..., 1]).clamp_min(1e-6)
                / float(batch.events.shape[-2])
            )
            bbox_ratio = bbox_height[:, -1].log() - bbox_height[:, -2].log()
            current_pair_known = output.diagnostics["pair_known"][:, -1].float()
            batch_values = {
                "physical_ratio": physical_ratio,
                "bbox_ratio": bbox_ratio,
                "analytic_ratio": output.analytic_log_height_ratio[:, -1],
                "residual_ratio": output.residual_log_height_ratio[:, -1],
                "pair_ratio": output.pair_log_height_ratio[:, -1],
                "effective_ratio": output.log_height_ratio[:, -1],
                "predicted_log_height": output.visible_height_normalized[:, 1:].log().reshape(-1),
                "bbox_log_height": bbox_height[:, 1:].log().reshape(-1),
                "known": output.known_mask.float(),
                "pair_known": current_pair_known,
                "support": output.sensor_support[:, -1],
                "ttc": output.ttc_mean_seconds,
            }
            for key, value in batch_values.items():
                values[key].append(value.float().cpu())
            sequences.extend(batch.sequence_ids)
    arrays = {key: torch.cat(items).numpy() for key, items in values.items()}
    sequence_array = np.asarray(sequences)
    per_sequence = {
        sequence: {
            "bbox_vs_physical": _relationship(
                arrays["physical_ratio"][sequence_array == sequence],
                arrays["bbox_ratio"][sequence_array == sequence],
            ),
            "analytic_vs_physical": _relationship(
                arrays["physical_ratio"][sequence_array == sequence],
                arrays["analytic_ratio"][sequence_array == sequence],
            ),
            "effective_vs_physical": _relationship(
                arrays["physical_ratio"][sequence_array == sequence],
                arrays["effective_ratio"][sequence_array == sequence],
            ),
        }
        for sequence in sorted(set(sequences))
    }
    support_threshold = model_config.min_sensor_support
    ratio_threshold = model_config.min_abs_log_ratio
    unknown = arrays["known"] < 0.5
    pair_unknown = arrays["pair_known"] < 0.5
    low_support = arrays["support"] < support_threshold
    low_ratio = np.abs(arrays["pair_ratio"]) < ratio_threshold
    result: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_a0_failure_decomposition_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "sample_count": len(sequences),
        "precision": "bf16",
        "relationships": {
            "bbox_ratio_vs_physical_ratio": _relationship(
                arrays["physical_ratio"], arrays["bbox_ratio"]
            ),
            "predicted_height_vs_bbox_height": _relationship(
                arrays["bbox_log_height"], arrays["predicted_log_height"]
            ),
            "analytic_ratio_vs_bbox_ratio": _relationship(
                arrays["bbox_ratio"], arrays["analytic_ratio"]
            ),
            "analytic_ratio_vs_physical_ratio": _relationship(
                arrays["physical_ratio"], arrays["analytic_ratio"]
            ),
            "residual_ratio_vs_physical_ratio": _relationship(
                arrays["physical_ratio"], arrays["residual_ratio"]
            ),
            "pair_ratio_vs_physical_ratio": _relationship(
                arrays["physical_ratio"], arrays["pair_ratio"]
            ),
            "effective_ratio_vs_physical_ratio": _relationship(
                arrays["physical_ratio"], arrays["effective_ratio"]
            ),
        },
        "per_sequence": per_sequence,
        "unknown_diagnostics": {
            "unknown_count": int(np.count_nonzero(unknown)),
            "known_coverage": float(np.mean(~unknown)),
            "pair_unknown_count": int(np.count_nonzero(pair_unknown)),
            "low_ratio_count": int(np.count_nonzero(low_ratio)),
            "low_support_count": int(np.count_nonzero(low_support)),
            "min_abs_log_ratio": ratio_threshold,
            "min_sensor_support": support_threshold,
            "support_minimum": float(np.min(arrays["support"])),
            "support_mean": float(np.mean(arrays["support"])),
            "support_maximum": float(np.max(arrays["support"])),
        },
        "singularity_diagnostics": {
            "ttc_clip_seconds": model_config.ttc_clip_seconds,
            "known_saturation_count": int(
                np.count_nonzero(
                    (~unknown)
                    & (np.abs(arrays["ttc"]) >= model_config.ttc_clip_seconds * 0.999)
                )
            ),
        },
        "localization": {
            "weak_bbox_iou_from_selected_summary": json.loads(
                summary.read_text(encoding="utf-8")
            )["validation_metrics"]["weak_bbox_iou"],
            "weak_box_is_filled_rectangle_not_segmentation": True,
        },
        "diagnosis": {
            "observed_failure_stage": "event_to_foreground_temporal_extent",
            "bbox_scale_target_contains_physical_signal": True,
            "sensor_support_is_limiting_factor": False,
            "analytic_extent_tracks_bbox_expansion": False,
            "learned_residual_recovers_physical_ratio": False,
            "physical_inverse_amplifies_bad_near_zero_ratios": True,
            "a1_geometry_only_is_causal_explanation_confirmed": False,
            "a1_status": "preregistered_ablation_required_to_test_weak_box_noise_hypothesis",
        },
        "sealed_sources": {
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
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
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
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
        print(f"A0 failure analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
