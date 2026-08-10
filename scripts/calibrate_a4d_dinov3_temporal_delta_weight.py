#!/usr/bin/env python
"""Train-only calibration for the post-A4 temporal relational-delta objective.

The existing A4 endpoint objective remains frozen at weight 4.0.  This script
uses the same 64 equispaced public-train samples and the same random-init seed
as A4, takes zero optimizer steps, and chooses a conservative temporal weight
such that at initialization its median contribution targets 25% of the
*weighted* A4 endpoint-relational contribution:

    lambda_delta_raw = 0.25 * lambda_endpoint
                       * median(L_endpoint_raw) / median(L_delta_raw)

with lambda_endpoint read from the frozen A4 config.  The result is clipped to
[0.25, 4.0].  Validation/test are never instantiated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DINOv3RelationalTeacherDataset,
)
from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
    collate_object_event_v4,
)
from e_jepa_ttc.distillation.dinov3_relational import (  # noqa: E402
    A4_RELATION_OFFSETS,
    local_cosine_relation_maps,
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
)
from e_jepa_ttc.reproducibility import seed_everything  # noqa: E402

CLAMP_RANGE = (0.25, 4.0)
TARGET_FRACTION_OF_WEIGHTED_ENDPOINT = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration distribution is empty or non-finite")
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = _read_yaml(path)
    if raw.pop("model", None) != "e_jepa_causal_scale_event_v8":
        raise ValueError("A4D calibration requires causal-scale event v8")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def run(
    config_path: Path,
    *,
    samples: int,
    seed: int,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    if samples != 64 or seed != 7:
        raise ValueError("A4D preregistration fixes samples=64 and seed=7")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("commit tracked changes before A4D calibration")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("commit staged changes before A4D calibration")

    raw = _read_yaml(config_path)
    training = raw.get("training")
    data = raw.get("data")
    if not isinstance(training, dict) or not isinstance(data, dict):
        raise ValueError("A4 config requires training/data mappings")
    if training.get("representation_supervision") != "dinov3_local_relational":
        raise ValueError("calibrate A4D from the frozen A4 endpoint-relational config")
    endpoint_weight = float(training.get("representation_distillation_weight", 0.0))
    if endpoint_weight != 4.0:
        raise ValueError("A4D calibration requires the frozen A4 endpoint weight 4.0")
    dino_teacher = data.get("dinov3_relational_teacher")
    if not isinstance(dino_teacher, dict):
        raise ValueError("A4 config lacks data.dinov3_relational_teacher")

    seed_everything(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    event_manifest = (ROOT / str(data["cache_manifest"])).resolve(strict=True)
    teacher_manifest = (ROOT / str(dino_teacher["manifest"])).resolve(strict=True)
    train_dataset = GarlTTCObjectEventV4Dataset(str(event_manifest), splits=("train",))
    wrapped = DINOv3RelationalTeacherDataset(
        train_dataset,
        manifest_path=teacher_manifest,
        expected_artifact_sha256=str(dino_teacher["artifact_sha256"]),
        expected_manifest_sha256=str(dino_teacher["manifest_sha256"]),
    )
    if len(wrapped) < samples:
        raise ValueError(f"train dataset has {len(wrapped)} rows, need {samples}")
    indices = np.linspace(0, len(wrapped) - 1, samples, dtype=int).tolist()
    loader = DataLoader(
        Subset(wrapped, indices),
        batch_size=min(samples, int(training["batch_size"])),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_object_event_v4,
    )

    model = CausalScaleTTC(
        _model_config((ROOT / str(raw["model_config"])).resolve(strict=True))
    ).to(device)
    model.eval()

    endpoint_errors: list[float] = []
    delta_errors: list[float] = []
    teacher_delta_abs: list[float] = []
    per_offset_teacher_delta_abs: list[list[float]] = [
        [] for _ in A4_RELATION_OFFSETS
    ]
    valid_pair_fractions: list[float] = []
    tokens: list[str] = []

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        for host_batch in loader:
            batch = host_batch.to(device)
            delta_t = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            output = model(batch.events, delta_t, return_dense_features=True)
            if output.endpoint_dense_features is None:
                raise RuntimeError("model did not return dense features")
            student_features = output.endpoint_dense_features[:, 1:3]
            student = local_cosine_relation_maps(student_features)
            teacher = batch.dinov3_relation_targets.float()
            teacher_valid = batch.dinov3_relation_valid.bool()
            endpoint_valid = teacher_valid & student.valid
            pair_valid = endpoint_valid[:, 0] & endpoint_valid[:, 1]
            teacher_pair_valid = teacher_valid[:, 0] & teacher_valid[:, 1]

            endpoint_error_map = (student.values - teacher).abs()
            student_delta = student.values[:, 1] - student.values[:, 0]
            teacher_delta = teacher[:, 1] - teacher[:, 0]
            delta_error_map = (student_delta - teacher_delta).abs()
            teacher_delta_map = teacher_delta.abs()

            for index, token in enumerate(batch.sample_tokens):
                if not endpoint_valid[index].any() or not pair_valid[index].any():
                    raise ValueError(f"empty relational validity for train token {token}")
                if not teacher_pair_valid[index].any():
                    raise ValueError(f"empty teacher temporal validity for train token {token}")
                endpoint_errors.append(
                    float(endpoint_error_map[index][endpoint_valid[index]].mean().cpu())
                )
                delta_errors.append(
                    float(delta_error_map[index][pair_valid[index]].mean().cpu())
                )
                teacher_delta_abs.append(
                    float(teacher_delta_map[index][teacher_pair_valid[index]].mean().cpu())
                )
                valid_pair_fractions.append(float(pair_valid[index].float().mean().cpu()))
                for offset_index in range(len(A4_RELATION_OFFSETS)):
                    mask = teacher_pair_valid[index, offset_index]
                    if mask.any():
                        per_offset_teacher_delta_abs[offset_index].append(
                            float(teacher_delta_map[index, offset_index][mask].mean().cpu())
                        )
                tokens.append(str(token))

    endpoint = np.asarray(endpoint_errors, dtype=np.float64)
    delta = np.asarray(delta_errors, dtype=np.float64)
    teacher_energy = np.asarray(teacher_delta_abs, dtype=np.float64)
    if endpoint.size != samples or delta.size != samples or teacher_energy.size != samples:
        raise RuntimeError("A4D calibration did not collect exactly 64 valid samples")

    median_endpoint = float(np.median(endpoint))
    median_delta = float(np.median(delta))
    median_teacher_energy = float(np.median(teacher_energy))
    if min(median_endpoint, median_delta) <= 0.0:
        raise ValueError("A4D calibration losses must be strictly positive")
    if median_teacher_energy <= 0.0:
        raise ValueError("cached DINO teacher has degenerate zero temporal relation change")

    lambda_raw = (
        TARGET_FRACTION_OF_WEIGHTED_ENDPOINT
        * endpoint_weight
        * median_endpoint
        / median_delta
    )
    selected = float(np.clip(lambda_raw, *CLAMP_RANGE))
    if not math.isfinite(selected):
        raise FloatingPointError("A4D calibrated temporal weight is non-finite")

    payload: dict[str, Any] = {
        "artifact_type": "a4d_dinov3_temporal_delta_weight_calibration_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "optimizer_steps": 0,
            "ttc_labels_used_for_calibration": False,
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "tracked_dirty": False,
        "source_a4_config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "teacher_artifact_sha256": str(dino_teacher["artifact_sha256"]),
        "teacher_manifest_sha256": str(dino_teacher["manifest_sha256"]),
        "samples_requested": samples,
        "samples_collected": len(tokens),
        "seed": seed,
        "tokens": tokens,
        "endpoint_distillation_weight": endpoint_weight,
        "target_fraction_of_weighted_endpoint": TARGET_FRACTION_OF_WEIGHTED_ENDPOINT,
        "calibration_formula": (
            "0.25 * endpoint_weight * median_endpoint_relation_error "
            "/ median_temporal_delta_error"
        ),
        "median_endpoint_relation_error": median_endpoint,
        "median_temporal_delta_error": median_delta,
        "median_teacher_temporal_delta_abs": median_teacher_energy,
        "endpoint_relation_error_distribution": _quantiles(endpoint),
        "temporal_delta_error_distribution": _quantiles(delta),
        "teacher_temporal_delta_abs_distribution": _quantiles(teacher_energy),
        "temporal_pair_valid_fraction_distribution": _quantiles(
            np.asarray(valid_pair_fractions, dtype=np.float64)
        ),
        "teacher_temporal_delta_abs_by_offset": {
            f"dy{dy}_dx{dx}": _quantiles(np.asarray(values, dtype=np.float64))
            for (dy, dx), values in zip(
                A4_RELATION_OFFSETS, per_offset_teacher_delta_abs, strict=True
            )
        },
        "lambda_raw": float(lambda_raw),
        "clamp_range": list(CLAMP_RANGE),
        "selected_weight": selected,
    }
    sign_artifact(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "artifact": str(output_path),
        "artifact_sha256": payload["artifact_sha256"],
        "file_sha256": _sha256(output_path),
        "median_endpoint_relation_error": median_endpoint,
        "median_temporal_delta_error": median_delta,
        "median_teacher_temporal_delta_abs": median_teacher_energy,
        "lambda_raw": lambda_raw,
        "selected_weight": selected,
    }, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(
            args.experiment_config.resolve(),
            samples=args.samples,
            seed=args.seed,
            device_name=args.device,
            output_path=args.output.resolve(),
        )
    except Exception as error:
        parser.exit(2, f"A4D calibration failed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
