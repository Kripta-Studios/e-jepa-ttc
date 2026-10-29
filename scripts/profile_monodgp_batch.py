"""Compare batch sizes on real RGB frames with the pretrained velocity variant."""

# ruff: noqa: E402 -- direct execution from the isolated detector environment.

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from profile_monodgp_precision import memory_snapshot
from smoke_monodgp_portable import VELOCITY_VARIANT, build_detector, require_finite, sha256
from train_monodgp_portable import collate_on_gpu, write_json

from e_jepa_ttc.detection.backbone_initialization import (
    OFFICIAL_SHA256,
    load_imagenet_resnet50_backbone,
)
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample


def main() -> int:
    """Run three warmup and five measured updates for each independent profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--portable-root",
        type=Path,
        default=ROOT / "output/detector_sources/MonoDGP-portable-velocity",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    tokens = json.loads(args.manifest.read_text())["overfit16"]["sample_tokens"][:4]
    with EAPRGBDataset(args.data_root, tokens) as dataset:
        prepared = [
            prepare_sample(dataset[i], width=640, height=384, include_velocity=True)
            for i in range(4)
        ]
    report = {
        "variant": VELOCITY_VARIANT,
        "scope": "real_four_train_frame_throughput_not_accuracy",
        "image_hw": [384, 640],
        "group_num": 11,
        "precision": "fp32_highest",
        "image_layout": "contiguous_NCHW",
        "sample_tokens": tokens,
        "seed": 7,
        "warmup_steps": 3,
        "measured_steps": 5,
        "manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(Path(__file__)),
        "source_receipt_sha256": sha256(args.portable_root / ".portable_source.json"),
        "profiles": [],
    }
    for batch_size in (1, 2, 4):
        torch.manual_seed(7)
        model, criterion, config = build_detector(
            args.portable_root, 11, predict_velocity=True, velocity_loss_coef=0.1
        )
        receipt = load_imagenet_resnet50_backbone(model, args.backbone_weights, OFFICIAL_SHA256)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        model.train()
        criterion.train()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for step in range(8):
            subset = [prepared[(step * batch_size + i) % len(prepared)] for i in range(batch_size)]
            images, calibration, targets, sizes = collate_on_gpu(subset)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            outputs = model(images, calibration, targets, sizes)
            require_finite(outputs)
            losses = criterion(outputs, targets)
            require_finite(losses)
            loss = sum(
                value * criterion.weight_dict[key]
                for key, value in losses.items()
                if key in criterion.weight_dict
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            rows.append(
                {
                    "step": step,
                    "seconds": time.perf_counter() - tick,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "memory": memory_snapshot(),
                }
            )
            del outputs, losses, loss
        median = statistics.median(row["seconds"] for row in rows[3:])
        item = {
            "batch_size": batch_size,
            "status": "PASS",
            "median_step_seconds": median,
            "images_per_second": batch_size / median,
            "estimated_compute_seconds_for_2048_image_exposures": 2048 * median / batch_size,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "steps": rows,
            "config": config,
            "backbone_initialization": receipt,
        }
        report["profiles"].append(item)
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    k: item[k]
                    for k in (
                        "batch_size",
                        "median_step_seconds",
                        "images_per_second",
                        "peak_allocated_bytes",
                    )
                }
            ),
            flush=True,
        )
        del model, criterion, optimizer
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
