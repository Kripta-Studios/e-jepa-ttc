#!/usr/bin/env python
"""Audit a built Object Event TTC v4 cache before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = GarlTTCLHRCacheDataset(
        args.manifest.resolve(), splits=("train", "validation")
    )
    count = min(len(dataset), args.samples)
    endpoints = []
    channel_values = []
    scale_ratios = []
    valid = []
    precontext_sources: dict[str, int] = {}
    for index in range(count):
        record = dataset[index]
        events = torch.as_tensor(record["event_v4_common_roi"], dtype=torch.float32)
        boxes = torch.as_tensor(record["event_v4_boxes_xyxy"], dtype=torch.float32)
        endpoints.append(
            float(
                torch.nn.functional.cosine_similarity(
                    events[1].reshape(1, -1),
                    events[2].reshape(1, -1),
                    dim=1,
                    eps=1.0e-8,
                )[0]
            )
        )
        channel_values.append(events.permute(1, 0, 2, 3).reshape(12, -1))
        first_height = float(boxes[1, 3] - boxes[1, 1])
        second_height = float(boxes[2, 3] - boxes[2, 1])
        scale_ratios.append(second_height / max(first_height, 1.0e-6))
        valid.append(bool(record.get("event_v4_precontext_valid", False)))
        source = str(record.get("event_v4_precontext_source", "unknown"))
        precontext_sources[source] = precontext_sources.get(source, 0) + 1
    stacked = torch.cat(channel_values, dim=1)
    report = {
        "artifact_type": "object_event_v4_cache_audit_v1",
        "manifest": args.manifest.resolve().as_posix(),
        "sample_count": count,
        "precontext_valid_fraction": float(np.mean(valid)),
        "precontext_source_counts": dict(sorted(precontext_sources.items())),
        "event_shape": list(dataset[0]["event_v4_common_roi"].shape),
        "per_channel_std": stacked.std(dim=1).tolist(),
        "per_channel_nonzero_fraction": (stacked != 0).float().mean(dim=1).tolist(),
        "endpoint_cosine_mean": float(np.mean(endpoints)),
        "endpoint_cosine_std": float(np.std(endpoints)),
        "mapped_box_scale_ratio_mean": float(np.mean(scale_ratios)),
        "mapped_box_scale_ratio_std": float(np.std(scale_ratios)),
        "all_channels_nonconstant": bool((stacked.std(dim=1) > 0).all()),
        "passed": bool(
            np.mean(valid) >= 0.80
            and list(dataset[0]["event_v4_common_roi"].shape)[:2] == [3, 12]
            and (stacked.std(dim=1) > 0).all()
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise RuntimeError("Object Event v4 cache audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
