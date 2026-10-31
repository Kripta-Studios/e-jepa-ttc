"""Replay both original Slider recordings, seal predictions, then join public GT."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from operational.evttc_competition.geometry import load_calibration
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_competition.runtime import CompetitionRuntime
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.evaluation import metrics
from operational.phase_campaign.external import join_external_targets
from operational.phase_campaign.review_controls import shared_scores
from operational.phase_robust.cache import SensorCache
from operational.phase_robust.runtime import load_fusion, predict_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sensors", "candidate", "fusion", "public", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    fusion = load_fusion(args.fusion)
    runtime = CompetitionRuntime.from_candidate(args.candidate, device="cuda")
    atomic_receipt(
        args.output / "FREEZE.json",
        {
            "fusion_freeze_sha256": file_sha(args.fusion / "FINAL_FREEZE.json"),
            "sources": {
                str(p): file_sha(p)
                for p in [
                    Path(__file__),
                    Path("operational/phase_robust/cache.py"),
                    Path("operational/phase_robust/runtime.py"),
                ]
            },
            "scope": (
                "current gate fit/selection reserve; historical benchmark exposure; "
                "public bbox exact equality and frame/clock target join required"
            ),
            "upstream": (
                "original nine-component candidate, eager producers unchanged; "
                "lossless bounded sensor cache"
            ),
        },
    )
    for speed in (750, 1000):
        name = f"Slider-{speed}"
        output = args.output / name
        calibration = load_calibration(args.sensors / name, "slider_candidate")
        with SensorSequence(args.sensors, name) as reader:
            public_box = np.loadtxt(args.public / f"{speed}_bbox.csv", delimiter=",")
            local_box = np.loadtxt(args.sensors / name / "bbox.csv", delimiter=",")
            if not np.array_equal(public_box, local_box):
                raise ValueError("Public release and sensor query boxes differ")
            rows = reader.bbox_rows()
            boxes = np.array([r["box_xyxy"] for r in rows])
            clocks = np.array([r["anchor_us"] for r in rows])
            cached = SensorCache(reader)
            native, preds, geometric, quality, latency, parity = [], [], [], [], [], []
            for i, row in enumerate(rows):
                start = time.perf_counter()
                prepared = prepare_query(cast(SensorSequence, cached), rows, row, calibration)
                if i in {0, len(rows) // 2, len(rows) - 1}:
                    plain = prepare_query(reader, rows, row, calibration)
                    # Prepared observations include tensors and timestamp contracts.
                    for a, b in zip(
                        prepared.events + prepared.rgb, plain.events + plain.rgb, strict=True
                    ):
                        for field in a.__dataclass_fields__:
                            x, y = getattr(a, field), getattr(b, field)
                            if isinstance(x, torch.Tensor):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)
                            elif isinstance(x, np.ndarray):
                                np.testing.assert_array_equal(x, y)
                            elif x != y:
                                raise AssertionError(f"Cache metadata differs: {field}")
                    parity.append(i)
                result = runtime.predict(prepared.events, prepared.rgb, row["anchor_us"])
                values = np.array(
                    [result.model_ttc, result.rgb_ttc, result.event_ttc], dtype=np.float32
                )
                value, physical = predict_history(fusion, values, boxes, clocks, row["anchor_us"])
                native.append(values)
                preds.append(value)
                geometric.append(physical["ttc"] if physical["valid"] else np.nan)
                quality.append(physical["quality"])
                latency.append(time.perf_counter() - start)
                print(f"{name} {i + 1}/{len(rows)} elapsed={latency[-1]:.3f}s", flush=True)
            arrays = {
                "native": np.array(native),
                "prediction": np.array(preds),
                "geometry": np.clip(np.array(geometric), -60, 60),
                "quality": np.array(quality),
                "clocks": clocks,
            }
            save_npz(output / "PREDICTIONS.npz", arrays)
            atomic_receipt(
                output / "SEALED.json",
                {
                    "predictions_sha256": file_sha(output / "PREDICTIONS.npz"),
                    "GT_opened": False,
                    "public_bbox_sha256": file_sha(args.public / f"{speed}_bbox.csv"),
                    "sensor_bag_sha256": file_sha(args.sensors / name / f"slider_{speed}.bag"),
                    "calibration_sha256": file_sha(args.sensors / name / "calibParams.mat"),
                    "sensor_metadata": reader.source_metadata(),
                    "cache_parity_queries": parity,
                    "cache_hits": cached.hits,
                    "cache_misses": cached.misses,
                    "mean_seconds": float(np.mean(latency)),
                    "p95_seconds": float(np.quantile(latency, 0.95)),
                },
            )
            labels = np.loadtxt(args.public / f"{speed}_gt.csv", delimiter=",", ndmin=2)
            target = join_external_targets(rows, labels)
            save_npz(output / "TARGETS.npz", {"target": target})
            comparisons = {
                "F0": arrays["native"][:, 0],
                "candidate": arrays["prediction"],
                "geometry": arrays["geometry"],
            }
            atomic_receipt(
                output / "REPORT.json",
                {
                    "full": {k: metrics(target, p) for k, p in comparisons.items()},
                    "shared": shared_scores(target, comparisons),
                    "target_sha256": file_sha(args.public / f"{speed}_gt.csv"),
                    "official_equivalence": False,
                    "extrinsics": "provisional rotation only, unchanged",
                },
            )
            print(
                f"COMPLETE {name}: public labels joined by exact frame and <=2us clock", flush=True
            )


if __name__ == "__main__":
    main()
