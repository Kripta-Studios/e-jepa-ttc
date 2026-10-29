"""Native TRAIN40 feature and head parity before official test predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.sota_evidence.predict_test import predict
from operational.sota_evidence.test_inputs import (
    EXPOSURE_COLUMNS,
    INPUT_COLUMNS,
    describe_input,
    supported_job,
)
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.history_features import load_source


def run(campaign: Path, raw_root: Path, output: Path) -> None:
    """Verify three predeclared train examples against native cached model inputs."""
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = FrozenModels(campaign, "cuda")
    manifest = json.loads((campaign / "H8_FEATURE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = digest(campaign / "H8_FEATURE_MANIFEST.json")
    source = load_source(campaign, manifest)
    rows = pd.read_parquet(campaign / "TRAIN40_ROWS.parquet", columns=list(INPUT_COLUMNS))
    media = pd.read_parquet(raw_root / "data/train.parquet", columns=list(EXPOSURE_COLUMNS))
    exposures = {(r["sequence_id"], r["rgb_member_path"]): r for r in media.to_dict("records")}
    pool = ReaderPool()
    checks = []
    try:
        for ordinal in (0, len(rows) // 2, len(rows) - 1):
            job = describe_input(rows.iloc[ordinal].to_dict(), exposures, raw_root, split="train")
            reader = pool.get(job["path"])
            timestamps = reader.datasets["events/t"]
            job = supported_job(job, int(timestamps[0]), int(timestamps[-1]))
            events = encode_union(
                reader,
                job["windows"],
                np.arange(750000, -1, -50000, dtype=np.int64),
                np.concatenate((np.zeros(8, bool), job["valid"])),
                tuple(job["square"]),
                sequence_id=job["sequence"],
                roi_size=128,
                event_pixel_diff=5,
            ).numpy()[-8:]
            history = source.history[ordinal, -8:]
            raw = np.array(source.features[np.maximum(history, 0)], copy=True)
            raw[history < 0] = 0
            predicted = predict(model, events, job["valid"], job, reference_features=raw)
            xs = source.gather(torch.tensor([ordinal]))[:4]
            with torch.inference_mode():
                for seed in (7, 13, 23):
                    reference = float(
                        phase_to_ttc(
                            model.heads[seed](*(x.to(model.device) for x in xs))["point_phase"]
                        ).item()
                    )
                    np.testing.assert_allclose(
                        predicted[f"H8_seed{seed}"], reference, atol=0.01, rtol=1e-4
                    )
                    checks.append(
                        {
                            "ordinal": ordinal,
                            "seed": seed,
                            "reference_ttc": reference,
                            "current_ttc": predicted[f"H8_seed{seed}"],
                        }
                    )
    finally:
        pool.close()
        atomic_json(
            output / "GPU_QA_RESERVATION.json",
            {
                "seconds": time.monotonic() - started,
                "optimizer_updates": 0,
            },
        )
    atomic_json(
        output / "GPU_TRAIN_PARITY.json",
        {
            "status": "PASSED",
            "checks": checks,
            "feature_atol": 1e-5,
            "feature_rtol": 1e-4,
            "ttc_atol": 0.01,
            "ttc_rtol": 1e-4,
            "runtime_sha256": digest(Path(__file__).with_name("predict_test.py")),
            "source_sha256": digest(Path(__file__)),
            "model_bindings": model.bindings,
            "optimizer_updates": 0,
            "test_rows_evaluated": 0,
        },
    )


def main() -> None:
    """Check already-trained endpoints without backward or optimizer creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "raw-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        run(args.campaign, args.raw_root, args.output)
    finally:
        atomic_json(
            args.output / f"GPU_QA_ACCOUNTING_{time.time_ns()}.json",
            {
                "seconds": time.monotonic() - started,
                "optimizer_updates": 0,
            },
        )


if __name__ == "__main__":
    main()
