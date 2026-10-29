"""Check the label-free adapter against frozen TRAIN40 geometry and real tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.efficient_context.common import digest
from operational.sota_evidence.test_inputs import (
    EXPOSURE_COLUMNS,
    INPUT_COLUMNS,
    describe_input,
    supported_job,
)
from operational.train40_system.durable_io import atomic_json


def run(campaign: Path, raw_root: Path, output: Path) -> None:
    """Compare all TRAIN metadata and three frozen raw-input examples exactly."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = pd.read_parquet(campaign / "TRAIN40_ROWS.parquet", columns=list(INPUT_COLUMNS))
    media = pd.read_parquet(raw_root / "data/train.parquet", columns=list(EXPOSURE_COLUMNS))
    exposures = {(r["sequence_id"], r["rgb_member_path"]): r for r in media.to_dict("records")}
    with np.load(campaign / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        expected = {k: stored[k] for k in ("tokens", "windows_us", "square_xyxy", "delta_t_s")}
    jobs = []
    for i, row in enumerate(rows.to_dict("records")):
        job = describe_input(row, exposures, raw_root, split="train")
        assert job["sample_token"] == expected["tokens"][i]
        np.testing.assert_array_equal(job["windows"], expected["windows_us"][i])
        np.testing.assert_array_equal(job["square"], expected["square_xyxy"][i])
        np.testing.assert_array_equal(np.float32(job["delta"]), expected["delta_t_s"][i])
        jobs.append(job)
    freeze = json.loads((campaign / "PREPARE_FREEZE.json").read_text(encoding="utf-8"))
    cache = Path(freeze["cache_root"])
    # Fixed coverage across beginning, middle and end; no metric-based selection.
    checks = []
    pool = ReaderPool()
    try:
        for i in (0, len(rows) // 2, len(rows) - 1):
            job = jobs[i]
            reader = pool.get(job["path"])
            timestamps = reader.datasets["events/t"]
            job = supported_job(job, int(timestamps[0]), int(timestamps[-1]))
            valid = np.zeros(16, bool)
            valid[-1] = True
            tensor = encode_union(
                reader,
                job["windows"],
                np.arange(750000, -1, -50000, dtype=np.int64),
                valid,
                tuple(job["square"]),
                sequence_id=job["sequence"],
                roi_size=128,
                event_pixel_diff=5,
            ).numpy()[-1]
            path = cache / f"shard_{i // 32:05d}.npz"
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if digest(path) != receipt["sha256"]:
                raise ValueError("Frozen training shard changed")
            with np.load(path, allow_pickle=False) as stored:
                position = int(np.flatnonzero(stored["ordinals"] == i)[0])
                np.testing.assert_array_equal(tensor, stored["events"][position])
            checks.append(
                {
                    "ordinal": i,
                    "sample_token": job["sample_token"],
                    "tensor_exact_equal": True,
                    "reference_sha256": receipt["sha256"],
                }
            )
    finally:
        pool.close()
    atomic_json(
        output,
        {
            "status": "PASSED",
            "metadata_rows_exact": len(jobs),
            "tensor_checks": checks,
            "optimizer_updates": 0,
            "test_labels_read": False,
            "adapter_sha256": digest(Path(__file__).with_name("test_inputs.py")),
            "qa_source_sha256": digest(Path(__file__)),
            "train_index_sha256": digest(campaign / "TRAIN40_INDEX.npz"),
        },
    )


def main() -> None:
    """Run only train-input parity; never opens test targets or fits a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "raw-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.raw_root, args.output)


if __name__ == "__main__":
    main()
