"""Verify sensor-only CPU inference workers against32 original cached TRAIN inputs."""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
from typing import cast

from .common import ROOT, Campaign, atomic_json, digest, read


def main() -> int:
    """No neural models, optimizer updates, held-out queries or inference labels are used."""
    import numpy as np
    import torch

    from .garl_feature_inputs import InferenceWorkers
    from .garl_train import records

    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    c.freeze()
    rows = records(c)
    samples = read(c.out / "garl/DECODE_INPUT_QA.json")["samples"][:32]
    reference = {}
    identity = {}
    pin = digest(c.out / "garl/PROTOCOL.json")
    for sample in samples:
        token = sample["sample_token"]
        row = rows[token]
        stat = (c.raw / row["sequence_id"] / "events.h5").stat()
        identity[token] = stat.st_size, stat.st_mtime_ns
        key = hashlib.sha256(
            f"{pin}:{token}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()
        payload = (c.out / f"garl/native_cache/{key}.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != sample["original_NPZ_sha256"]:
            raise ValueError("sensor-only QA reference differs from the original TRAIN cache")
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            reference[token] = z["events"].copy()
    torch_rng = torch.get_rng_state()
    numpy_rng = cast(tuple, np.random.get_state())
    python_rng = random.getstate()
    workers = InferenceWorkers(c)
    try:
        checks = []
        # Match the future query's bounded eight-record submission and ordered consumption.
        for start in range(0, len(samples), 8):
            batch = samples[start : start + 8]
            futures = []
            for sample in batch:
                token = sample["sample_token"]
                row = rows[token]
                native = {k: row[k] for k in ("sequence_id", "boxes_xyxy", "event_windows_us")}
                futures.append(workers.submit(native, identity[token]))
            for sample, future in zip(batch, futures, strict=True):
                actual = future.result()
                token = sample["sample_token"]
                if not np.array_equal(actual, reference[token]):
                    raise ValueError(
                        "sensor-only CPU preparation differs from original FP32 TRAIN input"
                    )
                checks.append(sample)
        after = cast(tuple, np.random.get_state())
        if (
            not torch.equal(torch_rng, torch.get_rng_state())
            or numpy_rng[0] != after[0]
            or not np.array_equal(numpy_rng[1], after[1])
            or numpy_rng[2:] != after[2:]
            or python_rng != random.getstate()
        ):
            raise ValueError("native CPU inference preparation consumed parent scientific RNG")
        atomic_json(
            c.out / "garl/FEATURE_INPUT_QA.json",
            {
                "status": "PASSED",
                "native_protocol_sha256": pin,
                "implementation_sha256": digest(Path(__file__).with_name("garl_feature_inputs.py")),
                "QA_implementation_sha256": digest(Path(__file__)),
                "samples": checks,
                "CPU_preparation_workers": 4,
                "max_pending_inputs": 8,
                "exact_original_FP32_input_and_parent_RNG_parity": True,
                "inference_labels_or_geometry_supervision_passed_to_workers": False,
                "GPU_models_constructed": 0,
                "optimizer_updates": 0,
                "raw_event_inputs_prepared": len(checks),
                "holdout_or_OLD_DEV_inputs_prepared": 0,
            },
        )
        print("FEATURE_INPUT_QA_PASSED", len(checks), flush=True)
    finally:
        workers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
