"""Measure cached CUDA extraction against the unchanged sensor-to-head oracle."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_adapt_runtime import CachedAdaptPredictor
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = json.loads(args.population.read_text(encoding="utf-8"))["rows"]
    # Every source, plus adjacent queries to expose warm-cache behavior.
    indices = sorted(
        {
            next(i for i, r in enumerate(rows) if r["sequence_id"] == name)
            for name in {r["sequence_id"] for r in rows}
        }
        | {550, 551, 552, 1450}
    )
    paths, candidate = verify_candidate(args.candidate)
    oracle = FinalTransferPredictor(paths, "cuda")
    fast = CachedAdaptPredictor(paths, "cuda")
    captured: dict[str, np.ndarray] = {}

    def capture(_model: Any, inputs: Any) -> None:
        captured["features"] = inputs[0].detach().cpu().numpy().copy()
        captured["rgb_width_logits"] = inputs[1].detach().cpu().numpy().copy()

    hooks = [obj.models[HEAD].register_forward_pre_hook(capture) for obj in (oracle, fast)]
    results = []
    try:
        for index in indices:
            row = {k: v for k, v in rows[index].items() if k != "ttc_csv_path"}
            captured.clear()
            start = time.perf_counter()
            original = oracle.predict(row)
            original_seconds = time.perf_counter() - start
            expected = {k: v.copy() for k, v in captured.items()}
            captured.clear()
            start = time.perf_counter()
            actual = fast.predict(row)
            seconds = time.perf_counter() - start
            if set(expected) != set(captured):
                raise ValueError("Fusion availability differs")
            errors = {k: float(np.max(np.abs(v - captured[k]))) for k, v in expected.items()}
            for key, value in expected.items():
                np.testing.assert_array_equal(captured[key], value)
            if actual["predictions"] != original["predictions"]:
                raise ValueError("Predictions differ")
            result = dict(
                index=index,
                sequence=row["sequence_id"],
                original_seconds=original_seconds,
                cached_seconds=seconds,
                errors=errors,
                exact=True,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        atomic_receipt(
            args.output,
            {
                "candidate": candidate,
                "population_sha256": file_sha(args.population),
                "runtime_sha256": file_sha(Path("operational/phase_adapt_runtime.py")),
                "results": results,
                "exact": True,
                "producers": fast.producers.diagnostics(),
                "sensor_cache": dict(fast.inputs.stats),
            },
        )
    finally:
        for hook in hooks:
            hook.remove()
        fast.close()


if __name__ == "__main__":
    main()
