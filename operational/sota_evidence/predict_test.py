"""Frozen native H8 inference for a complete official test12 submission."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.simplex_t_shared_route.adapter import extract, inputs
from operational.sota_evidence.codabench import package_predictions
from operational.sota_evidence.test_inputs import LAGS_US
from operational.train40_system.contracts import environment
from operational.train40_system.durable_io import atomic_json


@torch.inference_mode()
def predict(
    model: FrozenModels,
    events: np.ndarray,
    valid: np.ndarray,
    job: dict[str, Any],
    *,
    reference_features: np.ndarray | None = None,
) -> dict[str, float]:
    """Use canonical FP32 B16 producers and measured positive exposure delay."""
    checked, mask = model._own_arrays(events, valid)
    padded = np.zeros((16, 3, 12, 128, 128), np.float32)
    padded[-8:] = checked
    tensor = torch.from_numpy(padded).to(model.device)
    delta = torch.full((16, 2), job["delta"], dtype=torch.float32, device=model.device)
    raw = extract("H8_SEED7", model.models, tensor, delta)[-8:]
    raw[~mask] = 0
    if reference_features is not None:
        np.testing.assert_allclose(raw, reference_features, atol=1e-5, rtol=1e-4)
    head_inputs = inputs(
        "H8_SEED7", raw, mask, LAGS_US, job["anchor"], job["available"], model.mean, model.scale
    )
    head_inputs = tuple(value.to(model.device) for value in head_inputs)
    result = {}
    for seed in (7, 13, 23):
        value = phase_to_ttc(model.heads[seed](*head_inputs)["point_phase"])
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise ValueError("Frozen H8 produced a nonfinite or nonscalar TTC")
        result[f"H8_seed{seed}"] = float(value.item())
    result["prediction"] = float(np.median(list(result.values())))
    return result


def run(
    campaign: Path,
    output: Path,
    cache: Path,
    *,
    budget_seconds: float,
    maximum_queries: int | None = None,
) -> None:
    """Resume immutable per-query predictions; package only after full coverage."""
    if budget_seconds <= 0:
        raise ValueError("A positive remaining GPU wall-time budget is required")
    source_manifest = json.loads((output / "CODE_FREEZE.json").read_text(encoding="utf-8"))
    for name, expected in source_manifest["sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Inference source changed after train parity: {name}")
    prepared = json.loads((output / "PREPARATION_RESULT.json").read_text(encoding="utf-8"))
    if prepared["status"] != "COMPLETE" or prepared["completed"] != 6762:
        raise ValueError("All public inputs must be prepared before GPU reservation")
    qa = json.loads((output / "TRAIN_INPUT_QA.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASSED" or qa["metadata_rows_exact"] != 88744:
        raise ValueError("TRAIN40 input parity is required before official inference")
    frozen = json.loads((output / "INPUT_FREEZE.json").read_text(encoding="utf-8"))
    for name, expected in frozen["sources"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen preprocessing source changed")
    if qa["adapter_sha256"] != digest(Path(__file__).with_name("test_inputs.py")):
        raise ValueError("QA adapter source changed")
    gpu_qa = json.loads((output / "GPU_TRAIN_PARITY.json").read_text(encoding="utf-8"))
    if gpu_qa["status"] != "PASSED" or gpu_qa["runtime_sha256"] != digest(Path(__file__)):
        raise ValueError("Native training feature/head parity required for this runtime")
    if "execution_sha256" in prepared:
        execution_path = output / "PARALLEL_PREPARATION.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if digest(execution_path) != prepared["execution_sha256"] or execution[
            "source_sha256"
        ] != digest(Path(__file__).with_name("prepare_parallel.py")):
            raise ValueError("Parallel preparation identity changed")
    if digest(output / "INPUT_FREEZE.json") != prepared["input_freeze_sha256"]:
        raise ValueError("Prepared input freeze changed")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = FrozenModels(campaign, "cuda")
    if model.bindings != gpu_qa["model_bindings"]:
        raise ValueError("Model weights or normalization changed after GPU parity")
    binding = {
        "model": model.bindings,
        "input_freeze_sha256": digest(output / "INPUT_FREEZE.json"),
        "qa_sha256": digest(output / "TRAIN_INPUT_QA.json"),
        "gpu_qa_sha256": digest(output / "GPU_TRAIN_PARITY.json"),
        "code_freeze_sha256": digest(output / "CODE_FREEZE.json"),
        "runtime_sha256": digest(Path(__file__)),
        "extractor_sha256": digest(Path("operational/simplex_t_shared_route/adapter.py")),
        "candidate": "H8_median_of_seeds_7_13_23",
        "producer_batch": 16,
        "precision": "float32_no_tf32",
        "optimizer_updates": 0,
        "test_labels_read": False,
    }
    binding_path = output / "MODEL_FREEZE.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Frozen inference identity changed")
    else:
        atomic_json(binding_path, binding)
    identity = digest(binding_path)
    atomic_json(output / "INFERENCE_ENVIRONMENT.json", environment())
    directory = output / "predictions"
    directory.mkdir(exist_ok=True)
    results = []
    new_queries = 0
    try:
        for number, job in enumerate(frozen["rows"]):
            path = directory / f"query_{number:05d}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    record["model_freeze_sha256"] != identity
                    or record["sample_token"] != job["sample_token"]
                ):
                    raise ValueError("Existing prediction belongs to another model or input")
            else:
                if time.monotonic() - started > budget_seconds - 30 or (
                    maximum_queries is not None and new_queries >= maximum_queries
                ):
                    break
                data = cache / f"query_{number:05d}.npz"
                receipt = json.loads(data.with_suffix(".json").read_text(encoding="utf-8"))
                if (
                    digest(data) != receipt["sha256"]
                    or receipt["input_freeze_sha256"] != binding["input_freeze_sha256"]
                    or receipt["sample_token"] != job["sample_token"]
                ):
                    raise ValueError("Prepared sensor input changed")
                begin = time.monotonic()
                with np.load(data, allow_pickle=False) as stored:
                    predictions = predict(model, stored["events"], stored["valid"], job)
                record = {
                    "sample_token": job["sample_token"],
                    "sequence_id": job["sequence"],
                    **predictions,
                    "model_freeze_sha256": identity,
                    "prepared_input_sha256": receipt["sha256"],
                    "seconds": time.monotonic() - begin,
                }
                atomic_json(path, record)
                new_queries += 1
            if not np.isfinite([record[f"H8_seed{s}"] for s in (7, 13, 23)]).all():
                raise ValueError("Invalid resumed head predictions")
            if record["prediction"] != float(
                np.median([record[f"H8_seed{s}"] for s in (7, 13, 23)])
            ):
                raise ValueError("Prediction aggregation changed")
            results.append(record)
            atomic_json(
                output / "INFERENCE_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed": len(results),
                    "total": len(frozen["rows"]),
                    "gpu_reservation_s": time.monotonic() - started,
                    "utc": datetime.now(UTC).isoformat(),
                },
            )
    finally:
        atomic_json(
            output / f"GPU_RESERVATION_{time.time_ns()}.json",
            {
                "seconds": time.monotonic() - started,
                "budget_seconds": budget_seconds,
                "new_queries": new_queries,
                "optimizer_updates": 0,
            },
        )
    if len(results) != 6762:
        atomic_json(
            output / "INFERENCE_RESULT.json",
            {
                "status": "PAUSED_BUDGET_OR_QUERY_LIMIT",
                "completed": len(results),
                "total": 6762,
            },
        )
        return
    frame = pd.DataFrame(results)
    temporary = output / "TEST_PREDICTIONS.pending.csv"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output / "TEST_PREDICTIONS.csv")
    submission = output / "H8_GarlTTC_submission.zip"
    if submission.exists():
        raise ValueError(
            "Submission already exists; verify existing artifact instead of overwriting"
        )
    receipt = package_predictions(
        [r["sample_token"] for r in frozen["rows"]], frame, binding_path, submission
    )
    receipt.update(
        predictions_sha256=digest(output / "TEST_PREDICTIONS.csv"),
        checkpoint_identity="MODEL_FREEZE.json binds all six trained endpoints and normalizer",
        official_test_score_available=False,
        test_labels_read=False,
    )
    atomic_json(output / "INFERENCE_RESULT.json", receipt)


def main() -> None:
    """Run inference only, under a caller-provided GPU reservation budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "output", "cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--maximum-queries", type=int)
    args = parser.parse_args()
    run(
        args.campaign,
        args.output,
        args.cache,
        budget_seconds=args.budget_seconds,
        maximum_queries=args.maximum_queries,
    )


if __name__ == "__main__":
    main()
