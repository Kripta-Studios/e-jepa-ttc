"""Chronological FCWD raw-event streaming evaluation on CPU, without fitting."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.garl_comparison.mid import load_scorer, paired_mid, scores
from operational.sota_eval.fcwd_inputs import FCWDEventReader, _validate_row
from operational.sota_evidence.metrics import observations
from operational.sota_evidence.run import json_safe
from operational.train40_system.durable_io import atomic_json

from .distill import load_student
from .preparation import IncrementalPreparer, Query
from .runtime import StreamRuntime
from .state import Policy


def run(args: argparse.Namespace) -> None:
    """Freeze the complete input population; join targets only after predictions finish."""
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        Path(__file__).parent,
        args.output / "source_archive",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = sorted(document["rows"], key=lambda r: (r["sequence_id"], r["anchor_us"]))
    if len(rows) != 630 or len({r["query_id"] for r in rows}) != 630:
        raise ValueError("complete frozen 630-query FCWD population required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = FrozenModels(args.campaign, "cpu")
    runtimes = {
        "H8_reference": StreamRuntime(frozen, Policy(reuse=False)),
        "H8_warp": StreamRuntime(frozen, Policy()),
        "H8_warp_student": StreamRuntime(frozen, Policy(), student=load_student(args.student)),
    }
    receipt = {
        "status": "RUNNING",
        "scope": "Full previously exposed FCWD population; chronological CPU inference",
        "manifest_sha256": digest(args.manifest),
        "gpu_seconds": 0,
        "optimizer_updates": 0,
        "models": frozen.bindings,
        "student_sha256": digest(args.student),
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "selection_uses_targets": False,
        "timing_is_not_gpu_latency": True,
        "queries": [r["query_id"] for r in rows],
    }
    atomic_json(args.output / "RESULT.json", receipt)
    sequence, reader, preparers = "", None, {}
    results = []
    began = time.monotonic()
    try:
        for index, row in enumerate(rows):
            if time.monotonic() - began > args.max_seconds:
                raise TimeoutError("bounded CPU external evaluation stopped; journal retained")
            path = _validate_row(row)
            if sequence != row["sequence_id"]:
                if reader is not None:
                    reader.close()
                sequence = row["sequence_id"]
                reader = FCWDEventReader(path).open()
                preparers = {
                    name: IncrementalPreparer(
                        reader.read_window,
                        raw_bytes=128 * 1024**2,
                        voxel_bytes=16 * 1024**2,
                        approximate_voxels=name != "H8_reference",
                    )
                    for name in runtimes
                }
            query = Query(
                sequence,
                "published_target",
                int(row["anchor_us"]),
                int(row["anchor_us"]),
                tuple((int(start), int(end)) for start, end in row["windows_us"]),
                tuple(map(float, row["square_xyxy"])),
                offset=0.0,
            )
            for name, runtime in runtimes.items():
                value = runtime.predict(query, preparers[name])
                record = {
                    "query_id": row["query_id"],
                    "sequence_id": sequence,
                    "variant": name,
                    **value,
                }
                results.append(record)
                with (args.output / "ROWS.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            atomic_json(
                args.output / "PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed": index + 1,
                    "total": len(rows),
                    "elapsed_s": time.monotonic() - began,
                    "gpu_seconds": 0,
                },
            )
            if index % 20 == 0:
                print(f"FCWD {index + 1}/{len(rows)}", flush=True)
    except Exception as exc:
        atomic_json(
            args.output / "RESULT.json",
            {
                **receipt,
                "status": "FAILED",
                "error": repr(exc),
                "recorded_predictions": len(results),
            },
        )
        raise
    finally:
        if reader is not None:
            reader.close()
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "PREDICTIONS.csv", index=False)
    parent = pd.read_csv(args.prior)
    if set(parent.query_id) != set(frame.query_id) or parent.query_id.duplicated().any():
        raise ValueError("frozen target/baseline table coverage differs")
    merged = frame.merge(
        parent[["query_id", "truth_ttc_seconds", "H8_median3", "public_Garl_event_lhr"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    baseline = merged.loc[merged.variant == "H8_reference"]
    difference = np.abs(np.asarray(baseline.ttc) - np.asarray(baseline.H8_median3))
    admitted = bool(np.all(difference <= 0.01 + 1e-4 * np.abs(np.asarray(baseline.H8_median3))))
    scorer = load_scorer(args.scorer)
    metric_rows, paired = [], []
    for sequence_group in ["ALL", *sorted(set(frame.sequence_id))]:
        local = (
            merged if sequence_group == "ALL" else merged.loc[merged.sequence_id == sequence_group]
        )
        valid = np.isfinite(local.truth_ttc_seconds) & (local.truth_ttc_seconds != 0)
        local = local.loc[valid]
        predictions = {}
        for name, group in local.groupby("variant"):
            group = group.sort_values("query_id")
            truth, prediction = (
                np.asarray(group.truth_ttc_seconds, float),
                np.asarray(group.ttc, float),
            )
            predictions[name] = observations(truth, prediction)["garl_mid"]
            metric_rows.append(
                {
                    "sequence": sequence_group,
                    "variant": name,
                    "eligible_n": len(group),
                    "MAE_s": float(np.mean(np.abs(prediction - truth))),
                    "signed_bias_s": float(np.mean(prediction - truth)),
                    **scores(scorer, truth, prediction),
                }
            )
            if name == "H8_reference":
                comparator = np.asarray(group.public_Garl_event_lhr, float)
                predictions["Garl_event"] = observations(truth, comparator)["garl_mid"]
                metric_rows.append(
                    {
                        "sequence": sequence_group,
                        "variant": "Garl_event",
                        "eligible_n": len(group),
                        "MAE_s": float(np.mean(np.abs(comparator - truth))),
                        "signed_bias_s": float(np.mean(comparator - truth)),
                        **scores(scorer, truth, comparator),
                    }
                )
        if sequence_group == "ALL":
            sequences = np.asarray(
                local.loc[local.variant == "H8_reference"].sort_values("query_id").sequence_id
            )
            for name in ("H8_warp", "H8_warp_student"):
                paired.append(
                    {
                        "variant": name,
                        "reference": "H8_reference",
                        **paired_mid(predictions[name], predictions["H8_reference"], sequences),
                    }
                )
    pd.DataFrame(metric_rows).to_csv(args.output / "METRICS.csv", index=False)
    atomic_json(
        args.output / "RESULT.json",
        cast(
            dict,
            json_safe(
                {
                    **receipt,
                    "status": "COMPLETE" if admitted else "BASELINE_PARITY_FAILED",
                    "baseline_cpu_parity_admitted": admitted,
                    "baseline_max_difference_s": float(difference.max()),
                    "scorer_sha256": digest(args.scorer),
                    "target_table_sha256": digest(args.prior),
                    "metrics": metric_rows,
                    "paired": paired,
                    "wall_seconds": time.monotonic() - began,
                    "predictions_sha256": digest(args.output / "PREDICTIONS.csv"),
                }
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "campaign", "student", "prior", "scorer", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=3600)
    run(parser.parse_args())
