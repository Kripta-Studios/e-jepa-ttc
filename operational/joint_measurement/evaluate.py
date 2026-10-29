"""Endpoint-frozen, cohort-complete V18 inference and prospective H decisions."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, config, gpu_charge, require_resources
from .data import SensorDataset, cache_root
from .train import initialize, setup


def freeze_primary() -> dict:
    """Admit all four complete endpoints before opening any H outcome."""
    endpoints = {}
    for fit_id in config()["fits"]:
        directory = RUN / "fits" / fit_id
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        if (
            receipt["status"] != "COMPLETE"
            or receipt["completed_updates"] != config()["updates"]
            or not receipt["scientific_endpoint"]
            or sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError(f"Primary endpoint not complete: {fit_id}")
        endpoints[fit_id] = receipt["checkpoint_sha256"]
    endpoints["A5-P2-FROZEN"] = read(RUN / "ADMISSION.json")["parent_sha256"]
    result = {"endpoints": endpoints, "config_hash": digest(config())}
    freeze(RUN / "PRIMARY_ENDPOINTS.json", result)
    return result


def predict(fit_id: str, role: str) -> dict:
    """Predict durably by chunks, passing only event tensors and real deltas."""
    endpoints = freeze_primary()["endpoints"]
    if fit_id not in endpoints or role not in {"P", "H", "V"}:
        raise ValueError("Unregistered frozen endpoint or role")
    directory = RUN / "predictions" / role / fit_id
    binding = {
        "fit_id": fit_id,
        "role": role,
        "endpoint": endpoints[fit_id],
        "sensor_complete": sha(cache_root(role) / "COMPLETE.json"),
        "evaluator": sha(ROOT / "operational/joint_measurement/evaluate.py"),
        "batch_size": 4,
        "precision": "FP32",
    }
    freeze(directory / "IDENTITY.json", binding)
    if (directory / "COMPLETE.json").exists():
        result = read(directory / "COMPLETE.json")
        if sha(directory / "predictions.parquet") != result["sha256"]:
            raise ValueError("Completed predictions changed")
        return result
    require_resources(starting=True)
    setup(config()["seed"])
    dataset = SensorDataset(role)
    model = initialize(fit_id)
    if fit_id != "A5-P2-FROZEN":
        payload = torch.load(
            RUN / "fits" / fit_id / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
    try:
        with gpu_charge("inference/" + role + "/" + fit_id, reserve=900) as deadline:
            model = model.cuda().eval().requires_grad_(False)
            with torch.inference_mode():
                for begin in range(0, dataset.population, 256):
                    path = directory / f"chunk_{begin // 256:05d}.parquet"
                    receipt_path = path.with_suffix(".json")
                    if receipt_path.exists():
                        if sha(path) != read(receipt_path)["sha256"]:
                            raise ValueError("Prediction fragment changed")
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": begin}
                    require_resources()
                    stop = min(begin + 256, dataset.population)
                    predictions, phases, timings = [], [], []
                    for start in range(begin, stop, 4):
                        batch = dataset.batch(list(range(start, min(start + 4, stop))))
                        events = batch.events.cuda()
                        delta = batch.delta_t_s.cuda().reshape(len(events), 1)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        if isinstance(model, CausalScaleTTC):
                            pred = model(events, delta).ttc_mean_seconds.reshape(-1)
                            raw = ttc_to_phase(pred)
                        else:
                            raw = model(events, delta)["raw_phase"].reshape(-1)
                            pred = phase_to_ttc(raw)
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter() - tick)
                        predictions.extend(pred.cpu().double().tolist())
                        phases.extend(raw.cpu().double().tolist())
                    frame = dataset.rows.iloc[begin:stop][
                        ["sample_token", "group_id", "sequence_id", "target_ttc"]
                    ].copy()
                    frame["prediction_ttc"] = predictions
                    frame["raw_phase"] = phases
                    frame["role"] = role
                    pending = path.with_suffix(".pending")
                    frame.to_parquet(pending, index=False)
                    os.replace(pending, path)
                    write(
                        receipt_path,
                        {
                            "sha256": sha(path),
                            "binding": digest(binding),
                            "start": begin,
                            "stop": stop,
                            "forward_batch_seconds": timings,
                            "utc": utc(),
                        },
                    )
                    print(f"PREDICT {role} {fit_id} {stop}/{dataset.population}", flush=True)
    finally:
        dataset.close()
    frames = [pd.read_parquet(p) for p in sorted(directory.glob("chunk_*.parquet"))]
    result_frame = pd.concat(frames, ignore_index=True)
    if len(result_frame) != dataset.population or not np.array_equal(
        result_frame.sample_token, dataset.rows.sample_token
    ):
        raise ValueError("Complete population identity/order mismatch")
    path = directory / "predictions.parquet"
    result_frame.to_parquet(path, index=False)
    result = {
        "status": "COMPLETE",
        "rows": len(result_frame),
        "sha256": sha(path),
        "binding": digest(binding),
        "official_comparison": False,
    }
    write(
        directory / "METRICS.json",
        metrics(
            result_frame.target_ttc.to_numpy(),
            result_frame.prediction_ttc.to_numpy(),
            result_frame.group_id.to_numpy(),
        ),
    )
    groups = {
        str(group): metrics(
            frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
        )
        for group, frame in result_frame.groupby("group_id")
    }
    write(directory / "GROUP_METRICS.json", groups)
    freeze(directory / "COMPLETE.json", result)
    return result


def development_decision(reference: dict, candidate: dict) -> dict:
    """Apply the prospective H mechanism/development gates, never claim promotion."""
    if reference["overall_MiD"] is None or candidate["overall_MiD"] is None:
        return {"eligible": False, "reason": "INCOMPLETE_METRIC_COVERAGE"}
    selection = config()["selection"]
    overall_ratio = candidate["overall_MiD"] / reference["overall_MiD"]
    ratios = {
        band: candidate["bins"][band]["group_macro_mid"]
        / reference["bins"][band]["group_macro_mid"]
        for band in reference["bins"]
    }
    checks = {
        "coverage": candidate["complete_prediction_and_mid_coverage"],
        "sign": candidate["diagnostics"]["sign_error_rate"]
        <= reference["diagnostics"]["sign_error_rate"] + selection["all_routes_sign_increase_max"],
        "bands": all(v <= selection["all_routes_band_ratio_max"] for v in ratios.values()),
    }
    routes = selection["routes"]
    signals = {
        "aggregate": overall_ratio <= routes["overall_ratio_max"],
        "crucial": ratios["crucial"] <= routes["crucial_ratio_max"]
        and overall_ratio <= routes["crucial_route_overall_ratio_max"],
    }
    return {
        "eligible": all(checks.values()) and any(signals.values()),
        "checks": checks,
        "signals": signals,
        "ratios": ratios,
        "overall_ratio": overall_ratio,
        "cost_route": "requires separate paired latency evidence",
        "status_scope": "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION",
    }


def select_h() -> dict:
    """Freeze the H-based order after all required predictions are complete."""
    freeze_primary()
    results = {}
    for fit_id in ["A5-P2-FROZEN", *config()["fits"]]:
        folder = RUN / "predictions/H" / fit_id
        if read(folder / "COMPLETE.json")["status"] != "COMPLETE":
            raise ValueError("H selection requires the complete comparison")
        results[fit_id] = read(folder / "METRICS.json")
    decisions = {
        fit_id: development_decision(results["A5-P2-FROZEN"], results[fit_id])
        for fit_id in config()["fits"]
    }
    eligible = [fit_id for fit_id, value in decisions.items() if value["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda f: (
            results[f]["overall_MiD"],
            results[f]["bins"]["crucial"]["group_macro_mid"],
            f,
        ),
    )[: config()["selection"]["max_integrations"]]
    result = {
        "decisions": decisions,
        "ranked_metric_candidates": ranked,
        "state": "DEVELOPMENT_CANDIDATE" if ranked else "NO_METRIC_CANDIDATE",
        "next": "paired cost audit and observation-driven registered continuation",
    }
    freeze(RUN / "H_METRIC_SELECTION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "predict", "select"])
    parser.add_argument("--fit-id")
    parser.add_argument("--role", choices=["P", "H", "V"])
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze_primary())
    elif args.action == "select":
        print(select_h())
    else:
        print(predict(args.fit_id, args.role))
