"""Cohort-complete RGB evaluation after both registered adaptation endpoints freeze."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, gpu_charge, owner, require_resources
from .evaluate import development_decision
from .rgb_data import RGBPairDataset
from .rgb_model import load_parent, pair_from_parent
from .rgb_train import REGISTRATION, registered
from .train import setup

BASELINE = "R_A5-P2-FROZEN"


def freeze_endpoints() -> dict:
    """Require both fixed endpoints and matching durable journals before evaluation."""
    record = registered()
    endpoints = {}
    for fit_id in record["fits"]:
        directory = RUN / "rgb/fits" / fit_id
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        journal = read(directory / "UPDATE_JOURNAL.json")
        if (
            receipt["status"] != "COMPLETE"
            or not receipt["scientific_endpoint"]
            or receipt["completed_updates"] != record["updates_per_fit"]
            or journal["completed_updates"] != record["updates_per_fit"]
            or journal["durable_updates"] != record["updates_per_fit"]
            or journal["pending_update_upper"] != 0
            or sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError(f"RGB endpoint not complete and durable: {fit_id}")
        identity = read(directory / "IDENTITY.json")
        if identity["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("RGB endpoint belongs to another registration")
        endpoints[fit_id] = receipt["checkpoint_sha256"]
    parent = owner() / "fits/R_A5/checkpoint_last.pt"
    if sha(parent) != record["parent"]["parent_sha256"]:
        raise ValueError("Frozen RGB baseline changed")
    endpoints[BASELINE] = record["parent"]["parent_sha256"]
    result = {"endpoints": endpoints, "registration_sha256": sha(REGISTRATION)}
    freeze(RUN / "rgb/ENDPOINTS.json", result)
    return result


def load_model(fit_id: str) -> CausalScaleTTC | PairMeasurement:
    """Load admitted weights into a private RGB model, never an event-channel surrogate."""
    record = registered()
    parent, identity = load_parent()
    if identity != record["parent"]:
        raise ValueError("RGB parent provenance changed")
    if fit_id == BASELINE:
        return parent
    if fit_id not in record["fits"]:
        raise ValueError("Unregistered RGB endpoint")
    model = pair_from_parent(
        parent,
        correlation=record["variant"] == "CORR",
        frozen="FROZEN" in fit_id,
        seed=record["seed"],
    )
    payload = torch.load(
        RUN / "rgb/fits" / fit_id / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def verify_chunk(path: Path, binding: str, start: int, stop: int) -> None:
    """Reject a reused prediction fragment from another source, interval or byte stream."""
    receipt = read(path.with_suffix(".json"))
    if (
        receipt["binding"] != binding
        or receipt["start"] != start
        or receipt["stop"] != stop
        or receipt["sha256"] != sha(path)
    ):
        raise ValueError("RGB prediction fragment identity differs")


def predict(fit_id: str, role: str) -> dict:
    """Infer every query with only pixels and RGB sensor time, retaining resumable chunks."""
    endpoints = freeze_endpoints()["endpoints"]
    if fit_id not in endpoints or role not in {"P", "H", "V"}:
        raise ValueError("Unregistered RGB endpoint or population")
    if role == "V" and not (RUN / "rgb/SELECTION_FINAL.json").exists():
        raise ValueError("Complete and freeze RGB H metrics and paired cost before V")
    require_resources(starting=True)
    dataset = RGBPairDataset(role)
    directory = RUN / "rgb/predictions" / role / fit_id
    try:
        cache = dataset.source.prepared_cache
        assert cache is not None
        binding = {
            "fit_id": fit_id,
            "role": role,
            "endpoint": endpoints[fit_id],
            "registration_sha256": sha(REGISTRATION),
            "role_manifest_sha256": sha(owner() / f"{role}_MANIFEST.json"),
            "rows_sha256": sha(dataset.source.rows_path),
            "rgb_cache_binding": cache.binding_sha256,
            "rgb_cache_manifest": cache.manifest_sha256,
            "source": sha(Path(__file__)),
            "scorer_source": sha(ROOT / "operational/range_transfer/analyze.py"),
            "rgb_model_source": sha(ROOT / "operational/joint_measurement/rgb_model.py"),
            "batch_size": 4,
            "precision": "FP32_no_TF32",
            "context": "two real RGB frames, native frame separation, query anchor",
        }
        freeze(directory / "IDENTITY.json", binding)
        binding_sha = digest(binding)
        if (directory / "COMPLETE.json").exists():
            receipt = read(directory / "COMPLETE.json")
            if (
                receipt["binding"] != binding_sha
                or sha(directory / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("RGB completed predictions changed")
            return receipt
        setup(7)
        model = load_model(fit_id)
        with gpu_charge(f"rgb_inference/{role}/{fit_id}", reserve=900) as deadline:
            model = model.cuda().eval().requires_grad_(False)
            with torch.inference_mode():
                for begin in range(0, dataset.population, 256):
                    stop = min(begin + 256, dataset.population)
                    path = directory / f"chunk_{begin // 256:05d}.parquet"
                    if path.with_suffix(".json").exists():
                        verify_chunk(path, binding_sha, begin, stop)
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": begin}
                    require_resources()
                    predictions, phases, times = [], [], []
                    for start in range(begin, stop, 4):
                        batch = dataset.batch(list(range(start, min(start + 4, stop))))
                        pixels = batch.events.cuda()
                        delta = batch.delta_t_s.cuda().reshape(len(pixels), 1)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        if isinstance(model, CausalScaleTTC):
                            ttc = model(pixels, delta).ttc_mean_seconds.reshape(-1)
                            raw = ttc_to_phase(ttc)
                        else:
                            raw = model(pixels, delta)["raw_phase"].reshape(-1)
                            ttc = phase_to_ttc(raw)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - tick)
                        predictions.extend(ttc.cpu().double().tolist())
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
                        path.with_suffix(".json"),
                        {
                            "binding": binding_sha,
                            "sha256": sha(path),
                            "start": begin,
                            "stop": stop,
                            "forward_batch_seconds": times,
                            "utc": utc(),
                        },
                    )
                    print(f"RGB PREDICT {role} {fit_id} {stop}/{dataset.population}", flush=True)
        paths = [
            directory / f"chunk_{i // 256:05d}.parquet" for i in range(0, dataset.population, 256)
        ]
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if len(frame) != dataset.population or not np.array_equal(
            frame.sample_token.to_numpy(), dataset.rows.sample_token.to_numpy()
        ):
            raise ValueError("RGB evaluation omitted, repeated or reordered queries")
        final = directory / "predictions.parquet"
        frame.to_parquet(final, index=False)
        outcome = metrics(
            frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
        )
        write(directory / "METRICS.json", outcome)
        write(
            directory / "GROUP_METRICS.json",
            {
                str(group): metrics(
                    part.target_ttc.to_numpy(),
                    part.prediction_ttc.to_numpy(),
                    part.group_id.to_numpy(),
                )
                for group, part in frame.groupby("group_id")
            },
        )
        receipt = {
            "status": "COMPLETE",
            "binding": binding_sha,
            "rows": len(frame),
            "sha256": sha(final),
            "official_comparison": False,
        }
        freeze(directory / "COMPLETE.json", receipt)
        return receipt
    finally:
        dataset.close()


def select_h() -> dict:
    """Freeze metric-only development evidence; full R_CTX/F_TRUE promotion remains separate."""
    freeze_endpoints()
    record = registered()
    outcomes, evidence = {}, {}
    for role in ("P", "H"):
        for name in [BASELINE, *record["fits"]]:
            folder = RUN / "rgb/predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            if (
                receipt["status"] != "COMPLETE"
                or sha(folder / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("RGB selection requires all complete P/H predictions")
            evidence[f"{role}/{name}"] = sha(folder / "COMPLETE.json")
            if role == "H":
                outcomes[name] = read(folder / "METRICS.json")
    decisions = {
        name: development_decision(outcomes[BASELINE], outcomes[name]) for name in record["fits"]
    }
    ranked = sorted(
        [name for name, row in decisions.items() if row["eligible"]],
        key=lambda name: (
            outcomes[name]["overall_MiD"],
            outcomes[name]["bins"]["crucial"]["group_macro_mid"],
            name,
        ),
    )
    result = {
        "status": "DEVELOPMENT_METRIC_CANDIDATE" if ranked else "NO_METRIC_CANDIDATE",
        "reference": BASELINE,
        "decisions": decisions,
        "ranked_metric_candidates": ranked,
        "evidence": evidence,
        "cost_route": "pending paired RGB cost evidence",
        "system_promotion": False,
        "official_comparison": False,
    }
    freeze(RUN / "rgb/H_METRIC_SELECTION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "predict", "select"])
    parser.add_argument("--fit-id")
    parser.add_argument("--role", choices=["P", "H", "V"])
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze_endpoints())
    elif args.action == "select":
        print(select_h())
    else:
        print(predict(args.fit_id, args.role))
