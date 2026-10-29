"""Complete, resumable P/H evaluation after the registered geometry endpoint freezes."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, config, gpu_charge, owner, require_resources
from .data import SensorDataset, cache_root
from .evaluate import development_decision, freeze_primary
from .geometry_model import GeometricPairMeasurement
from .geometry_train import CONTROL, DIRECTORY, FIT, REGISTRATION, registered
from .rgb_evaluate import verify_chunk
from .train import initialize, setup
from .verify_rgb_endpoint import inspect_payload

BASELINE = "A5-P2-FROZEN"


def freeze_endpoint() -> dict:
    """Require complete checkpoint, durable journal and exact preregistered control."""
    record = registered()
    directory = DIRECTORY / "fits" / FIT
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    pointer = read(directory / "CHECKPOINT_POINTER.json")
    identity = read(directory / "IDENTITY.json")
    if (
        receipt["status"] != "COMPLETE"
        or not receipt["scientific_endpoint"]
        or receipt["completed_updates"] != 12000
        or journal["completed_updates"] != 12000
        or journal["durable_updates"] != 12000
        or journal["pending_update_upper"] != 0
        or pointer["completed_updates"] != 12000
        or identity["registration_sha256"] != sha(REGISTRATION)
        or journal["identity_sha256"] != receipt["identity_sha256"]
        or pointer["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
        or sha(directory / "checkpoint_versions" / pointer["version"])
        != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Geometry endpoint is not complete, durable and hash-bound")
    primary = freeze_primary()["endpoints"]
    control = record["paired_control"]
    if (
        primary[CONTROL] != control["checkpoint_sha256"]
        or sha(RUN / "fits" / CONTROL / "IDENTITY.json") != control["identity_sha256"]
    ):
        raise ValueError("Reused geometry control changed")
    result = {
        "endpoints": {
            FIT: receipt["checkpoint_sha256"],
            CONTROL: primary[CONTROL],
            BASELINE: primary[BASELINE],
        },
        "registration_sha256": sha(REGISTRATION),
    }
    freeze(DIRECTORY / "ENDPOINTS.json", result)
    return result


def load_model() -> GeometricPairMeasurement:
    """Load a private geometry endpoint after its admission, on CPU."""
    freeze_endpoint()
    setup(7)
    model = GeometricPairMeasurement(initialize(BASELINE))
    payload = torch.load(
        DIRECTORY / "fits" / FIT / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def verify_endpoint() -> dict:
    """Regenerate all 384k sampler IDs and inspect actual optimizer/encoder state on CPU."""
    endpoint = freeze_endpoint()
    directory = DIRECTORY / "fits" / FIT
    payload = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    identity = read(directory / "IDENTITY.json")
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    if (
        payload["identity"] != identity
        or payload["identity_sha256"] != receipt["identity_sha256"]
        or payload["recovery_upper"] != journal["recovery_upper"]
    ):
        raise ValueError("Geometry checkpoint identity or recovery accounting changed")
    setup(7)
    model = GeometricPairMeasurement(initialize(BASELINE))
    result = inspect_payload(
        model,
        payload,
        updates=12000,
        population=registered()["population_size"],
        batch_size=32,
        seed=7,
        frozen=False,
    )
    result.update(
        utc=utc(),
        fit_id=FIT,
        checkpoint_sha256=endpoint["endpoints"][FIT],
        registration_sha256=sha(REGISTRATION),
        cuda_initialized=torch.cuda.is_initialized(),
        scope="Final event geometry endpoint; no inference or optimizer updates",
    )
    write(DIRECTORY / "verification/ENDPOINT.json", result)
    return result


def predict(role: str) -> dict:
    """Retain every query and signed output; only P/H are admitted by this evaluator."""
    if role not in {"P", "H"}:
        raise ValueError("Geometry development evaluator admits only P/H")
    endpoints = freeze_endpoint()["endpoints"]
    proof = read(DIRECTORY / "verification/ENDPOINT.json")
    if proof["status"] != "PASS" or proof["checkpoint_sha256"] != endpoints[FIT]:
        raise ValueError("Verify geometry optimizer and trajectory before evaluation")
    require_resources(starting=True)
    manifest_path = owner() / f"{role}_MANIFEST.json"
    manifest = read(manifest_path)
    if (
        sha(manifest_path) != read(cache_root(role) / "BINDING.json")["manifest_sha256"]
        or sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]
    ):
        raise ValueError("Geometry cohort manifest or row metadata changed")
    directory = DIRECTORY / "predictions" / role / FIT
    binding = {
        "fit_id": FIT,
        "endpoint": endpoints[FIT],
        "role": role,
        "registration_sha256": sha(REGISTRATION),
        "sensor_complete": sha(cache_root(role) / "COMPLETE.json"),
        "manifest_sha256": sha(manifest_path),
        "rows_sha256": manifest["rows_sha256"],
        "sources": {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/geometry_evaluate.py",
                "operational/joint_measurement/geometry_model.py",
                "operational/joint_measurement/rgb_evaluate.py",
                "operational/range_transfer/analyze.py",
            )
        },
        "batch_size": 4,
        "precision": "FP32_no_TF32",
    }
    freeze(directory / "IDENTITY.json", binding)
    key = digest(binding)
    if (directory / "COMPLETE.json").exists():
        result = read(directory / "COMPLETE.json")
        if result["binding"] != key or sha(directory / "predictions.parquet") != result["sha256"]:
            raise ValueError("Geometry completed prediction table changed")
        return result
    dataset = SensorDataset(role)
    try:
        model = load_model()
        with gpu_charge(f"geometry_inference/{role}", reserve=900) as deadline:
            model = model.cuda().eval().requires_grad_(False)
            with torch.inference_mode():
                for begin in range(0, dataset.population, 256):
                    stop = min(begin + 256, dataset.population)
                    path = directory / f"chunk_{begin // 256:05d}.parquet"
                    if path.with_suffix(".json").exists():
                        verify_chunk(path, key, begin, stop)
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": begin}
                    require_resources()
                    predictions, phases = [], []
                    for start in range(begin, stop, 4):
                        batch = dataset.batch(list(range(start, min(start + 4, stop))))
                        raw = model(batch.events.cuda(), batch.delta_t_s.cuda().reshape(-1))[
                            "raw_phase"
                        ]
                        predictions.extend(phase_to_ttc(raw).cpu().double().tolist())
                        phases.extend(raw.cpu().double().tolist())
                    frame = dataset.rows.iloc[begin:stop][
                        ["sample_token", "group_id", "sequence_id", "target_ttc"]
                    ].copy()
                    frame["prediction_ttc"], frame["raw_phase"], frame["role"] = (
                        predictions,
                        phases,
                        role,
                    )
                    pending = path.with_suffix(".pending")
                    frame.to_parquet(pending, index=False)
                    os.replace(pending, path)
                    write(
                        path.with_suffix(".json"),
                        {
                            "binding": key,
                            "sha256": sha(path),
                            "start": begin,
                            "stop": stop,
                            "utc": utc(),
                        },
                    )
                    print(f"GEOMETRY PREDICT {role} {stop}/{dataset.population}", flush=True)
        paths = []
        for begin in range(0, dataset.population, 256):
            path = directory / f"chunk_{begin // 256:05d}.parquet"
            verify_chunk(path, key, begin, min(begin + 256, dataset.population))
            paths.append(path)
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if len(frame) != dataset.population or not np.array_equal(
            frame.sample_token.to_numpy(), dataset.rows.sample_token.to_numpy()
        ):
            raise ValueError("Geometry predictions omit, repeat or reorder queries")
        final = directory / "predictions.parquet"
        frame.to_parquet(final, index=False)
        write(
            directory / "METRICS.json",
            metrics(
                frame.target_ttc.to_numpy(),
                frame.prediction_ttc.to_numpy(),
                frame.group_id.to_numpy(),
            ),
        )
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
        result = {
            "status": "COMPLETE",
            "binding": key,
            "rows": len(frame),
            "sha256": sha(final),
            "official_comparison": False,
        }
        freeze(directory / "COMPLETE.json", result)
        return result
    finally:
        dataset.close()


def prediction_folder(role: str, name: str) -> Path:
    """Resolve new geometry and reused controls without copying their journals or counts."""
    if role not in {"P", "H"} or name not in {FIT, CONTROL, BASELINE}:
        raise ValueError("Unknown geometry comparison route")
    return (DIRECTORY if name == FIT else RUN) / "predictions" / role / name


def select_metrics() -> dict:
    """Metric-only decision; paired cost must complete before a final development choice."""
    endpoints = freeze_endpoint()["endpoints"]
    record = registered()
    if record["development_gates"] != config()["selection"]:
        raise ValueError("Registered geometry exploration gates changed")
    evidence, outcomes = {}, {}
    for role in ("P", "H"):
        for name in (BASELINE, CONTROL, FIT):
            directory = prediction_folder(role, name)
            receipt = read(directory / "COMPLETE.json")
            identity = read(directory / "IDENTITY.json")
            if (
                receipt["status"] != "COMPLETE"
                or identity["endpoint"] != endpoints[name]
                or sha(directory / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("Geometry selection requires intact complete P/H comparisons")
            evidence[f"{role}/{name}"] = {
                "predictions": receipt["sha256"],
                "metrics": sha(directory / "METRICS.json"),
            }
            if role == "H":
                outcomes[name] = read(directory / "METRICS.json")
    decision = development_decision(outcomes[BASELINE], outcomes[FIT])
    result = {
        "status": "DEVELOPMENT_METRIC_CANDIDATE" if decision["eligible"] else "NO_METRIC_CANDIDATE",
        "decision": decision,
        "reference": BASELINE,
        "reused_control": CONTROL,
        "free_control_contrast": development_decision(outcomes[CONTROL], outcomes[FIT]),
        "evidence": evidence,
        "cost_route": "pending paired cost",
        "system_promotion": False,
        "official_comparison": False,
    }
    freeze(DIRECTORY / "H_METRIC_SELECTION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "predict", "select"])
    parser.add_argument("--role", choices=["P", "H"])
    args = parser.parse_args()
    print(
        verify_endpoint()
        if args.action == "verify"
        else predict(args.role)
        if args.action == "predict"
        else select_metrics()
    )
