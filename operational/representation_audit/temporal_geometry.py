"""Registered temporally supervised geometry calibration with paired controls."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import wrap
from operational.range_transfer.admission import revalidate
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, resources, sha, write
from operational.range_transfer.qa import same
from operational.rgb_port.train_producers import ProducerCheckpoint

from .geometry import binding as geometry_binding
from .geometry_model import CALIBRATION as CONTROL_CALIBRATION
from .geometry_model import GeometryReadout
from .precision import Vectors, learning_rate, objective
from .precision import initialize as native_initialize
from .run import OLD, OWNER, ROOT, RUN, accounting, gpu_charge
from .temporal_calibration import TEMPORAL_CALIBRATION as CALIBRATION

F_QUEUE = tuple((arm, seed) for seed in (7, 13, 23) for arm in ("F0", "F1"))


def binding() -> dict:
    result = geometry_binding()
    paths = [
        Path(__file__),
        Path(__file__).with_name("temporal_calibration.py"),
        ROOT / "docs/representation_audit_20261012/PROTOCOL_F_TEMPORAL_GEOMETRY.md",
    ]
    result.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    return result


def initialize(seed: int, arm: str, *, device: str = "cuda") -> tuple:
    if arm not in ("F0", "F1"):
        raise ValueError("Unregistered temporal calibration arm")
    native, config, optimizer, scheduler, generator = native_initialize(seed, device=device)
    # wrap reuses the same parameter objects, preserving optimizer references.
    model = GeometryReadout(
        wrap(native.model, ResidualSpec(0.05), reset=True),
        read(CONTROL_CALIBRATION if arm == "F0" else CALIBRATION),
        "Q1",
    ).to(device)
    model.model.residual.scorer.train()
    if {id(p) for p in model.parameters() if p.requires_grad} != {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }:
        raise AssertionError("Optimizer/scorer parameter identity mismatch")
    return model, config, optimizer, scheduler, generator


def fit(
    data: Vectors, arm: str, seed: int, *, technical: str | None = None, stop: int | None = None
) -> dict:
    if arm == "F0" and seed == 7 and not technical:
        receipt = read(RUN / "fits/Q1-S7/CHECKPOINT_RECEIPT.json")
        if (
            receipt["status"] != "COMPLETE"
            or sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]
        ):
            raise ValueError("E Q1 seed7 control must be complete and unchanged")
        return receipt
    if (arm, seed) not in F_QUEUE:
        raise ValueError("Unregistered arm or seed")
    if technical:
        if data.identity.get("status") != "SYNTHETIC_SHAPES_ONLY":
            raise ValueError("Technical updates require synthetic inputs")
    elif data.identity.get("role") != "P" or data.n != 53890:
        raise ValueError("Readout scientific training requires the admitted P cohort")
    if not technical and read(RUN / "qa/TEMP_GEOMETRY_PASS.json")["sources"] != binding():
        raise ValueError("Geometry QA required")
    if not technical and seed != 7:
        if read(RUN / "TEMP_GEOMETRY_SCREEN_DECISION.json")["status"] != "REPLICATE":
            raise ValueError("Geometry replicas not admitted")
    name = technical or f"{arm}-S{seed}"
    endpoint = 10 if technical else 6000
    identity = {
        "fit_id": name,
        "arm": arm,
        "seed": seed,
        "endpoint": endpoint,
        "sources": binding(),
        "calibration": sha(CONTROL_CALIBRATION if arm == "F0" else CALIBRATION),
        "operator": asdict(ResidualSpec(0.05)),
        "cache": digest(data.identity),
        "parent": read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"],
        "device": "cuda",
        "dtype": "float32_TF32off",
        "torch": str(torch.__version__),
        "batch": 128,
        "technical": bool(technical),
    }
    identity["identity_sha256"] = digest(identity)
    directory = RUN / ("qa" if technical else "fits") / name
    state = ProducerCheckpoint(directory, identity, endpoint)
    if state.receipt.exists() and read(state.receipt)["status"] == "COMPLETE":
        receipt = read(state.receipt)
        if (
            receipt["identity_sha256"] != identity["identity_sha256"]
            or sha(state.path) != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Existing geometry fit changed")
        return receipt
    revalidate(read(OLD / "ADMISSION.json"))
    with gpu_charge(name) as deadline:
        model, config, optimizer, scheduler, generator = initialize(seed, arm)
        frozen = {
            k: v.detach().cpu().clone() for k, v in model.named_parameters() if not v.requires_grad
        }
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        arrays = {
            k: torch.from_numpy(v.copy()).to("cuda")
            for k, v in data.values.items()
            if k != "sample_token"
        }
        cursor = {"curve": [], "draws": [], "elapsed_s": 0.0}
        if state.pointer.exists():
            cursor = state.restore(model, optimizer, scheduler, generator)
        else:
            state.save(model, optimizer, scheduler, generator, cursor, status="READY")
        count = accounting()["physical_upper"]
        started = time.perf_counter()
        executed = 0
        while state.completed < endpoint:
            if count >= 1_000_000 or time.perf_counter() >= deadline or (RUN / "PAUSE").exists():
                break
            if state.completed % 25 == 0 and not resources(RUN)["allowed"]:
                break
            ids = torch.randint(data.n, (128,), generator=generator)
            idx = ids.to("cuda")
            optimizer.param_groups[0]["lr"] = learning_rate(state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            loss, nll = objective(model, {k: v[idx] for k, v in arrays.items()}, config, "U0")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite geometry loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.model.residual.scorer.parameters(), 1, error_if_nonfinite=True
            )
            state.begin()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            state.commit()
            executed += 1
            count += 1
            cursor["curve"].append(
                {
                    "update": state.completed,
                    "loss": float(loss.detach()),
                    "nll": float(nll.detach()),
                    "gradient_l2": float(norm),
                }
            )
            cursor["draws"].append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
            if (
                state.completed % 100 == 0
                or state.completed == endpoint
                or (stop and executed >= stop)
            ):
                cursor["elapsed_s"] += time.perf_counter() - started
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    cursor,
                    status="COMPLETE" if state.completed == endpoint else "IN_PROGRESS",
                )
                pd.DataFrame(cursor["curve"]).to_csv(directory / "CURVE.csv", index=False)
                accounting()
                print(name, state.completed, round(cursor["elapsed_s"], 2), flush=True)
                started = time.perf_counter()
            if stop and executed >= stop:
                break
        if state.completed != state.durable:
            cursor["elapsed_s"] += time.perf_counter() - started
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED")
        if any(
            not torch.equal(v.detach().cpu(), frozen[k])
            for k, v in model.named_parameters()
            if k in frozen
        ):
            raise AssertionError("Frozen parent parameters changed")
        if any(not torch.equal(v.detach().cpu(), buffers[k]) for k, v in model.named_buffers()):
            raise AssertionError("Frozen calibration buffers changed")
        write(
            directory / "FROZEN_PARENT_AUDIT.json",
            {"status": "PASS", "unchanged_tensors": len(frozen)},
        )
        accounting()
    return read(state.receipt)


def qa() -> None:
    data = Vectors("P", synthetic=True)
    fit(data, "F1", 7, technical="temporal_geometry_continuous10")
    fit(data, "F1", 7, technical="temporal_geometry_split5_restore5", stop=5)
    fit(data, "F1", 7, technical="temporal_geometry_split5_restore5")
    a = torch.load(
        RUN / "qa/temporal_geometry_continuous10/checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    b = torch.load(
        RUN / "qa/temporal_geometry_split5_restore5/checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    )
    checks = {key: same(a[key], b[key]) for key in keys}
    checks["curve"] = same(a["cursor"]["curve"], b["cursor"]["curve"])
    checks["draws"] = same(a["cursor"]["draws"], b["cursor"]["draws"])
    if not all(checks.values()):
        raise AssertionError(checks)
    freeze(
        RUN / "qa/TEMP_GEOMETRY_PASS.json",
        {"status": "PASS", "sources": binding(), "checks": checks, "updates": 20},
    )


def evaluate(*, replicas: bool = False) -> None:
    seeds = (7, 13, 23) if replicas else (7,)
    endpoints = {}
    for seed in seeds:
        for arm in ("F0", "F1"):
            name = f"{arm}-S{seed}"
            source_name = "Q1-S7" if name == "F0-S7" else name
            receipt = read(RUN / f"fits/{source_name}/CHECKPOINT_RECEIPT.json")
            if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 6000:
                raise ValueError("Freeze the complete registered queue before evaluation")
            if sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]:
                raise ValueError("Endpoint changed")
            endpoints[name] = receipt
    suffix = "REPLICAS" if replicas else "SCREEN"
    freeze(RUN / f"TEMP_GEOMETRY_{suffix}_ENDPOINTS.json", endpoints)
    scores = {}
    for role in ("P", "H"):
        data = Vectors(role)
        manifest = read(OWNER / f"{role}_MANIFEST.json")
        rows = pd.read_parquet(manifest["rows_path"]).set_index("sample_token")
        ordered = rows.loc[data.values["sample_token"].astype(str)]
        target = ordered.target_ttc.to_numpy(np.float64)
        groups = ordered.group_id.to_numpy(str)
        for seed in seeds:
            for arm in ("F0", "F1"):
                name = f"{arm}-S{seed}"
                folder = RUN / f"temporal_geometry_evaluation/{role}/{name}"
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / "RECEIPT.json").exists():
                    receipt = read(folder / "RECEIPT.json")
                    if (
                        receipt["endpoint"] != endpoints[name]["checkpoint_sha256"]
                        or receipt["predictions_sha256"] != sha(folder / "PREDICTIONS.csv")
                        or receipt["metrics_sha256"] != sha(folder / "METRICS.json")
                    ):
                        raise ValueError("Saved evaluation changed")
                    scores[f"{role}/{name}"] = read(folder / "METRICS.json")
                    continue
                with gpu_charge("temporal_geometry_evaluation_" + role + name):
                    model, *_ = initialize(seed, arm)
                    payload = torch.load(
                        endpoints[name]["checkpoint_path"], map_location="cpu", weights_only=False
                    )
                    model.load_state_dict(payload["model_state_dict"], strict=True)
                    model.eval().requires_grad_(False)
                    predictions = []
                    with torch.inference_mode():
                        for begin in range(0, data.n, 128):
                            inputs = {
                                k: torch.from_numpy(v[begin : begin + 128].copy()).to("cuda")
                                for k, v in data.values.items()
                                if k != "sample_token" and not k.startswith(("target_", "dino_"))
                            }
                            predictions.append(model(inputs).ttc_mean_seconds.cpu().numpy())
                    pred = np.concatenate(predictions)
                pd.DataFrame(
                    {
                        "sample_token": ordered.index,
                        "group_id": groups,
                        "target_ttc": target,
                        "prediction_ttc": pred,
                    }
                ).to_csv(folder / "PREDICTIONS.csv", index=False)
                score = metrics(target, pred, groups)
                write(folder / "METRICS.json", score)
                write(
                    folder / "RECEIPT.json",
                    {
                        "endpoint": endpoints[name]["checkpoint_sha256"],
                        "predictions_sha256": sha(folder / "PREDICTIONS.csv"),
                        "metrics_sha256": sha(folder / "METRICS.json"),
                    },
                )
                scores[f"{role}/{name}"] = score
    original = read(OLD / "PRODUCER_METRICS_P_H.json")["scores"]["H/historical_A5"]
    gates = {}
    for seed in seeds:
        gates[f"paired_S{seed}"] = gate(scores[f"H/Q0-S{seed}"], scores[f"H/Q1-S{seed}"])
        gates[f"historical_S{seed}"] = gate(original, scores[f"H/Q1-S{seed}"])
    passed = all(row["eligible"] for row in gates.values())
    status = (
        ("HEAD_INTEGRATION_ADMITTED" if replicas else "REPLICATE") if passed else "NO_CANDIDATE"
    )
    write(RUN / f"TEMP_GEOMETRY_{suffix}_RESULTS.json", scores)
    write(
        RUN / f"TEMP_GEOMETRY_{suffix}_DECISION.json",
        {"status": status, "gates": gates, "selection_role": "H", "V_opened": False},
    )
    print(read(RUN / f"TEMP_GEOMETRY_{suffix}_DECISION.json"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("qa", "train", "evaluate", "replicate", "evaluate-replicas")
    )
    args = parser.parse_args()
    lock = RUN / "PROCESS.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage == "qa":
            qa()
        elif args.stage in ("train", "replicate"):
            freeze(RUN / "TEMP_GEOMETRY_SOURCES_FREEZE.json", binding())
            data = Vectors("P")
            seeds = (7,) if args.stage == "train" else (13, 23)
            for seed in seeds:
                for arm in ("F0", "F1"):
                    if fit(data, arm, seed)["status"] != "COMPLETE":
                        return
        else:
            evaluate(replicas=args.stage == "evaluate-replicas")
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
