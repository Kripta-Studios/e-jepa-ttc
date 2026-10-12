"""Conditional nonlinear height readout with separately frozen geometry/scorer stages."""

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
from operational.range_transfer.admission import revalidate
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, resources, sha, write
from operational.range_transfer.qa import same
from operational.rgb_port.train_producers import ProducerCheckpoint

from .nonlinear_model import NonlinearSpatialReadout
from .precision import binding as precision_binding
from .precision import learning_rate, objective
from .provenance import record_start
from .run import OLD, OWNER, ROOT, RUN, accounting, gpu_charge
from .selection import claim_family
from .spatial_geometry import initialize as init_spatial
from .spatial_model import SpatialVectors, calibration_path

NL_QUEUE = tuple((arm, seed) for seed in (7, 13, 23) for arm in ("NLH", "NL0", "NL1"))


def binding() -> dict:
    result = precision_binding()
    paths = [
        Path(__file__),
        Path(__file__).with_name("geometry_model.py"),
        Path(__file__).with_name("spatial_model.py"),
        Path(__file__).with_name("spatial_geometry.py"),
        Path(__file__).with_name("nonlinear_model.py"),
        Path(__file__).with_name("spatial_cache.py"),
        Path(__file__).with_name("selection.py"),
        Path(__file__).with_name("provenance.py"),
        ROOT / "docs/representation_audit_20261012/PROTOCOL_I_NONLINEAR_HEIGHT.md",
    ]
    result.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    return result


def geometry_parent(seed: int) -> dict:
    path = RUN / f"fits/NLH-S{seed}/CHECKPOINT_RECEIPT.json"
    receipt = read(path)
    if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 6000:
        raise ValueError("Freeze the complete nonlinear height fit before training its scorer")
    if sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]:
        raise ValueError("Nonlinear height parent changed")
    return receipt


def prerequisites() -> None:
    for filename in ("SPATIAL_SCREEN_DECISION.json", "NOCLIP_SCREEN_DECISION.json"):
        if read(RUN / filename)["status"] != "NO_CANDIDATE":
            raise ValueError("Queue I requires G and H closed as NO_CANDIDATE")
    selected = RUN / "SELECTED_REPLICATION_FAMILY.json"
    if selected.exists() and read(selected)["family"] != "NONLINEAR":
        raise ValueError("Another family already claimed the global conditional replication slot")


def initialize(seed: int, arm: str, *, device: str = "cuda", technical: bool = False) -> tuple:
    if arm not in ("NLH", "NL0", "NL1"):
        raise ValueError("Unregistered nonlinear arm")
    base, config, optimizer, scheduler, generator = init_spatial(seed, "SP1", device=device)
    if arm == "NL0":
        return base, config, optimizer, scheduler, generator
    model = NonlinearSpatialReadout(base.model, read(calibration_path("SP1"))).to(device)
    if arm == "NLH":
        model.eval().requires_grad_(False)
        model.height_mlp.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.height_mlp.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        if not technical:
            receipt = geometry_parent(seed)
            state = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=False)
            model.height_mlp.load_state_dict(
                {
                    k.removeprefix("height_mlp."): value
                    for k, value in state["model_state_dict"].items()
                    if k.startswith("height_mlp.")
                },
                strict=True,
            )
        model.height_mlp.eval().requires_grad_(False)
        model.model.residual.scorer.train().requires_grad_(True)
    if {id(p) for p in model.parameters() if p.requires_grad} != {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }:
        raise AssertionError("Optimizer stage parameters differ")
    return model, config, optimizer, scheduler, generator


def fit(
    data: SpatialVectors,
    arm: str,
    seed: int,
    *,
    technical: str | None = None,
    stop: int | None = None,
) -> dict:
    if not technical:
        prerequisites()
    if arm == "NL0" and seed == 7 and not technical:
        receipt = read(RUN / "fits/SP1-S7/CHECKPOINT_RECEIPT.json")
        if (
            receipt["status"] != "COMPLETE"
            or sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Frozen SP1 control changed")
        return receipt
    if (arm, seed) not in NL_QUEUE:
        raise ValueError("Unregistered arm or seed")
    if technical:
        if data.identity.get("status") != "SYNTHETIC_SHAPES_ONLY":
            raise ValueError("Technical updates require synthetic inputs")
    elif data.identity.get("role") != "P" or data.n != 53890:
        raise ValueError("Readout scientific training requires the admitted P cohort")
    qa_stage = "GEOMETRY" if arm == "NLH" else "SCORER"
    if not technical and read(RUN / f"qa/NONLINEAR_{qa_stage}_PASS.json")["sources"] != binding():
        raise ValueError("Geometry QA required")
    if not technical and seed != 7:
        if read(RUN / "NONLINEAR_SCREEN_DECISION.json")["status"] != "REPLICATE":
            raise ValueError("Geometry replicas not admitted")
        claim_family("NONLINEAR", RUN / "NONLINEAR_SCREEN_DECISION.json")
    name = technical or f"{arm}-S{seed}"
    endpoint = 10 if technical else 6000
    identity = {
        "fit_id": name,
        "arm": arm,
        "seed": seed,
        "endpoint": endpoint,
        "sources": binding(),
        "calibration": sha(calibration_path("SP1")),
        "geometry_parent": geometry_parent(seed)["checkpoint_sha256"]
        if arm == "NL1" and not technical
        else None,
        "stage": "geometry" if arm == "NLH" else "scorer",
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
        model, config, optimizer, scheduler, generator = initialize(
            seed, arm, technical=bool(technical)
        )
        frozen = {
            k: v.detach().cpu().clone() for k, v in model.named_parameters() if not v.requires_grad
        }
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        arrays = {
            k: torch.from_numpy(v.copy()).to("cuda")
            for k, v in data.values.items()
            if k != "sample_token"
            and (
                arm != "NLH"
                or k
                in (
                    "height_features",
                    "target_valid",
                    "target_height_normalized",
                    "visible_height",
                    "analytic",
                )
            )
        }
        cursor = {"curve": [], "draws": [], "elapsed_s": 0.0}
        if state.pointer.exists():
            cursor = state.restore(model, optimizer, scheduler, generator)
        else:
            record_start(directory, identity, "SYNTHETIC" if technical else "P")
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
            optimizer.param_groups[0]["lr"] = learning_rate(state.completed + 1) / (
                3 if arm == "NLH" else 1
            )
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v[idx] for k, v in arrays.items()}
            if arm == "NLH":
                if not isinstance(model, NonlinearSpatialReadout):
                    raise TypeError("Wrong geometry model")
                loss = model.geometry_loss(batch)
                nll = loss.detach()
            else:
                loss, nll = objective(model, batch, config, "U0")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite geometry loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True
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
                    "nll": float(nll.detach()) if arm != "NLH" else None,
                    "objective_kind": "geometry_only" if arm == "NLH" else "native_producer",
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


def qa(stage: str) -> None:
    arm = "NLH" if stage == "geometry" else "NL1"
    data = SpatialVectors("P", synthetic=True)
    calibration = read(calibration_path("SP1"))
    rng = np.random.default_rng(92)
    data.values["height_features"] = np.asarray(calibration["mean"], np.float32) + np.asarray(
        calibration["scale"], np.float32
    ) * rng.normal(0, 0.05, (64, 3, 512)).astype(np.float32)
    data.identity["height_feature_distribution"] = "synthetic seed92 at frozen P normalizer scales"
    first = f"nonlinear_{stage}_continuous10"
    second = f"nonlinear_{stage}_split5_restore5"
    fit(data, arm, 7, technical=first)
    fit(data, arm, 7, technical=second, stop=5)
    fit(data, arm, 7, technical=second)
    payloads = []
    for name in (first, second):
        receipt = read(RUN / f"qa/{name}/CHECKPOINT_RECEIPT.json")
        if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 10:
            raise RuntimeError("Technical fit paused; resume before admitting scientific training")
        payloads.append(
            torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=False)
        )
    a, b = payloads
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
        RUN / f"qa/NONLINEAR_{stage.upper()}_PASS.json",
        {"status": "PASS", "sources": binding(), "checks": checks, "updates": 20},
    )


def evaluate(*, replicas: bool = False) -> None:
    prerequisites()
    seeds = (7, 13, 23) if replicas else (7,)
    endpoints = {}
    for seed in seeds:
        for arm in ("NL0", "NL1"):
            name = f"{arm}-S{seed}"
            receipt = read(
                RUN
                / (
                    "fits/SP1-S7/CHECKPOINT_RECEIPT.json"
                    if name == "NL0-S7"
                    else f"fits/{name}/CHECKPOINT_RECEIPT.json"
                )
            )
            if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 6000:
                raise ValueError("Freeze the complete registered queue before evaluation")
            if sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]:
                raise ValueError("Endpoint changed")
            endpoints[name] = receipt
    suffix = "REPLICAS" if replicas else "SCREEN"
    freeze(
        RUN / f"NONLINEAR_{suffix}_GEOMETRY_PARENTS.json",
        {str(seed): geometry_parent(seed) for seed in seeds},
    )
    freeze(RUN / f"NONLINEAR_{suffix}_ENDPOINTS.json", endpoints)
    scores = {}
    for role in ("P", "H"):
        data = SpatialVectors(role)
        manifest = read(OWNER / f"{role}_MANIFEST.json")
        rows = pd.read_parquet(manifest["rows_path"]).set_index("sample_token")
        ordered = rows.loc[data.values["sample_token"].astype(str)]
        target = ordered.target_ttc.to_numpy(np.float64)
        groups = ordered.group_id.to_numpy(str)
        for seed in seeds:
            for arm in ("NL0", "NL1"):
                name = f"{arm}-S{seed}"
                folder = RUN / f"nonlinear_evaluation/{role}/{name}"
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
                with gpu_charge("nonlinear_evaluation_" + role + name):
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
        gates[f"paired_S{seed}"] = gate(scores[f"H/NL0-S{seed}"], scores[f"H/NL1-S{seed}"])
        gates[f"historical_S{seed}"] = gate(original, scores[f"H/NL1-S{seed}"])
    passed = all(row["eligible"] for row in gates.values())
    status = (
        ("HEAD_INTEGRATION_ADMITTED" if replicas else "REPLICATE") if passed else "NO_CANDIDATE"
    )
    write(RUN / f"NONLINEAR_{suffix}_RESULTS.json", scores)
    write(
        RUN / f"NONLINEAR_{suffix}_DECISION.json",
        {"status": status, "gates": gates, "selection_role": "H", "V_opened": False},
    )
    print(read(RUN / f"NONLINEAR_{suffix}_DECISION.json"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qa-geometry",
            "qa-scorer",
            "geometry",
            "scorer",
            "evaluate",
            "replicate",
            "evaluate-replicas",
        ),
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
        if args.stage.startswith("qa-"):
            qa(args.stage.removeprefix("qa-"))
        elif args.stage in ("geometry", "scorer", "replicate"):
            prerequisites()
            freeze(RUN / "NONLINEAR_SOURCES_FREEZE.json", binding())
            data = SpatialVectors("P")
            seeds = (13, 23) if args.stage == "replicate" else (7,)
            arms = (
                ("NLH", "NL0", "NL1")
                if args.stage == "replicate"
                else (("NLH",) if args.stage == "geometry" else ("NL0", "NL1"))
            )
            for seed in seeds:
                for arm in arms:
                    if fit(data, arm, seed)["status"] != "COMPLETE":
                        raise SystemExit(75)
        else:
            evaluate(replicas=args.stage == "evaluate-replicas")
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
