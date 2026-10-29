"""Frozen-budget directional bypass, retaining all A controls and inference contracts."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import training_loss
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.admission import revalidate
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, resources, sha, write
from operational.range_transfer.qa import same
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .run import INPUTS, OLD, ROOT, RUN, accounting, cache, gpu_charge, init, raw_side, sources


def binding() -> dict:
    result = sources()
    for path in (
        Path(__file__),
        ROOT / "docs/representation_audit_20261012/PROTOCOL_D_DIRECTIONAL.md",
    ):
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def directions(raw: np.ndarray) -> np.ndarray:
    """512 temporal changes; common additive token level cancels."""
    if raw.ndim != 2 or raw.shape[1] != 896 or not np.isfinite(raw).all():
        raise ValueError("Expected finite original vectors")
    geometry = raw[:, :384].reshape(-1, 3, 128)
    forward = raw[:, 384:640].reshape(-1, 2, 128)
    reverse = raw[:, 640:].reshape(-1, 2, 128)
    return np.concatenate(
        (np.diff(geometry, axis=1).reshape(-1, 256), (forward - reverse).reshape(-1, 256)), axis=1
    )


def prepare(role: str) -> dict:
    if role not in ("H", "V") or (role == "V" and not (RUN / "D_ENDPOINTS.json").exists()):
        raise ValueError("D V requires all three frozen endpoints")
    values = cache(role)
    raw, source = raw_side(role, values["sample_token"].astype(str))
    directional = directions(raw.astype(np.float64))
    path = RUN / "D_H_NORMALIZER.json"
    if role == "H":
        freeze(
            path,
            {
                "role": "H",
                "mean": directional.mean(0).tolist(),
                "std": np.maximum(directional.std(0), 1e-6).tolist(),
            },
        )
    norm = read(path)
    if norm["role"] != "H":
        raise ValueError("Only H fits normalizer")
    side = np.zeros((len(raw), 896), np.float32)
    side[:, :512] = ((directional - np.asarray(norm["mean"])) / np.asarray(norm["std"])).astype(
        np.float32
    )
    values["side"] = side
    destination = RUN / f"cache/D_{role}.npz"
    if not destination.exists():
        np.savez_compressed(destination, **values)
    identity = {
        "role": role,
        "sources": source,
        "native": sha(RUN / f"cache/{role}.npz"),
        "normalizer": sha(path),
        "output": sha(destination),
    }
    freeze(destination.with_suffix(".json"), identity)
    return identity


def load_h() -> tuple[dict, dict]:
    path = RUN / "cache/D_H.npz"
    identity = read(path.with_suffix(".json"))
    if identity["role"] != "H" or identity["output"] != sha(path):
        raise ValueError("D H cache changed")
    with np.load(path, allow_pickle=False) as arc:
        values = {k: arc[k] for k in arc.files}
    if set(values["role"].astype(str)) != {"H"}:
        raise ValueError("Training role differs")
    return values, identity


def fit(seed: int, *, technical: str | None = None, stop: int | None = None) -> dict:
    if seed not in (7, 13, 23):
        raise ValueError("Unregistered seed")
    if technical:
        rng = np.random.default_rng(192)
        n = 64
        values = {
            "features": rng.normal(size=(n, 8, 17)).astype(np.float32),
            "timing": rng.normal(size=(n, 8, 4)).astype(np.float32),
            "valid": np.ones((n, 8), bool),
            "expert_phase": rng.uniform(0.005, 0.1, (n, 3)).astype(np.float32),
            "target_phase": rng.uniform(0.002, 0.15, n).astype(np.float32),
            "mass": np.full(n, 1 / n, np.float32),
            "side": rng.normal(size=(n, 896)).astype(np.float32),
        }
        identity_cache = {"status": "SYNTHETIC_ONLY"}
    else:
        if read(RUN / "qa/D_PASS.json")["sources"] != binding():
            raise ValueError("D continuity QA required")
        revalidate(read(OLD / "ADMISSION.json"))
        values, identity_cache = load_h()
        n = len(values["mass"])
        if n != 18422:
            raise ValueError("H population differs")
    name = technical or f"D-S{seed}"
    endpoint = 10 if technical else 2500
    identity = {
        "fit_id": name,
        "seed": seed,
        "arm": "D",
        "endpoint": endpoint,
        "sources": binding(),
        "cache": digest(identity_cache),
        "device": "cuda",
        "precision": "FP32_TF32off",
        "batch": 128,
        "torch": str(torch.__version__),
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
            raise ValueError("D endpoint changed")
        return receipt
    with gpu_charge(name) as deadline:
        model, optimizer, scheduler, generator = init("G", seed)
        arrays = {
            k: torch.from_numpy(values[k].copy()).to("cuda")
            for k in (*INPUTS, "target_phase", "mass")
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
            ids = torch.randint(n, (128,), generator=generator)
            idx = ids.to("cuda")
            optimizer.param_groups[0]["lr"] = _lr("E_CTX_MATCHED", state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*(arrays[k][idx] for k in INPUTS))
            loss = training_loss(
                prediction,
                arrays["target_phase"][idx],
                arrays["expert_phase"][idx],
                arrays["mass"][idx],
                n,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite D objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
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
        accounting()
    return read(state.receipt)


def qa() -> None:
    fit(7, technical="D_continuous10")
    fit(7, technical="D_split5_restore5", stop=5)
    fit(7, technical="D_split5_restore5")
    a = torch.load(
        RUN / "qa/D_continuous10/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        RUN / "qa/D_split5_restore5/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    )
    checks = {k: same(a[k], b[k]) for k in keys}
    checks["curves"] = same(a["cursor"]["curve"], b["cursor"]["curve"])
    checks["draws"] = same(a["cursor"]["draws"], b["cursor"]["draws"])
    if not all(checks.values()):
        raise AssertionError(checks)
    freeze(
        RUN / "qa/D_PASS.json",
        {"status": "PASS", "sources": binding(), "checks": checks, "updates": 20},
    )


def evaluate() -> None:
    endpoints = {}
    for seed in (7, 13, 23):
        name = f"D-S{seed}"
        row = read(RUN / f"fits/{name}/CHECKPOINT_RECEIPT.json")
        if (
            row["status"] != "COMPLETE"
            or row["completed_updates"] != 2500
            or sha(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]
        ):
            raise ValueError("D endpoints incomplete")
        endpoints[name] = row
    freeze(RUN / "D_ENDPOINTS.json", endpoints)
    prepare("V")
    with np.load(RUN / "cache/D_V.npz", allow_pickle=False) as arc:
        values = {k: arc[k] for k in arc.files}
    target = -0.1 / np.expm1(-values["target_phase"].astype(np.float64))
    groups = values["group_id"].astype(str)
    results = {}
    for seed in (7, 13, 23):
        name = f"D-S{seed}"
        folder = RUN / f"evaluation/{name}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "METRICS.json").exists():
            receipt = read(folder / "RECEIPT.json")
            if receipt["predictions_sha256"] != sha(folder / "PREDICTIONS.csv"):
                raise ValueError("Saved D predictions changed")
            if receipt["endpoint"] != endpoints[name]["checkpoint_sha256"]:
                raise ValueError("Saved D endpoint identity differs")
            results[name] = read(folder / "METRICS.json")
            continue
        with gpu_charge("evaluate_" + name):
            model, *_ = init("G", seed)
            payload = torch.load(
                endpoints[name]["checkpoint_path"], map_location="cpu", weights_only=False
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval().requires_grad_(False)
            preds = []
            with torch.inference_mode():
                for start in range(0, len(target), 128):
                    args = [
                        torch.from_numpy(values[k][start : start + 128].copy()).to("cuda")
                        for k in INPUTS
                    ]
                    preds.append(phase_to_ttc(model(*args)["raw_location"]).cpu().numpy())
            pred = np.concatenate(preds)
        pd.DataFrame(
            {
                "sample_token": values["sample_token"],
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
            },
        )
        results[name] = score
    baseline = read(RUN / "RESULTS.json")
    historical = read(
        ROOT.parent
        / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
        / "evaluation/screen/RT-M0-L0-S7/METRICS.json"
    )
    gates = {
        f"S{seed}": gate(baseline[f"Z-S{seed}"], results[f"D-S{seed}"]) for seed in (7, 13, 23)
    }
    gates["historical_S7"] = gate(historical, results["D-S7"])
    directional_better = all(
        results[f"D-S{seed}"]["bins"]["crucial"]["group_macro_mid"]
        < baseline[f"G-S{seed}"]["bins"]["crucial"]["group_macro_mid"]
        for seed in (7, 13, 23)
    )
    passed = all(row["eligible"] for row in gates.values()) and directional_better
    write(RUN / "D_RESULTS.json", results)
    write(
        RUN / "D_NEXT_DECISION.json",
        {
            "status": "ACCURACY_CANDIDATE_COST_PENDING" if passed else "NO_CANDIDATE",
            "gates": gates,
            "all_seeds_better_than_G_crucial": directional_better,
            "SOTA": False,
        },
    )
    print(read(RUN / "D_NEXT_DECISION.json"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "qa", "train", "evaluate"))
    args = parser.parse_args()
    # CPU preparation is read-only with respect to the running GPU queue.
    if args.stage == "prepare":
        prepare("H")
        return
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
        elif args.stage == "train":
            freeze(RUN / "D_SOURCES_FREEZE.json", binding())
            for seed in (7, 13, 23):
                if fit(seed)["status"] != "COMPLETE":
                    return
        else:
            evaluate()
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
