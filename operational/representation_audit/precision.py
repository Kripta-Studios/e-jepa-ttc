"""Prospective scorer precision intervention, selected on H without opening V."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.object_event_v4 import BoxGeometryTargets
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig, causal_scale_ttc_loss
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    target_log_ratio_from_ttc,
)
from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import VectorReadout, wrap
from operational.range_transfer.admission import revalidate
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, resources, sha, write
from operational.range_transfer.qa import same
from operational.rgb_port.train_producers import ProducerCheckpoint

from .run import OLD, OWNER, ROOT, RUN, accounting, gpu_charge

B_QUEUE = tuple((arm, seed) for seed in (7, 13, 23) for arm in ("U0", "U1"))
PRECISION = 26551.97057752027


def binding() -> dict:
    paths = [__file__, str(ROOT / "docs/representation_audit_20261012/PROTOCOL_B_PRECISION.md")]
    paths += [
        str(ROOT / p)
        for p in (
            "operational/representation_audit/run.py",
            "src/e_jepa_ttc/range_readout/readout.py",
            "src/e_jepa_ttc/range_readout/operators.py",
            "src/e_jepa_ttc/losses/causal_scale_ttc.py",
            "src/e_jepa_ttc/models/causal_scale_ttc.py",
            "operational/rgb_port/train_producers.py",
        )
    ]
    paths.append(str(ROOT / "configs/rgb_port/producers.json"))
    paths.append(str(ROOT / "operational/rgb_port/accounting.py"))
    return {str(Path(p).relative_to(ROOT)): sha(Path(p)) for p in paths}


class Vectors:
    """Hash-checked V15 vector inputs, retaining native P labels for training."""

    def __init__(self, role: str, *, synthetic: bool = False) -> None:
        if role not in ("P", "H"):
            raise ValueError("No V in precision screen")
        folder = OLD / "vector_cache" / role
        complete = read(folder / "COMPLETE.json")
        if complete["binding"]["manifest_sha256"] != sha(OWNER / f"{role}_MANIFEST.json"):
            raise ValueError("Role manifest changed")
        parts = []
        hashes = {}
        for index in range(1 if synthetic else math.ceil(complete["rows"] / 128)):
            path = folder / f"part_{index:06d}.npz"
            actual = sha(path)
            if actual != read(path.with_suffix(".json"))["sha256"]:
                raise ValueError("Vector shard changed")
            hashes[path.name] = actual
            with np.load(path, allow_pickle=False) as arc:
                parts.append({k: arc[k] for k in arc.files})
        self.values = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
        if synthetic:
            rng = np.random.default_rng(88)
            generated = {}
            for key, value in self.values.items():
                if key == "sample_token":
                    continue
                shape = (64, *value.shape[1:])
                if value.dtype.kind == "b":
                    arr = np.ones(shape, bool)
                    arr[:, 0] = False
                elif value.dtype.kind in "iu":
                    arr = np.ones(shape, np.int64)
                else:
                    arr = np.full(shape, 0.5, np.float32)
                    if key in ("geometry_tokens", "forward_transport", "reverse_transport"):
                        arr = rng.normal(size=shape).astype(np.float32) * 0.1
                    elif key in ("analytic", "raw_foreground_ratio", "raw_foreground_width_ratio"):
                        arr = rng.normal(size=shape).astype(np.float32) * 0.01
                    elif key == "delta":
                        arr.fill(0.1)
                    elif key == "log_height_correction":
                        arr.fill(0)
                    elif key == "target_ttc":
                        arr = np.tile([0.5, 1.5, 5, -2], 16).astype(np.float32)
                generated[key] = arr
            self.values = generated
            self.identity = {"status": "SYNTHETIC_SHAPES_ONLY", "seed": 88, "population": 64}
            self.n = 64
        else:
            self.n = len(self.values["sample_token"])
            if self.n != complete["rows"] or len(set(self.values["sample_token"])) != self.n:
                raise ValueError("Incomplete/duplicated cohort")
            self.identity = {
                "role": role,
                "complete": sha(folder / "COMPLETE.json"),
                "shards": hashes,
            }


def initialize(seed: int, *, device: str = "cuda") -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt"
    if sha(path) != read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"]:
        raise ValueError("Parent changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    native = CausalScaleTTC(CausalScaleTTCConfig(**payload["identity"]["model_config"]))
    native.load_state_dict(payload["model_state_dict"], strict=True)
    model = VectorReadout(wrap(native, ResidualSpec(0.2), reset=True)).to(device)
    model.model.residual.scorer.train()
    config = CausalScaleTTCLossConfig(
        **read(ROOT / "configs/rgb_port/producers.json")["producers"]["E_A5_MATCHED"]["loss_config"]
    )
    optimizer = torch.optim.AdamW(
        model.model.residual.scorer.parameters(), lr=3e-4, weight_decay=1e-4, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate(step + 1) / 3e-4
    )
    return model, config, optimizer, scheduler, torch.Generator().manual_seed(seed)


def learning_rate(step: int) -> float:
    if step <= 100:
        return 3e-4 * step / 100
    return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (min(step, 6000) - 100) / 5900))


def homogeneous_nll(
    prediction: torch.Tensor, truth: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Same average precision P, without per-query heteroscedastic weighting."""
    return (0.5 * PRECISION * (prediction - truth).square() + 0.5 * logvar).mean()


def objective(
    model: VectorReadout, values: dict, config: CausalScaleTTCLossConfig, arm: str
) -> tuple:
    if arm not in ("U0", "U1"):
        raise ValueError("Unregistered precision intervention")
    out = model({k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))})
    geometry = BoxGeometryTargets(
        **{f.name: values["target_" + f.name] for f in fields(BoxGeometryTargets)}
    )
    base = causal_scale_ttc_loss(
        out,
        target_ttc_seconds=values["target_ttc"],
        delta_t_s=values["delta"],
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=torch.isfinite(values["target_ttc"]) & (values["target_ttc"] != 0),
        target_geometry=geometry,
        config=config,
    )
    dino = values["dino_numerator"].sum() / values["dino_count"].sum().clamp_min(1)
    total = base.total + 8 * dino
    nll = base.components["log_ratio_nll"]
    if arm == "U1":
        if not config.supervise_pair_ratio_before_temporal_blend:
            raise ValueError("Expected native pre-blend supervision")
        truth, valid = target_log_ratio_from_ttc(values["target_ttc"], values["delta"][:, -1])
        valid = valid & torch.isfinite(values["target_ttc"]) & (values["target_ttc"] != 0)
        if not valid.any():
            raise ValueError("No physical target in registered batch")
        nll = homogeneous_nll(
            out.pair_log_height_ratio[:, -1][valid],
            truth[valid],
            out.log_ratio_log_variance[:, -1][valid],
        )
        total = total + config.log_ratio_nll_weight * (nll - base.components["log_ratio_nll"])
    return total, nll


def fit(
    data: Vectors, arm: str, seed: int, *, technical: str | None = None, stop: int | None = None
) -> dict:
    if (arm, seed) not in B_QUEUE:
        raise ValueError("Unregistered arm or seed")
    if technical:
        if data.identity.get("status") != "SYNTHETIC_SHAPES_ONLY":
            raise ValueError("Technical updates require synthetic inputs")
    elif data.identity.get("role") != "P" or data.n != 53890:
        raise ValueError("Readout scientific training requires the admitted P cohort")
    if not technical and read(RUN / "qa/PRECISION_PASS.json")["sources"] != binding():
        raise ValueError("Precision QA required")
    name = technical or f"{arm}-S{seed}"
    endpoint = 10 if technical else 6000
    identity = {
        "fit_id": name,
        "arm": arm,
        "seed": seed,
        "endpoint": endpoint,
        "sources": binding(),
        "precision_P": PRECISION,
        "operator": asdict(ResidualSpec(0.2)),
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
            raise ValueError("Existing precision fit changed")
        return receipt
    revalidate(read(OLD / "ADMISSION.json"))
    with gpu_charge(name) as deadline:
        model, config, optimizer, scheduler, generator = initialize(seed)
        frozen = {
            k: v.detach().cpu().clone() for k, v in model.named_parameters() if not v.requires_grad
        }
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
            loss, nll = objective(model, {k: v[idx] for k, v in arrays.items()}, config, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite precision loss")
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
        write(
            directory / "FROZEN_PARENT_AUDIT.json",
            {"status": "PASS", "unchanged_tensors": len(frozen)},
        )
        accounting()
    return read(state.receipt)


def qa() -> None:
    data = Vectors("P", synthetic=True)
    fit(data, "U1", 7, technical="precision_continuous10")
    fit(data, "U1", 7, technical="precision_split5_restore5", stop=5)
    fit(data, "U1", 7, technical="precision_split5_restore5")
    a = torch.load(
        RUN / "qa/precision_continuous10/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        RUN / "qa/precision_split5_restore5/checkpoint_last.pt",
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
    checks = {k: same(a[k], b[k]) for k in keys}
    checks["curve"] = same(a["cursor"]["curve"], b["cursor"]["curve"])
    checks["draws"] = same(a["cursor"]["draws"], b["cursor"]["draws"])
    if not all(checks.values()):
        raise AssertionError(checks)
    freeze(
        RUN / "qa/PRECISION_PASS.json",
        {"status": "PASS", "sources": binding(), "checks": checks, "updates": 20},
    )


def evaluate() -> None:
    endpoints = {}
    for arm, seed in B_QUEUE:
        name = f"{arm}-S{seed}"
        row = read(RUN / f"fits/{name}/CHECKPOINT_RECEIPT.json")
        if row["status"] != "COMPLETE" or row["completed_updates"] != 6000:
            raise ValueError("Freeze six readouts first")
        endpoints[name] = row
    freeze(RUN / "PRECISION_ENDPOINTS.json", endpoints)
    scores = {}
    for role in ("P", "H"):
        data = Vectors(role)
        manifest = read(OWNER / f"{role}_MANIFEST.json")
        rows = pd.read_parquet(manifest["rows_path"]).set_index("sample_token")
        ordered = rows.loc[data.values["sample_token"].astype(str)]
        target = ordered.target_ttc.to_numpy(np.float64)
        groups = ordered.group_id.to_numpy(str)
        for arm, seed in B_QUEUE:
            name = f"{arm}-S{seed}"
            folder = RUN / f"precision_evaluation/{role}/{name}"
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "METRICS.json").exists():
                scores[f"{role}/{name}"] = read(folder / "METRICS.json")
                continue
            if (
                sha(__import__("pathlib").Path(endpoints[name]["checkpoint_path"]))
                != endpoints[name]["checkpoint_sha256"]
            ):
                raise ValueError("Endpoint changed")
            with gpu_charge("precision_evaluation_" + role + name):
                model, *_ = initialize(seed)
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
            frame = pd.DataFrame(
                {
                    "sample_token": ordered.index,
                    "group_id": groups,
                    "target_ttc": target,
                    "prediction_ttc": pred,
                }
            )
            frame.to_csv(folder / "PREDICTIONS.csv", index=False)
            score = metrics(target, pred, groups)
            write(folder / "METRICS.json", score)
            write(
                folder / "RECEIPT.json",
                {
                    "endpoint": endpoints[name]["checkpoint_sha256"],
                    "predictions_sha256": sha(folder / "PREDICTIONS.csv"),
                },
            )
            scores[f"{role}/{name}"] = score
    original = read(OLD / "PRODUCER_METRICS_P_H.json")["scores"]["H/historical_A5"]
    gates = {}
    for seed in (7, 13, 23):
        gates[f"paired_S{seed}"] = gate(scores[f"H/U0-S{seed}"], scores[f"H/U1-S{seed}"])
        gates[f"historical_S{seed}"] = gate(original, scores[f"H/U1-S{seed}"])
    passed = all(item["eligible"] for item in gates.values())
    write(RUN / "PRECISION_RESULTS.json", scores)
    write(
        RUN / "PRECISION_NEXT_DECISION.json",
        {
            "status": "HEAD_INTEGRATION_ADMITTED" if passed else "NO_CANDIDATE",
            "gates": gates,
            "selection_role": "H",
            "V_opened": False,
        },
    )
    print(read(RUN / "PRECISION_NEXT_DECISION.json"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qa", "train", "evaluate"))
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
        elif args.stage == "train":
            freeze(RUN / "PRECISION_SOURCES_FREEZE.json", binding())
            data = Vectors("P")
            for arm, seed in B_QUEUE:
                if fit(data, arm, seed)["status"] != "COMPLETE":
                    return
        else:
            evaluate()
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
