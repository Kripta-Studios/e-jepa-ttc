"""V17 isolated queue: admission, durable GPU fits, freeze and native evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import training_loss
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.admission import revalidate
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, resources, sha, utc, write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .model import RepresentationRefiner

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/representation_audit_20261012"
OLD = ROOT.parent / "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012"
OWNER = ROOT.parent / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
PROTOCOL = ROOT / "docs/representation_audit_20261012/PROTOCOL.md"
INPUTS = ("features", "timing", "valid", "expert_phase", "side")
QUEUE = tuple((arm, seed) for seed in (7, 13, 23) for arm in ("Z", "G"))


def sources() -> dict:
    paths = [Path(__file__), Path(__file__).with_name("model.py"), PROTOCOL]
    paths += [
        ROOT / p
        for p in (
            "src/e_jepa_ttc/simplex_t/model.py",
            "src/e_jepa_ttc/simplex_t/phase.py",
            "operational/rgb_port/train_producers.py",
            "operational/rgb_port/accounting.py",
            "operational/range_transfer/analyze.py",
            "operational/range_transfer/common.py",
        )
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def accounting() -> dict:
    result = {"scientific": 0, "technical": 0, "recovery_upper": 0, "pending_upper": 0}
    for path in RUN.glob("*/**/UPDATE_JOURNAL.json"):
        row = read(path)
        kind = "technical" if "qa" in path.relative_to(RUN).parts else "scientific"
        result[kind] += row["completed_updates"]
        result["recovery_upper"] += row["recovery_upper"]
        result["pending_upper"] += row["pending_update_upper"]
    result["physical_upper"] = sum(result.values())
    if result["physical_upper"] > 1_000_000:
        raise RuntimeError("Physical cap exceeded")
    write(RUN / "ACCOUNTING.json", result)
    return result


@contextmanager
def gpu_charge(scope: str, reserve: float = 900) -> Iterator[float]:
    """Reserve conservatively before GPU work; crashed calls retain their reserve."""
    path = RUN / "GPU_BUDGET.json"
    book = read(path) if path.exists() else {"cap_s": 64800, "entries": []}
    used = sum(r["charged_s"] for r in book["entries"])
    if used + reserve > book["cap_s"]:
        raise RuntimeError("GPU cap exhausted; no new reservation")
    row = {
        "scope": scope,
        "utc": utc(),
        "charged_s": reserve,
        "status": "RESERVED",
        "pid": os.getpid(),
    }
    book["entries"].append(row)
    write(path, book)
    started = time.perf_counter()
    try:
        yield started + reserve - 20
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row.update(charged_s=elapsed, status="SETTLED")
        book["charged_total_s"] = sum(r["charged_s"] for r in book["entries"])
        write(path, book)
        if elapsed > reserve:
            raise RuntimeError("GPU reservation overrun; halt campaign")


def raw_side(role: str, tokens: np.ndarray) -> tuple[np.ndarray, dict]:
    folder = OLD / "vector_cache" / role
    complete = read(folder / "COMPLETE.json")
    binding = complete["binding"]
    if complete["status"] != "COMPLETE" or binding["role"] != role:
        raise ValueError("Incomplete vector cache")
    if binding["manifest_sha256"] != sha(OWNER / f"{role}_MANIFEST.json"):
        raise ValueError("Role manifest changed")
    parent_sha = read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"]
    if binding["parent_sha256"] != parent_sha:
        raise ValueError("Wrong A5 parent")
    vectors, ids, checks = [], [], {}
    for index in range((complete["rows"] + 127) // 128):
        path = folder / f"part_{index:06d}.npz"
        receipt = read(path.with_suffix(".json"))
        if receipt["sha256"] != sha(path):
            raise ValueError("Vector shard changed")
        checks[path.name] = receipt["sha256"]
        with np.load(path, allow_pickle=False) as arc:
            n = len(arc["sample_token"])
            vectors.append(
                np.concatenate(
                    [
                        arc[k].reshape(n, -1)
                        for k in ("geometry_tokens", "forward_transport", "reverse_transport")
                    ],
                    axis=1,
                )
            )
            ids.extend(arc["sample_token"].astype(str).tolist())
    if len(ids) != complete["rows"] or len(set(ids)) != len(ids) or set(ids) != set(tokens):
        raise ValueError("Side/history cohort differs")
    positions = {token: i for i, token in enumerate(ids)}
    values = np.concatenate(vectors)[[positions[token] for token in tokens]]
    if values.shape != (len(tokens), 896) or not np.isfinite(values).all():
        raise ValueError("Wrong bypass shape/finiteness")
    return values, {"complete_sha256": sha(folder / "COMPLETE.json"), "shards": checks}


def prepare(role: str) -> dict:
    if role not in ("H", "V") or (role == "V" and not (RUN / "QUEUE_FREEZE.json").exists()):
        raise ValueError("V inputs require all six frozen endpoints")
    revalidate(read(OLD / "ADMISSION.json"))
    path = OWNER / f"features/{role}/E_HISTORY.npz"
    receipt = read(path.with_suffix(".json"))
    if sha(path) != receipt["output_sha256"]:
        raise ValueError("Native history hash differs")
    with np.load(path, allow_pickle=False) as arc:
        values = {k: arc[k] for k in arc.files}
    if set(values["role"].astype(str)) != {role}:
        raise ValueError("Wrong history role")
    side, binding = raw_side(role, values["sample_token"].astype(str))
    normalizer = RUN / "H_SIDE_NORMALIZER.json"
    if role == "H":
        freeze(
            normalizer,
            {
                "role": "H",
                "mean": side.mean(0, dtype=np.float64).tolist(),
                "std": np.maximum(side.std(0, dtype=np.float64), 1e-6).tolist(),
                "tokens_sha256": digest(values["sample_token"].astype(str).tolist()),
            },
        )
    norm = read(normalizer)
    if norm["role"] != "H":
        raise ValueError("Invalid normalizer training role")
    values["side"] = ((side - np.asarray(norm["mean"])) / np.asarray(norm["std"])).astype(
        np.float32
    )
    destination = RUN / f"cache/{role}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        np.savez_compressed(destination, **values)
    identity = {
        "role": role,
        "rows": len(side),
        "history_sha256": sha(path),
        "vector_binding": binding,
        "normalizer_sha256": sha(normalizer),
        "output_sha256": sha(destination),
    }
    freeze(destination.with_suffix(".json"), identity)
    return identity


def cache(role: str) -> dict:
    path = RUN / f"cache/{role}.npz"
    if sha(path) != read(path.with_suffix(".json"))["output_sha256"]:
        raise ValueError("Prepared cache changed")
    with np.load(path, allow_pickle=False) as arc:
        values = {k: arc[k] for k in arc.files}
    if set(values["role"].astype(str)) != {role}:
        raise ValueError("Cache role differs")
    return values


def init(arm: str, seed: int) -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = RepresentationRefiner(arm == "G").to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr("E_CTX_MATCHED", min(step + 1, 2500)) / 3e-4
    )
    return model, optimizer, scheduler, torch.Generator(device="cpu").manual_seed(seed)


def fit(arm: str, seed: int, *, technical: str | None = None, stop: int | None = None) -> dict:
    if (arm, seed) not in QUEUE:
        raise ValueError("Unregistered arm/seed")
    if not technical and read(RUN / "qa/PASS.json")["sources"] != sources():
        raise ValueError("Current sources require successful QA")
    name = technical or f"{arm}-S{seed}"
    endpoint = 10 if technical else 2500
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
        binding = {"status": "SYNTHETIC_ONLY"}
    else:
        revalidate(read(OLD / "ADMISSION.json"))
        values = cache("H")
        binding = read(RUN / "cache/H.json")
    n = len(values["mass"])
    identity = {
        "fit_id": name,
        "arm": arm,
        "seed": seed,
        "endpoint": endpoint,
        "sources": sources(),
        "binding": digest(binding),
        "device": "cuda",
        "precision": "FP32_TF32_OFF",
        "torch": str(torch.__version__),
        "batch": 128,
        "scientific": not bool(technical),
    }
    identity["identity_sha256"] = digest(identity)
    directory = RUN / ("qa" if technical else "fits") / name
    checkpoint = ProducerCheckpoint(directory, identity, endpoint)
    if checkpoint.receipt.exists():
        existing = read(checkpoint.receipt)
        if existing["status"] == "COMPLETE":
            if (
                existing["identity_sha256"] != identity["identity_sha256"]
                or sha(checkpoint.path) != existing["checkpoint_sha256"]
            ):
                raise ValueError("Completed fit changed")
            return existing
    with gpu_charge(name) as deadline:
        model, optimizer, scheduler, generator = init(arm, seed)
        arrays = {
            k: torch.from_numpy(values[k].copy()).to("cuda")
            for k in (*INPUTS, "target_phase", "mass")
        }
        cursor = {"curves": [], "draw_sha": [], "elapsed_s": 0.0}
        if checkpoint.pointer.exists():
            cursor = checkpoint.restore(model, optimizer, scheduler, generator)
        else:
            checkpoint.save(model, optimizer, scheduler, generator, cursor, status="READY")
        total = accounting()["physical_upper"]
        started = time.perf_counter()
        executed = 0
        while checkpoint.completed < endpoint:
            if time.perf_counter() >= deadline or total >= 1_000_000 or (RUN / "PAUSE").exists():
                break
            if checkpoint.completed % 25 == 0 and not resources(RUN)["allowed"]:
                break
            ids = torch.randint(n, (128,), generator=generator)
            gpu_ids = ids.to("cuda")
            optimizer.param_groups[0]["lr"] = _lr("E_CTX_MATCHED", checkpoint.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            output = model(*(arrays[k][gpu_ids] for k in INPUTS))
            loss = training_loss(
                output,
                arrays["target_phase"][gpu_ids],
                arrays["expert_phase"][gpu_ids],
                arrays["mass"][gpu_ids],
                n,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            checkpoint.begin()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            checkpoint.commit()
            total += 1
            executed += 1
            cursor["curves"].append(
                {
                    "update": checkpoint.completed,
                    "loss": float(loss.detach()),
                    "gradient_l2": float(grad),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            cursor["draw_sha"].append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
            if (
                checkpoint.completed % 100 == 0
                or checkpoint.completed == endpoint
                or (stop and executed >= stop)
            ):
                cursor["elapsed_s"] += time.perf_counter() - started
                status = "COMPLETE" if checkpoint.completed == endpoint else "IN_PROGRESS"
                checkpoint.save(model, optimizer, scheduler, generator, cursor, status=status)
                pd.DataFrame(cursor["curves"]).to_csv(directory / "CURVE.csv", index=False)
                accounting()
                print(name, checkpoint.completed, round(cursor["elapsed_s"], 2), flush=True)
                started = time.perf_counter()
            if stop and executed >= stop:
                break
        if checkpoint.completed != checkpoint.durable:
            cursor["elapsed_s"] += time.perf_counter() - started
            checkpoint.save(model, optimizer, scheduler, generator, cursor, status="PAUSED")
        accounting()
    return read(checkpoint.receipt)


def qa() -> None:
    fit("G", 7, technical="continuous10")
    fit("G", 7, technical="split5_restore5", stop=5)
    fit("G", 7, technical="split5_restore5")
    a = torch.load(
        RUN / "qa/continuous10/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        RUN / "qa/split5_restore5/checkpoint_last.pt", map_location="cpu", weights_only=False
    )

    def equal(left: Any, right: Any) -> bool:  # noqa: ANN401 -- recursive checkpoint tree
        if isinstance(left, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                equal(x, y) for x, y in zip(left, right, strict=True)
            )
        return left == right

    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
    ):
        if not equal(a[key], b[key]):
            raise AssertionError(f"Resume differs: {key}")
    if (
        a["cursor"]["curves"] != b["cursor"]["curves"]
        or a["cursor"]["draw_sha"] != b["cursor"]["draw_sha"]
    ):
        raise AssertionError("Resume trajectory differs")
    freeze(
        RUN / "qa/PASS.json",
        {"status": "PASS", "sources": sources(), "continuous_equals_resumed": True, "updates": 20},
    )


def freeze_queue() -> dict:
    rows = {}
    for arm, seed in QUEUE:
        name = f"{arm}-S{seed}"
        row = read(RUN / f"fits/{name}/CHECKPOINT_RECEIPT.json")
        if (
            row["status"] != "COMPLETE"
            or row["completed_updates"] != 2500
            or sha(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]
        ):
            raise ValueError("All six endpoints must be complete")
        rows[name] = row
    freeze(RUN / "QUEUE_FREEZE.json", rows)
    return rows


def evaluate() -> None:
    endpoints = freeze_queue()
    prepare("V")
    values = cache("V")
    target = -0.1 / np.expm1(-values["target_phase"].astype(np.float64))
    groups = values["group_id"].astype(str)
    results = {}
    for arm, seed in QUEUE:
        name = f"{arm}-S{seed}"
        out = RUN / f"evaluation/{name}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "METRICS.json").exists():
            results[name] = read(out / "METRICS.json")
            continue
        with gpu_charge("evaluation_" + name):
            model, *_ = init(arm, seed)
            payload = torch.load(
                endpoints[name]["checkpoint_path"], map_location="cpu", weights_only=False
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval().requires_grad_(False)
            predictions = []
            begin_time = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, len(target), 128):
                    inputs = [
                        torch.from_numpy(values[k][start : start + 128].copy()).to("cuda")
                        for k in INPUTS
                    ]
                    predictions.append(phase_to_ttc(model(*inputs)["raw_location"]).cpu().numpy())
            pred = np.concatenate(predictions)
            elapsed = time.perf_counter() - begin_time
        pd.DataFrame(
            {
                "sample_token": values["sample_token"],
                "group_id": groups,
                "target_ttc": target,
                "prediction_ttc": pred,
            }
        ).to_csv(out / "PREDICTIONS.csv", index=False)
        score = metrics(target, pred, groups)
        write(out / "METRICS.json", score)
        write(
            out / "GROUP_METRICS.json",
            {
                g: metrics(target[groups == g], pred[groups == g], groups[groups == g])
                for g in sorted(set(groups))
            },
        )
        write(
            out / "RECEIPT.json",
            {
                "predictions_sha256": sha(out / "PREDICTIONS.csv"),
                "endpoint": endpoints[name]["checkpoint_sha256"],
                "scope": "GPU_FP32_cached_head_batch128_including_copies_no_producers_no_IO",
                "seconds": elapsed,
            },
        )
        results[name] = score
    historic = read(
        ROOT.parent
        / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
        / "evaluation/screen/RT-M0-L0-S7/METRICS.json"
    )
    gates = {f"S{seed}": gate(results[f"Z-S{seed}"], results[f"G-S{seed}"]) for seed in (7, 13, 23)}
    gates["historical_S7"] = gate(historic, results["G-S7"])
    passed = all(row["eligible"] for row in gates.values())
    write(
        RUN / "NEXT_DECISION.json",
        {
            "status": "ACCURACY_CANDIDATE_COST_PENDING" if passed else "NO_CANDIDATE",
            "gates": gates,
            "SOTA": False,
            "evaluation": "exposed_internal_V",
        },
    )
    write(RUN / "RESULTS.json", results)
    print(read(RUN / "NEXT_DECISION.json"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "qa", "train", "evaluate"))
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
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
        if args.stage == "prepare":
            prepare("H")
        elif args.stage == "qa":
            qa()
        elif args.stage == "train":
            freeze(RUN / "SOURCES_FREEZE.json", sources())
            for arm, seed in QUEUE:
                if fit(arm, seed)["status"] != "COMPLETE":
                    return
            freeze_queue()
        else:
            evaluate()
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
