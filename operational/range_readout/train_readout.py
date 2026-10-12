"""Fixed 6000-update scorer-only fits with native loss and durable resume."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ruff: noqa: ANN401 -- shared native checkpoint/cache boundary
import numpy as np
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec, require_same_operator
from e_jepa_ttc.range_readout.readout import VectorReadout, wrap
from operational.rgb_port.infer_experts import Fragments
from operational.rgb_port.train_producers import ProducerCheckpoint

from .cache_readout import loss, parent
from .common import OWNER, ROOT, RUN, csv_write, digest, read, resources, sha, utc, write
from .ledger import accounting


class VectorCache:
    """Admitted vector shards only; preserve the exact P query order."""

    def __init__(self, role: str) -> None:
        directory = RUN / "vector_cache" / role
        complete = read(directory / "COMPLETE.json")
        if complete["binding"]["role"] != role or complete["binding"]["manifest_sha256"] != sha(
            OWNER / f"{role}_MANIFEST.json"
        ):
            raise ValueError("Vector cache role/manifest identity changed")
        self.binding = complete
        self.values = {}
        store = Fragments(directory, complete["binding"])
        parts = [store.read(i) for i in range(math.ceil(complete["rows"] / 128))]
        if any(p is None for p in parts):
            raise ValueError("Incomplete vector cache")
        parts = [p for p in parts if p is not None]
        for key in parts[0]:
            self.values[key] = np.concatenate([p[key] for p in parts])
        self.population = len(self.values["sample_token"])
        if (
            self.population != complete["rows"]
            or len(set(self.values["sample_token"])) != self.population
        ):
            raise ValueError("Vector cache population differs")

    def batch(self, ids: torch.Tensor) -> dict:
        return {
            k: torch.from_numpy(v[ids.numpy()].copy())
            for k, v in self.values.items()
            if k != "sample_token"
        }


def lr(update: int) -> float:
    """Prospective readout-only schedule, independent of producer epoch18."""
    if update <= 100:
        return 3e-4 * update / 100
    return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (min(update, 6000) - 100) / 5900))


def initialize(bound: float, seed: int) -> tuple[Any, Any, Any, Any, torch.Generator]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    native, config = parent()
    model = VectorReadout(wrap(native, ResidualSpec(bound), reset=True))
    model.model.residual.scorer.train()
    optimizer = torch.optim.AdamW(
        model.model.residual.scorer.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr(step + 1) / 3e-4)
    generator = torch.Generator().manual_seed(seed)
    return model, config, optimizer, scheduler, generator


def sources() -> dict:
    paths = [
        "operational/range_readout/train_readout.py",
        "operational/range_readout/cache_readout.py",
        "src/e_jepa_ttc/range_readout/readout.py",
        "src/e_jepa_ttc/range_readout/operators.py",
        "src/e_jepa_ttc/models/causal_scale_ttc.py",
        "src/e_jepa_ttc/losses/causal_scale_ttc.py",
        "operational/rgb_port/train_producers.py",
        "operational/rgb_port/accounting.py",
        "operational/range_readout/ledger.py",
    ]
    return {p: sha(ROOT / p) for p in paths}


def fit(
    cache: Any,
    bound: float,
    seed: int = 7,
    *,
    technical: str | None = None,
    stop: int | None = None,
) -> dict[str, Any]:
    if seed not in (7, 13, 23) or (stop is not None and stop < 1):
        raise ValueError("Invalid seed or partial update limit")
    if technical:
        if cache.binding.get("status") != "SYNTHETIC_ONLY":
            raise ValueError("Technical budget is synthetic-only")
    elif cache.binding.get("binding", {}).get("role") != "P" or cache.population != 53890:
        raise ValueError("Scientific readout fits require the complete P cache")
    name = technical or f"R-C{round(bound * 100):02d}-S{seed}"
    if seed != 7 and read(RUN / "SELECTION_FREEZE.json").get("family") != "R":
        raise ValueError("R replication not admitted")
    endpoint = 10 if technical else 6000
    if not technical and read(RUN / "qa/READOUT_RESUME.json")["status"] != "PASS":
        raise ValueError("Readout resume QA required")
    spec = ResidualSpec(bound)
    directory = RUN / ("qa" if technical else "fits") / name
    identity = {
        "fit_id": name,
        "seed": seed,
        "endpoint": endpoint,
        "operator": asdict(spec),
        "cache": cache.binding,
        "source_code": sources(),
        "technical": technical is not None,
        "precision": "float32",
        "device": "cpu",
        "batch": 128,
        "trainable": "residual.scorer",
        "torch": str(torch.__version__),
    }
    identity["identity_sha256"] = digest(identity)
    model, cfg, optimizer, scheduler, generator = initialize(bound, seed)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor = {"losses": [], "draw_hashes": [], "elapsed_s": 0.0}
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    if state.completed == endpoint:
        return read(state.receipt)
    started = time.perf_counter()
    executed = 0
    status = "IN_PROGRESS"
    try:
        while state.completed < endpoint:
            if state.completed % 25 == 0:
                observation = resources(RUN)
                write(RUN / "R_RESOURCES.json", observation)
                if not observation["allowed"] or (RUN / "PAUSE").exists():
                    status = "PAUSED_RESOURCE"
                    break
            ids = torch.randint(cache.population, (128,), generator=generator)
            values = cache.batch(ids)
            optimizer.param_groups[0]["lr"] = lr(state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            total, components = loss(model, values, cfg)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite native readout objective")
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.model.residual.scorer.parameters(), 1.0, error_if_nonfinite=True
            )
            accounting(RUN, prospective=1, technical=technical is not None)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            executed += 1
            cursor["losses"].append(
                {
                    "update": state.completed,
                    "loss": float(total.detach()),
                    "gradient_l2_before_clip": float(grad),
                    "lr": lr(state.completed),
                    **{k: float(v.detach()) for k, v in components.items()},
                }
            )
            cursor["draw_hashes"].append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
            if state.completed == endpoint:
                status = "COMPLETE"
            stopping = stop is not None and executed >= stop
            if state.completed % 100 == 0 or status == "COMPLETE" or stopping:
                cursor["elapsed_s"] += time.perf_counter() - started
                state.save(model, optimizer, scheduler, generator, cursor, status=status)
                started = time.perf_counter()
                csv_write(directory / "CURVE.csv", cursor["losses"])
            if stopping:
                break
        if state.completed != state.durable:
            cursor["elapsed_s"] += time.perf_counter() - started
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
    except BaseException as error:
        write(
            directory / f"FAILURE_{time.time_ns()}.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed": state.completed,
                "durable": state.durable,
                "utc": utc(),
            },
        )
        raise
    for n, p in model.named_parameters():
        if n in frozen and not torch.equal(p, frozen[n]):
            raise AssertionError("Frozen parameter changed: " + n)
    accounting(RUN)
    write(
        directory / "RUN_PROVENANCE.json",
        {
            "fit_id": name,
            "seed": seed,
            "parent": read(RUN / "ADMISSION.json")["parents"]["E_A5_MATCHED"],
            "operator": asdict(spec),
            "frozen_parameters_unchanged": True,
            "scope": "P_readout_only" if not technical else "synthetic_QA",
            "elapsed_s": cursor["elapsed_s"],
            "draw_sequence_sha256": digest(cursor["draw_hashes"]),
            "checkpoint_sha256": sha(state.path),
            "identity_sha256": identity["identity_sha256"],
            "status": status,
        },
    )
    return read(state.receipt)


def load_endpoint(path: Path, spec: ResidualSpec) -> VectorReadout:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require_same_operator(ResidualSpec(**value["identity"]["operator"]), spec)
    if value["completed_updates"] != 6000:
        raise ValueError("Readout endpoint incomplete")
    model, _, _, _, _ = initialize(spec.bound, int(value["identity"]["seed"]))
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=float, choices=[0.05, 0.20], required=True)
    parser.add_argument("--seed", type=int, choices=[7, 13, 23], default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(fit(VectorCache("P"), args.bound, args.seed), flush=True)


if __name__ == "__main__":
    main()
