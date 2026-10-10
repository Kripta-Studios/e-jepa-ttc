"""Fixed-endpoint TRAIN40-only experiment, with durable state and RNG resume."""

# ruff: noqa: ANN401 -- serialized experiment records have heterogeneous values.
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.efficient_context.common import digest
from operational.train40_system.contracts import environment
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.heads import head_source
from operational.train40_system.history_features import load_source
from operational.ttc_revision.head import DirectTTCHead, symmetric_ttc_loss


def save_state(path: Path, payload: dict[str, Any]) -> None:
    """Commit an entire optimizer boundary atomically, never a partial update."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(config_path: Path, source: Path, *, device: str = "cuda") -> None:
    """Train only the declared fixed seeds and updates; never read evaluation labels."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    recipe = config["training"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = head_source(source, "H8")
    queries = load_source(source, manifest)
    index_path = source / "TRAIN40_INDEX.npz"
    with np.load(index_path, allow_pickle=False) as index:
        truth = index["ttc_s"].astype(np.float32)
        if not np.array_equal(index["phase"].astype(np.float32), queries.target_phase):
            raise ValueError("training target order differs from frozen history")
    if not np.isfinite(truth).all() or len(truth) != queries.population:
        raise ValueError("invalid TRAIN40 targets")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Small fixed feature cache, no raw events or transfer annotations in fitter.
    blocks = [
        queries.gather(torch.arange(i, min(i + 4096, queries.population)))[:3]
        for i in range(0, queries.population, 4096)
    ]
    features, timing, valid = [
        torch.cat([block[k] for block in blocks]).to(device) for k in range(3)
    ]
    target = torch.from_numpy(truth).to(device)
    # Original train-only sequence/TTC-bucket weighting, explicitly preserved.
    weights = torch.from_numpy(queries.mass.astype(np.float32) * queries.population).to(device)
    contract = {
        "config_sha256": digest(config_path),
        "index_sha256": digest(index_path),
        "feature_manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": {
            p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("head.py"))
        },
        "device": device,
        "recipe": recipe,
        "seeds": config["seeds"],
    }
    contract_path = output / "TRAINING_FREEZE.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("training contract changed; use a new experiment directory")
    atomic_json(contract_path, contract)
    budget_path = output / "TRAINING_BUDGET.json"
    used = json.loads(budget_path.read_text())["elapsed_seconds"] if budget_path.exists() else 0.0
    begun = time.perf_counter()
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        model = DirectTTCHead().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=recipe["learning_rate"],
            weight_decay=recipe["weight_decay"],
            foreach=False,
        )
        path = output / f"direct_seed{seed}.pt"
        update, history = 0, []
        identity = {"environment": environment(), "start_time": datetime.now(UTC).isoformat()}
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["contract"] != contract or saved["seed"] != seed:
                raise ValueError("checkpoint identity mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            update, history, identity = saved["update"], saved["history"], saved["identity"]
        while update < recipe["updates"]:
            ids = torch.randint(queries.population, (recipe["batch_size"],), generator=generator)
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[ids], timing[ids], valid[ids])
            loss = (symmetric_ttc_loss(prediction, target[ids]) * weights[ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite TTC-space loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            update += 1
            history.append({"update": update, "loss": float(loss.detach())})
            elapsed = used + time.perf_counter() - begun
            stopping = elapsed >= config["gpu_budget_seconds"] or (output / "STOP_REQUEST").exists()
            if update % 100 == 0 or update == recipe["updates"] or stopping:
                status = (
                    "COMPLETE"
                    if update == recipe["updates"]
                    else "PAUSED"
                    if stopping
                    else "RUNNING"
                )
                save_state(
                    path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "generator": generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "update": update,
                        "seed": seed,
                        "contract": contract,
                        "history": history,
                        "identity": identity,
                        "status": status,
                    },
                )
                atomic_json(
                    path.with_suffix(".json"),
                    {
                        "seed": seed,
                        "update": update,
                        "status": status,
                        "sha256": digest(path),
                        "identity": identity,
                        "updated_utc": datetime.now(UTC).isoformat(),
                        "loss": history[-1]["loss"],
                    },
                )
                atomic_json(
                    budget_path,
                    {"elapsed_seconds": elapsed, "limit_seconds": config["gpu_budget_seconds"]},
                )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "update": update,
                            "loss": history[-1]["loss"],
                            "elapsed_seconds": elapsed,
                            "status": status,
                        }
                    ),
                    flush=True,
                )
            if stopping:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/ttc_revision_20261009.json")
    )
    parser.add_argument("--source", type=Path, default=Path("artifacts/train40_system_20261005"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.config, args.source, device=args.device)
