"""Retained PAIR and H8 recipes with atomic complete resume and physical accounting."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from operational.efficient_context.common import Lease, digest
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import environment, read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.models import resource_guard


def head_source(output: Path, kind: str) -> dict:
    """Require a complete manifest before accepting any training feature array."""
    manifest_path = output / f"{kind}_FEATURE_MANIFEST.json"
    manifest = read(manifest_path)
    if manifest["status"] != "COMPLETE_VERIFIED" or manifest["row_count"] != 88744:
        raise ValueError("Complete full TRAIN40 head feature population required")
    for item in manifest["files"]:
        path = output / item["path"]
        if not path.resolve().is_relative_to(output.resolve()) or digest(path) != item["sha256"]:
            raise ValueError("Head feature content changed")
    return {**manifest, "manifest_sha256": digest(manifest_path)}


def run(output: Path, kind: str, seed: int) -> None:
    """Fit the fixed endpoint; reserve and record each physical optimizer update."""
    from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
    from e_jepa_ttc.reproducibility import seed_everything
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
    from e_jepa_ttc.simplex_t.training import learning_rate
    from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule
    from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase, PairHeadTrainingConfig

    freeze_path = output / "HEADS_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
    if digest(Path(__file__)) != freeze["source_sha256"]:
        raise ValueError("Head engine changed after QA")
    protocol = read(output / "TRAINING_PROTOCOL.json")
    if digest(output / "TRAINING_PROTOCOL.json") != freeze["protocol_sha256"]:
        raise ValueError("Head scientific protocol changed")
    if kind == "PAIR" and seed != 7 or kind == "H8" and seed not in (7, 13, 23):
        raise ValueError("Unregistered head seed")
    source = head_source(output, kind)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(seed, deterministic=True)
    schedule = None
    if kind == "PAIR":
        config = PairHeadTrainingConfig(seed=seed)
        with np.load(output / "PAIR_FEATURES.npz", allow_pickle=False) as stored:
            features = torch.from_numpy(stored["features"])
            target = torch.from_numpy(stored["target_phase"])
            sequences, tokens = stored["sequences"].tolist(), stored["tokens"].tolist()
        PairFeatureBatch(features)
        schedule, schedule_sha = deterministic_sequence_grouped_schedule(
            sequences, tokens, seed=seed, batch_size=32, updates=6840
        )
        model = CachedPairDirectPhase(dropout=config.dropout)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        model_config = {"class": "CachedPairDirectPhase", "dropout": config.dropout}
        optimizer_config = {"class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4}
        recipe = asdict(config)
        limit = 6840
    else:
        from operational.train40_system.history_features import load_source

        queries = load_source(output, source)
        config_h8 = TemporalConfig(feature_count=17, hidden=160, backbone="gru")
        model = TemporalRefiner(config_h8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3, foreach=False)
        model_config = asdict(config_h8)
        optimizer_config = {
            "class": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "foreach": False,
        }
        recipe = protocol["H8"]
        schedule_sha = None
        limit = 2500
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    generator = torch.Generator().manual_seed(seed)
    directory = output / "fits" / f"{kind.lower()}_seed{seed}"
    identity_path = directory / "FIT_IDENTITY.json"
    if not identity_path.exists():
        atomic_json(
            identity_path,
            {"environment": environment(), "start_time": datetime.now(UTC).isoformat()},
        )
    contract = {
        "fit": kind,
        "seed": seed,
        "device": "cpu",
        "dtype": "float32",
        "model_config": model_config,
        "optimizer_config": optimizer_config,
        "training_recipe": recipe,
        "schedule_sha256": schedule_sha,
        "feature_manifest_sha256": source["manifest_sha256"],
        "protocol_sha256": freeze["protocol_sha256"],
        "engineering_freeze_sha256": digest(freeze_path),
        "environment_at_start": read(identity_path),
        "updates_limit": limit,
        "recovery_upper_limit": 2000,
    }
    state = DurableState(directory, contract)
    cursor: dict = {"pending_curve": []}
    if state.path.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    status = "RUNNING"
    try:
        model.train()
        while state.committed < limit:
            allowed, resource = resource_guard(output)
            if not allowed:
                status = "PAUSED_RESOURCE"
                atomic_json(directory / "RESOURCE_PAUSE.json", resource)
                break
            started = time.perf_counter()
            update = state.committed + 1
            optimizer.zero_grad(set_to_none=True)
            if kind == "PAIR":
                assert schedule is not None
                ids = torch.from_numpy(schedule[state.committed])
                prediction = model(PairFeatureBatch(features[ids]))
                loss = torch.nn.functional.smooth_l1_loss(prediction, target[ids])
            else:
                ids = torch.randint(queries.population, (128,), generator=generator)
                x, timing, valid, experts, truth, mass = queries.gather(ids)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(update)
                prediction_h8 = model(x, timing, valid, experts)
                loss = training_loss(prediction_h8, truth, experts, mass, queries.population)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TRAIN40 head loss")
            loss.backward()
            if kind == "H8":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            state.begin_update()
            optimizer.step()
            state.commit_update()
            cursor["pending_curve"].append(
                {
                    "update": state.committed,
                    "loss": float(loss.detach()),
                    "seconds": time.perf_counter() - started,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if state.committed % 100 == 0 or state.committed == limit:
                atomic_json(
                    directory / f"curve_{state.committed:06d}.json",
                    {"updates": cursor["pending_curve"]},
                )
                cursor["pending_curve"] = []
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    cursor,
                    status="COMPLETE" if state.committed == limit else "RUNNING",
                )
                atomic_json(
                    directory / "PROGRESS.json",
                    {
                        "status": "COMPLETE" if state.committed == limit else "RUNNING",
                        "committed_updates": state.committed,
                        "total_updates": limit,
                        "loss": float(loss.detach()),
                        "resources": resource,
                    },
                )
        if state.committed == limit:
            status = "COMPLETE"
    except BaseException as exc:
        status = "PAUSED_FAILURE"
        atomic_json(
            directory / "FAILURE.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "scientific_negative": False,
                "committed_updates": state.committed,
                "durable_updates": state.durable,
            },
        )
        raise
    finally:
        if not state.pending and status != "PAUSED_FAILURE":
            state.save(model, optimizer, scheduler, generator, cursor, status=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--kind", choices=("PAIR", "H8"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.kind, args.seed)
