"""Closed CPU FP32 head fits using native model, schedule and durable checkpoints."""

from __future__ import annotations

import hashlib
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
from operational.range_transfer.admission import revalidate
from operational.range_transfer.cache import HeadCache
from operational.range_transfer.diagnose import csv_write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import code_binding, digest, fit_id, read, resources, sha, utc, write
from .ledger import accounting


def initialize(seed: int) -> tuple[Any, Any, Any, torch.Generator]:
    """Exact native temporal initialization/AdamW/scheduler, parameterized only by seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TemporalRefiner(
        TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    ).float()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr("E_CTX_MATCHED", min(step + 1, 2500)) / 3e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return model, optimizer, scheduler, generator


def loss_on_batch(
    model: torch.nn.Module, cache: HeadCache, ids: torch.Tensor, mass_id: int, loss_id: int
) -> torch.Tensor:
    """Change only the global mass vector and optional fixed relative term."""
    device = torch.device("cpu")
    output = cache.forward(model, ids)
    if loss_id != 0 or mass_id not in (0, 1):
        raise ValueError("V15 S admits M0/M1 L0 only")
    # p is normalized in float64 once over H, NOT within the sampled batch.
    probability = torch.from_numpy(cache.plans[mass_id].astype(np.float64))
    probability /= probability.sum()
    ratio = torch.from_numpy(cache.plans[mass_id][ids.numpy()].copy()).double() / probability[ids]
    return training_loss(
        output,
        cache.tensor("target_phase", ids, device),
        cache.tensor("expert_phase", ids, device),
        ratio.float(),
        1,
    )


def fit(
    cache: HeadCache,
    run: Path,
    binding: dict[str, Any],
    *,
    mass_id: int,
    loss_id: int,
    seed: int,
    max_updates_this_call: int | None = None,
    technical_name: str | None = None,
    technical_endpoint: int = 10,
) -> dict[str, Any]:
    """Resume fixed-endpoint CPU training; technical mode is synthetic QA only."""
    technical = technical_name is not None
    name = technical_name if technical else fit_id(mass_id, loss_id, seed)
    if technical:
        if binding.get("status") != "SYNTHETIC_ONLY" or technical_endpoint != 10:
            raise ValueError("Technical tests must use the fixed 10-update synthetic fixture")
    elif binding.get("status") != "ADMITTED":
        raise ValueError("Scientific fits require admitted final parents and H cache")
    if max_updates_this_call is not None and max_updates_this_call < 1:
        raise ValueError("max_updates_this_call must be positive")
    if not technical and seed != 7:
        selection = read(run / "SELECTION_FREEZE.json")
        if selection.get("family") != "S" or mass_id not in (0, 1):
            raise ValueError("Replication recipe is not the frozen candidate/control")
    endpoint = 10 if technical else 2500
    directory = run / ("qa" if technical else "fits") / str(name)
    identity = {
        "fit_id": name,
        "seed": seed,
        "mass_id": mass_id,
        "loss_id": loss_id,
        "binding": binding,
        "source_code": code_binding(),
        "endpoint": endpoint,
        "torch": str(torch.__version__),
        "device": "cpu",
        "precision": "float32",
        "batch": 128,
        "hidden": 160,
        "technical": technical,
    }
    identity["identity_sha256"] = digest(identity)
    probability = torch.from_numpy(cache.plans[mass_id].astype(np.float64))
    probability /= probability.sum()
    identity["sampling"] = {
        "method": "multinomial_replacement",
        "dtype": "float64",
        "p_sha256": hashlib.sha256(probability.numpy().tobytes()).hexdigest(),
        "estimator": "w/p",
        "mass_sum": float(cache.plans[mass_id].astype(np.float64).sum()),
    }
    identity["identity_sha256"] = digest(identity)
    observation = resources(run, starting=True)
    write(run / "LAST_RESOURCE_CHECK.json", observation)
    if not observation["allowed"]:
        return {"status": "PAUSED_RESOURCE", "fit_id": name, "completed_updates": 0}
    if not technical:
        revalidate(binding)
    model, optimizer, scheduler, generator = initialize(seed)
    state = ProducerCheckpoint(directory, identity, endpoint)
    cursor: dict[str, Any] = {
        "losses": [],
        "draw_hashes": [],
        "elapsed_fit_s": 0.0,
        "exposure": [0] * cache.population,
    }
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    accounting(run)
    if state.completed == endpoint:
        return read(state.receipt)  # No reserialization of an existing endpoint.
    executed, started, status = 0, time.perf_counter(), "IN_PROGRESS"
    try:
        while state.completed < endpoint:
            if state.completed % 25 == 0:
                observation = resources(run)
                write(run / "LAST_RESOURCE_CHECK.json", observation)
            if not observation["allowed"] or (run / "PAUSE").exists():
                status = "PAUSED_RESOURCE"
                break
            ids = torch.multinomial(probability, 128, replacement=True, generator=generator)
            optimizer.param_groups[0]["lr"] = _lr("E_CTX_MATCHED", state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on_batch(model, cache, ids, mass_id, loss_id)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; no rows may be silently excluded")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            accounting(run, prospective=1, technical=technical)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            executed += 1
            exposure = np.asarray(cursor["exposure"], dtype=np.int64)
            np.add.at(exposure, ids.numpy(), 1)
            cursor["exposure"] = exposure.tolist()
            cursor["losses"].append(
                {
                    "update": state.completed,
                    "loss": float(loss.detach()),
                    "gradient_l2_before_clip": float(grad),
                    "lr": _lr("E_CTX_MATCHED", state.completed),
                }
            )
            cursor["draw_hashes"].append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
            stop = max_updates_this_call is not None and executed >= max_updates_this_call
            if state.completed == endpoint:
                status = "COMPLETE"
            if state.completed % 100 == 0 or status == "COMPLETE" or stop:
                if not technical:
                    revalidate(binding)
                cursor["elapsed_fit_s"] += time.perf_counter() - started
                state.save(model, optimizer, scheduler, generator, cursor, status=status)
                started = time.perf_counter()
                csv_write(directory / "CURVE.csv", cursor["losses"])
            if stop:
                break
        if state.completed != state.durable:
            cursor["elapsed_fit_s"] += time.perf_counter() - started
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
    except BaseException as error:
        # Never promote a possibly half-applied optimizer state. Journal conservatively
        # charges it and restore returns to the previous complete atomic pointer.
        write(
            directory / f"FAILURE_{time.time_ns()}.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "utc": utc(),
                "completed": state.completed,
                "durable": state.durable,
                "pending": state.pending,
            },
        )
        raise
    accounting(run)
    receipt = read(state.receipt)
    write(
        directory / "RUN_PROVENANCE.json",
        {
            "fit_id": name,
            "seed": seed,
            "status": receipt["status"],
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "precision": "float32",
            "scope": "synthetic_QA" if technical else "H_head_fit",
            "identity_sha256": identity["identity_sha256"],
            "checkpoint_sha256": sha(state.path),
            "draw_sequence_sha256": digest(cursor["draw_hashes"]),
            "elapsed_fit_s": cursor["elapsed_fit_s"],
            "updated_utc": utc(),
        },
    )
    return receipt
