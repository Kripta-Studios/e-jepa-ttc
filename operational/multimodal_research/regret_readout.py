"""Matched phase-unit readouts with a registered RGB-relative excess-risk loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.multimodal_validation.common import read, sha, write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import OUT, accounting, source_hashes
from .fusion_readout import guard, setup
from .phase_units import PhaseUnitAnchor
from .regret_loss import rgb_regret

STUDIES = {"RGB_REGRET": (7,)}


def contract(study: str) -> dict:
    assert study == "RGB_REGRET"
    spec = read(OUT / "RGB_REGRET_PREREGISTRATION.json")
    value = dict(
        **spec,
        preregistration_sha256=sha(OUT / "RGB_REGRET_PREREGISTRATION.json"),
        original_protocol_sha256=sha(OUT / "PROTOCOL.json"),
        frozen_scientific_sources=source_hashes(),
        phase_units_source_sha256=sha(Path(__file__).with_name("phase_units.py")),
        regret_source_sha256=sha(Path(__file__).with_name("regret_loss.py")),
        CPU_guard_source_sha256=sha(Path(__file__).with_name("fusion_readout.py")),
        trainer_source_sha256=sha(__file__),
        validation="Exposed V development; original gates; no external selection",
    )
    path = OUT / "studies" / study / "PROTOCOL.json"
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    return value


def fit(study: str, mode: str, seed: int) -> None:
    config = contract(study)
    destination = OUT / "studies" / study
    guard()
    cache = OUT / "cache/H/states.npz"
    assert sha(cache) == read(cache.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z:
        assert set(z["role"].tolist()) == {"H"}
        values = {
            k: torch.from_numpy(z[k]).to("cpu")
            for k in ("features", "rgb_width_logits", "target_phase", "mass")
        }
    identity: dict = dict(
        fit_id=f"R_ANCHOR_{study}_{mode}_S{seed}",
        cache_sha256=sha(cache),
        protocol_sha256=sha(destination / "PROTOCOL.json"),
        sources=source_hashes(),
        study_source_sha256=sha(__file__),
        device="cpu_FP32_one_thread",
        seed=seed,
    )
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = PhaseUnitAnchor(mode).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.001, foreach=False, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = ProducerCheckpoint(OUT / "fits" / identity["fit_id"], identity, 2500)
    cursor = {"losses": []}
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    population = len(values["features"])
    try:
        while state.completed < config["updates_per_fit"]:
            if state.completed % 25 == 0:
                guard(prospective_updates=25)
            ids = torch.randint(population, (128,), generator=generator).to("cpu")
            optimizer.param_groups[0]["lr"] = _lr("F_TRUE", state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            output = model(values["features"][ids], values["rgb_width_logits"][ids])
            loss = fusion_loss(output, values["target_phase"][ids], values["mass"][ids], population)
            anchor = emitted_phase(values["features"][ids, 160])
            loss = loss + rgb_regret(
                output["point_phase"],
                values["target_phase"][ids],
                anchor,
                values["mass"][ids],
                population,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite phase-unit loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            cursor["losses"].append(float(loss.detach().cpu()))
            if state.completed % 250 == 0:
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    cursor,
                    status="COMPLETE" if state.completed == 2500 else "RUNNING",
                )
                accounting()
                print(study, seed, mode, state.completed, cursor["losses"][-1], flush=True)
    except (RuntimeError, MemoryError):
        if not state.pending:
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
        raise
    accounting()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "train"))
    args = parser.parse_args()
    for study in STUDIES:
        contract(study)
    if args.action == "register":
        return
    setup()
    for study, seeds in STUDIES.items():
        for seed in seeds:
            for mode in ("EVENT", "ZERO"):
                fit(study, mode, seed)


if __name__ == "__main__":
    main()
