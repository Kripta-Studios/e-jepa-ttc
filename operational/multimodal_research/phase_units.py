"""Matched phase-unit study, motivated by H-only input-scale attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from operational.multimodal_validation.common import read, sha, write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import OUT, accounting, guard, setup, source_hashes
from .model import RGBAnchor

STUDY = OUT / "studies/PHASE_UNITS"


class PhaseUnitAnchor(RGBAnchor):
    def inputs(self, features: Tensor) -> Tensor:
        values = super().inputs(features).clone()
        for start in (160, 330):
            values[:, start : start + 5] /= 0.03
        return values


def contract() -> dict:
    base = read(OUT / "PROTOCOL.json")
    value = dict(
        study="PHASE_UNITS",
        arms=["EVENT", "ZERO"],
        seed=7,
        updates_each=2500,
        hypothesis="Small H phase contributions relative to hidden states",
        motivation_sha256=sha(OUT / "H_LINEAR_CONTRIBUTIONS_2425.json"),
        change="Divide five explicit phase coordinates per modality by native 0.03 unit inside MLP",
        unchanged="Raw RGB anchor, residual, hidden states, clocks, loss, optimizer",
        recipe=base,
        base_protocol_sha256=sha(OUT / "PROTOCOL.json"),
        role="New supervised H stage; not OOF; exposed V development; no external selection",
        optimizer_updates_authorized_for_study=5000,
    )
    path = STUDY / "PROTOCOL.json"
    if path.exists():
        assert read(path) == value
    else:
        assert not (OUT / "V_RESULTS.json").exists(), "Register this H-motivated study before new V"
        write(path, value)
    return value


def fit(mode: str) -> None:
    config = contract()
    guard()
    cache = OUT / "cache/H/states.npz"
    assert sha(cache) == read(cache.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z:
        assert set(z["role"].tolist()) == {"H"}
        values = {
            k: torch.from_numpy(z[k]).cuda()
            for k in ("features", "rgb_width_logits", "target_phase", "mass")
        }
    identity: dict = dict(
        fit_id=f"R_ANCHOR_UNITS_{mode}_S7",
        cache_sha256=sha(cache),
        protocol_sha256=sha(STUDY / "PROTOCOL.json"),
        sources=source_hashes(),
        study_source_sha256=sha(__file__),
        device="cuda_FP32_noTF32",
        seed=7,
    )
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    random.seed(7)
    model = PhaseUnitAnchor(mode).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.001, foreach=False, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    state = ProducerCheckpoint(OUT / "fits" / identity["fit_id"], identity, 2500)
    cursor = {"losses": []}
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    population = len(values["features"])
    try:
        while state.completed < config["updates_each"]:
            if state.completed % 25 == 0:
                guard(prospective_updates=25)
            ids = torch.randint(population, (128,), generator=generator).cuda()
            optimizer.param_groups[0]["lr"] = _lr("F_TRUE", state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            output = model(values["features"][ids], values["rgb_width_logits"][ids])
            loss = fusion_loss(output, values["target_phase"][ids], values["mass"][ids], population)
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
                print(mode, state.completed, cursor["losses"][-1], flush=True)
    except (RuntimeError, MemoryError):
        if not state.pending:
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
        raise
    accounting()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "train"))
    args = parser.parse_args()
    if args.action == "register":
        contract()
        print("H-motivated study registered before new V", flush=True)
        return
    setup()
    assert (OUT / "V_RESULTS.json").exists(), "Close the original matched pair first"
    for mode in ("EVENT", "ZERO"):
        fit(mode)


if __name__ == "__main__":
    main()
