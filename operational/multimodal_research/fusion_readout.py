"""Matched frozen fusion-state versus zero-fusion-state readouts on H."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from operational.multimodal_validation.common import read, sha, write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import OUT, accounting, source_hashes
from .common import guard as original_guard
from .fusion_anchor import FusionStateAnchor


def guard(*, prospective_updates: int = 0) -> None:
    assert not torch.cuda.is_initialized(), "This registered study must remain CPU-only"
    try:
        original_guard(prospective_updates=prospective_updates)
    except RuntimeError as error:
        if not str(error).startswith("Live research GPU owner must finish:"):
            raise
        # Native budget/deadline/RAM/disk checks precede the GPU owner check.
        # CPU-only fitting does not acquire or alter that owner's GPU resources.


def setup() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    guard()


STUDIES = {"FUSION_STATE": (7,)}


def contract(study: str) -> dict:
    assert study == "FUSION_STATE"
    spec = read(OUT / "FUSION_STATE_PREREGISTRATION.json")
    value = dict(
        **spec,
        preregistration_sha256=sha(OUT / "FUSION_STATE_PREREGISTRATION.json"),
        original_protocol_sha256=sha(OUT / "PROTOCOL.json"),
        frozen_scientific_sources=source_hashes(),
        phase_units_source_sha256=sha(Path(__file__).with_name("phase_units.py")),
        fusion_model_source_sha256=sha(Path(__file__).with_name("fusion_anchor.py")),
        trainer_source_sha256=sha(__file__),
        execution=read(OUT / "FUSION_STATE_EXECUTION.json"),
        fusion_H_binding_sha256=sha(OUT / "fusion_states/H/BINDING.json"),
        validation="Exposed V development; fixed original gates; no external selection",
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
    fusion_cache = OUT / "fusion_states/H/states.npz"
    assert sha(fusion_cache) == read(fusion_cache.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z, np.load(fusion_cache, allow_pickle=False) as f:
        assert np.array_equal(z["sample_token"], f["sample_token"])
        values["features"] = torch.cat(
            [values["features"], torch.from_numpy(f["features"]).to("cpu")], -1
        )
    identity: dict = dict(
        fit_id=f"R_ANCHOR_{study}_{mode}_S{seed}",
        cache_sha256=sha(cache),
        fusion_cache_sha256=sha(fusion_cache),
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
    model = FusionStateAnchor(mode).to("cpu")
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
            for mode in ("FUSED", "ZERO_FUSED"):
                fit(study, mode, seed)


if __name__ == "__main__":
    main()
