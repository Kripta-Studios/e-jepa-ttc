"""Two predeclared frozen-state readouts; H-only optimization and durable native state."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from operational.multimodal_validation.common import read, sha, write
from operational.rgb_port.train_heads import _lr
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import OUT, accounting, authorize, guard, setup, source_hashes
from .model import RGBAnchor


def protocol() -> dict:
    value = dict(
        name="R_ANCHOR_20261012",
        arms=["EVENT", "ZERO"],
        updates_each=2500,
        seed=7,
        optimizer="AdamW",
        lr="native head warmup100 cosine2500 3e-4 to 3e-5",
        weight_decay=0.001,
        batch_size=128,
        sampler="uniform randint seed7 global H mass",
        loss="native fusion_loss: emitted-phase L1 + 0.1 pinball, no cost auxiliary",
        input="340 coordinates; frozen actual R_CTX/E_CTX location input states",
        zero_control="event hidden160/raw/phase/experts3 all zero; same clocks/count retained",
        anchor="R_CTX raw_location; linear residual 0.03, no additional residual cap",
        quantile="native R_CTX width logits plus zero-initialized readout correction",
        normalization="frozen PHASE17 parents; no V-fit normalizer",
        stages="new supervised H stage on H-trained contexts, NOT OOF",
        validation="exposed V; scientific recipe frozen before new endpoint predictions",
        gates=dict(
            aggregate_vs_R_CTX_max_ratio=0.97,
            mae_vs_R_CTX_max_ratio=1.0,
            crucial_vs_R_CTX_max_ratio=1.02,
            other_bands_vs_R_CTX_max_ratio=1.02,
            aggregate_vs_F_TRUE_max_ratio=1.0,
            mae_vs_F_TRUE_max_ratio=1.0,
            sign_error_vs_R_CTX_max_increase=0.0,
            paired_group_CI_EVENT_minus_ZERO_upper=0.0,
        ),
        external="never used for training/normalization/gate selection; descriptive after seal",
        initial_updates_authorized=5000,
        authorization=authorize(),
    )
    path = OUT / "PROTOCOL.json"
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    return value


def fit(mode: str, stop_after: int | None) -> None:
    config = protocol()
    guard()
    cache_path = OUT / "cache/H/states.npz"
    seal = read(cache_path.parent / "SEALED.json")
    assert sha(cache_path) == seal["states_sha256"] and seal["role"] == "H"
    with np.load(cache_path, allow_pickle=False) as z:
        assert set(map(str, z["role"])) == {"H"}
        values = {
            key: torch.from_numpy(z[key]).cuda()
            for key in ("features", "rgb_width_logits", "target_phase", "mass")
        }
    identity: dict = dict(
        fit_id=f"R_ANCHOR_{mode}_S7",
        cache_sha256=sha(cache_path),
        protocol_sha256=sha(OUT / "PROTOCOL.json"),
        sources=source_hashes(),
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
    model = RGBAnchor(mode).cuda()
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
    if state.completed == 2500:
        return
    population = len(values["features"])
    start = state.completed
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
                raise FloatingPointError("Nonfinite readout loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            cursor["losses"].append(float(loss.detach().cpu()))
            if state.completed % 250 == 0 or state.completed == 2500:
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    cursor,
                    status="COMPLETE" if state.completed == 2500 else "RUNNING",
                )
                accounting()
                print(mode, state.completed, "loss", cursor["losses"][-1], flush=True)
            if stop_after is not None and state.completed - start >= stop_after:
                state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_REQUEST")
                break
    except (RuntimeError, MemoryError):
        if not state.pending:
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
        raise
    write(
        OUT / "LAST_PROGRESS.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            fit=identity["fit_id"],
            completed=state.completed,
            accounting=accounting(),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("EVENT", "ZERO", "both"), default="both")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    setup()
    for mode in ("EVENT", "ZERO") if args.mode == "both" else (args.mode,):
        fit(mode, args.stop_after)


if __name__ == "__main__":
    main()
