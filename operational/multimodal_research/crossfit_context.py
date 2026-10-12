"""Fit native temporal contexts on training groups only, then seal held-out predictions."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from e_jepa_ttc.simplex_t.model import training_loss
from operational.multimodal_validation.common import ROOT, RUN, read, sha, write
from operational.rgb_port.train_heads import SealedCache, _lr, _model
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import OUT, accounting, guard, setup, source_hashes
from .crossfit_data import DEST
from .model import frozen_state


def context_loss(
    model: torch.nn.Module, values: dict, ids: torch.Tensor, population: int
) -> torch.Tensor:
    experts = values["expert_phase"][ids]
    output = model(values["features"][ids], values["timing"][ids], values["valid"][ids], experts)
    return training_loss(
        output, values["target_phase"][ids], experts, values["mass"][ids], population
    )


def binding(fold: int, modality: str) -> dict:
    folder = DEST / f"fold{fold}" / modality
    seal = read(folder / "SEALED.json")
    assert sha(folder / "train.npz") == seal["train_sha256"]
    assert sha(folder / "heldout.npz") == seal["heldout_sha256"]
    assert not set(seal["train_groups"]) & set(seal["heldout_groups"])
    fit_id = f"GROUP_CROSSFIT_F{fold}_{modality}_S7"
    identity = dict(
        fit_id=fit_id,
        fold=fold,
        modality=modality,
        seed=7,
        fold_seal_sha256=sha(folder / "SEALED.json"),
        protocol_sha256=sha(OUT / "GROUP_CROSSFIT_PREREGISTRATION.json"),
        parent_freeze_sha256=sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
        source_sha256=sha(__file__),
        scientific_sources=source_hashes(),
        native_sources={
            name: sha(ROOT / name)
            for name in (
                "src/e_jepa_ttc/simplex_t/model.py",
                "src/e_jepa_ttc/simplex_t/phase.py",
                "operational/rgb_port/train_heads.py",
                "operational/rgb_port/train_producers.py",
            )
        },
        device="cuda_FP32_noTF32",
        architecture="Native160 two-layer masked GRU temporal refiner",
        target_role="H training groups only",
        normalization="Training unique observations only",
    )
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return identity


def fit(fold: int, modality: str) -> None:
    guard()
    identity = binding(fold, modality)
    folder = DEST / f"fold{fold}" / modality
    receipt_path = OUT / "fits" / identity["fit_id"] / "CHECKPOINT_RECEIPT.json"
    predictions_seal = folder / "OOF_PREDICTIONS_SEALED.json"
    if receipt_path.exists() and predictions_seal.exists():
        endpoint = read(receipt_path)
        sealed = read(predictions_seal)
        assert endpoint["status"] == "COMPLETE" and endpoint["completed_updates"] == 2500
        assert endpoint["identity_sha256"] == identity["identity_sha256"]
        assert sha(receipt_path.parent / "checkpoint_last.pt") == endpoint["checkpoint_sha256"]
        assert sealed["checkpoint_sha256"] == endpoint["checkpoint_sha256"]
        assert sealed["fold_seal_sha256"] == sha(folder / "SEALED.json")
        assert sealed["sha256"] == sha(folder / "OOF_PREDICTIONS.npz")
        return
    native_id = "E_CTX_MATCHED" if modality == "E" else "R_CTX"
    cache = SealedCache(folder / "train.npz", native_id)
    seal = read(folder / "SEALED.json")
    assert set(cache.values["group_id"]) == set(seal["train_groups"])
    tensors = {
        key: torch.from_numpy(cache.values[key]).cuda()
        for key in ("features", "timing", "valid", "expert_phase", "target_phase", "mass")
    }
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    random.seed(7)
    model = _model(native_id).float().cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.001, foreach=False, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr(native_id, min(step + 1, 2500)) / 3e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    state = ProducerCheckpoint(OUT / "fits" / identity["fit_id"], identity, 2500)
    cursor = {"losses": []}
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    try:
        while state.completed < 2500:
            if state.completed % 25 == 0:
                guard(prospective_updates=25)
            ids = torch.randint(cache.population, (128,), generator=generator).cuda()
            optimizer.param_groups[0]["lr"] = _lr(native_id, state.completed + 1)
            optimizer.zero_grad(set_to_none=True)
            loss = context_loss(model, tensors, ids, cache.population)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native context loss")
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
                print(
                    "GROUP_CROSSFIT",
                    fold,
                    modality,
                    state.completed,
                    cursor["losses"][-1],
                    flush=True,
                )
    except (RuntimeError, MemoryError):
        if not state.pending:
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
        raise
    # Only after the complete endpoint do held-out physical inputs reach inference.
    endpoint = read(OUT / "fits" / identity["fit_id"] / "CHECKPOINT_RECEIPT.json")
    assert endpoint["status"] == "COMPLETE" and endpoint["completed_updates"] == 2500
    heldout_path = folder / "heldout.npz"
    assert sha(heldout_path) == seal["heldout_sha256"]
    model.eval().requires_grad_(False)
    with np.load(heldout_path, allow_pickle=False) as z:
        assert "target_phase" not in z and "mass" not in z
        assert set(z["group_id"]) == set(seal["heldout_groups"])
        blocks, widths = [], []
        for start in range(0, len(z["sample_token"]), 128):
            guard()
            values = [
                torch.from_numpy(z[key][start : start + 128]).cuda()
                for key in ("features", "timing", "valid", "expert_phase")
            ]
            block, width, _ = frozen_state(model, *values)
            block[:, :160] = (
                0  # hidden coordinates have no common basis across independently fitted folds
            )
            blocks.append(block.cpu().numpy())
            widths.append(width.cpu().numpy())
        result = dict(
            block=np.concatenate(blocks),
            width_logits=np.concatenate(widths),
            global_index=z["global_index"],
            sample_token=z["sample_token"],
            group_id=z["group_id"],
        )
    path = folder / "OOF_PREDICTIONS.npz"
    from .crossfit_data import save

    digest = save(path, result)
    write(
        folder / "OOF_PREDICTIONS_SEALED.json",
        dict(
            sha256=digest,
            checkpoint_sha256=endpoint["checkpoint_sha256"],
            fold_seal_sha256=sha(folder / "SEALED.json"),
            population=len(result["sample_token"]),
            hidden_coordinates_zero=True,
            heldout_targets_read=False,
            source_sha256=sha(__file__),
            supervised_groups_disjoint=True,
        ),
    )
    accounting()


def main() -> None:
    setup()
    torch.cuda.set_per_process_memory_fraction(
        1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    assert read(DEST / "FOLDS_SEALED.json")["population"] == 18422
    for fold in range(4):
        for modality in ("E", "R"):
            fit(fold, modality)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
