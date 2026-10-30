"""Fixed-recipe F1 refitting with explicit resubstitution metrics and saved optimizer."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha, write_receipt
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt import balanced_mass, epoch_count


def load_head(path: Path, expected_sha256: str) -> PhaseOnlyFinal:
    """Verify bytes and strict-load a trained head on CPU."""
    if file_sha(path) != expected_sha256:
        raise ValueError("Adapted checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != "COMPLETE" or payload.get("model_class") != "PhaseOnlyFinal":
        raise ValueError("Incomplete or unsupported adapted checkpoint")
    model = PhaseOnlyFinal()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def evaluate(model: PhaseOnlyFinal, tensors: dict[str, torch.Tensor], domains: np.ndarray) -> dict:
    """Compute training-fit metrics only; no test or selection interpretation."""
    model.eval()
    with torch.inference_mode():
        output = model(tensors["features"], tensors["rgb_width_logits"])
        truth = tensors["target_phase"]
        loss = fusion_loss(output, truth, tensors["mass"], len(truth))
        predicted = output["ttc"].cpu().numpy()
        target = phase_to_ttc(truth).cpu().numpy()
        metrics = {}
        for domain in np.unique(domains):
            mask = domains == domain
            error = np.abs(predicted[mask] - target[mask])
            metrics[str(domain)] = {
                "rows": int(mask.sum()),
                "mae_seconds": float(error.mean()),
                "relative_absolute_error_pct": float((100 * error / np.abs(target[mask])).mean()),
                "nonfinite": int((~np.isfinite(predicted[mask])).sum()),
            }
        return {"loss": float(loss.cpu()), "scope": "training_resubstitution", "domains": metrics}


def fit_head(
    initial_state: dict[str, torch.Tensor],
    values: dict[str, np.ndarray],
    recipe: dict,
    output: Path,
    *,
    device: str,
    binding: dict,
) -> dict[str, Any]:
    """Train complete epochs with fixed seed/budget, then save the last endpoint."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; preserve earlier attempts")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(recipe["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    model = PhaseOnlyFinal().to(device)
    model.load_state_dict(initial_state, strict=True)
    tensors = {
        name: torch.from_numpy(values[name]).to(device)
        for name in ("features", "rgb_width_logits", "target_phase")
    }
    mass = balanced_mass(values["domain"], values["group_id"])
    tensors["mass"] = torch.from_numpy(mass).to(device)
    n = len(mass)
    epochs = epoch_count(
        n,
        batch_size=recipe["batch_size"],
        max_updates=recipe["max_updates"],
        max_epochs=recipe["max_epochs"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    started = datetime.now(UTC).isoformat()
    baseline = evaluate(model, tensors, values["domain"])
    write_receipt(
        output / "START.json",
        {
            "started": started,
            "recipe": recipe,
            "binding": binding,
            "epochs": epochs,
            "population": n,
            "baseline": baseline,
        },
    )
    history, updates = [], 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        seen = 0
        for start in range(0, n, recipe["batch_size"]):
            indices = torch.as_tensor(order[start : start + recipe["batch_size"]], device=device)
            prediction = model(tensors["features"][indices], tensors["rgb_width_logits"][indices])
            loss = fusion_loss(
                prediction, tensors["target_phase"][indices], tensors["mass"][indices], n
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), recipe["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            seen += len(indices)
        if seen != n:
            raise ValueError("Epoch did not consume all training examples")
        metrics = evaluate(model, tensors, values["domain"])
        history.append({"epoch": epoch + 1, "updates": updates, "rows_seen": seen, **metrics})
        write_receipt(output / f"epoch_{epoch + 1:03d}.json", history[-1])
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} updates={updates} loss={metrics['loss']:.6f}",
            flush=True,
        )
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "status": "COMPLETE",
            "model_class": "PhaseOnlyFinal",
            "model_state_dict": {
                key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "recipe": recipe,
            "binding": binding,
            "epochs": epochs,
            "updates": updates,
            "history": history,
            "baseline": baseline,
            "started": started,
            "ended": datetime.now(UTC).isoformat(),
            "selection": "last_endpoint_fixed_budget_no_test_selection",
        },
        checkpoint,
    )
    report = {
        "seed": seed,
        "epochs": epochs,
        "updates": updates,
        "history": history,
        "baseline": baseline,
        "checkpoint_sha256": file_sha(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "rows": n,
    }
    reloaded = load_head(checkpoint, report["checkpoint_sha256"]).to(device)
    check = evaluate(reloaded, tensors, values["domain"])
    if check != {
        k: v for k, v in history[-1].items() if k not in {"epoch", "updates", "rows_seen"}
    }:
        raise ValueError("Reloaded endpoint metrics differ")
    write_receipt(output / "COMPLETE.json", report)
    return report
