"""Fixed original QA objective and train-only preprocessing for support/capacity controls."""

import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_native_support.model import NativeSupportRouter
from operational.phase_robust.model import fit_normalization

SEEDS = (7, 13, 21)
EPOCHS = 300
METRICS = ("macro_record_mae_seconds", "macro_record_relative_error_pct")


def qa_corruption(native: torch.Tensor) -> torch.Tensor:
    """Exactly the original QA random draw order; support remains unchanged."""
    corrupted = native.clone()
    chosen = torch.rand(len(native)) < 0.25
    indices = chosen.nonzero().flatten()
    experts = torch.randint(0, 3, (len(indices),))
    factors = torch.tensor([-3.0, -1.0, 0.25, 4.0])[torch.randint(0, 4, (len(indices),))]
    corrupted[indices, experts] *= factors
    return corrupted


def fit_preprocessing(
    model: NativeSupportRouter,
    native: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    support: torch.Tensor,
) -> None:
    """Call only with admitted clean fitting rows; original33 remain unmasked."""
    x = model.inputs(native, geometry, quality, support).detach().numpy()
    center, scale = fit_normalization(x)
    model.center.copy_(torch.from_numpy(center))
    model.scale.copy_(torch.from_numpy(scale))
    if model.use_support:
        variance = support.detach().numpy().astype(np.float64).var(axis=0)
        model.support_mask.copy_(torch.from_numpy((variance >= 1e-12).astype(np.float32)))


def fit(
    arrays: dict, mask: np.ndarray, mode: str, seed: int, destination: Path
) -> NativeSupportRouter:
    """Run the preregistered300 QA updates; preserve every loss and failure."""
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = NativeSupportRouter(mode)
    expected = 868 if mode == "F2" else 878
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError("Model capacity differs from preregistration")
    valid = (
        mask
        & np.isfinite(arrays["target"])
        & (np.isfinite(arrays["native"]).any(1) | np.isfinite(arrays["geometry"]))
    )
    native, geo, quality, support, target = [
        torch.from_numpy(arrays[k][valid])
        for k in ("native", "geometry", "quality", "support", "target")
    ]
    if not len(target):
        raise ValueError("No admitted training rows")
    families = arrays["family"][valid]
    weights = np.array([1 / np.sum(families == f) for f in families], np.float32)
    mass = torch.from_numpy(weights / weights.sum())
    fit_preprocessing(model, native, geo, quality, support)
    seconds_scale = target.abs().median().clamp(min=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    binding = {
        "mode": mode,
        "seed": seed,
        "epochs": EPOCHS,
        "parameters": expected,
        "fit_groups": sorted(set(arrays["group"][mask])),
        "rows": len(target),
        "seconds_scale_train_only": float(seconds_scale),
        "active_native_mask": model.support_mask.tolist() if model.use_support else None,
        "support_mask_variance_threshold": 1e-12 if model.use_support else None,
        "normalization": "admitted clean train only; original33 unmasked",
    }
    atomic_receipt(destination / "FIT_CONFIG.json", binding)
    started, history = time.perf_counter(), []

    def objective(prediction: torch.Tensor) -> torch.Tensor:
        error = (prediction - target).abs()
        return (mass * (error / target.abs().clamp(min=0.2) + error / seconds_scale)).sum()

    try:
        for epoch in range(EPOCHS):
            clean = objective(model(native, geo, quality, support))
            perturbed = objective(model(qa_corruption(native), geo, quality, support))
            loss = 0.5 * clean + 0.5 * perturbed
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite QA training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "clean": float(clean.detach()),
                    "corrupted": float(perturbed.detach()),
                }
            )
    except Exception as exc:
        torch.save(
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            destination / "FAILED.pt",
        )
        atomic_receipt(
            destination / "FAILED.json", {**binding, "error": repr(exc), "loss": history}
        )
        raise
    torch.save(
        {"state_dict": model.state_dict(), "mode": mode, "seed": seed},
        destination / "checkpoint.pt",
    )
    atomic_receipt(
        destination / "COMPLETE.json",
        {
            **binding,
            "loss": history,
            "seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_sha(destination / "checkpoint.pt"),
        },
    )
    return model.eval()


def select_candidate(reports: dict) -> dict:
    """Use only the four primary clean metrics; tie-break F2, without causal attribution."""
    candidates = {}
    for mode in ("F2", "Fcap"):
        full = all(reports[mode][domain]["coverage"] == 1.0 for domain in ("all", "road", "slider"))
        gates, ratios = {}, []
        for domain in ("road", "slider"):
            for key in METRICS:
                value, base = reports[mode][domain][key], reports["IQ"][domain][key]
                if base is None or not np.isfinite(base) or base <= 0:
                    raise ValueError("Primary IQ denominator must be finite and positive")
                ok = value is not None and np.isfinite(value)
                gates[f"{domain}_{key}"] = bool(ok and value <= base)
                ratios.append(value / base if ok else float("inf"))
        candidates[mode] = {
            "full_coverage": full,
            "gates": gates,
            "passes": full and all(gates.values()),
            "mean_ratio_to_IQ": float(np.mean(ratios)) if np.isfinite(ratios).all() else None,
        }
    passed = [m for m in ("F2", "Fcap") if candidates[m]["passes"]]
    selected = (
        min(passed, key=lambda m: (candidates[m]["mean_ratio_to_IQ"], m != "F2"))
        if passed
        else None
    )
    differences = [
        reports["F2"][d][k] - reports["Fcap"][d][k]
        for d in ("road", "slider")
        for k in METRICS
        if reports["F2"][d][k] is not None and reports["Fcap"][d][k] is not None
    ]
    dominates = (
        len(differences) == 4
        and all(np.isfinite(differences))
        and all(x <= 0 for x in differences)
        and any(x < 0 for x in differences)
        and all(candidates[m]["full_coverage"] for m in candidates)
    )
    return {
        "selected": selected,
        "candidates": candidates,
        "support_dominates_capacity": bool(dominates),
        "support_attribution": "association consistent with support benefit, not causal proof"
        if dominates
        else "not established",
        "promoted": False,
        "secondary_used_for_selection": False,
    }
