"""Preregistered interaction of explicit support and actual native-context restarts."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_native_support.model import NativeSupportRouter
from operational.phase_native_support.training import EPOCHS
from operational.phase_native_support.training import select_candidate as select_base
from operational.phase_robust.model import fit_normalization
from scripts.train_phase_native_context_fusion import admitted, augmentation_weights, fold_masks


def prepare_fold(clean: dict, aug: dict, indices: np.ndarray, family: str) -> dict:
    """Admit training examples before fitting any statistic or assigning variant mass."""
    train, test, aug_train, aug_test = fold_masks(clean, aug, indices, family)
    cmask, amask = train & admitted(clean), aug_train & admitted(aug)
    cf, af = clean["family"][cmask], aug["family"][amask]
    if not len(cf) or set(cf) != set(af):
        raise ValueError("Every training family requires admitted clean and restart rows")
    cw = np.array([1 / np.sum(cf == f) for f in cf], np.float32)
    return {
        "train": train,
        "test": test,
        "aug_test": aug_test,
        "clean_mask": cmask,
        "aug_mask": amask,
        "clean_mass": cw / cw.sum(),
        "aug_mass": augmentation_weights(aug["token"][amask], af),
    }


def fit_preprocessing(
    model: NativeSupportRouter, clean: dict, aug: dict, cmask: np.ndarray, amask: np.ndarray
) -> None:
    """Fit original33 on clean train, added10 on admitted clean+restart train union."""
    base = model.base_inputs(
        *(torch.from_numpy(clean[k][cmask]) for k in ("native", "geometry", "quality"))
    ).numpy()
    center, scale = fit_normalization(base)
    model.center[:33].copy_(torch.from_numpy(center))
    model.scale[:33].copy_(torch.from_numpy(scale))
    if model.use_support:
        union = np.concatenate((clean["support"][cmask], aug["support"][amask]))
        center, scale = fit_normalization(union)
        model.center[33:].copy_(torch.from_numpy(center))
        model.scale[33:].copy_(torch.from_numpy(scale))
        model.support_mask.copy_(
            torch.from_numpy((union.astype(np.float64).var(axis=0) >= 1e-12).astype(np.float32))
        )


def fit(
    clean: dict, aug: dict, fold: dict, mode: str, seed: int, destination: Path
) -> NativeSupportRouter:
    """Run exactly300 half-clean half-native-restart updates, without QA corruption."""
    if mode not in ("F2n", "Fcapn"):
        raise ValueError("Expected preregistered joint candidate")
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = NativeSupportRouter(mode[:-1])
    cmask, amask = fold["clean_mask"], fold["aug_mask"]
    fit_preprocessing(model, clean, aug, cmask, amask)
    keys = ("native", "geometry", "quality", "support", "target")
    values = [torch.from_numpy(clean[k][cmask]) for k in keys]
    augmented = [torch.from_numpy(aug[k][amask]) for k in keys]
    seconds_scale = values[-1].abs().median().clamp(min=0.2)
    masses = [torch.from_numpy(fold[k]) for k in ("clean_mass", "aug_mass")]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    binding = {
        "mode": mode,
        "seed": seed,
        "epochs": EPOCHS,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fit_groups": sorted(set(clean["group"][fold["train"]])),
        "clean_rows": int(cmask.sum()),
        "augmented_rows": int(amask.sum()),
        "augmented_queries": len(set(aug["token"][amask])),
        "seconds_scale_clean_train_only": float(seconds_scale),
        "normalization": "base33 clean train; support10 admitted clean+restart train union",
        "center": model.center.tolist(),
        "scale": model.scale.tolist(),
        "active_native_mask": model.support_mask.tolist() if model.use_support else None,
        "support_mask_variance_threshold": 1e-12 if model.use_support else None,
        "clean_fit_token_sha256": hashlib.sha256(clean["token"][cmask].tobytes()).hexdigest(),
        "aug_fit_token_sha256": hashlib.sha256(aug["token"][amask].tobytes()).hexdigest(),
        "aug_fit_keep_sha256": hashlib.sha256(aug["keep"][amask].tobytes()).hexdigest(),
        "clean_mass_sha256": hashlib.sha256(fold["clean_mass"].tobytes()).hexdigest(),
        "aug_mass_sha256": hashlib.sha256(fold["aug_mass"].tobytes()).hexdigest(),
    }
    atomic_receipt(destination / "FIT_CONFIG.json", binding)
    history, started = [], time.perf_counter()

    def objective(inputs: list[torch.Tensor], mass: torch.Tensor) -> torch.Tensor:
        error = (model(*inputs[:4]) - inputs[-1]).abs()
        return (mass * (error / inputs[-1].abs().clamp(min=0.2) + error / seconds_scale)).sum()

    try:
        for epoch in range(EPOCHS):
            closs, aloss = objective(values, masses[0]), objective(augmented, masses[1])
            loss = 0.5 * closs + 0.5 * aloss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite joint native-context loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "clean": float(closs.detach()),
                    "native": float(aloss.detach()),
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
    """Apply the unchanged IQ four-gate rule, renaming only the two candidate identities."""
    result = select_base({"IQ": reports["IQ"], "F2": reports["F2n"], "Fcap": reports["Fcapn"]})
    result["selected"] = result["selected"] + "n" if result["selected"] else None
    result["candidates"] = {k + "n": v for k, v in result["candidates"].items()}
    return result
