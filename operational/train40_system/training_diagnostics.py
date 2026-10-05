"""TRAIN40 prediction diagnostics; these are fitting diagnostics, not holdout accuracy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.efficient_context.common import digest
from operational.train40_system.contracts import read, verified_endpoint, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json, replace
from operational.train40_system.heads import head_source
from operational.train40_system.history_features import load_source


def metrics(prediction: np.ndarray, truth_ttc: np.ndarray, truth_phase: np.ndarray) -> dict:
    """Retain all TRAIN queries and report finite-output coverage explicitly."""
    if prediction.shape != truth_ttc.shape or truth_phase.shape != truth_ttc.shape:
        raise ValueError("Diagnostic query alignment differs")
    finite = np.isfinite(prediction)
    valid_phase = finite & ((prediction < 0) | (prediction > 0.1))
    error = np.abs(prediction[finite] - truth_ttc[finite])
    phase = -np.log1p(-0.1 / prediction[valid_phase])
    return {
        "population": len(prediction),
        "finite_coverage": float(finite.mean()),
        "phase_domain_coverage": float(valid_phase.mean()),
        "MAE_seconds_finite": float(error.mean()) if error.size else None,
        "RMSE_seconds_finite": float(np.sqrt((error**2).mean())) if error.size else None,
        "MiD_phase_valid": float((10000 * np.abs(phase - truth_phase[valid_phase])).mean())
        if valid_phase.any()
        else None,
        "evaluation_role": "TRAIN_FIT_DIAGNOSTIC_NOT_GENERALIZATION",
    }


def run(output: Path) -> None:
    """Save predictions in independent fragments only after complete producer/head endpoints."""
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    freeze = read(output / "HEADS_FREEZE.json")
    verify_sources(freeze)
    source = head_source(output, "H8")
    queries = load_source(output, source)
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        truth_ttc, truth_phase = stored["ttc_s"], stored["phase"]
    torch.set_num_threads(4)
    heads, bindings = {}, {}
    for seed in (7, 13, 23):
        path, receipt = verified_endpoint(output, f"h8_seed{seed}", 2500)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head = TemporalRefiner(TemporalConfig(**payload["contract"]["model_config"]))
        head.load_state_dict(payload["model_state_dict"], strict=True)
        heads[seed] = head.float().eval().requires_grad_(False)
        bindings[str(seed)] = receipt["sha256"]
    binding = {
        "head_SHA256": bindings,
        "feature_manifest_sha256": source["manifest_sha256"],
        "source_sha256": digest(Path(__file__)),
        "role": "TRAIN_DIAGNOSTIC",
    }
    directory = output / "train_predictions"
    directory.mkdir(exist_ok=True)
    binding_path = directory / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Preserve earlier prediction lineage")
    atomic_json(binding_path, binding)
    predictions = {seed: np.empty(88744, np.float32) for seed in heads}
    with torch.inference_mode():
        for start in range(0, 88744, 128):
            stop = min(start + 128, 88744)
            path = directory / f"batch_{start:06d}.npz"
            receipt_path = path.with_suffix(".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                if digest(path) != receipt["sha256"] or receipt["binding_sha256"] != digest(
                    binding_path
                ):
                    raise ValueError("Saved prediction fragment changed")
                with np.load(path, allow_pickle=False) as stored:
                    values = {key: stored[key] for key in stored.files}
                if not np.array_equal(values["ordinals"], np.arange(start, stop)):
                    raise ValueError("Saved prediction fragment query identity changed")
            else:
                x, timing, valid, experts, _, _ = queries.gather(torch.arange(start, stop))
                values: dict[str, Any] = {"ordinals": np.arange(start, stop)}
                for seed, head in heads.items():
                    result = head(x, timing, valid, experts)
                    values[f"phase_seed{seed}"] = result["point_phase"].numpy()
                    values[f"ttc_seed{seed}"] = phase_to_ttc(result["point_phase"]).numpy()
                    values[f"q10_seed{seed}"] = result["q10"].numpy()
                    values[f"q90_seed{seed}"] = result["q90"].numpy()
                temporary = path.with_suffix(".pending.npz")
                np.savez_compressed(temporary, **values)
                with temporary.open("rb+") as handle:
                    os.fsync(handle.fileno())
                replace(temporary, path)
                atomic_json(
                    receipt_path,
                    {
                        "sha256": digest(path),
                        "binding_sha256": digest(binding_path),
                        "rows": stop - start,
                        "optimizer_updates": 0,
                    },
                )
            for seed in heads:
                predictions[seed][start:stop] = values[f"ttc_seed{seed}"]
    atomic_json(
        output / "TRAIN_FIT_DIAGNOSTICS.json",
        {
            "status": "COMPLETE",
            "population": 88744,
            "head_seeds": {
                str(seed): metrics(value, truth_ttc, truth_phase)
                for seed, value in predictions.items()
            },
            "no_holdout_generalization_claim": True,
            "no_published_Garl_score_reused": True,
            "weights_and_feature_lineage": binding,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output.resolve())
