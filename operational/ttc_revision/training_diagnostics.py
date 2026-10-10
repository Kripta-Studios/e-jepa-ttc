"""Descriptive TRAIN40 fit evaluation; never select or update a model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.efficient_context.common import digest
from operational.train40_system.contracts import verified_endpoint
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.heads import head_source
from operational.train40_system.history_features import load_source
from operational.ttc_revision.head import DirectTTCHead
from operational.ttc_revision.score import metrics


def evaluate_training(source: Path, campaign: Path) -> None:
    """Compare fixed endpoints on their seen training rows, on CPU only."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = head_source(source, "H8")
    queries = load_source(source, manifest)
    with np.load(source / "TRAIN40_INDEX.npz", allow_pickle=False) as data:
        truth = data["ttc_s"].astype(np.float32)
        sequences = data["sequences"]
        if not np.array_equal(data["phase"].astype(np.float32), queries.target_phase):
            raise ValueError("training target order changed")
    contract = json.loads((campaign / "TRAINING_FREEZE.json").read_text(encoding="utf-8"))
    if (
        contract["feature_manifest_sha256"] != manifest["manifest_sha256"]
        or contract["index_sha256"] != digest(source / "TRAIN40_INDEX.npz")
        or contract["source_sha256"]["head.py"] != digest(Path(__file__).with_name("head.py"))
    ):
        raise ValueError("training diagnostic identity mismatch")
    table = pd.DataFrame({"sequence_id": sequences, "truth_ttc_seconds": truth})
    bindings = {}
    with torch.inference_mode():
        for seed in (7, 13, 23):
            old_path, receipt = verified_endpoint(source, f"h8_seed{seed}", 2500)
            old_saved = torch.load(old_path, map_location="cpu", weights_only=False)
            old = TemporalRefiner(TemporalConfig(**old_saved["contract"]["model_config"])).eval()
            old.load_state_dict(old_saved["model_state_dict"], strict=True)
            path = campaign / f"direct_seed{seed}.pt"
            saved_receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if saved_receipt["status"] != "COMPLETE" or digest(path) != saved_receipt["sha256"]:
                raise ValueError("complete verified Direct checkpoint required")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["contract"] != contract or saved["update"] != contract["recipe"]["updates"]:
                raise ValueError("fixed endpoint mismatch")
            new = DirectTTCHead().eval()
            new.load_state_dict(saved["model"], strict=True)
            old_values, new_values = [], []
            for start in range(0, queries.population, 256):
                xs = queries.gather(torch.arange(start, min(start + 256, queries.population)))
                old_values.append(phase_to_ttc(old(*xs[:4])["point_phase"]).numpy())
                new_values.append(new(*xs[:3]).numpy())
            table[f"H8_seed{seed}"] = np.concatenate(old_values)
            table[f"Direct_seed{seed}"] = np.concatenate(new_values)
            bindings[f"H8_seed{seed}"] = receipt["sha256"]
            bindings[f"Direct_seed{seed}"] = saved_receipt["sha256"]
            print(json.dumps({"seed": seed, "rows": queries.population}), flush=True)
    for name in ("H8", "Direct"):
        table[f"{name}_median3"] = table[[f"{name}_seed{s}" for s in (7, 13, 23)]].median(axis=1)
    cohorts = {
        "all": np.ones(len(truth), bool),
        "negative": truth < 0,
        "positive_at_most_1": (truth > 0) & (truth <= 1),
        "positive_1_4": (truth > 1) & (truth < 4),
        "positive_4_8": (truth >= 4) & (truth < 8),
        "positive_at_least_8": truth >= 8,
    }
    rows = [
        {"method": method, "cohort": name, **metrics(truth[mask], table[method].to_numpy()[mask])}
        for method in table.columns[2:]
        for name, mask in cohorts.items()
    ]
    table.to_csv(campaign / "TRAIN_FIT_PREDICTIONS.csv", index=False)
    pd.DataFrame(rows).to_csv(campaign / "TRAIN_FIT_METRICS.csv", index=False)
    atomic_json(
        campaign / "TRAIN_FIT_DIAGNOSTICS.json",
        {
            "status": "COMPLETE",
            "population": len(truth),
            "scope": "Seen TRAIN40 data; unweighted descriptive fit, not validation/generalization",
            "optimizer_updates": 0,
            "model_selection": False,
            "device": "cpu",
            "source_sha256": digest(Path(__file__)),
            "training_contract_sha256": digest(campaign / "TRAINING_FREEZE.json"),
            "checkpoints": bindings,
            "metrics_sha256": digest(campaign / "TRAIN_FIT_METRICS.csv"),
            "predictions_sha256": digest(campaign / "TRAIN_FIT_PREDICTIONS.csv"),
        },
    )


if __name__ == "__main__":
    evaluate_training(
        Path("artifacts/train40_system_20261005"), Path("artifacts/ttc_revision_20261009")
    )
