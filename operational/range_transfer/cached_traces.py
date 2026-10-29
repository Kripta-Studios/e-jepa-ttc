"""Post-hoc V12 head attribution on verified stored FCWD features, CPU only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.simplex_t_shared_route.adapter import inputs
from operational.train40_system.contracts import verified_endpoint

from .common import ROOT, read, resources, sha, write
from .diagnose import csv_write


def trace_v12(v12: Path, run: Path) -> dict[str, Any]:
    """Execute historical heads only; never treat TRAIN40 as a held-out producer."""
    replay = v12 / "artifacts/ttc_revision_20261009/replay_fcwd"
    campaign = v12 / "artifacts/train40_system_20261005"
    output = run / "v12_cached_attribution"
    required = [
        replay / "REPLAY_COMPLETE.json",
        replay / "REPLAY_IDENTITY.json",
        campaign / "H8_FEATURES.npz",
        campaign / "H8_FEATURE_MANIFEST.json",
    ]
    if any(not p.exists() for p in required):
        result = {
            "status": "WAITING_HISTORICAL_ASSETS",
            "paths": [str(p) for p in required if not p.exists()],
        }
        write(output / "RESULT.json", result)
        return result
    if not resources(run, starting=True)["allowed"]:
        return {"status": "PAUSED_RESOURCE"}
    identity, complete = read(required[1]), read(required[0])
    norm_sha = sha(required[2])
    if (
        complete.get("status") != "COMPLETE"
        or norm_sha != identity["models"]["normalization"]["artifact_sha256"]
    ):
        raise ValueError("Historical replay/normalizer not verified")
    for relative in (
        "src/e_jepa_ttc/simplex_t/model.py",
        "src/e_jepa_ttc/simplex_t/phase.py",
        "operational/simplex_t_shared_route/adapter.py",
    ):
        if sha(ROOT / relative) != sha(v12 / relative):
            raise ValueError(f"V12 inference source differs: {relative}")
    with np.load(required[2], allow_pickle=False) as archive:
        mean, scale = archive["mean"], archive["scale"]
    heads, checkpoints = {}, {}
    for seed in (7, 13, 23):
        checkpoint, receipt = verified_endpoint(campaign, f"h8_seed{seed}", 2500)
        if receipt["sha256"] != identity["models"]["h8_heads"][str(seed)]["checkpoint_sha256"]:
            raise ValueError("Historical head endpoint differs from cached features")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["status"] != "COMPLETE" or payload["committed_updates"] != 2500:
            raise ValueError("Historical head is not final")
        head = TemporalRefiner(TemporalConfig(**payload["contract"]["model_config"]))
        head.load_state_dict(payload["model_state_dict"], strict=True)
        heads[seed] = head.float().eval().requires_grad_(False)
        checkpoints[str(checkpoint)] = receipt["sha256"]
    selected = pd.read_csv(run / "PUBLIC_DIAGNOSTIC_TRACES.csv")
    source = pd.read_csv(
        v12 / "docs/agent_handoff_20261010/evidence/comparison/FCWD_PREDICTIONS.csv"
    )
    ordinal = {token: i for i, token in enumerate(source.query_id)}
    traces, observations, fragment_hashes = [], [], {}
    started = time.perf_counter()
    with torch.inference_mode():
        for row in selected.to_dict("records"):
            token = row["query_id"]
            fragment = replay / "fragments" / f"{ordinal[token]:06d}.npz"
            expected = complete["fragments"][fragment.name]
            if sha(fragment) != expected:
                raise ValueError("Historical inference fragment changed")
            fragment_hashes[str(fragment)] = expected
            with np.load(fragment, allow_pickle=False) as archive:
                if str(archive["query_id"]) != token:
                    raise ValueError("Fragment/query join mismatch")
                raw, stored = archive["features"], archive["canonical"]
            xs = inputs(
                "H8_SEED7",
                raw,
                np.ones(8, bool),
                np.arange(350000, -1, -50000, dtype=np.int64),
                int(row["anchor_us"]),
                int(row["anchor_us"]),
                mean,
                scale,
            )
            anchor = float(np.median(raw[-1, 8:11]))
            for index in range(8):
                observations.append(
                    {
                        "query_id": token,
                        "observation_slot": index,
                        "anchor_us": int(row["anchor_us"]),
                        "observation_anchor_us": int(row["anchor_us"]) - (7 - index) * 50000,
                        "lag_s": float(xs[1][0, index, 0]),
                        **{f"phase17_{i}": float(raw[index, i]) for i in range(17)},
                    }
                )
            for offset, (seed, head) in enumerate(heads.items()):
                result = head(*xs)
                estimate = float(phase_to_ttc(result["raw_location"])[0])
                reference = float(stored[offset])
                traces.append(
                    {
                        "query_id": token,
                        "sequence_id": row["sequence_id"],
                        "seed": seed,
                        "truth_ttc": row["truth_ttc_seconds"],
                        "prediction_ttc": estimate,
                        "stored_prediction_ttc": reference,
                        "abs_cpu_replay_delta_s": abs(estimate - reference),
                        "expert_a5_phase": float(raw[-1, 8]),
                        "expert_c2f_phase": float(raw[-1, 9]),
                        "expert_pair_phase": float(raw[-1, 10]),
                        "current_median_phase": anchor,
                        "residual_phase": float(result["raw_residual"][0]),
                        "raw_location_phase": float(result["raw_location"][0]),
                        "emitted_phase": float(result["point_phase"][0]),
                        "residual_suppressed_ttc": float(phase_to_ttc(torch.tensor(anchor))),
                        "at_cap60": abs(estimate) >= 59.999,
                        "aux_cost_index_not_router": int(result["expert_index"][0]),
                        "scope": "V12_FCWD_exposed_cached_features_CPU_head_inference",
                    }
                )
    csv_write(output / "HEAD_TRACES.csv", traces)
    csv_write(output / "OBSERVATION_TRACES.csv", observations)
    table = pd.DataFrame(traces)
    summary = []
    for name, low, high in (("c", 0.1, 3), ("s", 3, 6), ("l", 6, 10), ("gt10", 10, np.inf)):
        band = table.loc[(table.truth_ttc > low) & (table.truth_ttc <= high)]
        if len(band):
            summary.append(
                {
                    "band": name,
                    "queries": band.query_id.nunique(),
                    "head_seed_rows": len(band),
                    "bias_head_s": float((band.prediction_ttc - band.truth_ttc).mean()),
                    "bias_current_median_s": float(
                        (band.residual_suppressed_ttc - band.truth_ttc).mean()
                    ),
                    "scope": "selected_top20_plus_neighbours_not_full_cohort",
                }
            )
    csv_write(output / "STAGE_BIAS.csv", summary)
    result = {
        "status": "COMPLETE_CACHED_HEAD_ATTRIBUTION",
        "queries": len(selected),
        "head_forwards": len(traces),
        "optimizer_updates": 0,
        "new_producer_inference": False,
        "max_abs_cpu_replay_delta_s": float(table.abs_cpu_replay_delta_s.max()),
        "checkpoints": checkpoints,
        "fragments": fragment_hashes,
        "normalizer_sha256": norm_sha,
        "cost_seconds": time.perf_counter() - started,
        "cost_scope": "cached_features_CPU_head_trace_including_IO",
        "missing_raw_internals": [
            "bbox_height_analytic_ratio",
            "producer_residual",
            "C2F_internal_gate",
            "raw_input_parity",
        ],
        "V13_holdout_claim": False,
    }
    write(output / "RESULT.json", result)
    return result
