"""H-only mass/exposure/gradient audits and exposed public query diagnostics."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.range_transfer.mass import fit_mass_plans
from e_jepa_ttc.range_transfer.objective import range_transfer_loss
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, pinball, relative_cost_targets

from .cache import HeadCache
from .common import read, sha, write

RANGES = (
    ("c_0.1_1", 0.1, 1),
    ("c_1_2", 1, 2),
    ("c_2_3", 2, 3),
    ("s_3_6", 3, 6),
    ("l_6_8", 6, 8),
    ("l_8_10", 8, 10),
    ("n", -10, 0),
)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a populated CSV; empty stages are represented by explicit receipts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_metadata(owner: Path, run: Path) -> dict[str, Any]:
    """Audit real H metadata now, explicitly distinct from future cached features."""
    manifest = read(owner / "H_MANIFEST.json")
    path = Path(manifest["rows_path"])
    if sha(path) != manifest["rows_sha256"]:
        raise ValueError("H rows hash mismatch")
    columns = [
        "sample_token",
        "group_id",
        "sequence_id",
        "track_id",
        "query_time_us",
        "role",
        "rgb_available",
        "target_ttc",
        "mass",
    ]
    rows = pd.read_parquet(path, columns=columns)
    if set(rows.role) != {"H"} or rows.sample_token.duplicated().any():
        raise ValueError("H metadata role or identity mismatch")
    available = rows.loc[rows.rgb_available].copy()
    target = available.target_ttc.to_numpy()
    plans = fit_mass_plans(
        target, available.group_id.to_numpy(), available.mass.to_numpy(), fit_role="H"
    )
    results = []
    for group in ["ALL", *sorted(available.group_id.unique())]:
        for name, low, high in RANGES:
            mask = (target > low) & (target <= high)
            if name == "n":
                mask &= target < 0
            if group != "ALL":
                mask &= available.group_id.to_numpy() == group
            subset = available.loc[mask]
            # Track IDs are reported as tracks, not asserted independent scenes.
            episodes = 0
            for _, track in subset.groupby("track_id"):
                times = np.sort(track.query_time_us.to_numpy())
                episodes += 1 + int(np.sum(np.diff(times) > 1_000_000))
            results.append(
                {
                    "scope": "H_ROLE_METADATA_NOT_FINAL_FEATURE_CACHE",
                    "group": group,
                    "range": name,
                    "rows": int(mask.sum()),
                    "groups": subset.group_id.nunique(),
                    "tracks": subset.track_id.nunique(),
                    "episodes_gap_gt_1s_descriptive": episodes,
                    "native_mass": float(plans.native[mask].sum()),
                    "M1_mass": float(plans.benchmark[mask].sum()),
                    "M2_mass": float(plans.subrange[mask].sum()),
                    "expected_uniform_draws": 2500 * 128 * int(mask.sum()) / len(available),
                }
            )
    csv_write(run / "MASS_AUDIT.csv", results)
    # Same uniform sampler as all arms; these are draws, not optimizer updates.
    generator = torch.Generator().manual_seed(7)
    draws = torch.randint(len(available), (2500, 128), generator=generator).numpy()
    counts = np.bincount(draws.ravel(), minlength=len(available))
    exposure = available[["sample_token", "group_id", "track_id", "target_ttc"]].copy()
    exposure["draw_count_seed7"] = counts
    exposure.to_csv(run / "H_EXPOSURE.csv", index=False)
    controls = []
    for _name, low, high in RANGES:
        subset = available.loc[(target > low) & (target <= high)]
        controls.extend(
            sorted(subset.sample_token, key=lambda s: hashlib.sha256(s.encode()).digest())[:4]
        )
    rest = sorted(
        set(available.sample_token) - set(controls),
        key=lambda s: hashlib.sha256(s.encode()).digest(),
    )
    controls = (controls + rest)[:32]
    available.set_index("sample_token").loc[controls].reset_index().to_csv(
        run / "H_DIAGNOSTIC_CONTROLS.csv", index=False
    )
    summary = {
        "status": "METADATA_COMPLETE_FEATURE_GRADIENTS_PENDING",
        "rows": len(rows),
        "available_rows": len(available),
        "unavailable_rows_retained": int((~rows.rgb_available).sum()),
        "groups": available.group_id.nunique(),
        "rows_path": str(path),
        "rows_sha256": sha(path),
        "max_draws_per_row": int(counts.max()),
        "unique_sampled_rows": int((counts > 0).sum()),
        "schedule_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "expected_draws_per_row": 320000 / len(available),
        "optimizer_updates": 0,
        "episode_definition": "track_id runs separated by >1 second; not independent scenes",
        "native_mass_sum": float(plans.native.sum()),
        "native_final_cache_audited": False,
    }
    write(run / "H_METADATA_AUDIT.json", summary)
    return summary


def public_traces(v12: Path, run: Path) -> None:
    """Select published worst queries and immediate sequence neighbours, without fitting."""
    source = v12 / "docs/agent_handoff_20261010/evidence/comparison/FCWD_PREDICTIONS.csv"
    frame = pd.read_csv(source)
    ranked = pd.read_csv(run / "public_analysis/ranked_queries.csv")
    selected = set(ranked.head(20).query_id)
    ids = set(selected)
    for _, sequence in frame.groupby("sequence_id"):
        sequence = sequence.sort_values("anchor_relative_seconds")
        tokens = sequence.query_id.tolist()
        for index, token in enumerate(tokens):
            if token in selected:
                ids.update(tokens[max(0, index - 1) : index + 2])
    chosen = frame.loc[frame.query_id.isin(ids)].copy()
    chosen["selection"] = np.where(chosen.query_id.isin(selected), "published_top20", "neighbour")
    chosen["scope"] = "V12_EXPOSED_PUBLIC_CSV_NOT_NEW_INFERENCE"
    chosen.to_csv(run / "PUBLIC_DIAGNOSTIC_TRACES.csv", index=False)
    write(
        run / "N1_STATUS.json",
        {
            "status": "PUBLIC_FIELDS_COMPLETE_INTERNAL_TENSORS_PENDING",
            "source_sha256": sha(source),
            "queries": len(chosen),
            "top_queries": 20,
            "available_columns": list(frame.columns),
            "new_producer_inference": False,
            "missing": [
                "raw exact/admitted parity",
                "internal analytical/residual decomposition",
                "internal C2F gate",
                "per-head residual traces with final matched parents",
            ],
            "reason": (
                "Preserve V13/R1 resources; no partial V13 weights or TRAIN40 holdout substitute"
            ),
        },
    )


def gradient_audit(cache: HeadCache, run: Path) -> None:
    """Measure fixed H per-term/per-range gradients without optimizer updates."""
    torch.manual_seed(7)
    model = TemporalRefiner(TemporalConfig(hidden=160))
    rows = []
    for name, low, high in RANGES:
        indices = np.flatnonzero((cache.target_ttc > low) & (cache.target_ttc <= high))
        indices = sorted(
            indices,
            key=lambda i: hashlib.sha256(str(cache.values["sample_token"][i]).encode()).digest(),
        )[:128]
        if not indices:
            continue
        ids = torch.tensor(indices)
        truth = cache.tensor("target_phase", ids, torch.device("cpu"))
        experts = cache.tensor("expert_phase", ids, torch.device("cpu"))
        for mass_id, mass_plan in enumerate(cache.plans):
            mass = torch.from_numpy(mass_plan[indices].copy())
            output = cache.forward(model, ids)
            cost = (
                (output["relative_cost"][:, 1:] - relative_cost_targets(experts, truth))
                .square()
                .mean(-1)
            )
            terms = {
                "phase": (output["point_phase"] - truth).abs() / 0.03,
                "quantile": 0.1
                * (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9))
                / 0.03,
                "cost": 0.01 * cost,
            }
            terms = {k: (cache.population * mass * v).mean() for k, v in terms.items()}
            native = range_transfer_loss(output, truth, experts, mass, cache.population)
            terms["relative_added"] = (
                range_transfer_loss(
                    output, truth, experts, mass, cache.population, relative_weight=0.25
                )
                - native
            )
            for term, value in terms.items():
                grads = torch.autograd.grad(
                    value, list(model.parameters()), retain_graph=True, allow_unused=True
                )
                norm = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** 0.5
                loc = output["raw_location"].detach()
                rows.append(
                    {
                        "range": name,
                        "mass": f"M{mass_id}",
                        "term": term,
                        "rows": len(ids),
                        "loss": float(value.detach()),
                        "gradient_l2": norm,
                        "flat_cap_fraction": float(
                            ((-torch.expm1(-loc) / 0.1).abs() <= 1 / 60).float().mean()
                        ),
                        "phase_clamp_fraction": float(
                            ((loc <= MIN_PHASE) | (loc >= MAX_PHASE)).float().mean()
                        ),
                        "scope": "fixed_hash_H_initialization_seed7_no_optimizer",
                    }
                )
    csv_write(run / "GRADIENT_AUDIT.csv", rows)
    write(run / "GRADIENT_AUDIT.json", {"status": "COMPLETE", "optimizer_updates": 0})
