"""Average paired losses across completed seeds, without constructing a TTC ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, npz, read


def aggregate(c: Campaign) -> None:
    """Report exploratory/all-seed and new-seed contrasts with original query masses."""
    import numpy as np
    import pandas as pd

    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass
    from operational.simplex_t_closure.runtime import resumable_hierarchical_losses

    c.freeze()
    decision = read(c.out / "H8_WIDE_RESULTS.json")
    if decision["status"] != "COMPLETE" or not decision["replication_authorized"]:
        raise ValueError("conditional replication was not authorized by the sealed seed7 result")
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    frames = {}
    endpoints = {}
    for seed in (7, 13, 23):
        seal = c.out / f"ENDPOINTS_seed{seed}.json"
        if len(read(seal)["fits"]) != 3:
            raise ValueError("all conditional endpoints must be sealed")
        endpoints[str(seed)] = digest(seal)
        for name in ("H8", "H16", "WIDE"):
            frame = pd.read_parquet(c.out / f"analysis/seed{seed}/{name}.parquet")
            frames[seed, name] = frame.sort_values("sample_token").reset_index(drop=True)
    base = frames[7, "H8"]
    if len(base) != 8192 or any(not f[identity].equals(base[identity]) for f in frames.values()):
        raise ValueError("three-seed paired populations differ")
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    result = {}
    for label, seeds in (("all_three_seeds", (7, 13, 23)), ("new_seeds_only", (13, 23))):
        names = ("H8", "H16", "WIDE")
        losses = np.column_stack(
            [
                np.mean([frames[s, name].loss.to_numpy(np.float64) for s in seeds], 0)
                for name in names
            ]
        )
        path = c.out / "analysis/replication" / label
        npz(path / "PAIRED_MEAN_LOSSES.npz", mass=mass, losses=losses, names=np.asarray(names))
        table = base[identity].copy()
        for column, name in enumerate(names):
            table[name + "_mean_loss"] = losses[:, column]
        atomic_bytes(path / "PER_QUERY_MEAN_LOSS.csv", table.to_csv(index=False).encode())
        sampled, receipt = resumable_hierarchical_losses(
            base.rename(columns={"target_ttc": "target_ttc_s"}),
            losses,
            path / "bootstrap",
            c.require_resources,
            draws_path=Path(c.h16["bootstrap_draws"]),
            binding={"parent_protocol_sha256": digest(c.out / "PROTOCOL.json"), "seals": endpoints},
        )
        contrasts = {}
        for index, name in enumerate(names[:2]):
            delta = losses[:, 2] - losses[:, index]
            sequence_deltas = []
            for sequence in sorted(base.sequence_id.unique()):
                use = base.sequence_id.to_numpy() == sequence
                sequence_deltas.append(float((mass[use] / mass[use].sum()) @ delta[use]))
            ci = np.percentile(sampled[:, 2] - sampled[:, index], [2.5, 97.5]).tolist()
            contrasts[name] = {
                "point_delta": float(mass @ delta),
                "hierarchical_ci95": ci,
                "sequence_only": exact_sequence_diagnostic(np.asarray(sequence_deltas)),
                "sequence_wins": sum(v < 0 for v in sequence_deltas),
                "noninferiority_margin_mid": 2,
                "noninferiority_supported": ci[1] <= 2,
            }
        result[label] = {
            "seeds": list(seeds),
            "scores": {name: float(mass @ losses[:, i]) for i, name in enumerate(names)},
            "contrasts": contrasts,
            "bootstrap": receipt,
        }
    atomic_json(
        c.out / "WIDE_REPLICATION_RESULTS.json",
        {
            "status": "COMPLETE",
            "results": result,
            "aggregation": "arithmetic mean of paired losses per query; no averaged TTC",
            "sign_finite_and_crucial_metrics": "retain the three separate per-seed reports",
            "candidate_promoted": False,
            "confirmation": False,
            "optimizer_updates": 0,
        },
    )


def main() -> int:
    """Regenerate only fully completed and prospectively authorized head replicas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    aggregate(Campaign(args.protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
