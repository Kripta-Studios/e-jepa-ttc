"""Complete historical frozen-weight diagnostics, explicitly POSTHOC_NONSELECTABLE."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_scientific_recovery_v9_stage66_69 import legacy_fit, read, references

from e_jepa_ttc.artifacts.risk_geometry_v10 import CampaignOwner, atomic_json, binding, verify
from e_jepa_ttc.data.frozen_expert_tables_v10 import arm_inputs, load_table
from e_jepa_ttc.evaluation.risk_geometry_v10 import diagnostics, loss_frame, verify_coverage
from e_jepa_ttc.evaluation.stage63_65 import paired_hierarchical_bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    root = parser.parse_args().output_root.resolve()
    indexes = read(root / "TABLE_INDEX.json")
    frames, quantiles = references(root, indexes)
    parts = []
    donor_maps = read(root / "CONTROL_MAPS.json")
    for fold in range(3):
        table = load_table(indexes[f"outer{fold}_outer_dev"], role="outer_dev")
        donor = np.load(verify(donor_maps[f"outer{fold}_outer_dev"]), allow_pickle=False)
        inputs = arm_inputs(table, "S67-SIMPLEX17-PAIRPERM", donor)
        model = legacy_fit(
            root / f"frozen_audit/extracted_input/run/stage65/outer{fold}/S65-RISK17.npz"
        )
        costs = model.predict_regret(inputs.features)
        parts.append(
            loss_frame(
                table.metadata,
                inputs.expert_ttc,
                costs,
                costs.argmin(1),
                arm="RISK17_FROZEN_PAIRPERM_USAGE",
                seed=7,
                fold=fold,
            )
        )
    frames["RISK17_FROZEN_PAIRPERM_USAGE"] = (
        pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
    )
    output = root / "historical_posthoc"
    with CampaignOwner(root) as owner:
        evidence = diagnostics(frames, "S65-RISK17", output, quantiles, owner.check)
        verify_coverage(output / "DIAGNOSTIC_COVERAGE.json")
        original = paired_hierarchical_bootstrap(
            frames["S65-RISK17"], frames["S65-RISK8"], resource_check=owner.check
        )
        new = evidence["comparisons"]["S65-RISK8"]
        if (
            original.draws_sha256 != evidence["bootstrap"]["draws_sha256"]
            or abs(original.ci95_high - new["hierarchical_ci95_high"]) > 1e-10
        ):
            raise ValueError("production loss-bootstrap parity failure")
        atomic_json(
            output / "BOOTSTRAP_REPOSITORY_PARITY.json",
            dict(passed=True, draws_sha256=original.draws_sha256, ci95_high=original.ci95_high),
        )
    # Classification logits are retained in original replay files, not interpreted as risk costs.
    for name in ("S65-CE17-REPLAY", "RouterR"):
        old = (
            pd.concat([pd.read_csv(root / f"tables/outer{f}_{name}_replay.csv") for f in range(3)])
            .sort_values("sample_token")
            .reset_index(drop=True)
        )
        for arm, frame in frames.items():
            if not old.sample_token.equals(frame.sample_token):
                raise ValueError("legacy comparison rows differ")
            delta = old[["sample_token", "sequence_id", "track_id", "selected_expert"]].copy()
            delta["new_selected"] = frame.selected_expert
            delta["delta_loss"] = frame.loss - old.loss
            delta.to_csv(output / f"{arm}_vs_{name}.csv", index=False)
    atomic_json(
        root / "HISTORICAL_POSTHOC_DIAGNOSTICS_INDEX.json",
        dict(
            status="POSTHOC_NONSELECTABLE",
            historical_acceptance="INTEGRITY_BLOCKED",
            no_refit=True,
            files=[binding(p) for p in output.rglob("*") if p.is_file()],
            frozen_permutation=(
                "Fixed original ridge weights; consistent PAIR feature/action "
                "permutation; usage diagnostic only, not retrained evidence"
            ),
        ),
    )
    print(json.dumps(evidence["summaries"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
