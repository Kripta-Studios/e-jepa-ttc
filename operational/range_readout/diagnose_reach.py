"""Population-wide P/H fixed-geometry reach; never claim an absolute system ceiling."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import VectorReadout

from .cache_readout import parent
from .common import OWNER, RUN, csv_write, read, write
from .train_readout import VectorCache, load_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    native, _ = parent()
    models = {"historical_A5": VectorReadout(native)}
    for bound in (0.05, 0.20):
        name = f"R-C{round(bound * 100):02d}-S7"
        models[name] = load_endpoint(
            RUN / "fits" / name / "checkpoint_last.pt", ResidualSpec(bound)
        )
    rows = []
    population = []
    for role in ("P", "H"):
        cache = VectorCache(role)
        manifest = read(OWNER / f"{role}_MANIFEST.json")
        metadata = pd.read_parquet(manifest["rows_path"]).set_index(
            "sample_token", verify_integrity=True
        )
        frame = metadata.loc[cache.values["sample_token"].astype(str)]
        population.append(
            {
                "role": role,
                "queries": len(frame),
                "groups": frame.group_id.nunique(),
                "sequences": frame.sequence_id.nunique(),
                "sequence_track_pairs": int(
                    frame.reset_index()[["sequence_id", "track_id"]].drop_duplicates().shape[0]
                ),
                "episode_count": None,
                "episode_reason": "No independent episode identifier is supplied",
                "metadata_sha256": manifest["rows_sha256"],
            }
        )
        with torch.inference_mode():
            for begin in range(0, cache.population, 128):
                ids = torch.arange(begin, min(begin + 128, cache.population))
                values = cache.batch(ids)
                inputs = {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
                outputs = {name: model(inputs) for name, model in models.items()}
                analytic = values["analytic"].numpy().astype(np.float64)
                delta = values["delta"].numpy().astype(np.float64)
                for local, index in enumerate(ids.tolist()):
                    record = frame.iloc[index]
                    target = float(record.target_ttc)
                    for name, out in outputs.items():
                        cap = 0.20 if name.startswith("R-C20") else 0.05
                        upper = np.expm1(analytic[local] + cap) / delta[local]
                        denominator = 1 - delta[local, -1] * upper[-2]
                        valid_bound = bool(denominator > 1e-4)
                        maximum = (
                            max(
                                upper[-1],
                                native.config.temporal_inverse_ttc_blend * upper[-1]
                                + (1 - native.config.temporal_inverse_ttc_blend)
                                * upper[-2]
                                / denominator,
                            )
                            if valid_bound
                            else np.nan
                        )
                        floor = 1 / maximum if maximum > 0 and np.isfinite(maximum) else None
                        residual = float(out.residual_log_height_ratio[local, -1])
                        rows.append(
                            {
                                "role": role,
                                "sample_token": str(frame.index[index]),
                                "group_id": str(record.group_id),
                                "producer": name,
                                "target_ttc": target,
                                "cap": cap,
                                "analytic_last": analytic[local, -1],
                                "analytic_previous": analytic[local, -2],
                                "delta_last_s": delta[local, -1],
                                "delta_previous_s": delta[local, -2],
                                "residual_last": residual,
                                "residual_previous": float(
                                    out.residual_log_height_ratio[local, -2]
                                ),
                                "residual_over_cap": residual / cap,
                                "required_current_log_ratio": float(
                                    np.log1p(delta[local, -1] / target)
                                )
                                if target > 0
                                else None,
                                "required_previous_log_ratio": float(
                                    np.log1p(delta[local, -2] / (target + delta[local, -1]))
                                )
                                if target > 0
                                else None,
                                "optimistic_min_positive_ttc": floor,
                                "transport_pole_excluded": not valid_bound,
                                "target_below_conditional_reach": bool(
                                    target > 0 and floor is not None and target < floor
                                )
                                if valid_bound
                                else None,
                                "known": bool(out.known_mask[local]),
                                "prediction_ttc": float(out.ttc_mean_seconds[local]),
                                "support_current": float(out.sensor_support[local, -1]),
                                "support_previous": float(out.sensor_support[local, -2]),
                                "query_delay_us_metadata": int(record.query_delay_us),
                            }
                        )
        del cache
    csv_write(RUN / "PRODUCER_REACH_P_H.csv", rows)
    write(
        RUN / "POPULATION_AUDIT.json",
        {
            "roles": population,
            "independent_episodes_invented": False,
            "crop_note": (
                "roi_xyxy is RGB metadata; never substituted for event common-square crops"
            ),
            "event_crop_from_vector_cache": "not stored; preserved original frozen raw inputs",
            "event_crop_population_audit": (
                "EVENT_CROP_AUDIT.json and EVENT_CROP_ROWS_P_H.csv use actual square_xyxy "
                "from the frozen TRAIN40 index, restricted to admitted P/H tokens"
            ),
            "scope": (
                "fixed geometry, independent maximal residuals, "
                "current-only or native transport mixture"
            ),
            "transport_poles": "marked unavailable, never converted to finite bounds",
            "does_not_bound_PAIR_H8_or_retrained_encoder": True,
            "optimizer_updates": 0,
        },
    )
    data = pd.DataFrame(rows)
    data["band"] = np.select(
        [
            (data.target_ttc > 0.1) & (data.target_ttc <= 1),
            (data.target_ttc > 1) & (data.target_ttc <= 2),
            (data.target_ttc > 2) & (data.target_ttc <= 3),
            data.target_ttc < 0,
        ],
        ["0.1-1", "1-2", "2-3", "negative"],
        default="other_positive",
    )
    data.groupby(["role", "producer", "band"], dropna=False).agg(
        queries=("sample_token", "size"),
        mean_residual_over_cap=("residual_over_cap", "mean"),
        overestimate_count=("target_below_conditional_reach", "sum"),
        unavailable_bounds=("transport_pole_excluded", "sum"),
        mean_prediction=("prediction_ttc", "mean"),
    ).reset_index().to_csv(RUN / "PRODUCER_REACH_SUMMARY.csv", index=False)


if __name__ == "__main__":
    main()
