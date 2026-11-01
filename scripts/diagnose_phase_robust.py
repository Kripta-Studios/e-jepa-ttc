"""Locate errors relative to a causal inverse-scale physical observer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_campaign.review_controls import shared_scores
from operational.phase_robust.data import histories, query_geometry
from operational.phase_robust.geometry import expansion_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "campaign", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    population = read(args.data / "POPULATION.json")["rows"]
    tracks = histories(population)
    lookup = {"evttc:" + r["query_id"]: r for r in population}
    with np.load(args.campaign / "diagnosis" / "predictions.npz") as a:
        source = {k: a[k] for k in a.files}
    rows = [lookup[str(t)] for t in source["sample_token"]]
    geometry = [query_geometry(tracks[r["sequence_id"]], r["anchor_us"]) for r in rows]
    geo = np.array([g["ttc"] if g["valid"] else np.nan for g in geometry])
    geo = np.clip(geo, -60, 60)
    quality = np.array([g["quality"] for g in geometry], dtype=np.float32)
    save_npz(
        args.output / "road.npz",
        {
            **source,
            "geometry": geo,
            "quality": quality,
            "geometry_valid": np.array([g["valid"] for g in geometry]),
            "clocks": np.array([r["anchor_us"] for r in rows]),
            "reason": np.array([g["reason"] for g in geometry]),
        },
    )
    excluded = read(args.campaign / "diagnosis" / "REPORT.json")["excluded_recordings"]
    eligible = ~np.isin(source["group"], excluded)
    report = {
        "scope": "exposed development; no new parameter selection yet",
        "road": {},
        "slider": {},
        "input_hashes": {
            str(p): file_sha(p)
            for p in (
                args.data / "POPULATION.json",
                args.campaign / "diagnosis" / "predictions.npz",
            )
        },
    }
    for label, mask in {
        "eligible24": eligible,
        "all32": np.ones(len(geo), dtype=bool),
        "eligible_under1": eligible & (source["target"] < 1),
        "eligible_over4": eligible & (source["target"] > 4),
    }.items():
        predictions = {
            "F0": source["F0"][mask],
            "F1": source["F1_all_data_resubstitution"][mask],
            "geometry": geo[mask],
        }
        report["road"][label] = {
            "full": {k: metrics(source["target"][mask], v) for k, v in predictions.items()},
            "shared": shared_scores(source["target"][mask], predictions),
        }
    slider_root = args.campaign / "Slider500"
    raw = np.loadtxt(slider_root / "inputs" / "Slider-500" / "bbox.csv", delimiter=",")
    clocks = np.rint(raw[:, 1] * 1e6).astype(np.int64)
    geometries = [expansion_features(raw[:, 2:6], clocks, int(t)) for t in clocks]
    with (
        np.load(slider_root / "predictions_sealed.npz") as p,
        np.load(slider_root / "targets.npz") as y,
    ):
        arrays = {k: p[k] for k in p.files}
        arrays.update(
            target=y["target"],
            quality=np.array([g["quality"] for g in geometries], np.float32),
            geometry=np.clip(
                np.array([g["ttc"] if g["valid"] else np.nan for g in geometries]), -60, 60
            ),
            geometry_valid=np.array([g["valid"] for g in geometries]),
            clocks=clocks,
            reason=np.array([g["reason"] for g in geometries]),
        )
    save_npz(args.output / "slider500.npz", arrays)
    names = ("F0", "F1_7", "RGB", "event", "geometry")
    report["slider"] = {
        "full": {k: metrics(arrays["target"], arrays[k]) for k in names},
        "shared": shared_scores(arrays["target"], {k: arrays[k] for k in names}),
    }
    atomic_receipt(args.output / "REPORT.json", report)
    print("COMPLETE: causal geometry diagnostics on road and exposed Slider500")


if __name__ == "__main__":
    main()
