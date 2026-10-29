"""Actual frozen event common-square crop and population metadata audit, P/H only."""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import OWNER, RUN, read, sha, write


def main() -> None:
    root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    path = root / "TRAIN40_INDEX.npz"
    expected = read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
        "train40_index_sha256"
    ]
    if sha(path) != expected:
        raise ValueError("Frozen event crop index changed")
    with np.load(path, allow_pickle=False) as z:
        tokens = z["tokens"].astype(str)
        square = z["square_xyxy"]
        delta = z["delta_t_s"]
        windows = z["windows_us"]
    lookup = {t: i for i, t in enumerate(tokens)}
    parts = []
    pop = []
    for role in ("P", "H"):
        manifest = read(OWNER / f"{role}_MANIFEST.json")
        metadata = pd.read_parquet(manifest["rows_path"])
        if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
            raise ValueError("Role metadata changed")
        ids = np.asarray([lookup[t] for t in metadata.sample_token.astype(str)])
        crop = square[ids]
        frame = metadata[
            ["sample_token", "sequence_id", "track_id", "group_id", "target_ttc"]
        ].copy()
        frame["role"] = role
        frame["crop_width_px"] = crop[:, 2] - crop[:, 0]
        frame["crop_height_px"] = crop[:, 3] - crop[:, 1]
        frame["crop_area_px2"] = frame.crop_width_px * frame.crop_height_px
        frame["delta_t_s"] = delta[ids]
        frame["input_span_us"] = windows[ids, -1, 1] - windows[ids, 0, 0]
        frame["band"] = np.select(
            [
                (frame.target_ttc > 0.1) & (frame.target_ttc <= 1),
                (frame.target_ttc > 1) & (frame.target_ttc <= 3),
                (frame.target_ttc > 3) & (frame.target_ttc <= 6),
                (frame.target_ttc > 6) & (frame.target_ttc <= 10),
                frame.target_ttc < 0,
            ],
            ["0.1-1", "1-3", "3-6", "6-10", "negative"],
            default="outside",
        )
        parts.append(frame)
        pop.append(
            {
                "role": role,
                "queries": len(frame),
                "groups": frame.group_id.nunique(),
                "sequences": frame.sequence_id.nunique(),
                "sequence_track_pairs": len(
                    set(zip(frame.sequence_id, frame.track_id, strict=True))
                ),
                "independent_episodes": None,
                "reason": "No independent episode identity in source",
            }
        )
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows.to_csv(RUN / "EVENT_CROP_ROWS_P_H.csv", index=False)
    all_rows.groupby(["role", "group_id", "band"]).agg(
        queries=("sample_token", "size"),
        width_mean_px=("crop_width_px", "mean"),
        width_min_px=("crop_width_px", "min"),
        width_max_px=("crop_width_px", "max"),
        width_p05_px=("crop_width_px", lambda x: x.quantile(0.05)),
        width_p95_px=("crop_width_px", lambda x: x.quantile(0.95)),
        area_mean_px2=("crop_area_px2", "mean"),
        input_span_mean_us=("input_span_us", "mean"),
    ).reset_index().to_csv(RUN / "EVENT_CROP_SUMMARY.csv", index=False)
    write(
        RUN / "EVENT_CROP_AUDIT.json",
        {
            "status": "COMPLETE",
            "source": str(path),
            "sha256": expected,
            "roles": pop,
            "scope": "Frozen event common-square coordinates; not RGB ROI proxies",
            "optimizer_updates": 0,
            "rows_sha256": sha(RUN / "EVENT_CROP_ROWS_P_H.csv"),
        },
    )


if __name__ == "__main__":
    main()
