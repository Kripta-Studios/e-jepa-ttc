"""Actual frozen-producer restart controls at fixed external sensor quartiles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from operational.evttc_competition.geometry import load_calibration
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_competition.runtime import CompetitionRuntime
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_adapt_training import configure_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("external", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    runtime = CompetitionRuntime.from_candidate(args.candidate, device="cuda")
    results = []
    with SensorSequence(
        args.external / "inputs", "Slider-500", slider_profile="event_aided_ttc"
    ) as reader:
        rows = reader.bbox_rows()
        calibration = load_calibration(reader.sequence_dir, "slider_candidate")
        for index in (len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4):
            row = rows[index]
            prepared = prepare_query(reader, rows, row, calibration)
            for available_ms in (100, 200, 400, 650):
                cutoff = row["anchor_us"] - available_ms * 1000
                events = [o for o in prepared.events if o.first_sensor_ts_us >= cutoff]
                rgb = [o for o in prepared.rgb if o.first_sensor_ts_us >= cutoff]
                result = runtime.predict(events, rgb, row["anchor_us"])
                results.append(
                    {
                        "query": index,
                        "available_ms": available_ms,
                        "event_observations": len(events),
                        "rgb_observations": len(rgb),
                        "result": asdict(result),
                    }
                )
    atomic_receipt(
        args.output / "REPORT.json",
        {
            "scope": "whole observations removed before restart; no fabricated partial windows",
            "interpretation": "support sensitivity, not a benchmark of a redesigned startup policy",
            "rows": results,
        },
    )


if __name__ == "__main__":
    main()
