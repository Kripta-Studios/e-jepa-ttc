"""Check acquisition exclusion and matched sensor-control populations end to end."""

from pathlib import Path

import numpy as np
import pytest

from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import read, verify_complete
from scripts.release_sensor_proxy import analyze


@pytest.mark.parametrize("extra", [None, "rgb_observed"])
def test_sensor_proxy_excludes_acquisition_and_recovers_fixed_scale(tmp_path, extra):
    capture = tmp_path / "capture"
    run = ResearchRun(
        capture,
        {"fixture_source": Path(__file__)},
        {
            "experiment_id": "S1-RGB-proxy-extract-v1"
            if extra is None
            else "S1-reference-extract-v1",
            "targets_read": False,
        },
    )
    rows = []
    texture = np.array([-2.0, -1.0, 1.0, 2.0])
    for group in range(3):
        for query in range(3):
            q = 0.2 + 0.3 * group + 0.1 * query
            token = f"fixture:{group}:{query}"
            filename = f"group{group}_query{query}.npz"
            np.savez_compressed(
                capture / filename,
                target_rate=3 * q * texture + 2,
                previous_rate=np.zeros(4),
                texture_motion=texture,
                rgb_observed=(3 * q * texture + 2) / 6,
                valid=np.ones(4, dtype=bool),
                allow_pickle=False,
            )
            rows.append(
                {
                    "token": token,
                    "group": str(group),
                    "query_us": query * 100000,
                    "status": "measured",
                    "admitted_cells": 4,
                    "q_at_window_start": q,
                    "arrays": filename,
                }
            )
    rows += [
        {
            "token": f"missing:{i}",
            "group": "fixture_missing",
            "query_us": i,
            "status": "no_event_or_R0_state",
        }
        for i in range(140)
    ]
    write_json(capture / "FEATURES.json", {"records": rows, "targets_read": False})
    run.finish()
    output = tmp_path / "analysis"
    analyze(capture, output, extra)
    verify_complete(output)
    metrics = read(output / "METRICS.json")
    assert metrics["queries_with_cells"] == 9
    assert metrics["population"] == 149
    assert metrics["arms"]["physical"]["macro_mse"] < 1e-24
    assert metrics["arms"]["physical"]["macro_mse"] < metrics["arms"]["permuted"]["macro_mse"]
    assert metrics["S2_admitted"] is False
    fits = read(output / "FITS.json")
    assert len(fits) == (12 if extra is None else 15)
    for fit in fits:
        assert fit["held_out_group"] not in fit["training_groups"]
        assert len(fit["training_groups"]) == 2
    scored = read(output / "SCORED.json")
    for arm in ("activity", "texture", "physical", "permuted"):
        assert {r["token"] for r in scored if r["arm"] == arm} == {r["token"] for r in rows[:9]}
    if extra is not None:
        assert metrics["arms"][extra]["macro_mse"] < 1e-24
