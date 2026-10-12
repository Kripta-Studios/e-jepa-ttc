"""Q0: physical cohort identities and the missing paired V contrast."""

from __future__ import annotations

import datetime
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import OUT, ROOT, RUN, V12, V13, csvwrite, read, sha, write


def main() -> None:
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import joined_by_sensor_time, number

    from operational.rgb_port.evaluate import BUCKETS, WEIGHTS, score_signed_ttc

    binding = {
        "base_commit": "62cf4dbe914ea077c68e42983f869cd5422ab71c",
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "optimizer_updates": 0,
        "endpoints": {},
        "dependencies": {},
    }
    for repo in (ROOT, V13, ROOT.parent / "e-jepa-ttc-v15-range-readout"):
        binding[repo.name] = {
            cmd: subprocess.check_output(["git", "-C", str(repo), *args], text=True)
            for cmd, args in [("head", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])]
        }
    for name in ("PRODUCERS_PAIR_ENDPOINT_FREEZE.json", "HEADS_V_ROUND_FREEZE.json"):
        p = RUN / "rounds" / name
        freeze = read(p)
        binding["dependencies"][str(p)] = sha(p)
        for key, item in freeze["endpoints"].items():
            actual = sha(item["checkpoint_path"])
            assert actual == item["checkpoint_sha256"], key
            binding["endpoints"][key] = dict(item, verified_sha256=actual)
    for role in ("P", "H", "V"):
        p = RUN / f"{role}_MANIFEST.json"
        binding["dependencies"][str(p)] = sha(p)
    for modality in ("E", "R"):
        p = RUN / f"features/H/{modality}_NORMALIZER.json"
        binding["dependencies"][str(p)] = sha(p)
    v15 = ROOT.parent / "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012"
    binding["V15_observed"] = {
        name: read(v15 / name)
        for name in ("ACTIVE_STAGE.json", "NEXT_DECISION.json", "ACCOUNTING.json")
    }
    write(OUT / "STATE_BINDING.json", binding)

    paths = {
        "current_1024": ROOT / "docs/dev32_transfer_20261012/SCORED_PREDICTIONS.csv",
        "original_1024": V12
        / "docs/research_review_20261004_08/evidence/rgb_event/SCORED_PREDICTIONS.csv",
        "expanded_3947": V12
        / "docs/agent_handoff_20261010/evidence/comparison/DEV32_PREDICTIONS.csv",
    }
    tables = {k: pd.read_csv(p).fillna("").to_dict("records") for k, p in paths.items()}
    review = []
    for name in ("original_1024", "expanded_3947"):
        pairs = joined_by_sensor_time(tables["current_1024"], tables[name])
        deltas = []
        for field in (
            "truth_ttc_seconds",
            "H8_seed7",
            "H8_seed13",
            "H8_seed23",
            "public_Garl_event_lhr",
            "public_Garl_rgb_event_full",
        ):
            a = np.array([number(str(x[field])) for x, _ in pairs])
            b = np.array([number(str(y[field])) for _, y in pairs])
            assert np.array_equal(np.isfinite(a), np.isfinite(b))
            delta = float(np.max(np.abs(a[np.isfinite(a)] - b[np.isfinite(b)])))
            assert delta < 1e-9
            deltas.append(delta)
        assert len(pairs) == 1024
        review.append(
            dict(
                cohort=name,
                rows=len(tables[name]),
                physical_matches=len(pairs),
                identical_query_id=sum(x["query_id"] == y["query_id"] for x, y in pairs),
                max_numeric_delta=max(deltas),
                sha256=sha(paths[name]),
            )
        )
    csvwrite(OUT / "COHORT_COMPARABILITY.csv", review)

    a = pd.read_csv(RUN / "predictions/V/F_TRUE.csv").set_index(
        "sample_token", verify_integrity=True
    )
    b = pd.read_csv(RUN / "predictions/V/R_CTX.csv").set_index(
        "sample_token", verify_integrity=True
    )
    assert set(a.index) == set(b.index)
    b = b.loc[a.index]
    assert np.array_equal(a.target_ttc, b.target_ttc) and np.array_equal(a.group_id, b.group_id)
    group_rows = []
    for group in sorted(a.group_id.unique()):
        g = a.group_id == group
        row: dict = dict(group_id=group, count=int(g.sum()))
        for band, lo, hi in BUCKETS:
            mask = g & (a.target_ttc > lo) & (a.target_ttc <= hi)
            assert mask.any()
            target = a.target_ttc[mask].to_numpy()
            losses = []
            for frame in (a, b):
                pred = frame.prediction_ttc[mask].to_numpy()
                mid = np.abs(np.log1p(-0.1 / target) - np.log1p(-0.1 / pred)) * 1e4
                assert np.isfinite(mid).all()
                losses.append(float(mid.mean()))
            row[band + "_F_TRUE"] = losses[0]
            row[band + "_R_CTX"] = losses[1]
            row[band + "_delta"] = losses[0] - losses[1]
        row["weighted_delta"] = sum(WEIGHTS[band] * row[band + "_delta"] for band in WEIGHTS)
        group_rows.append(row)
    csvwrite(OUT / "V_GROUP_LOSSES.csv", group_rows)
    deltas = np.array([r["weighted_delta"] for r in group_rows])
    draws = np.random.default_rng(20261012).choice(deltas, size=(4000, len(deltas))).mean(1)
    scores = {
        name: score_signed_ttc(a.target_ttc, frame.prediction_ttc, a.group_id)
        for name, frame in [("F_TRUE", a), ("R_CTX", b)]
    }
    write(
        OUT / "MATCHED_PAIRED_REVIEW.json",
        dict(
            role="posthoc_not_preregistered_gate",
            rows=len(a),
            groups=len(deltas),
            scores=scores,
            delta=float(deltas.mean()),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            draws=4000,
            seed=20261012,
            source_sha256={n: sha(RUN / f"predictions/V/{n}.csv") for n in scores},
        ),
    )
    print(
        "Q0 complete",
        len(a),
        "paired V delta",
        deltas.mean(),
        np.quantile(draws, [0.025, 0.975]),
        flush=True,
    )


if __name__ == "__main__":
    main()
