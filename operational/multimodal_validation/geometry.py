"""Q3: label-blind panels, causal priors and 20/40/80 ms experiments."""

from __future__ import annotations

import hashlib
import time

import cv2
import h5py
import numpy as np

from operational.fcwd_rgb_registration.geometry import SETTINGS, estimate_shift
from operational.fcwd_rgb_registration.prepare import counts
from operational.sota_eval.fcwd_inputs import FCWDEventReader, _validate_row

from .common import MANIFEST, OUT, REG, admission, csvwrite, read, registrations, sha, write


def causal_prior(row: dict, previous: dict | None, current: dict) -> dict:
    """An experimental prior is admissible only with current support/drift evidence."""
    if previous is None:
        return dict(eligible=False, reason="NO_PREVIOUS_ACCEPTANCE", prior_query_id=None)
    age = int(row["anchor_us"]) - int(previous["query_us"])
    assert age > 0
    drift = (
        float(np.linalg.norm(np.subtract(current["shift_px"], previous["shift_px"])))
        if "shift_px" in current
        else None
    )
    support = current.get("support_pixels", 0) >= SETTINGS["minimum_event_pixels"]
    stable = (
        current.get("half_window_disagreement_px", float("inf"))
        <= SETTINGS["maximum_split_disagreement_px"]
    )
    eligible = age <= 200000 and support and stable and drift is not None and drift <= 5.0
    return dict(
        eligible=eligible,
        reason="GEOMETRIC_REFERENCE_PENDING" if eligible else "AGE_SUPPORT_OR_DRIFT_NOT_VERIFIED",
        prior_query_id=previous["query_id"],
        age_us=age,
        prior_shift_px=previous["shift_px"],
        measured_drift_px=drift,
        promoted=False,
    )


def main() -> None:
    cv2.setNumThreads(1)
    rows = read(MANIFEST)["rows"]
    regs = registrations()
    calibration = read(REG / "BINDING.json")["calibration"]
    selected = []
    for sequence in sorted({r["sequence_id"] for r in rows}):
        for accepted in (False, True):
            candidates = [
                r
                for r in rows
                if r["sequence_id"] == sequence and regs[r["query_id"]]["accepted"] == accepted
            ]
            selected.extend(
                sorted(
                    candidates, key=lambda r: hashlib.sha256(r["query_id"].encode()).hexdigest()
                )[:2]
            )
    plan: dict = dict(
        selection=(
            "2 accepted and 2 rejected per sequence sorted SHA256(query_id); no TTC or error read"
        ),
        query_ids=[r["query_id"] for r in selected],
        windows_us=[20000, 40000, 80000],
        prior_max_age_us=200000,
        prior_max_drift_px=5.0,
        independent_correspondences_available=False,
        promoted=False,
        manifest_sha256=sha(MANIFEST),
        registration_sha256=sha(REG / "REGISTRATION_SEALED.json"),
    )
    write(OUT / "geometry/SELECTION_SEALED.json", plan)
    prior_rows = []
    last = {}
    for row in sorted(rows, key=lambda r: (r["sequence_id"], r["anchor_us"])):
        current = regs[row["query_id"]]
        prior = causal_prior(row, last.get(row["sequence_id"]), current)
        prior_rows.append(dict(query_id=row["query_id"], **prior))
        if current["accepted"]:
            last[row["sequence_id"]] = current
    write(
        OUT / "geometry/CAUSAL_PRIOR.json",
        dict(rows=prior_rows, labels_read=False, independent_geometric_validation=False),
    )
    trials = []
    annotations = []
    for row in selected:
        admission()
        reg = regs[row["query_id"]]
        path = _validate_row(row)
        with h5py.File(path, "r") as h:
            # _validate_row pins this HDF5 image Dataset.
            rgb = np.asarray(h["blackflys/right/data"][reg["rgb_frame_index"]], np.uint8)  # pyright: ignore[reportIndexIssue]
        timestamp = reg["rgb_sensor_us"]
        base_count = None
        for window in plan["windows_us"]:
            start = time.perf_counter()
            with FCWDEventReader(path) as reader:
                events = reader.read_window(timestamp - window, timestamp)
            early = counts(events, events["t"] < timestamp - window // 2)
            late = counts(events, events["t"] >= timestamp - window // 2)
            ready = time.perf_counter()
            result = estimate_shift(rgb, early, late, row["boxes_xyxy3"][-1], calibration)
            if window == 20000:
                base_count = early + late
            trial: dict = dict(
                query_id=row["query_id"],
                window_us=window,
                read_counts_ms=(ready - start) * 1000,
                registration_ms=(time.perf_counter() - ready) * 1000,
                **result,
            )
            trial["same_as_sealed_on_20ms"] = (
                all(result.get(k) == reg.get(k) for k in result) if window == 20000 else None
            )
            trials.append(trial)
        # Side-by-side raw event and raw RGB views. Coordinates are explicitly independent.
        assert base_count is not None
        e = np.log1p(base_count)
        e = (255 * e / max(float(e.max()), 1.0)).astype(np.uint8)
        ep = cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)
        x0, y0, x1, y1 = map(int, row["boxes_xyxy3"][-1])
        cv2.rectangle(ep, (x0, y0), (x1, y1), (255, 80, 0), 2)
        panel = np.full((770, 2560, 3), 255, np.uint8)
        panel[50:770, :1280] = ep
        panel[50:770, 1280:] = cv2.resize(rgb, (1280, 720))
        cv2.putText(
            panel,
            row["query_id"] + " EVENTS (1280x720) | RGB (display 1280x720; native 1920x1200)",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
        )
        name = row["query_id"].replace(":", "_") + ".png"
        directory = OUT / "geometry/panels"
        directory.mkdir(parents=True, exist_ok=True)
        encoded_ok, encoded = cv2.imencode(".png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        if not encoded_ok:
            raise OSError("PNG encoding failed")
        (directory / name).write_bytes(encoded.tobytes())
        for point in range(1, 7):
            annotations.append(
                dict(
                    query_id=row["query_id"],
                    point_id=point,
                    event_x="",
                    event_y="",
                    rgb_x_native="",
                    rgb_y_native="",
                    annotator="",
                    uncertain="",
                    panel=name,
                )
            )
        print("geometry", row["query_id"], flush=True)
        write(
            OUT / "geometry/MULTISCALE_TRIALS.json",
            dict(rows=trials, labels_read=False, reference_error_px=None, promoted=False),
        )
    csvwrite(OUT / "geometry/CORRESPONDENCES_TO_ANNOTATE.csv", annotations)
    (OUT / "geometry/ANNOTATION_INSTRUCTIONS.md").write_text(
        "Select six independently identifiable corresponding edges per panel. "
        "Enter native event coordinates (1280x720), and native RGB (1920x1200). "
        "The displayed right panel is rescaled. Mark uncertainty and occlusion. "
        "Do not consult TTC/predictions. A second annotator adjudicates disagreements. "
        "NCC and half-window consistency are not reference errors. "
        "All 72 correspondences are empty. No registration candidate is promoted.\n",
        encoding="utf-8",
    )
    write(
        OUT / "geometry/SUMMARY.json",
        dict(
            queries=len(selected),
            trials=len(trials),
            prior_eligible=sum(r["eligible"] for r in prior_rows),
            baseline_parity=all(
                t["same_as_sealed_on_20ms"] for t in trials if t["window_us"] == 20000
            ),
            independent_correspondences=0,
            optimizer_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
