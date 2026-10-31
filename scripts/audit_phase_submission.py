"""Bind the historical submitted ZIP to its actual runtime and sensor metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_submission.protocol import interpolate_negative_predictions
from operational.phase_submission.verify import verify_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("historical_run", "original_zip", "pdf", "cpu_json", "slider_contract", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    root = args.historical_run
    records = [json.loads(line) for line in (root / "all_rows.jsonl").read_text().splitlines()]
    expected: dict[str, list[tuple[int, str]]] = {}
    for row in records:
        identity = row["identity"]
        expected.setdefault(identity["sequence_id"], []).append(
            (identity["index"], identity["timestamp_text"])
        )
    manifest = read(root / "manifest.json")
    digest = file_sha(args.original_zip)
    if digest != manifest["results_zip_sha256"]:
        raise ValueError("The actual submitted ZIP does not match the historical run")
    verified = verify_results(args.original_zip, records, expected)
    if verified["rows"] != 434:
        raise ValueError("Historical population is incomplete")
    freeze = read(root / "RUN_FREEZE.json")
    cameras = {}
    for sequence, calibration in freeze["calibrations"].items():
        road = calibration["source_kind"] == "road_right"
        cameras[sequence] = {
            "pair": "Prophesee EVK-v4 + FLIR Blackfly-S (right)"
            if road
            else "Inivation DVXplorer + DAHENG MER2 (beam-splitter testbed)",
            "nominal_lens_mm": 16 if road else None,
            "lens_evidence": "inferred by matching calibrated FoV to paper Table II 16mm pair"
            if road
            else "nominal mm not established; use supplied calibration and identify testbed",
            "calibration": calibration,
            "horizontal_fov_deg": {
                kind: math.degrees(
                    2
                    * math.atan(
                        calibration[kind + "_resolution"][0]
                        / (2 * calibration[kind + "_intrinsics"][0])
                    )
                )
                for kind in ("event", "rgb")
            },
        }
    local_slider = {}
    for speed in (750, 1000):
        sequence = f"Slider-{speed}"
        rows = [r for r in records if r["identity"]["sequence_id"] == sequence]
        gt_path = args.slider_contract / f"{speed}_gt.csv"
        gt = np.loadtxt(gt_path, delimiter=",")
        lookup = {int(row[0]): row for row in gt}
        targets = []
        for row in rows:
            identity = row["identity"]
            target = lookup[identity["index"]]
            if abs(target[1] - float(identity["timestamp_text"])) > 2e-6:
                raise ValueError("Public Slider target clock differs")
            targets.append(target[4])
        times = np.array([float(r["identity"]["timestamp_text"]) for r in rows])
        raw = np.array([r["prediction_ttc_seconds"] for r in rows])
        processed, changed = interpolate_negative_predictions(times, raw)
        target = np.array(targets)
        local_slider[sequence] = {
            "scope": "public targets only; does not certify organizer target equivalence",
            "target_sha256": file_sha(gt_path),
            "rows": len(rows),
            "raw_relative_error_pct": float(np.mean(abs(raw - target) / abs(target)) * 100),
            "processed_relative_error_pct": float(
                np.mean(abs(processed - target) / abs(target)) * 100
            ),
            "negative_changes": [
                {"index": rows[i]["identity"]["index"], "raw": raw[i], "processed": processed[i]}
                for i in changed
            ],
        }
    atomic_receipt(
        args.output / "HISTORICAL_AUDIT.json",
        {
            "original_zip_sha256": digest,
            "evaluation_pdf_sha256": file_sha(args.pdf),
            "original_zip_matches_historical_manifest": True,
            "verification": verified,
            "historical_environment": read(root / "environment.json"),
            "cpu_observed_same_host_2026_10_31": json.loads(
                args.cpu_json.read_text(encoding="utf-8-sig")
            ),
            "cpu_provenance": (
                "current hardware queried on the same laptop; historical run did not log CPU"
            ),
            "author": "Álvaro Schwiedop Souto",
            "affiliation": None,
            "sole_author_user_confirmed": True,
            "cameras": cameras,
            "local_slider_protocol_audit": local_slider,
            "organizer_email_rules": {
                "source": "email supplied verbatim by user, no date headers supplied",
                "all_bbox_timestamps_required": True,
                "all_434_received": True,
                "negative_values_permitted_in_raw_files": True,
                "negative_processing": "linear interpolation between surrounding valid predictions",
                "ground_truth": "matched or interpolated to bbox reference timestamps",
            },
            "primary_sources": [
                "https://nail-hnu.github.io/EvTTC/about/sensor/",
                "https://arxiv.org/html/2412.05053v1",
                "https://nail-hnu.github.io/EvTTC/download/data_format/",
                "https://nail-hnu.github.io/EventAidedTTC/",
            ],
            "public_source_revisions": {
                "NAIL-HNU/event_aided_ttc": "79ff0842955304ec4f6164ec09baddc71386d225",
                "NAIL-HNU/EvTTC": "015e62d21c942356f2e85ff50b9dd0dcf33dedce",
            },
            "source_hashes": {
                name: file_sha(root / name)
                for name in (
                    "RUN_FREEZE.json",
                    "all_rows.jsonl",
                    "environment.json",
                    "manifest.json",
                )
            },
        },
    )
    print(f"Verified historical434 rows,2 negatives,ZIP {digest}")


if __name__ == "__main__":
    main()
