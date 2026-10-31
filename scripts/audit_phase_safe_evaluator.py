"""Versioned public-target audit; preserves historical artifacts and never fits clocks.

Run with ``python -m scripts.audit_phase_safe_evaluator`` from the repository root.
The only repaired target assignment follows the official download page's sequence
links. The source files and their SHA256 values are pinned before any scoring.
Slider-750 remains a public-target diagnostic, not an official-equivalent score.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from operational.phase_submission.protocol import interpolate_negative_predictions

ROAD_PAGE = "https://nail-hnu.github.io/EvTTC/download/download/"
SLIDER_PAGE = "https://nail-hnu.github.io/EventAidedTTC/"
PUBLIC_SOURCES = {
    "CCRs1-low": {
        "drive_id": "1l5_doDb30GsscNAtJquthr96RYZUXGBy",
        "sha256": "1c5adba8667acb8096eb80c26e9a9229ca3a7d76d0a6d998085c2d9b4e820fed",
        "source_page": ROAD_PAGE,
        "page_sequence": "CCRs-1-low-100%",
    },
    "CCRs1-medium": {
        "drive_id": "1cTwpIZIsRucXClgz8loDjxfaCFMGzN9C",
        "sha256": "780ee30ff9775da8452102bbbb9f5120ae903055007348096ecbc24b594b8c76",
        "source_page": ROAD_PAGE,
        "page_sequence": "CCRs-1-medium-100%",
    },
    "Slider-750": {
        "drive_id": "1mrhJplOKqOKwsOqqDe1fFcoXkbNLk1pP",
        "sha256": "4fb29e203ff057a262d0d118641769ee22306e57a6361f23b53313a0e03c2641",
        "source_page": SLIDER_PAGE,
        "page_sequence": "Slider 750",
    },
    "Slider-1000": {
        "drive_id": "1iX5GeyhrnxixolBY2U40cfo3DHTRL9HB",
        "sha256": "1d71a51e4fbd21280f53425e05c63367f08ea3662e32017cde913128b8e60f78",
        "source_page": SLIDER_PAGE,
        "page_sequence": "Slider 1000",
    },
}


def sha(data: bytes) -> str:
    """Return a byte-exact SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_verified_csv(data: bytes, expected_sha256: str) -> list[list[Any]]:
    """Verify original source bytes and parse timestamp seconds into integer us."""
    if sha(data) != expected_sha256:
        raise ValueError("Public source SHA256 differs; manual provenance review required")
    text = data.decode("utf-8-sig")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    return [
        [int(Decimal(row[1]) * 1_000_000), float(row[4])]
        for row in csv.reader(io.StringIO(text), delimiter=delimiter)
    ]


def assign_public_road_labels(
    local_labels: dict[str, list[list[Any]]], public_labels: dict[str, list[list[Any]]]
) -> dict[str, list[list[Any]]]:
    """Assign road GT by official sequence links, retaining original local inputs."""
    result = copy.deepcopy(local_labels)
    for name in ("CCRs1-low", "CCRs1-medium"):
        result[name] = copy.deepcopy(public_labels[name])
    return result


def verified_clock_map(clocks: dict[str, int], evidence: dict[str, Any]) -> dict[str, int]:
    """Accept only unchanged offsets independently established by every RGB frame."""
    if evidence.get("labels_read") is not False:
        raise ValueError("Clock evidence must be label-blind")
    result = {}
    for sequence in evidence["sequences"]:
        name = sequence["competition_sequence"]
        matches = sequence["matches"]
        if sequence.get("status") != "unique" or len(matches) != 1:
            raise ValueError("Clock requires unique independent RGB identity")
        match = matches[0]
        frames = match["all_rgb_frames"]
        if (
            not match["all_query_rgb_timestamps_match"]
            or not frames
            or not all(
                f["equal"] and f["competition_sha256"] == f["original_sha256"] for f in frames
            )
            or clocks.get(name) != match["timestamp_offset_us"]
        ):
            raise ValueError("Clock differs from independent RGB identity")
        result[name] = clocks[name]
    if set(result) != set(clocks):
        raise ValueError("Clock population differs from independent RGB identity")
    return result


def bounded_target(labels: list[list[Any]], timestamp_us: int) -> float:
    """Linearly interpolate TTC within public label support without extrapolation."""
    series = np.asarray(labels, dtype=np.float64)
    times, values = series[:, 0], series[:, 1]
    if not np.isfinite(series).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Labels must be finite with strictly increasing timestamps")
    if timestamp_us < times[0] or timestamp_us > times[-1]:
        raise ValueError("Query outside public label support")
    value = float(np.interp(timestamp_us, times, values))
    if value <= 0:
        raise ValueError("Relative-error audit requires positive targets")
    return value


def download(url: str) -> bytes:
    """Read a small public source with a bounded network timeout."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read(2_000_000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/phase_safe_fusion/evaluator_audit")
    )
    parser.add_argument("--replay", type=Path, default=Path("output/phase_submission/replay"))
    parser.add_argument(
        "--historical-run",
        type=Path,
        default=Path("output/phase_submission/extracted/historical_run"),
    )
    parser.add_argument("--inputs", type=Path, default=Path("output/evttc_transfer_v2/inputs"))
    parser.add_argument(
        "--identity", type=Path, default=Path("output/evttc_transfer_v2/D00_identity/report.json")
    )
    parser.add_argument(
        "--official-text", type=Path, default=Path("output/phase_startup/official_pdf.txt")
    )
    parser.add_argument("--offline", action="store_true", help="Use already preserved source bytes")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "sources"
    source_dir.mkdir(exist_ok=True)
    source_receipts = {}
    for name, specification in PUBLIC_SOURCES.items():
        path = source_dir / f"{name}.csv"
        url = "https://drive.google.com/uc?export=download&id=" + specification["drive_id"]
        data = path.read_bytes() if args.offline else download(url)
        parse_verified_csv(data, specification["sha256"])
        if not args.offline:
            path.write_bytes(data)
        source_receipts[name] = {**specification, "download_url": url, "bytes": len(data)}
    for label, url in (
        ("official_download_page", ROAD_PAGE),
        ("official_slider_page", SLIDER_PAGE),
    ):
        path = source_dir / f"{label}.html"
        data = path.read_bytes() if args.offline else download(url)
        if not args.offline:
            path.write_bytes(data)
        source_receipts[label] = {"url": url, "sha256": sha(data), "bytes": len(data)}
    legacy_labels = read(args.inputs / "labels.json")
    public_road = {
        name: parse_verified_csv(
            (source_dir / f"{name}.csv").read_bytes(), PUBLIC_SOURCES[name]["sha256"]
        )
        for name in ("CCRs1-low", "CCRs1-medium")
    }
    labels = assign_public_road_labels(legacy_labels, public_road)
    clocks = verified_clock_map(read(args.inputs / "clock_map.json"), read(args.identity))
    ledger_path = args.replay / "all_rows.jsonl"
    if sha(ledger_path.read_bytes()) != read(args.replay / "COMPLETE.json")["ledger_sha256"]:
        raise ValueError("Replay ledger seal changed")
    historical_path = args.historical_run / "all_rows.jsonl"
    if (
        sha(historical_path.read_bytes())
        != "e420f65ef05ce54a82502ceaaf9cc18b0c339ce7761d9aa049a687c60382126f"
    ):
        raise ValueError("Original 2026-10-28 ledger hash changed")
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    history = {
        (r["identity"]["sequence_id"], r["identity"]["row_index"]): r
        for r in map(json.loads, historical_path.read_text().splitlines())
    }
    slider = {
        name: {int(r[0]): r for r in np.loadtxt(source_dir / f"{name}.csv", delimiter=",")}
        for name in ("Slider-750", "Slider-1000")
    }
    rows, original = [], []
    for record in records:
        identity = record["identity"]
        name = identity["sequence_id"]
        old = history[(name, identity["row_index"])]
        if identity != old["identity"]:
            raise ValueError("Historical and replay query identities differ")
        original.append(old["prediction_ttc_seconds"])
        if name in slider:
            label = slider[name][identity["index"]]
            if abs(label[1] - float(identity["timestamp_text"])) > 2e-6:
                raise ValueError("Slider frame timestamp mismatch")
            target = float(label[4])
            offset = 0
        else:
            offset = clocks[name]
            target = bounded_target(labels[name], identity["anchor_us"] + offset)
        rows.append(
            {
                "identity": identity,
                "token": name + ":" + str(identity["row_index"]),
                "target_ttc_seconds": target,
                "clock_offset_us": offset,
                "official_equivalence": "unverified"
                if name == "Slider-750"
                else "aggregate_reproduced_to_pdf_precision",
            }
        )
    if len(rows) != 434 or len({r["token"] for r in rows}) != 434:
        raise ValueError("Expected all 434 unique reference queries")
    target = np.array([r["target_ttc_seconds"] for r in rows])
    if not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("Expected 434 finite positive supported targets")
    sequence = np.array([r["identity"]["sequence_id"] for r in rows])
    timestamps = np.array([float(r["identity"]["timestamp_text"]) for r in rows])
    original_array = np.asarray(original)
    pdf = args.official_text.read_text(encoding="utf-8")
    official = {
        name: {"count": int(count), "rte_pct": float(score), "runtime_seconds": float(runtime)}
        for name, count, score, runtime in re.findall(
            r"^\s*(CCRs[12]-(?:low|medium|high)|CCRm-(?:low|medium)|Slider-(?:750|1000))\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$",
            pdf,
            re.MULTILINE,
        )
    }
    if len(official) != 10:
        raise ValueError("Official PDF text must supply ten sequence results")
    comparison = {}
    current_array = np.asarray([r["prediction_ttc_seconds"] for r in records])
    original_processed = original_array.copy()
    current_processed = current_array.copy()
    for name, reference in official.items():
        selected = sequence == name
        processed, changed = interpolate_negative_predictions(
            timestamps[selected], original_array[selected]
        )
        original_processed[selected] = processed
        current_processed[selected], current_changed = interpolate_negative_predictions(
            timestamps[selected], current_array[selected]
        )
        score = float(np.mean(abs(processed - target[selected]) / target[selected]) * 100)
        comparison[name] = {
            "local_count": int(selected.sum()),
            "local_original_rte_pct": score,
            "official_pdf": reference,
            "matches_pdf_rounding": round(score, 2) == reference["rte_pct"],
            "interpolated_negative_row_indices": changed,
            "frozen_submission_negative_row_indices": current_changed,
        }
    predictions = {
        "original_raw": original_array,
        "original_email_processed": original_processed,
        "frozen_submission_raw": current_array,
        "frozen_submission_email_processed": current_processed,
    }
    populations = {
        "all434_public_diagnostic": np.ones(434, dtype=bool),
        "road365": ~np.char.startswith(sequence, "Slider"),
        "slider69_public_diagnostic": np.char.startswith(sequence, "Slider"),
        "pdf_reproduced_sequences383": sequence != "Slider-750",
        **{name: sequence == name for name in official},
    }
    baseline_metrics = {
        population: {
            method: {
                "population": int(mask.sum()),
                "scored": int(mask.sum()),
                "relative_error_pct": float(
                    np.mean(abs(value[mask] - target[mask]) / target[mask]) * 100
                ),
                "mae_seconds": float(np.mean(abs(value[mask] - target[mask]))),
                "rmse_seconds": float(np.sqrt(np.mean((value[mask] - target[mask]) ** 2))),
            }
            for method, value in predictions.items()
        }
        for population, mask in populations.items()
    }
    target_path = args.output / "TARGETS.npz"
    np.savez_compressed(
        target_path,
        target=target,
        token=np.array([r["token"] for r in rows]),
        sequence=sequence,
        index=np.array([r["identity"]["index"] for r in rows]),
        row_index=np.array([r["identity"]["row_index"] for r in rows]),
        timestamp=timestamps,
        timestamp_text=np.array([r["identity"]["timestamp_text"] for r in rows]),
        anchor_us=np.array([r["identity"]["anchor_us"] for r in rows]),
    )
    (args.output / "TARGET_ROWS.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path = args.output / "BASELINE_PREDICTIONS.npz"
    np.savez_compressed(prediction_path, token=np.array([r["token"] for r in rows]), **predictions)
    receipt = {
        "schema": "phase_safe_evaluator_audit_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "434 public supported targets; Slider-750 official equivalence unresolved",
        "rows": 434,
        "clock_fit_performed": False,
        "clock_offsets_us_unchanged": clocks,
        "repair_basis": "Official sequence-to-GT download links, verified before score calculation",
        "public_low_equals_legacy_medium": public_road["CCRs1-low"]
        == legacy_labels["CCRs1-medium"],
        "public_medium_equals_legacy_low": public_road["CCRs1-medium"]
        == legacy_labels["CCRs1-low"],
        "source_receipts": source_receipts,
        "original_vs_official": comparison,
        "frozen_baseline_metrics": baseline_metrics,
        "frozen_baseline_binding": {
            "target_and_predictions_share_exact_replay_order": True,
            "historical_and_replay_full_identity_equal_all_rows": True,
            "candidate_selection_performed": False,
            "original_ledger_sha256": sha(historical_path.read_bytes()),
            "frozen_submission_ledger_sha256": sha(ledger_path.read_bytes()),
        },
        "official_equivalence": False,
        "remaining_blocker": (
            "Organizer's exact Slider-750 aligned target vector and evaluator/version; "
            "public GT hash is unchanged"
        ),
        "benchmark_requirements": [
            "Freeze candidate before reading targets; exclude all benchmark recordings "
            "from training and selection",
            "Report all 434 reference rows and per-sequence mean absolute relative error",
            "Keep raw negative outputs; apply organizer interpolation only for evaluation",
            "Do not compare local Slider-750 scores directly with official leaderboard scores",
            "Compare runtime only with matching hardware, input support and measured scope",
        ],
        "input_sha256": {
            str(p): sha(p.read_bytes())
            for p in (
                args.inputs / "labels.json",
                args.inputs / "clock_map.json",
                args.identity,
                ledger_path,
                historical_path,
                args.official_text,
            )
        },
        "output_sha256": {
            "TARGETS.npz": sha(target_path.read_bytes()),
            "BASELINE_PREDICTIONS.npz": sha(prediction_path.read_bytes()),
            "TARGET_ROWS.json": sha((args.output / "TARGET_ROWS.json").read_bytes()),
        },
        "script_sha256": sha(Path(__file__).read_bytes()),
    }
    (args.output / "AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"rows": 434, "comparison": comparison, "official_equivalence": False}, indent=2)
    )


if __name__ == "__main__":
    main()
