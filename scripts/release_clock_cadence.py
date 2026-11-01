"""Separate fixed clock readout effects from the previously sealed PAIR cadence change."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.release_research.evaluation import compare
from operational.release_research.physics import (
    ClockResult,
    Measurement,
    observe,
    q_to_raw_phase,
    raw_phase_to_q,
)
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/CLOCK_CADENCE_CONTROL.md")
MODES = ("persistence", "ewma", "transport", "observer")


def inputs() -> dict[str, Path]:
    return {
        **sources(),
        "source:scripts/release_clock_cadence.py": Path(__file__),
        "preregistration": PLAN,
    }


def clock_readout(
    measurements: list[Measurement], query_s: float, frame: str, fallback_q: float, mode: str
) -> ClockResult:
    """Use the existing fixed observer and an explicit untransported persistence control."""
    if mode == "persistence":
        # First use the validated causal admission path; do not let persistence accept future data.
        admitted = [
            m for m in measurements if query_s - 0.65 <= m.time_s <= m.available_s <= query_s
        ]
        validation = observe(measurements, query_s, frame, fallback_q, "ewma")
        if not admitted:
            return validation
        return ClockResult(max(admitted, key=lambda m: m.time_s).q, None, len(admitted), "ok")
    if mode not in ("ewma", "transport", "observer"):
        raise ValueError("Unknown clock mode")
    return observe(measurements, query_s, frame, fallback_q, mode)


def extract(screen: Path, output: Path) -> None:
    """Reuse sealed raw PAIR phases and histories; this stage never reads TTC targets."""
    parent = verify_complete(screen)
    freeze = read(screen / "FREEZE.json")
    rows = read(screen / "PREDICTIONS.json")["records"]
    if (
        parent["targets_read"]
        or not parent["baseline_exact"]
        or freeze["schema"] != "phase_canonical_screen149_v1"
        or len(rows) != 149
        or [r["token"] for r in rows] != freeze["tokens"]
    ):
        raise ValueError("Require the intact blind screen149 population")
    files = {
        **inputs(),
        "parent": screen / "COMPLETE.json",
        "predictions": screen / "PREDICTIONS.json",
    }
    for row in rows:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        for name in (
            "CALLS.json",
            "original/PAIR_REPLAY.npz",
            "canonical/PAIR_REPLAY.npz",
            "original/CALLS.json",
            "canonical/CALLS.json",
        ):
            files[identity + ":" + name] = screen / "queries" / identity / name
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "clock-cadence-extract-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "modes": MODES,
            "cadences": ["original", "canonical"],
            "seed": 7,
            "scope": "fixed readout diagnostic, not integrated fusion replacement",
        },
    )
    try:
        results = []
        for row in rows:
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            folder = screen / "queries" / identity
            source_calls = read(folder / "CALLS.json")["calls"]
            result = {k: row[k] for k in ("token", "group", "query_us")}
            result["cadences"] = {}
            for cadence in ("original", "canonical"):
                baseline = row["results"][cadence]["native"][2]
                calls = {c["call_id"]: c for c in read(folder / cadence / "CALLS.json")["calls"]}
                measurements = []
                with np.load(folder / cadence / "PAIR_REPLAY.npz", allow_pickle=False) as arrays:
                    for call in source_calls:
                        if call["prefix"] != "E" or calls[call["call_id"]]["status"] != "accepted":
                            continue
                        phase = arrays[f"call{call['call_id']}.raw_phase"]
                        for item in call["observations"]:
                            if item["first_sensor_ts_us"] < row["query_us"] - 650000:
                                continue
                            measurements.append(
                                Measurement(
                                    item["anchor_us"] / 1e6,
                                    item["available_us"] / 1e6,
                                    raw_phase_to_q(float(phase[item["batch_index"]])),
                                    0.01,
                                    row["token"],
                                )
                            )
                readouts = {}
                for mode in MODES:
                    if baseline is None:
                        readouts[mode] = {"ttc": None, "reason": "event_context_absent"}
                        continue
                    # Zero is only a placeholder; every fallback returns the exact baseline below.
                    estimate = clock_readout(
                        measurements, row["query_us"] / 1e6, row["token"], 0.0, mode
                    )
                    value, reason = baseline, estimate.reason
                    if reason not in ("no_causal_history", "contact_fallback"):
                        try:
                            phase = q_to_raw_phase(estimate.q)
                            value = float(
                                phase_to_ttc(torch.tensor([phase], dtype=torch.float32))[0]
                            )
                        except ValueError:
                            reason = "phase_domain_fallback"
                    readouts[mode] = {"ttc": value, "reason": reason, "trace": asdict(estimate)}
                result["cadences"][cadence] = {
                    "event_context": baseline,
                    "readouts": readouts,
                    "measurements": [asdict(m) for m in measurements],
                }
            results.append(result)
        write_json(output / "PREDICTIONS.json", {"records": results, "targets_read": False})
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def evaluate(predictions: Path, evaluation: Path, output: Path) -> None:
    """Compare every predefined readout with its matching context, preserving missing output."""
    for folder in (predictions, evaluation):
        verify_complete(folder)
    freeze = read(predictions / "FREEZE.json")
    if freeze["config"]["targets_read"] or freeze["inputs"]["preregistration"] != sha256(PLAN):
        raise ValueError("Require unchanged blind clock predictions")
    rows = read(predictions / "PREDICTIONS.json")["records"]
    scored = read(evaluation / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Target identity mismatch")
    run = ResearchRun(
        output,
        {
            **inputs(),
            "parent": predictions / "COMPLETE.json",
            "predictions": predictions / "PREDICTIONS.json",
            "evaluation": evaluation / "COMPLETE.json",
            "targets": evaluation / "DIAGNOSTICS.json",
        },
        {
            "experiment_id": "clock-cadence-evaluate-v1",
            "targets_read": True,
            "optimizer_updates": 0,
            "D4_admitted": False,
        },
    )
    try:
        target = np.array([r["target"] for r in scored], float)
        groups = np.array([r["group"] for r in rows])
        metrics = {}
        lines = [
            "# Reloj y cadencia: control de dependencia",
            "",
            "Desarrollo expuesto, parámetros fijos, comparación con cada contexto de eventos.",
            "",
            "| Cadencia | Lectura | MAE contexto | MAE lectura | Delta CI95 |",
            "|---|---|---:|---:|---|",
        ]
        for cadence in ("original", "canonical"):
            baseline = np.array([r["cadences"][cadence]["event_context"] for r in rows], float)
            metrics[cadence] = {}
            for mode in MODES:
                candidate = np.array(
                    [r["cadences"][cadence]["readouts"][mode]["ttc"] for r in rows], float
                )
                value = compare(target, baseline, candidate, groups)
                metrics[cadence][mode] = value
                lines.append(
                    f"| {cadence} | {mode} | {value['baseline']['macro_mae']:.9f} | "
                    f"{value['candidate']['macro_mae']:.9f} | {value['paired_macro_delta_ci95']} |"
                )
        write_json(output / "METRICS.json", {"cadences": metrics, "D4_admitted": False})
        lines += [
            "",
            "No cambia la cabeza de fusión ni selecciona un nuevo candidato.",
            "Las ausencias y fallbacks se conservan en las predicciones selladas.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("mode", choices=("extract", "evaluate"))
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--evaluation", type=Path, default=root / "evaluation149_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args.screen, args.output)
    elif args.predictions is None:
        parser.error("evaluate requires --predictions")
    else:
        evaluate(args.predictions, args.evaluation, args.output)


if __name__ == "__main__":
    main()
