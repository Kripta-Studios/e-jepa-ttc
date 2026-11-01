"""Run isolated PHASE release diagnostics; no official inference or checkpoints are mutated."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from operational.release_research.evaluation import compare
from operational.release_research.photometric import recover_q_and_light, simulate_patch
from operational.release_research.physics import (
    Measurement,
    estimate_expansion,
    observe,
    q_to_raw_phase,
    raw_phase_to_q,
    transport,
)
from operational.release_research.protocol import ResearchRun, sha256, verify_artifacts, write_json
from operational.release_research.visibility import sensor_log_scale_change, surface_log_expansion

HEAD = "PHASE_ONLY_EVENT_S7_TRAIN40"


def read(path: Path) -> dict:
    """Read a local receipt without changing it."""
    return json.loads(path.read_text(encoding="utf-8"))


def sources() -> dict[str, Path]:
    """Bind every module of this opt-in implementation plus the unchanged emission contract."""
    paths = [
        Path(__file__),
        Path("src/e_jepa_ttc/simplex_t/phase.py"),
        *Path("operational/release_research").glob("*.py"),
    ]
    return {"source:" + p.as_posix(): p for p in paths}


def verify_complete(folder: Path) -> dict:
    """Reject partial/failed receipts and verify their complete artifact inventory."""
    complete = read(folder / "COMPLETE.json")
    if complete.get("status") != "COMPLETE" or (folder / "FAILED.json").exists():
        raise ValueError("Require a completed, nonfailed parent")
    verify_artifacts(folder, complete["artifacts"])
    return complete


def mechanisms(output: Path) -> None:
    """Execute a small deterministic law/degeneracy bank, not an R0 accuracy benchmark."""
    run = ResearchRun(
        output, sources(), {"experiment_id": "mechanisms-v1", "seed": 7, "updates": 0}
    )
    try:
        trajectories = []
        for q in (-1.0, -0.1, 0.0, 0.1, 1.0):
            for alpha in (0.5, 0.75, 1.0, 1.25, 2.0):
                moved, derivative = transport(q, 0.1)
                warped, _ = transport(q / alpha, 0.1 * alpha)
                trajectories.append(
                    {
                        "q": q,
                        "alpha": alpha,
                        "transport_q": moved,
                        "jacobian": derivative,
                        "equivariance_error": abs(alpha * warped - moved),
                    }
                )
        identifiable = np.zeros((2, 6))
        identifiable[:, [0, 3]] = [[1, 0], [0, 1]]
        ambiguous = np.zeros((2, 6))
        ambiguous[:, 0] = 1
        descriptors = {
            "rank2_q_identifiable": asdict(estimate_expansion(identifiable, np.array([0.4, 0.4]))),
            "aperture_ambiguous": asdict(estimate_expansion(ambiguous, np.array([0.4, 0.4]))),
        }
        scales = np.array([0.75, 1.0, 1.25])
        corrected = sensor_log_scale_change(0.2 + np.log(scales), np.ones(3), scales)
        expansion, valid = surface_log_expansion(
            np.stack((np.eye(2) * 1.1, np.eye(2) * 0.9, np.zeros((2, 2))))
        )
        photometric = []
        xy = np.column_stack((np.linspace(-2.0, 2.0, 64), np.zeros(64)))
        gradient = np.array([1.0, 0.0])
        for q in (-0.5, 0.0, 0.5):
            for light in (0.0, 0.3):
                for threshold in (0.01, 0.03):
                    simulation = simulate_patch(
                        xy,
                        gradient,
                        np.linspace(0.0, 0.5, 101),
                        q,
                        light,
                        threshold,
                        threshold * 0.8,
                    )
                    estimate = recover_q_and_light(simulation, xy, gradient)
                    photometric.append(
                        {
                            "q_true": q,
                            "light_true": light,
                            "threshold": threshold,
                            "events": len(simulation.events),
                            "estimate": estimate,
                        }
                    )
        write_json(
            output / "METRICS.json",
            {
                "scope": "synthetic mathematical mechanisms only, not model improvement",
                "trajectories": trajectories,
                "observability": descriptors,
                "photometric_S0": photometric,
                "crop_invariance_max_error": float(np.max(np.abs(corrected - 0.2))),
                "surface_expansion": [
                    float(x) if v else None for x, v in zip(expansion, valid, strict=True)
                ],
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def diagnose(screen: Path, output: Path) -> None:
    """O0 constant controls and D1-D3 on raw PAIR histories, target blind and CPU only."""
    complete = verify_complete(screen)
    freeze = read(screen / "FREEZE.json")
    if not complete.get("baseline_exact") or complete.get("targets_read") is not False:
        raise ValueError("Need target-blind, exact R0 parent replay")
    if freeze.get("schema") not in (
        "phase_canonical_screen149_v1",
        "phase_canonical_population2608_v1",
    ):
        raise ValueError("Unsupported screen lineage")
    if complete.get("freeze_sha256") != sha256(screen / "FREEZE.json"):
        raise ValueError("Parent freeze mismatch")
    records = read(screen / "PREDICTIONS.json")["records"]
    if [r["token"] for r in records] != freeze["tokens"] or [r["group"] for r in records] != freeze[
        "groups"
    ]:
        raise ValueError("Screen query identity differs from preregistration")
    if len({r["token"] for r in records}) != len(records):
        raise ValueError("Duplicate physical query")
    inputs = {
        **sources(),
        "parent_complete": screen / "COMPLETE.json",
        "parent_freeze": screen / "FREEZE.json",
    }
    # Bind all consumed inputs, not merely the parent's receipt bytes.
    inputs.update({"parent:" + n: screen / n for n in complete["artifacts"]})
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "O0-D123-frozen-v1",
            "seed": 7,
            "updates": 0,
            "exposure": "historically exposed development",
            "weights": [0.0, 0.5, 1.0],
            "history_max_s": 0.65,
            "clock_measurement": "raw event PAIR; not postfiltered R0",
            "measurement_variance": 0.01,
            "targets_read": False,
            "missing_policy": "retain original missing; clock exceptions fallback to R0",
            "intervals": "point-only; no recalibration or interval claim",
        },
    )
    rows, timings = [], []
    try:
        for record in records:
            start = time.perf_counter()
            folder = screen / "queries" / hashlib.sha256(record["token"].encode()).hexdigest()[:24]
            row = {k: record[k] for k in ("token", "group", "query_us")}
            baseline = record["results"]["original"]["native"][0]
            row.update(
                R0=baseline,
                EVENT_CONTEXT=record["results"]["original"]["native"][1],
                RGB=None,
                OBS_HALF=None,
                raw_rgb=None,
                raw_r0=None,
                correction=None,
                reason="original_missing",
            )
            if baseline is not None:
                with np.load(folder / "original/CONTEXT.npz", allow_pickle=False) as a:
                    rgb = a["R_CTX.output_raw_location"]
                    original = a[HEAD + ".output_raw_location"]
                    # GPU expm1 and the diagnostic CPU kernel can differ by a few
                    # FP32 ULPs. R0 remains the captured GPU value, bit for bit.
                    cpu_ttc = phase_to_ttc(torch.from_numpy(original)).numpy()
                    np.testing.assert_allclose(
                        cpu_ttc, a[HEAD + ".output_ttc"], rtol=1e-6, atol=1e-6
                    )
                    row["cpu_emission_abs_difference"] = float(abs(cpu_ttc[0] - baseline))
                    if float(a[HEAD + ".output_ttc"][0]) != baseline:
                        raise ValueError("Captured baseline output differs")
                    half = rgb + np.float32(0.5) * (original - rgb)
                    row.update(
                        raw_rgb=float(rgb[0]),
                        raw_r0=float(original[0]),
                        correction=float(original[0] - rgb[0]),
                        RGB=record["results"]["original"]["native"][2],
                        OBS_HALF=float(phase_to_ttc(torch.from_numpy(half))[0]),
                        reason="ok",
                    )
                measurements = []
                clocks = []
                with np.load(folder / "PAIR_INPUTS.npz", allow_pickle=False) as a:
                    for call in read(folder / "CALLS.json")["calls"]:
                        if call["prefix"] != "E" or call["status"] != "accepted":
                            continue
                        prefix = f"call{call['call_id']}.PAIR_E_MATCHED"
                        for item in call["observations"]:
                            i = item["batch_index"]
                            # Include the entire sensor support, not merely the anchor age.
                            if item["first_sensor_ts_us"] < record["query_us"] - 650000:
                                continue
                            measured = Measurement(
                                item["anchor_us"] / 1e6,
                                item["available_us"] / 1e6,
                                raw_phase_to_q(float(a[prefix + ".raw_phase"][i])),
                                0.01,
                                record["token"],
                            )
                            measurements.append(measured)
                            endpoints = item["timestamps_us"]
                            clocks.append(
                                {
                                    "dt_feature": float(a[prefix + ".input133"][i, 128]),
                                    "dt_sensor": (endpoints[-1] - endpoints[-2]) / 1e6,
                                }
                            )
                row["event_clocks"] = clocks
                for mode in ("ewma", "transport", "observer"):
                    result = observe(
                        measurements,
                        record["query_us"] / 1e6,
                        record["token"],
                        raw_phase_to_q(row["raw_r0"]),
                        mode,
                    )
                    if result.reason in ("no_causal_history", "contact_fallback"):
                        value = baseline
                    else:
                        try:
                            phase = q_to_raw_phase(result.q)
                        except ValueError:
                            value = baseline
                            result = type(result)(
                                result.q,
                                None,
                                result.observations,
                                "phase_domain_fallback",
                                result.innovations,
                            )
                        else:
                            value = float(
                                phase_to_ttc(torch.tensor([phase], dtype=torch.float32))[0]
                            )
                    row["CLOCK_" + mode] = value
                    row["clock_" + mode + "_trace"] = asdict(result)
            else:
                row.update({"CLOCK_" + mode: None for mode in ("ewma", "transport", "observer")})
            rows.append(row)
            timings.append((time.perf_counter() - start) * 1000)
        write_json(output / "PREDICTIONS.json", {"records": rows, "targets_read": False})
        write_json(
            output / "RUNTIME.json",
            {
                "scope": "CPU cached diagnostic including NPZ reads; not end-to-end latency",
                "queries": len(rows),
                "milliseconds_p50_p95_p99": np.quantile(timings, [0.5, 0.95, 0.99]).tolist(),
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def evaluate(run_path: Path, population: Path, output: Path) -> None:
    """Join authorized development targets only after diagnostic predictions are sealed."""
    verify_complete(run_path)
    parent = read(run_path / "FREEZE.json")
    if parent["config"].get("experiment_id") != "O0-D123-frozen-v1":
        raise ValueError("Expected target-blind diagnostic run")
    screen_freeze = read(Path(parent["input_paths"]["parent_freeze"]))
    if sha256(population) not in screen_freeze["inputs"].values():
        raise ValueError("Targets differ from the development population bound by the screen")
    rows = read(run_path / "PREDICTIONS.json")["records"]
    run = ResearchRun(
        output,
        {
            **sources(),
            "predictions": run_path / "PREDICTIONS.json",
            "parent_complete": run_path / "COMPLETE.json",
            "population": population,
        },
        {
            "experiment_id": "O0-D123-evaluation-v1",
            "seed": 7,
            "targets_read": True,
            "exposure": "development",
            "updates": 0,
        },
    )
    try:
        with np.load(population, allow_pickle=False) as a:
            tokens, targets, groups = a["token"].astype(str), a["target"], a["group"].astype(str)
        index = {t: i for i, t in enumerate(tokens)}
        if len(index) != len(tokens):
            raise ValueError("Duplicate target identity")
        ids = [index[row["token"]] for row in rows]
        target, group = targets[ids].astype(np.float64), groups[ids]
        if group.tolist() != [r["group"] for r in rows]:
            raise ValueError("Target acquisition mismatch")
        baseline = np.array([r["R0"] if r["R0"] is not None else np.nan for r in rows])
        comparisons = {}
        for name in ("RGB", "OBS_HALF", "CLOCK_ewma", "CLOCK_transport", "CLOCK_observer"):
            values = np.array([r[name] if r[name] is not None else np.nan for r in rows])
            comparisons[name] = compare(target, baseline, values, group)
            if name.startswith("CLOCK_"):
                event_context = np.array(
                    [r["EVENT_CONTEXT"] if r["EVENT_CONTEXT"] is not None else np.nan for r in rows]
                )
                comparisons[name + "_vs_event_context"] = compare(
                    target, event_context, values, group
                )
        for row, truth in zip(rows, target, strict=True):
            row["target"] = float(truth)
            if row["R0"] is not None:
                gain = abs(row["RGB"] - truth) - abs(row["R0"] - truth)
                target_phase = float(ttc_to_phase(torch.tensor([truth], dtype=torch.float64))[0])
                row.update(
                    gain_seconds=gain,
                    gain_raw_phase=abs(row["raw_rgb"] - target_phase)
                    - abs(row["raw_r0"] - target_phase),
                    oracle_error_seconds=min(abs(row["RGB"] - truth), abs(row["R0"] - truth)),
                )
        write_json(
            output / "DIAGNOSTICS.json",
            {"records": rows, "oracle": "unrealizable labeled opportunity only"},
        )
        write_json(output / "METRICS.json", comparisons)
        write_json(
            output / "DECISION.json",
            {
                "status": "diagnose",
                "release_modified": False,
                "learned_family_admitted": False,
                "reason": (
                    "Constant attenuation and raw PAIR controls do not establish "
                    "incremental physical quality; no native confirmation"
                ),
                "missing": [
                    "normal-flow q indicators on release ROIs",
                    "native full-chain conservation",
                    "photometric RAW/reference calibration",
                    "independent confirmation acquisitions",
                ],
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def report(evaluation: Path, output: Path) -> None:
    """Regenerate a concise report from sealed results, never hand-entered metrics."""
    verify_complete(evaluation)
    inputs = {
        **sources(),
        "metrics": evaluation / "METRICS.json",
        "evaluation_complete": evaluation / "COMPLETE.json",
    }
    run = ResearchRun(output, inputs, {"experiment_id": "research-report-v1", "updates": 0})
    try:
        metrics = read(evaluation / "METRICS.json")
        lines = [
            "# Diagnóstico congelado O0/D1–D3",
            "",
            "Población de desarrollo históricamente expuesta. No es confirmación independiente.",
            "Los controles CLOCK usan PAIR de eventos; su referencia es EVENT_CONTEXT.",
            "No se han reajustado pesos ni modificado el release oficial.",
            "",
            "| Comparación | MAE macro referencia (s) | MAE macro control (s) | "
            "IC95 delta (s) | Nuevas ausencias |",
            "|---|---:|---:|---|---:|",
        ]
        for name, values in metrics.items():
            old = values["baseline"]["macro_mae"]
            new = values["candidate"]["macro_mae"]
            ci = values["paired_macro_delta_ci95"]
            old_text = f"{old:.9f}" if old is not None else "no estimable"
            new_text = f"{new:.9f}" if new is not None else "no estimable"
            lines.append(f"| {name} | {old_text} | {new_text} | {ci} | {values['newly_missing']} |")
        lines.extend(
            [
                "",
                "El bootstrap conserva adquisiciones completas; los intervalos son descriptivos.",
                "La pérdida original de cobertura se conserva en el denominador.",
                "OBS_HALF es un control constante prefijado, no un gate de observabilidad.",
                "No se infiere Overall nativo desde esta población ni se promociona un candidato.",
                "",
            ]
        )
        (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(output / "METRICS.json", metrics)
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Expose separate mechanism, blind extraction, and authorized development evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("mechanisms", "diagnose", "evaluate", "report"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "diagnose":
            command.add_argument("--screen", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--run", type=Path, required=True)
            command.add_argument("--population", type=Path, required=True)
        if name == "report":
            command.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "mechanisms":
        mechanisms(args.output)
    elif args.command == "diagnose":
        diagnose(args.screen, args.output)
    elif args.command == "evaluate":
        evaluate(args.run, args.population, args.output)
    else:
        report(args.evaluation, args.output)


if __name__ == "__main__":
    main()
