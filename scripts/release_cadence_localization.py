"""Localize the fixed canonical PAIR intervention without selecting a replacement candidate."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from operational.phase_observation_cache.replay import replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.phase_timewarp.screen import replay_pair_batches
from operational.phase_timewarp.screen_metrics import compare_screen
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.build_phase_native_context_augmentations import native_vector
from scripts.final_train40_transfer import load_models
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/CADENCE_LOCALIZATION_PREREGISTRATION.md")
ARMS = ("original", "event", "rgb", "both")


def source_paths() -> dict[str, Path]:
    """Bind all executed numerical dependencies without editing the active population sources."""
    result = sources()
    for name in (
        *DEPENDENCIES,
        "scripts/release_cadence_localization.py",
        "scripts/build_phase_native_context_augmentations.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_observation_cache/replay.py",
        "operational/phase_timewarp/screen.py",
        "operational/phase_timewarp/screen_metrics.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_robust/bundle.py",
    ):
        result["source:" + name] = Path(name)
    return result


def extract(screen: Path, upstream: Path, output: Path) -> None:
    """Replay the same compact batches in four fixed branches, without reading targets."""
    parent = verify_complete(screen)
    freeze = read(screen / "FREEZE.json")
    if (
        not parent["baseline_exact"]
        or parent["targets_read"]
        or freeze["schema"] != "phase_canonical_screen149_v1"
    ):
        raise ValueError("Require the sealed target-blind screen149")
    rows = read(screen / "PREDICTIONS.json")["records"]
    if len(rows) != 149 or [r["token"] for r in rows] != freeze["tokens"]:
        raise ValueError("Population differs")
    paths = verified_upstream_paths(upstream)
    inputs = {
        **source_paths(),
        **{"model:" + k: v for k, v in paths.items()},
        "parent_complete": screen / "COMPLETE.json",
        "parent_predictions": screen / "PREDICTIONS.json",
        "preregistration": PLAN,
    }
    for row in rows:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        for name in ("PAIR_INPUTS.npz", "CALLS.json", "original/CONTEXT.npz"):
            inputs[identity + ":" + name] = screen / "queries" / identity / name
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "T1-modality-localization-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "exposure": "development",
            "device": "cpu",
            "threads": 1,
            "arms": ARMS,
            "seed": 7,
            "historical_ttc_rtol": 1e-4,
            "historical_ttc_atol": 1e-4,
        },
    )
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        models, normalizers = load_models(paths, "cpu")
        variants = {
            "original": models,
            "event": {**models, "PAIR_E_MATCHED": CanonicalPair(models["PAIR_E_MATCHED"])},
            "rgb": {**models, "PAIR_R": CanonicalPair(models["PAIR_R"])},
            "both": {
                **models,
                **{k: CanonicalPair(models[k]) for k in ("PAIR_E_MATCHED", "PAIR_R")},
            },
        }
        results = []
        for index, row in enumerate(rows):
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            folder = screen / "queries" / identity
            calls = read(folder / "CALLS.json")["calls"]
            with np.load(folder / "PAIR_INPUTS.npz", allow_pickle=False) as stored:
                arrays = {k: stored[k] for k in stored.files}
            result = {k: row[k] for k in ("token", "group", "query_us")}
            result["arms"] = {}
            for arm, current in variants.items():
                computed = replay_pair_batches(current, calls, arrays, "cpu")
                _, context = replay(
                    current, normalizers, computed["observations"], row["query_us"], "cpu"
                )
                values = native_vector(context, "cpu")
                result["arms"][arm] = {
                    "native": [float(v) if np.isfinite(v) else None for v in values],
                    "rejected_calls": [c for c in computed["calls"] if c["status"] != "accepted"],
                    "observations": {k: len(v) for k, v in computed["observations"].items()},
                }
                destination = output / "queries" / identity / arm
                destination.mkdir(parents=True)
                np.savez_compressed(destination / "CONTEXT.npz", allow_pickle=False, **context)
                if arm == "original":
                    historical = np.asarray(row["results"]["original"]["native"], float)
                    if not np.array_equal(np.isfinite(values), np.isfinite(historical)):
                        raise ValueError("Historical availability differs")
                    valid = np.isfinite(values)
                    if not np.allclose(values[valid], historical[valid], rtol=1e-4, atol=1e-4):
                        raise ValueError("Original CPU TTC fails fixed historical tolerance")
                    result["historical_ttc_max_abs"] = (
                        float(np.max(np.abs(values[valid] - historical[valid])))
                        if valid.any()
                        else None
                    )
                    with np.load(folder / "original/CONTEXT.npz", allow_pickle=False) as archive:
                        if set(archive.files) != set(context):
                            raise ValueError("Historical context fields differ")
                        result["historical_field_max_abs"] = {
                            k: float(
                                np.max(np.abs(context[k].astype(float) - archive[k].astype(float)))
                            )
                            for k in context
                        }
            event_calls = [c for c in calls if c["prefix"] == "E" and c["status"] == "accepted"]
            identity_clock = bool(event_calls) and all(
                np.all(arrays[f"call{c['call_id']}.delta"] == np.float32(0.1)) for c in event_calls
            )
            result["nominal_event_identity"] = identity_clock
            if identity_clock:
                for left, right in (("original", "event"), ("rgb", "both")):
                    if result["arms"][left] != result["arms"][right]:
                        raise ValueError("Nominal event identity failed in full output")
            results.append(result)
            print(f"{index + 1}/149 {row['token']}", flush=True)
        write_json(output / "PREDICTIONS.json", {"records": results})
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "queries": len(rows),
                "producer_calls": 0,
                "scope": "CPU compact replay; not sensor-to-output latency",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def evaluate(predictions: Path, evaluated: Path, output: Path) -> None:
    """Describe all four arms using the paired CPU original, without changing the F0 gate."""
    verify_complete(predictions)
    verify_complete(evaluated)
    freeze = read(predictions / "FREEZE.json")
    if freeze["config"]["targets_read"] or freeze["inputs"]["preregistration"] != sha256(PLAN):
        raise ValueError("Prediction preregistration differs")
    rows = read(predictions / "PREDICTIONS.json")["records"]
    scored = read(evaluated / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Target identities differ")
    inputs = {
        **source_paths(),
        "predictions": predictions / "PREDICTIONS.json",
        "predictions_complete": predictions / "COMPLETE.json",
        "preregistration": PLAN,
        "evaluation": evaluated / "DIAGNOSTICS.json",
        "evaluation_complete": evaluated / "COMPLETE.json",
    }
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "T1-localization-evaluation-v1",
            "targets_read": True,
            "optimizer_updates": 0,
            "scope": "mechanism localization only; no candidate selection",
        },
    )
    try:
        groups = np.asarray([r["group"] for r in rows])
        target = np.asarray([r["target"] for r in scored], float)
        values = {
            arm: np.asarray([r["arms"][arm]["native"][0] for r in rows], float) for arm in ARMS
        }
        report = {}
        for domain, mask in (("road", groups != "Slider-500"), ("slider", groups == "Slider-500")):
            report[domain] = {
                arm: compare_screen(
                    target[mask], values["original"][mask], values[arm][mask], groups[mask]
                )
                for arm in ARMS
            }
        interaction = values["both"] - values["event"] - values["rgb"] + values["original"]
        result = {
            "domains": report,
            "identity_event_queries": sum(r["nominal_event_identity"] for r in rows),
            "historical_ttc_max_abs": max(r["historical_ttc_max_abs"] or 0 for r in rows),
            "prediction_interaction_max_abs": float(np.nanmax(np.abs(interaction))),
            "candidate_changed": False,
            "independent_confirmation": False,
        }
        write_json(output / "METRICS.json", result)
        lines = [
            "# Localización de la corrección PAIR",
            "",
            "Desarrollo expuesto; control original CPU emparejado. No selecciona candidato.",
            "",
            "| Dominio | Rama corregida | MAE macro (s) | RAE macro (%) | Pares |",
            "|---|---|---:|---:|---:|",
        ]
        for domain, arms in report.items():
            for arm, metrics in arms.items():
                m = metrics["canonical"]
                lines.append(
                    f"| {domain} | {arm} | {m['mae_seconds']:.9f} | "
                    f"{m['relative_error_pct']:.9f} | {metrics['paired']} |"
                )
        lines += [
            "",
            f"Consultas con identidad exacta de eventos: {result['identity_event_queries']}.",
            "Deriva máxima TTC del original CPU frente al archivo GPU: "
            f"{result['historical_ttc_max_abs']:.9g} s.",
            "Interacción máxima en predicción ambas−E−RGB+original: "
            f"{result['prediction_interaction_max_abs']:.9g} s.",
            "El candidato de population2608 y sus criterios permanecen intactos.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Separate label-blind replay from the subsequent development evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "evaluate"))
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--evaluated", type=Path, default=Path("output/release_research_20261031/evaluation149_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args.screen, args.upstream, args.output)
    elif args.predictions is None:
        parser.error("--predictions required for evaluate")
    else:
        evaluate(args.predictions, args.evaluated, args.output)


if __name__ == "__main__":
    main()
