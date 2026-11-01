"""Measure causal clock error and lag under fixed synthetic kinematic regimes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from operational.release_research.physics import ClockConfig, Measurement, observe
from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import sources


def state(times: np.ndarray, velocity: float, acceleration: float) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise constant acceleration after one second, with continuous depth and speed."""
    elapsed = np.maximum(times - 1.0, 0.0)
    depth = 10.0 + velocity * times + 0.5 * acceleration * elapsed**2
    speed = velocity + acceleration * elapsed
    if np.any(depth <= 0):
        raise ValueError("Fixture crossed contact")
    return -speed / depth, depth


def first_time(times: np.ndarray, mask: np.ndarray) -> float | None:
    """Return the first observed crossing, leaving absent crossings explicit."""
    found = np.flatnonzero(mask)
    return float(times[found[0]]) if len(found) else None


def experiment(output: Path) -> None:
    """Run all fixed regimes and retain both favorable and unfavorable filter responses."""
    config = ClockConfig()
    run = ResearchRun(
        output,
        {
            **sources(),
            "source:scripts/release_clock_regimes.py": Path(__file__),
            "preregistration": Path(
                "docs/release_research_implementation_20261031/CLOCK_REGIMES_PREREGISTRATION.md"
            ),
        },
        {
            "experiment_id": "clock-regime-mechanism-v1",
            "real_targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
            "clock": asdict(config),
            "scope": "mechanism only; no R0 comparison or learned-family admission",
        },
    )
    try:
        query = np.arange(201, dtype=float) / 100
        sample = [0.0]
        cadence = (0.04, 0.14, 0.07, 0.09, 0.16)
        while sample[-1] < 2:
            sample.append(sample[-1] + cadence[(len(sample) - 1) % len(cadence)])
        times = np.array([t for t in sample if t <= 2 and not 0.7 <= t <= 0.95])
        noise = np.random.default_rng(7).normal(size=len(times))
        metrics, traces = {}, {}
        modes: tuple[Literal["ewma", "transport", "observer"], ...] = (
            "ewma",
            "transport",
            "observer",
        )
        for sigma in (0.0, 0.05):
            for regime, velocity, acceleration in (
                ("constant", -3.0, 0.0),
                ("braking", -3.0, 6.0),
                ("accelerating", -3.0, -4.0),
                ("receding", 3.0, 0.0),
            ):
                measured, _ = state(times, velocity, acceleration)
                target, depth = state(query, velocity, acceleration)
                measurements = [
                    Measurement(
                        float(t),
                        float(t + 0.025),
                        float(q + sigma * e),
                        max(sigma**2, 1e-8),
                        "fixture",
                    )
                    for t, q, e in zip(times, measured, noise, strict=True)
                ]
                predictions: dict[str, list] = {"persistence": [], **{m: [] for m in modes}}
                reasons: dict[str, list] = {m: [] for m in modes}
                valid = []
                for now in query:
                    available = [
                        m
                        for m in measurements
                        if now - config.history_s <= m.time_s <= m.available_s <= now
                    ]
                    valid.append(bool(available))
                    fallback = available[-1].q if available else 0.0
                    predictions["persistence"].append(fallback)
                    for mode in modes:
                        result = observe(
                            measurements, float(now), "fixture", fallback, mode, config
                        )
                        predictions[mode].append(result.q)
                        reasons[mode].append(result.reason)
                valid_array = np.array(valid)
                arrays = {name: np.array(values) for name, values in predictions.items()}
                case = f"{regime}_noise_{sigma}"
                traces[case] = {
                    "time": query.tolist(),
                    "target_q": target.tolist(),
                    "depth": depth.tolist(),
                    "available": valid,
                    "predictions": predictions,
                    "reasons": reasons,
                    "measurements": [asdict(m) for m in measurements],
                }
                metrics[case] = {}
                for name, values in arrays.items():
                    error = np.abs(values - target)
                    transition = valid_array & (query >= 1) & (query <= 1.25)
                    true_reversal = (
                        first_time(query, (query >= 1) & (target <= 0))
                        if regime == "braking"
                        else None
                    )
                    predicted_reversal = (
                        first_time(query, valid_array & (query >= 1) & (values <= 0))
                        if regime == "braking"
                        else None
                    )
                    metrics[case][name] = {
                        "queries": int(valid_array.sum()),
                        "q_mae": float(error[valid_array].mean()),
                        "q_p95": float(np.quantile(error[valid_array], 0.95)),
                        "transition_q_mae": float(error[transition].mean()),
                        "sign_errors": int(
                            np.sum(
                                valid_array
                                & (np.abs(target) > 1e-6)
                                & (np.sign(values) != np.sign(target))
                            )
                        ),
                        "true_first_warning_s": first_time(query, target >= 0.5),
                        "predicted_first_warning_s": first_time(
                            query, valid_array & (values >= 0.5)
                        ),
                        "true_reversal_s": true_reversal,
                        "predicted_reversal_s": predicted_reversal,
                        "reversal_delay_s": predicted_reversal - true_reversal
                        if predicted_reversal is not None and true_reversal is not None
                        else None,
                    }
                if sigma == 0 and acceleration == 0:
                    np.testing.assert_allclose(
                        arrays["transport"][valid_array], target[valid_array], atol=1e-12, rtol=0
                    )
        prefix = query <= 1.0
        for sigma in (0.0, 0.05):
            for mode in ("persistence", *modes):
                for alternative in ("braking", "accelerating"):
                    np.testing.assert_array_equal(
                        np.array(traces[f"constant_noise_{sigma}"]["predictions"][mode])[prefix],
                        np.array(traces[f"{alternative}_noise_{sigma}"]["predictions"][mode])[
                            prefix
                        ],
                    )
        write_json(output / "TRACES.json", traces)
        write_json(output / "METRICS.json", metrics)
        lines = [
            "# Reloj causal bajo cambios de régimen",
            "",
            "Banco sintético prefijado; no es evidencia de mejora frente a R0.",
            "El transporte sin ruido y velocidad constante es exacto (error <=1e-12).",
            "Los futuros distintos conservan predicciones idénticas hasta su divergencia.",
            "",
            "| Régimen/ruido | Método | MAE q | MAE q tras cambio | "
            "Errores de signo | Demora inversión (s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for case, results in metrics.items():
            for name, result in results.items():
                delay = result["reversal_delay_s"]
                text_delay = "—" if delay is None else f"{delay:.3f}"
                lines.append(
                    f"| {case} | {name} | {result['q_mae']:.6f} | "
                    f"{result['transition_q_mae']:.6f} | {result['sign_errors']} | {text_delay} |"
                )
        lines += [
            "",
            "El ruido de medida es conocido solo en este banco; "
            "las ventanas reales pueden compartir evidencia.",
            "Los tiempos de régimen son etiquetas de análisis, nunca entradas del estimador.",
            "La ausencia de aviso o inversión queda registrada como null; "
            "no se transforma en demora cero.",
            "El banco no admite D4 y no modifica parámetros tras observar sus resultados.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Execute the fixed synthetic bank in a new evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    experiment(parser.parse_args().output)


if __name__ == "__main__":
    main()
