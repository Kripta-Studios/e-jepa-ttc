"""S0 ambiguity controls: equal observed event streams need not identify expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.release_research.photometric import recover_q_and_light, simulate_patch
from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import sources


def stream_comparison(left: np.ndarray, right: np.ndarray) -> dict:
    """Compare complete event observations, not only their counts."""
    # Simultaneous events may be serialized in another order after FP rounding.
    # Match within each pixel/polarity stream while retaining every timestamp.
    left = left[np.lexsort((left[:, 1], left[:, 2], left[:, 0]))]
    right = right[np.lexsort((right[:, 1], right[:, 2], right[:, 0]))]
    shape_equal = left.shape == right.shape
    categorical_equal = shape_equal and bool(np.array_equal(left[:, [0, 2]], right[:, [0, 2]]))
    time_error = (
        float(np.max(np.abs(left[:, 1] - right[:, 1]))) if shape_equal and len(left) else None
    )
    return {
        "left_events": len(left),
        "right_events": len(right),
        "same_count": shape_equal,
        "same_pixel_and_polarity": categorical_equal,
        "timestamp_max_abs_seconds": time_error,
        "same_stream_at_1ps": categorical_equal and (time_error is None or time_error <= 1e-12),
    }


def experiment(output: Path) -> None:
    """Run a fixed analytic fixture bank; no fit to real TTC or change to any R0 component."""
    inputs = {**sources(), "source:scripts/release_sensor_identifiability.py": Path(__file__)}
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "S0-identification-ambiguities-v1",
            "real_targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
            "scope": "affine log-texture fixture only; not a universal sensor impossibility result",
            "q_initial": [-0.4, 0.2, 0.4],
            "texture_scale": [0.5, 2.0],
            "illumination": [0.0, 0.2],
            "thresholds": [0.03, 0.024],
            "duration_s": 0.3,
            "pixels": 64,
        },
    )
    try:
        xx, yy = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-1, 1, 8))
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        gradient = np.array([0.8, 0.2])
        times = np.linspace(0, 0.3, 61)
        cases = []
        for q in (-0.4, 0.2, 0.4):
            for light in (0.0, 0.2):
                original = simulate_patch(xy, gradient, times, q, light, 0.03, 0.024)
                recovered = recover_q_and_light(original, xy, gradient)
                for scale in (0.5, 2.0):
                    # q' g' = q g: static references differ, observed event increments do not.
                    changed_q = simulate_patch(
                        xy, gradient / scale, times, q * scale, light, 0.03, 0.024
                    )
                    comparison = stream_comparison(original.events, changed_q.events)
                    if not comparison["same_stream_at_1ps"]:
                        raise ValueError("Known analytic q/texture ambiguity failed")
                    cases.append(
                        {
                            "kind": "q_texture_ambiguity",
                            "q_initial": q,
                            "alternative_q_initial": q * scale,
                            "texture_scale": 1 / scale,
                            "illumination": light,
                            "comparison": comparison,
                            "known_reference_recovery": recovered,
                            "alternative_known_reference_recovery": recover_q_and_light(
                                changed_q, xy, gradient / scale
                            ),
                            "static_reference_max_abs_difference": float(
                                np.max(
                                    np.abs(original.reference_initial - changed_q.reference_initial)
                                )
                            ),
                        }
                    )
                    changed_photo = simulate_patch(
                        xy, gradient * scale, times, q, light * scale, 0.03 * scale, 0.024 * scale
                    )
                    comparison = stream_comparison(original.events, changed_photo.events)
                    if not comparison["same_stream_at_1ps"]:
                        raise ValueError("Known photometric scale ambiguity failed")
                    cases.append(
                        {
                            "kind": "photometric_scale_ambiguity",
                            "q_initial": q,
                            "scale": scale,
                            "comparison": comparison,
                        }
                    )
        rank_one_xy = np.column_stack((np.ones(8), np.linspace(-1, 1, 8)))
        rank_one = simulate_patch(rank_one_xy, np.array([1.0, 0.0]), times, 0.4, 0.1, 0.03, 0.024)
        unidentifiable = recover_q_and_light(rank_one, rank_one_xy, np.array([1.0, 0.0]))
        if unidentifiable["q"] is not None:
            raise ValueError("Rank-one motion/light ambiguity was hidden")
        write_json(output / "CASES.json", {"cases": cases, "rank_one_motion_light": unidentifiable})
        summary = {
            "cases": len(cases),
            "identical_stream_cases": sum(c["comparison"]["same_stream_at_1ps"] for c in cases),
            "distinct_q_identical_stream_cases": sum(
                c["kind"] == "q_texture_ambiguity" for c in cases
            ),
            "known_texture_recovery_max_q_error": max(
                abs(c["known_reference_recovery"]["q"] - c["q_initial"])
                for c in cases
                if c["kind"] == "q_texture_ambiguity"
            ),
            "real_S1_admitted": False,
            "release_modified": False,
            "scope": "affine log-texture only; events omit the differing static reference",
        }
        write_json(output / "METRICS.json", summary)
        lines = [
            "# S0: identificación frente a recuento y textura desconocida",
            "",
            f"Casos: {len(cases)}; streams coincidentes: {summary['identical_stream_cases']}.",
            f"q distinto, mismo stream: {summary['distinct_q_identical_stream_cases']} casos.",
            "Coinciden píxel, polaridad y timestamp (tolerancia 1 ps), no solo recuento.",
            "",
            "En esta textura log-afín, el incremento contiene el producto q·gradiente.",
            "Cambiar q y escalar inversamente el gradiente deja iguales esos incrementos;",
            "las referencias estáticas iniciales son distintas y no están en el stream de eventos.",
            "Al proporcionar la referencia/calibración conocida se pueden separar las condiciones.",
            "Error máximo q con referencia conocida: "
            f"{summary['known_texture_recovery_max_q_error']:.9f} s^-1.",
            "",
            "Escalar textura, luz y umbrales juntos también conserva los eventos.",
            "Sin diversidad espacial persiste la ambigüedad entre luz y movimiento.",
            "Este banco no demuestra imposibilidad general: muestra por qué S1 necesita anclaje",
            "o identificación y por qué un auxiliar con textura y umbral libres puede ignorar q.",
            "No habilita S2 ni JEPA y no afirma una mejora del release.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Execute the bounded S0 identification controls in a fresh immutable run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    experiment(parser.parse_args().output)


if __name__ == "__main__":
    main()
