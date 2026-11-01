"""Evaluate the sealed V0/V1 diagnostic without selecting a surface threshold on TTC."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.release_research.evaluation import compare
from operational.release_research.physics import q_to_raw_phase
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_observability_analysis import within_group_correlations
from scripts.release_research import read, sources, verify_complete


def analyze(features: Path, evaluation: Path, output: Path) -> None:
    """Keep full-population R0 fallback and accepted-only estimates separate."""
    verify_complete(features)
    verify_complete(evaluation)
    prereg = Path("docs/release_research_implementation_20261031/SURFACE_PREREGISTRATION.md")
    freeze = read(features / "FREEZE.json")
    if (
        freeze["config"]["profiling_only"]
        or freeze["config"]["targets_read"]
        or freeze["inputs"]["preregistration"] != sha256(prereg)
    ):
        raise ValueError("Require full label-blind preregistered extraction")
    rows = read(features / "FEATURES.json")["records"]
    scored = read(evaluation / "DIAGNOSTICS.json")["records"]
    if len(rows) != 149 or [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Evaluation identity differs")
    inputs = {
        **sources(),
        "source:scripts/release_surface_analysis.py": Path(__file__),
        "source:scripts/release_observability_analysis.py": Path(
            "scripts/release_observability_analysis.py"
        ),
        "features": features / "FEATURES.json",
        "features_complete": features / "COMPLETE.json",
        "targets": evaluation / "DIAGNOSTICS.json",
        "targets_complete": evaluation / "COMPLETE.json",
        "preregistration": prereg,
    }
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "V0-V1-evaluation-v1",
            "seed": 7,
            "exposure": "development",
            "optimizer_updates": 0,
            "targets_read": True,
            "prediction": "signed endpoint q, causal transport, official emission, R0 fallback",
            "scope": "diagnostic only; no inference about a learned surface gate",
        },
    )
    try:
        baseline = np.array([r["R0"] for r in scored], float)
        target = np.array([r["target"] for r in scored], float)
        groups = np.array([r["group"] for r in rows])
        measurement = np.full(len(rows), np.nan)
        for index, row in enumerate(rows):
            q = row["q_at_query"]
            row["emission_reason"] = "unavailable_surface"
            if q is not None:
                try:
                    phase = q_to_raw_phase(q)
                    measurement[index] = float(
                        phase_to_ttc(torch.tensor([phase], dtype=torch.float32))[0]
                    )
                    row["emission_reason"] = "ok"
                except ValueError as error:
                    row["emission_reason"] = str(error)
            row.update(
                R0=scored[index]["R0"],
                target=float(target[index]),
                surface_ttc=float(measurement[index]) if np.isfinite(measurement[index]) else None,
            )
        available = np.isfinite(measurement)
        candidate = np.where(available, measurement, baseline)
        matched = available & np.isfinite(baseline)
        metrics = {
            "all_queries_with_R0_fallback": compare(target, baseline, candidate, groups),
            "surface_only": compare(target, baseline, measurement, groups),
            "surface_available": int(available.sum()),
            "fallback_queries": int((~available).sum()),
            "raw_surface_status": dict(Counter(r["status"] for r in rows)),
            "emission_status": dict(Counter(r["emission_reason"] for r in rows)),
            "within_triplet_crop_changes": sum(
                r.get("within_triplet_same_crop") is False for r in rows
            ),
            "history_crop_changes": sum(
                r.get("history_same_requested_crop") is False for r in rows
            ),
            "anisotropic_resize": sum(r.get("anisotropic_resize") is True for r in rows),
            "missing_coordinate_audit": sum("effective_crops" not in r for r in rows),
            "surface_beats_R0": int(
                (
                    np.abs(measurement[matched] - target[matched])
                    < np.abs(baseline[matched] - target[matched])
                ).sum()
            ),
            "surface_loses_to_R0": int(
                (
                    np.abs(measurement[matched] - target[matched])
                    > np.abs(baseline[matched] - target[matched])
                ).sum()
            ),
        }
        quality = np.asarray(
            [
                [
                    r.get("surface", {}).get("coverage"),
                    abs(r["surface"]["composition_logscale_residual"])
                    if r.get("surface", {}).get("composition_logscale_residual") is not None
                    else None,
                    r.get("surface", {})
                    .get("routes", {})
                    .get("bc", {})
                    .get("median_residual_pixels"),
                ]
                for r in rows
            ],
            float,
        )
        gain = np.abs(baseline - target) - np.abs(measurement - target)
        metrics["within_acquisition_quality_vs_surface_gain"] = within_group_correlations(
            quality[matched],
            gain[matched],
            groups[matched],
            ("coverage", "absolute_composition_residual", "affine_pixel_residual"),
        )
        write_json(output / "DIAGNOSTICS.json", {"records": rows})
        write_json(output / "METRICS.json", metrics)
        write_json(
            output / "DECISION.json",
            {
                "family": "visibility",
                "decision": "diagnose",
                "conditions": ["coordinate_audit"],
                "missing": ["visibility_incremental"],
                "release_modified": False,
                "reason": "A geometric diagnostic does not establish an incremental learned gate",
            },
        )
        full = metrics["all_queries_with_R0_fallback"]
        lines = [
            "# Superficies RGB y coordenadas del release",
            "",
            "Diagnóstico de desarrollo expuesto, sin ajuste de umbrales ni pesos.",
            "",
            f"Medida representable: {available.sum()}/{len(rows)}; "
            f"fallback R0: {(~available).sum()}.",
            f"MAE macro R0: {full['baseline']['macro_mae']:.9f} s.",
            f"MAE macro superficie con fallback: {full['candidate']['macro_mae']:.9f} s.",
            f"CI95 delta macro por adquisición: {full['paired_macro_delta_ci95']}.",
            f"Cambios de crop dentro del triplete: {metrics['within_triplet_crop_changes']}.",
            f"Cambios de crop entre snapshots: {metrics['history_crop_changes']}.",
            f"Resize anisótropo: {metrics['anisotropic_resize']}; "
            f"auditables: {149 - metrics['missing_coordinate_audit']}.",
            "",
            "Esta medida usa RGB; no prueba recuperación de q desde eventos.",
            "La correspondencia consistente no acredita por sí sola TTC ni visibilidad del objeto.",
            "R0 permanece intacto. V2/V3 requieren señal incremental y controles de igual coste.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Evaluate an immutable full surface screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--evaluation", type=Path, default=Path("output/release_research_20261031/evaluation149_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.features, args.evaluation, args.output)


if __name__ == "__main__":
    main()
