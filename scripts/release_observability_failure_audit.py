"""Explain a failed O0 screen without refitting, changing its population or selecting thresholds."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from operational.release_research.normal_quality import PHYSICAL_FIELDS
from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import read, sources, verify_complete


def audit(raw: Path, analysis: Path, evaluated: Path, output: Path) -> None:
    """Measure spatial-fit degrees of freedom and each fixed fold's contribution to failure."""
    for folder in (raw, analysis, evaluated):
        verify_complete(folder)
    rows = read(raw / "FEATURES.json")["records"]
    scored = read(evaluated / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Query/target identities differ")
    metrics = read(analysis / "METRICS.json")
    fits = read(analysis / "FITS.json")["physical"]
    inputs = {
        **sources(),
        "source:scripts/release_observability_failure_audit.py": Path(__file__),
        "raw_complete": raw / "COMPLETE.json",
        "features": raw / "FEATURES.json",
        "analysis_complete": analysis / "COMPLETE.json",
        "fits": analysis / "FITS.json",
        "metrics": analysis / "METRICS.json",
        "evaluation_complete": evaluated / "COMPLETE.json",
        "scored": evaluated / "DIAGNOSTICS.json",
    }
    for row in rows:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        inputs["raw_query:" + identity] = raw / "queries" / (identity + ".json")
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "O0-failure-localization-v1",
            "scope": "post-result descriptive audit; no retuning or new admission decision",
            "targets_read": True,
            "new_fits": 0,
            "optimizer_updates": 0,
        },
    )
    try:
        diagnostics = []
        by_fold = {fit["held_out_group"]: fit for fit in fits}
        for row, score in zip(rows, scored, strict=True):
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            stored = read(raw / "queries" / (identity + ".json"))
            snapshots = stored.get("snapshots", [])
            current = snapshots[-1] if snapshots else None
            fit = current["all"] if current else None
            record = {
                "token": row["token"],
                "group": row["group"],
                "planes": current["accepted_planes"] if current else 0,
                "q_estimable": bool(fit and fit["estimable"]),
            }
            if fit:
                record.update(
                    rank=fit["rank"],
                    residual_dof=record["planes"] - fit["rank"],
                    normalized_residual=fit["normalized_residual"],
                    q=fit["q"],
                    sensitivity=fit["sensitivity"],
                )
                if fit["q"] is not None:
                    record.update(
                        q_target=1 / score["target"],
                        q_absolute_error=abs(fit["q"] - 1 / score["target"]),
                        canonical_domain=1 - 0.1 * fit["q"] > 0,
                        sign_correct=bool(np.sign(fit["q"]) == np.sign(score["target"])),
                    )
            if score.get("gain_seconds") is not None:
                saved = by_fold[row["group"]]
                preprocessing = saved["preprocessing"]
                features = np.array(row["physical"], float)
                finite = np.isfinite(features)
                scaled = (
                    np.where(finite, features, preprocessing["medians"]) - preprocessing["mean"]
                ) / preprocessing["scale"]
                design = np.r_[1.0, scaled, finite.astype(float)]
                contributions = design * np.array(saved["coefficient"])
                record["ridge_contributions"] = dict(
                    zip(
                        (
                            "intercept",
                            *PHYSICAL_FIELDS,
                            *("present:" + name for name in PHYSICAL_FIELDS),
                        ),
                        contributions.tolist(),
                        strict=True,
                    )
                )
                record["maximum_standardized_magnitude"] = float(np.max(np.abs(scaled)))
                record["predicted_gain"] = float(contributions.sum())
                record["actual_gain"] = score["gain_seconds"]
            diagnostics.append(record)
        estimable = [r for r in diagnostics if r["q_estimable"]]
        influence = []
        physical = metrics["arms"]["physical"]["by_acquisition"]
        support = metrics["arms"]["support"]["by_acquisition"]
        total = sum(physical[g] - support[g] for g in physical)
        for group in sorted(physical):
            difference = physical[group] - support[group]
            influence.append(
                {
                    "group": group,
                    "physical_mse": physical[group],
                    "support_mse": support[group],
                    "difference": difference,
                    "fraction_of_total_signed_difference": difference / total if total else None,
                    "leave_this_group_out_delta": (total - difference) / (len(physical) - 1),
                }
            )
        summary = {
            "required": len(rows),
            "q_estimable": len(estimable),
            "no_residual_degrees_of_freedom": sum(r["residual_dof"] == 0 for r in estimable),
            "outside_canonical_domain": sum(not r["canonical_domain"] for r in estimable),
            "q_wrong_sign": sum(not r["sign_correct"] for r in estimable),
            "absolute_q_quantiles": np.quantile(
                [abs(r["q"]) for r in estimable], [0, 0.5, 0.9, 1]
            ).tolist(),
            "q_absolute_error_quantiles": np.quantile(
                [r["q_absolute_error"] for r in estimable], [0.5, 0.9, 1]
            ).tolist(),
            "by_acquisition_influence": influence,
            "original_gate_decision_unchanged": metrics["gate_pilot_admitted"],
            "no_new_admission": True,
            "interpretation": "q estimability does not establish accuracy or calibrated confidence",
        }
        write_json(output / "DIAGNOSTICS.json", {"records": diagnostics})
        write_json(output / "METRICS.json", summary)
        lines = [
            "# Localización del fallo de observabilidad",
            "",
            "Auditoría posterior al resultado. No se reajusta la sonda ni se excluyen datos.",
            "",
            f"q estimable: {len(estimable)}/{len(rows)}; sin grados de libertad residuales: "
            f"{summary['no_residual_degrees_of_freedom']}.",
            f"Signo incorrecto: {summary['q_wrong_sign']}; fuera del dominio canónico: "
            f"{summary['outside_canonical_domain']}.",
            f"Cuantiles |q| (0/50/90/100%): {summary['absolute_q_quantiles']}.",
            f"Cuantiles de error |q−1/TTC| (50/90/100%): {summary['q_absolute_error_quantiles']}.",
            "",
            "Identificar algebraicamente q no garantiza un valor físicamente correcto.",
            "Con cero grados de libertad, residuo casi nulo no demuestra fiabilidad.",
            "El target 1/TTC tampoco certifica el modelo de expansión local bajo giro o paralaje.",
            "",
            "| Adquisición | MSE físico | MSE soporte | Delta |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {r['group']} | {r['physical_mse']:.6f} | "
            f"{r['support_mse']:.6f} | {r['difference']:.6f} |"
            for r in influence
        ]
        lines += [
            "",
            "Se guardan las contribuciones de cada campo y la influencia de cada adquisición.",
            "Excluir un grupo solo mide sensibilidad; no sustituye la población prefijada.",
            "O1 y JEPA requieren más evidencia: primero hace falta una medición de q fiable.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Audit existing sealed evidence without new extraction, fitting or release changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("--raw", type=Path, default=root / "observability149_v1")
    parser.add_argument("--analysis", type=Path, default=root / "observability_analysis149_v1")
    parser.add_argument("--evaluated", type=Path, default=root / "evaluation149_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw, args.analysis, args.evaluated, args.output)


if __name__ == "__main__":
    main()
