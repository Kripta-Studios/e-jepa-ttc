"""Compare O0's q readout with the fixed historical robust solver on identical planes."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.models.normal_flow_routing_v10 import fit_affine_normal_flow
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.release_research.evaluation import compare
from operational.release_research.physics import q_to_raw_phase, transport
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/ROBUST_MEASUREMENT_PREREGISTRATION.md")


def source_paths() -> dict[str, Path]:
    """Include the unchanged historical IRLS source in the executed manifest."""
    return {
        **sources(),
        "source:scripts/release_robust_measurement.py": Path(__file__),
        "source:src/e_jepa_ttc/models/normal_flow_routing_v10.py": Path(
            "src/e_jepa_ttc/models/normal_flow_routing_v10.py"
        ),
    }


def extract(raw: Path, output: Path) -> None:
    """Read the sealed planes only; no target-dependent acceptance or tuning."""
    verify_complete(raw)
    rows = read(raw / "FEATURES.json")["records"]
    inputs = {
        **source_paths(),
        "raw_complete": raw / "COMPLETE.json",
        "features": raw / "FEATURES.json",
        "preregistration": PLAN,
    }
    for row in rows:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        inputs["query:" + identity] = raw / "queries" / (identity + ".json")
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "O0-robust-readout-control-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "scope": "exposed development, fixed historical solver",
        },
    )
    try:
        results = []
        for row in rows:
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            snapshots = read(raw / "queries" / (identity + ".json")).get("snapshots", [])
            result = {k: row[k] for k in ("token", "group", "query_us")}
            result["readouts"] = {}
            q_values: dict[str, float | None] = {"ols": None, "historical_robust": None}
            if snapshots:
                current = snapshots[-1]
                planes = np.asarray(current["planes"], float).reshape(-1, 10)
                fit = fit_affine_normal_flow(planes[:, :2], planes[:, 2:4], planes[:, 4])
                q_values.update(
                    ols=current["all"]["q"] if current["all"] else None,
                    historical_robust=fit.kappa if fit.valid else None,
                )
                result["robust_diagnostics"] = {
                    "valid": fit.valid,
                    "mode": fit.mode,
                    "planes": len(planes),
                    "condition": fit.condition,
                    "residual": fit.residual,
                    "inlier_fraction": fit.inlier_fraction,
                    "orientation_balance": fit.orientation_balance,
                }
                delta = (row["query_us"] - current["endpoint_us"]) * 1e-6
            else:
                delta = 0.0
            for name, q in q_values.items():
                record = {
                    "q_measurement": q,
                    "q_at_query": None,
                    "ttc": None,
                    "reason": "no_measurement",
                }
                if q is not None:
                    try:
                        at_query = transport(q, delta)[0]
                        record["q_at_query"] = at_query
                        phase = q_to_raw_phase(at_query)
                        record["ttc"] = float(
                            phase_to_ttc(torch.tensor([phase], dtype=torch.float32))[0]
                        )
                        record["reason"] = "ok"
                    except ValueError as error:
                        record["reason"] = str(error)
                result["readouts"][name] = record
            results.append(result)
        write_json(output / "PREDICTIONS.json", {"records": results})
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def evaluate(predictions: Path, evaluation: Path, output: Path) -> None:
    """Retain all missingness and prohibit treating accepted-only accuracy as full coverage."""
    verify_complete(predictions)
    verify_complete(evaluation)
    freeze = read(predictions / "FREEZE.json")
    if freeze["config"]["targets_read"] or freeze["inputs"]["preregistration"] != sha256(PLAN):
        raise ValueError("Require unchanged label-blind protocol")
    rows = read(predictions / "PREDICTIONS.json")["records"]
    scored = read(evaluation / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Target identities differ")
    run = ResearchRun(
        output,
        {
            **source_paths(),
            "predictions": predictions / "PREDICTIONS.json",
            "parent_complete": predictions / "COMPLETE.json",
            "evaluation_complete": evaluation / "COMPLETE.json",
            "evaluation": evaluation / "DIAGNOSTICS.json",
            "preregistration": PLAN,
        },
        {
            "experiment_id": "O0-robust-readout-evaluation-v1",
            "targets_read": True,
            "optimizer_updates": 0,
            "new_gate_admission": False,
        },
    )
    try:
        groups = np.asarray([r["group"] for r in rows])
        baseline = np.asarray([r["R0"] for r in scored], float)
        target = np.asarray([r["target"] for r in scored], float)
        result = {}
        for arm in ("ols", "historical_robust"):
            values = np.asarray([r["readouts"][arm]["ttc"] for r in rows], float)
            q = np.asarray([r["readouts"][arm]["q_at_query"] for r in rows], float)
            valid = np.isfinite(q)
            result[arm] = {
                "q_available": int(valid.sum()),
                "ttc_available": int(np.isfinite(values).sum()),
                "median_q_error": float(np.median(np.abs(q[valid] - 1 / target[valid])))
                if valid.any()
                else None,
                "q_sign_errors": int((np.sign(q[valid]) != np.sign(target[valid])).sum()),
                "all_queries_with_fallback": compare(
                    target, baseline, np.where(np.isfinite(values), values, baseline), groups
                ),
                "measurement_only": compare(target, baseline, values, groups),
                "reasons": dict(Counter(r["readouts"][arm]["reason"] for r in rows)),
            }
        modes = dict(
            Counter(r.get("robust_diagnostics", {}).get("mode", "no_planes") for r in rows)
        )
        write_json(
            output / "METRICS.json",
            {"readouts": result, "robust_modes": modes, "new_gate_admission": False},
        )
        lines = [
            "# Control de lectura robusta sobre los mismos planos",
            "",
            "Desarrollo expuesto; parámetros históricos intactos, sin entrenar gate.",
            "",
            "| Lectura | q disponible | TTC disponible | Signos q erróneos | MAE con fallback R0 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, data in result.items():
            mae = data["all_queries_with_fallback"]["candidate"]["macro_mae"]
            lines.append(
                f"| {arm} | {data['q_available']} | {data['ttc_available']} | "
                f"{data['q_sign_errors']} | {mae:.9f} |"
            )
        lines += [
            "",
            f"Modos del solver robusto: {modes}.",
            "El modo isotrópico introduce una hipótesis de movimiento explícita.",
            "Se conserva la población y ausencia original; no se cambia la admisión O0.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Separate target-blind readout from evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("mode", choices=("extract", "evaluate"))
    parser.add_argument("--raw", type=Path, default=root / "observability149_v1")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--evaluation", type=Path, default=root / "evaluation149_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args.raw, args.output)
    elif args.predictions is None:
        parser.error("--predictions required for evaluation")
    else:
        evaluate(args.predictions, args.evaluation, args.output)


if __name__ == "__main__":
    main()
