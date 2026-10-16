"""Registered CPU evidence stages; GPU training has its own source-bound admission."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from .common import ROOT, RUN, load_population, read_json, sha, write_json
from .registration import REGISTRATION, register, verify


def diagnostics() -> dict[str, Any]:
    from .diagnostics import run

    verify()
    result = run()
    result["registration_sha256"] = sha(REGISTRATION)
    write_json(RUN / "diagnostics/COMPLETE.json", result)
    return result


def routers() -> dict[str, Any]:
    from .router import PAIRS, evaluate_pair

    registration = verify()
    tables = load_population("H", ["A5", "NATIVE", "CORR", "FULL"])
    results = {}
    for base, special in PAIRS:
        directory = RUN / "router" / f"{base}__{special}"
        if (directory / "RESULT.json").exists():
            raise ValueError("Do not overwrite an executed router result; use verified replay")
        result = evaluate_pair(tables, base, special)
        result["registration_sha256"] = sha(REGISTRATION)
        result["ancestry"] = registration["ancestry"]
        write_json(directory / "RESULT.json", result)
        results[f"{base}__{special}"] = {
            "path": str(directory.relative_to(RUN) / "RESULT.json"),
            "sha256": sha(directory / "RESULT.json"),
        }
    result = {
        "status": "COMPLETE_FIXED_MATRIX_DEVELOPMENT_ONLY",
        "utc": datetime.now(UTC).isoformat(),
        "registration_sha256": sha(REGISTRATION),
        "pairs": results,
        "optimizer_updates_to_producers": 0,
        "GPU_used": False,
        "system_promotion": False,
    }
    write_json(RUN / "router/COMPLETE.json", result)
    return result


def report() -> dict[str, Any]:
    """Publish all fixed comparisons, never select a favorable pair retrospectively."""
    verify()
    complete = read_json(RUN / "router/COMPLETE.json")
    rows, decisions = [], {}
    for name, binding in complete["pairs"].items():
        path = RUN / binding["path"]
        if sha(path) != binding["sha256"]:
            raise ValueError("Executed router evidence changed")
        result = read_json(path)
        candidate = result["results"]["R0"]
        metrics = candidate["metrics"]
        negative = result["results"]["SHUFFLED"]["metrics"]
        base, special = result["pair"]
        best_fixed = min(result["references"][k]["overall_MiD"] for k in (base, special))
        a5 = result["references"]["A5"]
        ratios = {
            band: metrics["bins"][band]["group_macro_mid"] / a5["bins"][band]["group_macro_mid"]
            for band in a5["bins"]
        }
        checks = {
            "A5_development_gate": candidate["development_gate_A5"]["eligible"],
            "A5_coverage": metrics["complete_prediction_and_mid_coverage"],
            "A5_bands": all(r <= 1.1 for r in ratios.values()),
            "A5_sign": metrics["diagnostics"]["sign_error_rate"]
            <= a5["diagnostics"]["sign_error_rate"] + 0.005,
            "base_original_system_numerical_gates": candidate["gates"][base]["eligible"],
            "beats_best_fixed_aggregate": metrics["overall_MiD"] < best_fixed,
            "beats_shuffled_aggregate": metrics["overall_MiD"] < negative["overall_MiD"],
        }
        decisions[name] = {
            "checks": checks,
            "further_evaluation_enabled": all(checks.values()),
            "system_promotion": False,
            "primary": name == "NATIVE__FULL",
            "A5_band_ratios": ratios,
        }
        for label, value in [(k, v) for k, v in result["references"].items()] + [
            ("R0", metrics),
            ("SHUFFLED", negative),
        ]:
            rows.append(
                f"| {name} | {label} | {value['overall_MiD']:.3f} | "
                f"{value['bins']['crucial']['group_macro_mid']:.3f} | "
                f"{value['bins']['negative']['group_macro_mid']:.3f} | "
                f"{value['diagnostics']['mae_s']:.3f} | "
                f"{100 * value['diagnostics']['sign_error_rate']:.3f}% |"
            )
    decision = {
        "status": "PRIMARY_ROUTER_ADVANCES"
        if decisions["NATIVE__FULL"]["further_evaluation_enabled"]
        else "NO_PRIMARY_ROUTER_CANDIDATE",
        "pairs": decisions,
        "scope": (
            "Fixed-producer LOGO on repeatedly exposed H; "
            "no cost or independent confirmation yet"
        ),
        "loss_screen": "Independent registered L0/LW question remains authorized",
        "official_SOTA": False,
    }
    write_json(RUN / "router/DECISION.json", decision)
    text = "\n".join(
        [
            "# V19: selección de expertos sobre H",
            "",
            f"Estado: **{decision['status']}**. "
            "H sigue siendo desarrollo expuesto; no hay promoción ni SOTA.",
            "",
            "Ridge fijo 0,01, umbral 0, normalizador y ajuste excluyen el grupo evaluado. "
            "Los pesos siguen el scorer por banda/grupo. Todos los pares registrados "
            "y el control permutado se conservan.",
            "",
            "| Pareja | Modelo | Agregado MiD | MiDc | MiDn | MAE s | Error signo |",
            "|---|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Los intervalos de los JSON remuestrean grupos sobre predicciones OOF fijas. "
            "No repiten el ajuste ni estiman incertidumbre de selección de arquitectura. "
            "El export ALL_H es un modelo de desarrollo, no evidencia OOF sobre H.",
            "",
            "Fuente: artifacts/decision_recovery_20261017/router/{pareja}/RESULT.json "
            "y DECISION.json. Se requiere ejecutar ambos expertos; "
            "no hay coste sensorial completo medido para estos routers.",
            "",
        ]
    )
    destination = ROOT / "docs/decision_recovery_20261017"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ROUTER_RESULTS.md").write_text(text, encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "diagnostics", "router", "report"))
    args = parser.parse_args()
    actions = {
        "register": register,
        "diagnostics": diagnostics,
        "router": routers,
        "report": report,
    }
    output = actions[args.action]()
    print(json.dumps({"action": args.action, "status": output["status"]}, indent=2))
