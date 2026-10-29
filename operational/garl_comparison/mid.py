"""Apply the byte-pinned CodaBench scorer to local public-label diagnostic cohorts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

from operational.sota_evidence.metrics import observations
from operational.sota_evidence.run import digest, json_safe, read
from operational.train40_system.durable_io import atomic_json

SCORER_SHA256 = "f3bbf089ba6e47edfce1de522c92ddc969db2260fac7544e82bc017fedf82dce"
METHODS = ("H8_median3", "Direct_median3", "public_Garl_event_lhr", "public_Garl_rgb_event_full")


def load_scorer(path: Path) -> ModuleType:
    """Execute only the previously audited official scorer bytes."""
    if digest(path) != SCORER_SHA256:
        raise ValueError("Official scorer differs from the audited release")
    spec = importlib.util.spec_from_file_location("garl_official_mid", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the pinned official scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(scorer: ModuleType, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Keep official finite-only means alongside explicit invalid-MiD accounting."""
    if truth.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("Aligned one-dimensional arrays required")
    if not np.isfinite(truth).all() or (truth == 0).any():
        raise ValueError("Apply finite nonzero GT eligibility first")
    reference = [{"sample_token": str(i), "ttc": float(t)} for i, t in enumerate(truth)]
    values = {str(i): float(t) for i, t in enumerate(prediction)}
    result = scorer.evaluate_rows(reference, values, dT=0.1)
    row_mid = observations(truth, prediction)["garl_mid"]
    assigned = np.array([scorer.bin_suffix(float(t)) or "outside" for t in truth])
    result["invalid_mid_total"] = int((~np.isfinite(row_mid)).sum())
    result["strict_mean_MiD"] = float(row_mid.mean()) if np.isfinite(row_mid).all() else None
    result["outside_bands_n"] = int((assigned == "outside").sum())
    for suffix in "csln":
        mask = assigned == suffix
        result[f"n_{suffix}"] = int(mask.sum())
        result[f"invalid_mid_{suffix}"] = int((~np.isfinite(row_mid[mask])).sum())
    result["overall_available"] = bool(np.isfinite(result["overall_MiD"]))
    return result


def paired_mid(ours: np.ndarray, comparator: np.ndarray, sequence: np.ndarray) -> dict[str, Any]:
    """Bootstrap paired sequence clusters for the sample-weighted mean MiD difference."""
    if ours.ndim != 1 or ours.shape != comparator.shape or ours.shape != sequence.shape:
        raise ValueError("Aligned MiD arrays and sequence IDs required")
    names = np.unique(sequence)
    base = {"sequences": len(names), "samples": len(ours), "seed": 20261010, "replicates": 10000}
    if not len(ours) or not np.isfinite(ours).all() or not np.isfinite(comparator).all():
        return {**base, "status": "UNAVAILABLE_INVALID_MID_RETAINED"}
    delta = ours - comparator
    if len(names) < 2:
        return {**base, "status": "INSUFFICIENT_SEQUENCES", "delta_mean_MiD": float(delta.mean())}
    sums = np.array([delta[sequence == name].sum() for name in names])
    counts = np.array([(sequence == name).sum() for name in names])
    draws = np.random.default_rng(base["seed"]).integers(
        0, len(names), (base["replicates"], len(names))
    )
    distribution = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        **base,
        "status": "EXPLORATORY_SEQUENCE_CLUSTER_BOOTSTRAP",
        "delta_mean_MiD": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def run(root: Path, output: Path) -> None:
    """Report exact scorer outputs without replacing missing bands or inventing test labels."""
    output.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer(root / "artifacts/garlttc_submission_20261010/official_score.py")
    bindings = read(root / "artifacts/sota_evidence_20261010/INPUTS.json")
    rows = []
    pairs = []
    worst = []
    for dataset in ("DEV32", "FCWD"):
        path = root / f"artifacts/ttc_revision_20261009/{dataset}_PREDICTIONS.csv"
        if digest(path) != bindings[str(path.relative_to(root))]:
            raise ValueError("Frozen predictions changed")
        frame = pd.read_csv(path)
        eligible = np.isfinite(frame.truth_ttc_seconds) & (frame.truth_ttc_seconds != 0)
        frame = frame.loc[eligible]
        truth = frame.truth_ttc_seconds.to_numpy(float)
        sequence = frame.sequence_id.to_numpy(str)
        mid_by_method = {}
        for method in METHODS:
            prediction = frame[method].to_numpy(float)
            row_mid = observations(truth, prediction)["garl_mid"]
            mid_by_method[method] = row_mid
            for index in np.argsort(np.where(np.isfinite(row_mid), row_mid, -1))[-20:][::-1]:
                if np.isfinite(row_mid[index]):
                    worst.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "query_id": frame.iloc[index].query_id,
                            "sequence_id": sequence[index],
                            "truth_s": truth[index],
                            "prediction_s": prediction[index],
                            "MiD": row_mid[index],
                        }
                    )
            for policy in ("native", "equal_clip_60_sensitivity"):
                value = (
                    prediction
                    if policy == "native"
                    else np.where(np.isfinite(prediction), np.clip(prediction, -60, 60), prediction)
                )
                base = {
                    "dataset": dataset,
                    "method": method,
                    "policy": policy,
                    "prediction_coverage": float(np.isfinite(value).mean()),
                    "gt_ineligible_n": int((~eligible).sum()),
                }
                if not np.isfinite(value).all():
                    rows.append({**base, "status": "UNAVAILABLE_PREDICTIONS_RETAINED"})
                    continue
                rows.append(
                    {
                        **base,
                        "status": "PUBLIC_COHORT_NOT_OFFICIAL_TEST",
                        **scores(scorer, truth, value),
                    }
                )
        for comparator in METHODS[2:]:
            pairs.append(
                {
                    "dataset": dataset,
                    "ours": METHODS[0],
                    "comparator": comparator,
                    "policy": "native",
                    **paired_mid(mid_by_method[METHODS[0]], mid_by_method[comparator], sequence),
                }
            )
    pd.DataFrame(rows).to_csv(output / "MID_OFFICIAL_SCORER.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "MID_PAIRED_SEQUENCE_BOOTSTRAP.csv", index=False)
    pd.DataFrame(worst).to_csv(output / "MID_LARGEST_ERRORS.csv", index=False)
    atomic_json(
        output / "MID_RESULT.json",
        cast(
            dict,
            json_safe(
                {
                    "status": "COMPLETE_LOCAL_PUBLIC_COHORTS",
                    "scorer_sha256": SCORER_SHA256,
                    "source_sha256": digest(Path(__file__)),
                    "dt_seconds": 0.1,
                    "rows": rows,
                    "paired_sequence_bootstrap": pairs,
                    "official_test12_score_available": False,
                    "absent_bands_renormalized": False,
                    "optimizer_updates": 0,
                    "gpu_seconds": 0,
                }
            ),
        ),
    )
    lines = [
        "# MiD con el evaluador oficial de GarlTTC",
        "",
        "Menor es mejor. Se ejecuta el scorer CodaBench conservado y verificado por SHA-256 "
        "sobre predicciones y etiquetas locales. No son puntuaciones oficiales de test12.",
        "",
        "| Datos | Método | MiD medio | MiDc | MiDs | MiDl | MiDn | overall_MiD | FR (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["policy"] != "native":
            continue
        values = [
            f"{row[k]:.3f}" if row.get(k) is not None and np.isfinite(row[k]) else "N/D"
            for k in ("mean_MiD", "MiDc", "MiDs", "MiDl", "MiDn", "overall_MiD", "failed_rate")
        ]
        lines.append(f"| {row['dataset']} | {row['method']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "MiD = 10⁴ · |log(1 − 0,1/GT) − log(1 − 0,1/predicción)|.",
        "",
        "`overall_MiD = 0,5 MiDc + 0,3 MiDs + 0,1 MiDl + 0,1 MiDn`. "
        "Dev32 y FCWD no contienen GT negativos: MiDn y overall_MiD quedan N/D. "
        "No se redistribuye el peso de esa banda.",
        "",
        "El scorer calcula medias de MiD finitos. El CSV añade el número de MiD inválidos "
        "por banda y una media estricta que queda N/D ante cualquier MiD inválido. "
        "Las medias condicionales no eliminan esos fallos.",
        "",
        "MiD medio pondera muestras y puede incluir GT fuera de las cuatro bandas; "
        "no equivale a overall_MiD. FR oficial cuenta TTC no finitos o de magnitud <0,1 s, "
        "y no equivale a la tasa de avisos urgentes fallidos.",
        "",
        "En TTC largos, diferencias grandes en segundos pueden producir diferencias pequeñas "
        "en la razón de alturas. Por eso ordenar modelos por MAE y por MiD puede dar "
        "resultados distintos. No se debe concluir superioridad en MiD a partir del MAE.",
        "",
    ]
    lines += [
        "Diferencia de MiD medio: H8 menos comparador; un valor negativo favorece a H8. "
        "IC percentil del 95 % con 10.000 remuestreos pareados de secuencias completas. "
        "La media conserva la ponderación por muestras; no se remuestrean ventanas independientes. "
        "FCWD solo tiene tres secuencias y su intervalo es especialmente frágil.",
        "",
        "| Datos | Comparador | Δ MiD | IC 95 % | Secuencias |",
        "|---|---|---:|---|---:|",
    ]
    for pair in pairs:
        if "ci95_low" in pair:
            lines.append(
                f"| {pair['dataset']} | {pair['comparator']} | {pair['delta_mean_MiD']:.3f} | "
                f"[{pair['ci95_low']:.3f}, {pair['ci95_high']:.3f}] | {pair['sequences']} |"
            )
    lines += [
        "",
        "Casos con mayor MiD: `MID_LARGEST_ERRORS.csv`. "
        "Los intervalos exploratorios no corrigen la exposición previa de estas cohortes.",
        "",
    ]
    (output / "MID_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Calculate and publish MiD using only the existing public-label evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
