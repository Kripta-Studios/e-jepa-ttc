"""Replay frozen queue metrics and preserve tables, clipping and group contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import ROOT, RUN


def main() -> None:
    rows, replay, clips, contrasts = [], [], [], []
    sources = [
        ("B", "PRECISION_RESULTS.json", "precision_evaluation"),
        ("D", "D_RESULTS.json", "evaluation"),
        ("E", "GEOMETRY_SCREEN_RESULTS.json", "geometry_evaluation"),
    ]
    for queue, filename, directory in sources:
        if not (RUN / filename).exists():
            continue
        results = read(RUN / filename)
        for key, score in results.items():
            role, name = key.split("/") if "/" in key else ("V", key)
            folder = RUN / directory / role / name if queue != "D" else RUN / directory / name
            frame = pd.read_csv(
                folder / "PREDICTIONS.csv",
                converters={"target_ttc": float, "prediction_ttc": np.float32},
            )
            receipt = read(folder / "RECEIPT.json")
            if sha(folder / "PREDICTIONS.csv") != receipt["predictions_sha256"]:
                raise ValueError("Predictions changed")
            target = frame.target_ttc.to_numpy(np.float64)
            prediction = frame.prediction_ttc.to_numpy(np.float32)
            groups = frame.group_id.to_numpy(str)
            actual = metrics(target, prediction, groups)
            if actual != score:
                raise AssertionError(f"Metric replay differs: {queue}/{key}")
            write(
                folder / "GROUP_METRICS.json",
                {
                    group: metrics(
                        target[groups == group],
                        prediction[groups == group],
                        groups[groups == group],
                    )
                    for group in sorted(set(groups))
                },
            )
            replay.append({"queue": queue, "fit": key, "rows": len(frame), "exact": True})
            rows.append(
                {
                    "queue": queue,
                    "role": role,
                    "fit": name,
                    **{
                        band: score["bins"][band]["group_macro_mid"]
                        for band in ("crucial", "small", "large", "negative")
                    },
                    "overall": score["overall_MiD"],
                    "mae_s": score["diagnostics"]["mae_s"],
                    "late_warning": score["late_warning"]["rate"],
                    "false_alert": score["false_alert"]["rate"],
                }
            )
            if role == "P" or queue == "D":
                curve = pd.read_csv(RUN / f"fits/{name}/CURVE.csv")
                clips.append(
                    {
                        "fit": name,
                        "updates": len(curve),
                        "clipped_fraction": float((curve.gradient_l2 > 1).mean()),
                        "last250_clipped_fraction": float((curve.gradient_l2.tail(250) > 1).mean()),
                        "last250_loss": float(curve.loss.tail(250).mean()),
                    }
                )
        arms = ("U0", "U1") if queue == "B" else (("Q0", "Q1") if queue == "E" else ("Z", "D"))
        for seed in (7, 13, 23):
            role = "V" if queue == "D" else "H"
            key = f"{role}/{arms[1]}-S{seed}" if queue != "D" else f"D-S{seed}"
            if key not in results:
                continue
            baseline = (
                read(RUN / "RESULTS.json")[f"Z-S{seed}"]
                if queue == "D"
                else results[f"{role}/{arms[0]}-S{seed}"]
            )
            candidate = results[key]
            groups = sorted(baseline["group_crucial_mid"])
            diff = np.array(
                [
                    candidate["group_crucial_mid"][g] - baseline["group_crucial_mid"][g]
                    for g in groups
                ]
            )
            rng = np.random.default_rng(12345)
            draws = rng.integers(len(groups), size=(2000, len(groups)))
            low, high = np.quantile(diff[draws].mean(1), [0.025, 0.975])
            contrasts.append(
                {
                    "queue": queue,
                    "role": role,
                    "seed": seed,
                    "contrast": f"{arms[1]} minus {arms[0]}",
                    "MiDc_difference": float(diff.mean()),
                    "CI95_low": float(low),
                    "CI95_high": float(high),
                    "groups": len(groups),
                    "draws": 2000,
                }
            )
    pd.DataFrame(rows).to_csv(RUN / "QUEUES_TABLE.csv", index=False)
    pd.DataFrame(clips).to_csv(RUN / "QUEUES_CLIPPING.csv", index=False)
    pd.DataFrame(contrasts).to_csv(RUN / "QUEUES_BOOTSTRAP.csv", index=False)
    write(
        RUN / "QUEUES_METRIC_REPLAY.json",
        {"exact": True, "tables": replay, "rows": sum(r["rows"] for r in replay)},
    )
    lines = [
        "# V17: resultados de las colas adicionales",
        "",
        "Tablas regeneradas desde predicciones, con replay exacto de métricas. P es ajuste; H es cribado de productores; V es desarrollo interno expuesto. No son resultados oficiales ni SOTA.",
        "",
        "| Cola | Rol | Fit | MiDc | MiDs | MiDl | MiDn | Overall | MAE s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[k]) if k in ("queue", "role", "fit") else f"{row[k]:.6f}"
                for k in (
                    "queue",
                    "role",
                    "fit",
                    "crucial",
                    "small",
                    "large",
                    "negative",
                    "overall",
                    "mae_s",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Las curvas comparan trayectorias dentro de cada objetivo; sus pérdidas absolutas no son una métrica común entre objetivos. El clipping cuenta normas previas al recorte >1. Los intervalos se remuestrean sobre grupos completos (2000 draws, seed12345), no ventanas independientes.",
        "",
        "Los costes GPU incluyen entrenamiento, QA y evaluación registrados en GPU_BUDGET.json. No equivalen a latencia de inferencia desde eventos crudos. No se ejecuta evaluación externa para rescatar candidatos rechazados.",
    ]
    (ROOT / "docs/representation_audit_20261012/QUEUES_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
