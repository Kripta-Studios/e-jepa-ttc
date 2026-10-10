"""Generate revision tables and a Spanish report directly from result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json
from operational.ttc_revision.diagnose import diagnose
from operational.ttc_revision.latency_analysis import paired_latency


def markdown(table: object) -> str:
    """Render a compact table without adding an optional tabulate dependency."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("report table must be a DataFrame")

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "—" if pd.isna(value) else f"{value:.4f}"
        return str(value).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
            *(
                "| " + " | ".join(cell(v) for v in row) + " |"
                for row in table.itertuples(index=False, name=None)
            ),
        ]
    )


def figures(campaign: Path, destination: Path, latency: pd.DataFrame) -> None:
    """Plot saved metrics only, with explicit common bounds and seed averaging."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = destination / "figures"
    directory.mkdir(exist_ok=True)
    metrics = pd.read_csv(campaign / "dev32" / "METRICS.csv")
    selected = metrics[
        (metrics.policy == "common_bound_60") & metrics.cohort.str.startswith("positive_")
    ]
    buckets = [
        "positive_0_0.5",
        "positive_0.5_1",
        "positive_1_2",
        "positive_2_4",
        "positive_4_8",
        "positive_8_inf",
    ]
    labels = ["<0.5", "0.5–1", "1–2", "2–4", "4–8", "≥8"]
    groups = {
        "H8 (media 3 cabezas)": [f"H8_seed{s}" for s in (7, 13, 23)],
        "Direct (media 3 cabezas)": [f"Direct_seed{s}" for s in (7, 13, 23)],
        "Garl eventos": ["public_Garl_event_lhr"],
        "Garl RGB+eventos": ["public_Garl_rgb_event_full"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, methods in groups.items():
        matched = pd.DataFrame(selected[selected.method.isin(methods)])
        values = pd.DataFrame(matched.groupby("cohort")[["mae", "bias"]].mean())
        values = values.reindex(buckets)
        for axis, column in zip(axes, ("mae", "bias"), strict=True):
            axis.plot(np.arange(6), values[column], marker="o", label=label)
            axis.set_xticks(np.arange(6), labels)
            axis.set_xlabel("TTC real (s)")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("MAE (s)")
    axes[1].set_ylabel("Sesgo: predicción − GT (s)")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dev32 · límite común ±60 s · medias entre cabezas, no réplicas independientes")
    fig.savefig(directory / "errors.png", dpi=170)
    plt.close(fig)
    latency_figure(destination, latency)


def latency_figure(destination: Path, latency: pd.DataFrame) -> None:
    """Render same-pass CPU/wrapper costs without requiring accuracy results."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = destination / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    labels = {
        "h8_legacy_three": "H8 original · 3 cabezas",
        "h8_fast_three": "H8 optimizado · 3 cabezas",
        "h8_fast_one": "H8 optimizado · 1 cabeza",
        "garl_event_only": "Garl eventos",
        "garl_full": "Garl RGB + eventos",
    }
    modes = latency["mode"].unique()
    fig, axes = plt.subplots(
        1, len(modes), figsize=(6 * len(modes), 4.5), sharex=True, constrained_layout=True
    )
    for axis, mode in zip(np.atleast_1d(axes), modes, strict=True):
        subset = latency[latency["mode"] == mode]
        x = np.arange(len(subset))
        axis.barh(x, subset.cpu_ms, label="Preparación CPU")
        axis.barh(x, subset.wrapper_ms, left=subset.cpu_ms, label="Wrapper sincronizado")
        axis.set_yticks(x, [labels.get(system, system) for system in subset.system])
        axis.invert_yaxis()
        axis.set_xlabel("Milisegundos por consulta")
        axis.set_title(
            "Independiente"
            if mode.startswith("independent")
            else "Cronológico 512 MiB"
            if "512MiB" in mode
            else "Cronológico 128 MiB"
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("Misma pasada · modelos residentes · filesystem caliente · medias medidas")
    fig.savefig(directory / "latency.png", dpi=170)
    plt.close(fig)


def build(campaign: Path, destination: Path) -> None:
    """Require complete results and publish both positive and negative evidence."""
    benchmark = json.loads((campaign / "BENCHMARK.json").read_text(encoding="utf-8"))
    if benchmark["status"] != "COMPLETE":
        raise ValueError("complete benchmark required")
    diagnose(campaign, Path("artifacts/train40_system_20261005/H8_FEATURES.npz"))
    destination.mkdir(parents=True, exist_ok=True)
    tables = destination / "tables"
    tables.mkdir(exist_ok=True)
    raw = pd.read_csv(campaign / "LATENCY_RAW.csv")
    cache_receipt = campaign / "cache512" / "BENCHMARK.json"
    if cache_receipt.exists():
        if json.loads(cache_receipt.read_text(encoding="utf-8"))["status"] != "COMPLETE":
            raise ValueError("cache-capacity experiment not complete")
        raw = pd.concat(
            [raw, pd.read_csv(campaign / "cache512" / "LATENCY_RAW.csv")], ignore_index=True
        )
    measured = raw[~raw.warmup]
    latency_intervals = paired_latency(raw, tables)
    summary = (
        measured.groupby(["mode", "system"])
        .agg(
            n=("e2e_ms", "size"),
            cpu_ms=("cpu_ms", "mean"),
            wrapper_ms=("wrapper_ms", "mean"),
            e2e_ms=("e2e_ms", "mean"),
            e2e_p50_ms=("e2e_ms", "median"),
            e2e_p95_ms=("e2e_ms", lambda x: x.quantile(0.95)),
            cuda_stream_ms=("cuda_stream_ms", "mean"),
            raw_cache_hit_fraction=("raw_cache_hit", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(tables / "latency.csv", index=False)
    figures(campaign, destination, summary)
    outcomes = []
    for cohort in ("dev32", "fcwd"):
        table = pd.read_csv(campaign / cohort / "METRICS.csv")
        native = table[(table.policy == "native") & (table.cohort == "all")].set_index("method")
        old_mae = native.loc["H8_median3", "mae"]
        new_mae = native.loc["Direct_median3", "mae"]
        outcomes.append(
            f"{cohort.upper()}: la salida fija mediana de tres cabezas pasa de "
            f"MAE {old_mae:.4f} s (H8) a {new_mae:.4f} s (Direct), "
            f"{(new_mae / old_mae - 1) * 100:+.2f} %."
        )
        urgent_n = int(native.loc["Direct_median3", "urgent_n"])
        if urgent_n:
            old_miss = native.loc["H8_median3", "urgent_miss_fraction"]
            new_miss = native.loc["Direct_median3", "urgent_miss_fraction"]
            outcomes.append(
                f"Avisos urgentes en {cohort.upper()}: fallos {100 * old_miss:.2f} % "
                f"(H8) → {100 * new_miss:.2f} % (Direct), sobre {urgent_n} consultas "
                "con GT positivo ≤1 s. El aviso exige una predicción positiva ≤1 s; "
                "una mejora de MAE no garantiza mejorar este criterio."
            )
        tail = table[(table.policy == "native") & (table.cohort == "positive_8_inf")].set_index(
            "method"
        )
        if int(tail.loc["Direct_median3", "n"]):
            outcomes.append(
                f"TTC ≥8 s en {cohort.upper()}: MAE "
                f"{tail.loc['H8_median3', 'mae']:.4f} → "
                f"{tail.loc['Direct_median3', 'mae']:.4f} s; sesgo "
                f"{tail.loc['H8_median3', 'bias']:.4f} → "
                f"{tail.loc['Direct_median3', 'bias']:.4f} s."
            )
    text = [
        "# Revisión V12: resultados medidos\n",
        "Informe generado por `python -m operational.ttc_revision.report`. "
        "Los datos originales de la campaña del 8 de octubre permanecen intactos. "
        "[Protocolo y comandos](PROTOCOL.md).\n",
        "\n".join(outcomes) + "\n",
        "## Precisión\n",
        "![Error y sesgo por TTC](figures/errors.png)\n",
        "Los tres checkpoints nuevos son endpoints fijos de 2.500 actualizaciones sobre "
        "TRAIN40. Dev32 y FCWD son poblaciones de diagnóstico ya conocidas, no tests "
        "ciegos. Las semillas comparten productores. Ninguna métrica certifica SOTA. "
        "El replay no consulta GT TTC al inferir, pero consume las ROI oráculo del "
        "manifest; `targets_read=false` se refiere al TTC, no a esas anotaciones.\n",
        "### Problemas que motivan los cambios\n",
        "H8 aprende error en fase, no MAE en segundos. Cerca de fase cero, "
        "TTC ≈ 0.1/fase: una pequeña desviación puede producir un error grande o "
        "cambiar el signo. El decodificador limita magnitud a 60 s y cambia entre "
        "−60 y +60 al cruzar cero. El residual original es lineal y no está "
        "restringido al intervalo de los expertos. Direct elimina ese salto "
        "mediante una salida continua; su beneficio de generalización debe medirse.\n",
        "La subestimación de TTC largos coincide con una supervisión TRAIN40 "
        "concentrada entre −10 y +10 s y un cambio de dominio. Los productores "
        "congelados solo aportan 17 características por observación: cambiar la "
        "cabeza no recupera información visual perdida. Los grupos y diagnósticos "
        "publicados localizan asociaciones, no demuestran por sí solos causalidad.\n",
    ]
    training = []
    registry = []
    freeze = json.loads((campaign / "TRAINING_FREEZE.json").read_text(encoding="utf-8"))
    for seed in (7, 13, 23):
        receipt = json.loads((campaign / f"direct_seed{seed}.json").read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETE":
            raise ValueError("incomplete training seed")
        training.append(
            {
                "seed": seed,
                "updates": receipt["update"],
                "status": receipt["status"],
                "checkpoint_sha256": receipt["sha256"],
            }
        )
        registry.append(
            {
                "experiment_id": "ttc_revision_20261009",
                "run_name": f"direct_seed{seed}",
                "seed": seed,
                **receipt["identity"]["environment"],
                "config_hash": freeze["config_sha256"],
                "dataset_manifest_hash": freeze["feature_manifest_sha256"],
                "dataset_index_hash": freeze["index_sha256"],
                "split_version": "frozen_TRAIN40_20261005_no_new_split",
                "start_time": receipt["identity"]["start_time"],
                "end_time": receipt["updated_utc"],
                "status": receipt["status"],
                "checkpoint_path": str(campaign / f"direct_seed{seed}.pt"),
                "checkpoint_sha256": receipt["sha256"],
                "metrics_path": [str(campaign / c / "METRICS.csv") for c in ("dev32", "fcwd")],
            }
        )
    pd.DataFrame(training).to_csv(tables / "training.csv", index=False)
    atomic_json(destination / "EXPERIMENT_REGISTRY.json", {"runs": registry})
    fit_receipt_path = campaign / "TRAIN_FIT_DIAGNOSTICS.json"
    if fit_receipt_path.exists():
        fit_receipt = json.loads(fit_receipt_path.read_text(encoding="utf-8"))
        if fit_receipt["status"] != "COMPLETE" or fit_receipt["metrics_sha256"] != digest(
            campaign / "TRAIN_FIT_METRICS.csv"
        ):
            raise ValueError("complete verified training-fit diagnostics required")
        fit = pd.read_csv(campaign / "TRAIN_FIT_METRICS.csv")
        fit.to_csv(tables / "train_fit_metrics.csv", index=False)
        text += [
            "### Ajuste sobre TRAIN40\n",
            "Evaluación descriptiva de los endpoints sobre las 88.744 consultas ya "
            "utilizadas al entrenar. Es error sin ponderar en entrenamiento, "
            "no validación ni evidencia de generalización. No consume actualizaciones.\n",
            markdown(
                fit[fit.method.isin(["H8_median3", "Direct_median3"])][
                    ["method", "cohort", "n", "mae", "bias", "sign_error_fraction"]
                ]
            ),
            "\n",
        ]
        tail = fit[
            (fit.cohort == "positive_at_least_8")
            & fit.method.isin(["H8_median3", "Direct_median3"])
        ].set_index("method")
        text.append(
            f"La subestimación larga ya aparece en TRAIN40: sesgo para TTC ≥8 s "
            f"de {tail.loc['H8_median3', 'bias']:.4f} s en H8 y "
            f"{tail.loc['Direct_median3', 'bias']:.4f} s en Direct. Por tanto, "
            "no puede atribuirse exclusivamente al cambio de dominio ni a los "
            "targets de transferencia superiores al rango de entrenamiento.\n"
        )
    for cohort in ("dev32", "fcwd"):
        metrics = pd.read_csv(campaign / cohort / "METRICS.csv")
        metrics.to_csv(tables / f"{cohort}_metrics.csv", index=False)
        paired = pd.read_csv(campaign / cohort / "PAIRED_BOOTSTRAP.csv")
        paired.to_csv(tables / f"{cohort}_paired_bootstrap.csv", index=False)
        overview = metrics[metrics.cohort == "all"]
        selected_columns = [
            "method",
            "n",
            "coverage",
            "mae",
            "median_ae",
            "rmse",
            "bias",
            "rte_percent",
        ]
        text += [
            f"### {cohort.upper()}\n",
            "Salidas nativas:\n",
            markdown(overview[overview.policy == "native"][selected_columns]),
            "\nMismo límite operativo de ±60 s para todos los valores finitos:\n",
            markdown(overview[overview.policy == "common_bound_60"][selected_columns]),
        ]
        native = pd.DataFrame(overview[overview.policy == "native"]).set_index("method")
        old = native.loc[[f"H8_seed{s}" for s in (7, 13, 23)], "mae"].mean()
        new = native.loc[[f"Direct_seed{s}" for s in (7, 13, 23)], "mae"].mean()
        old_std = native.loc[[f"H8_seed{s}" for s in (7, 13, 23)], "mae"].std(ddof=1)
        new_std = native.loc[[f"Direct_seed{s}" for s in (7, 13, 23)], "mae"].std(ddof=1)
        text.append(
            f"\nMAE, media ± desviación entre cabezas: H8 {old:.4f} ± {old_std:.4f} s; "
            f"Direct {new:.4f} ± {new_std:.4f} s "
            f"({(new / old - 1) * 100:+.2f} %). "
            "Esta media describe tres cabezas sobre los mismos productores; no son "
            "réplicas independientes del sistema completo.\n"
        )
        median_comparison = paired[(paired.policy == "native") & (paired.left == "Direct_median3")]
        text += [
            "\nDiferencia de MAE de la mediana fija Direct menos cada comparador; "
            "un valor negativo favorece Direct. IC del 95 % por clusters:\n",
            markdown(
                median_comparison[
                    ["right", "grouping", "status", "difference_mae", "ci_low", "ci_high"]
                ]
            ),
            "\n",
        ]
        families = metrics[
            (metrics.policy == "native")
            & metrics.cohort.str.startswith("family:")
            & metrics.method.isin(["H8_median3", "Direct_median3"])
        ]
        text += [
            "Errores y sesgo por familia, usando una salida fija por sistema:\n",
            markdown(families[["method", "cohort", "n", "mae", "bias", "p95_ae"]]),
            "\n",
        ]
        buckets = metrics[(metrics.policy == "native") & metrics.cohort.str.startswith("positive_")]
        text += [
            "Errores por intervalo de TTC positivo; todos los seeds se conservan en CSV:\n",
            markdown(
                buckets[
                    buckets.method.isin(
                        [
                            "H8_seed7",
                            "Direct_seed7",
                            "public_Garl_event_lhr",
                            "public_Garl_rgb_event_full",
                        ]
                    )
                ][["method", "cohort", "n", "mae", "bias"]]
            ),
            "\n",
        ]
        if "urgent_miss_fraction" in overview:
            text += [
                "Aviso urgente (GT positivo ≤1 s):\n",
                markdown(
                    overview[overview.policy == "native"][
                        [
                            "method",
                            "urgent_n",
                            "urgent_miss_fraction",
                            "urgent_false_alarm_fraction",
                        ]
                    ]
                ),
                "\nUn resultado negativo predicho para un contacto positivo urgente cuenta "
                "como fallo de aviso, no como una alarma correcta.\n",
            ]
    upstream = pd.read_csv(campaign / "UPSTREAM_METRICS.csv")
    shift = pd.read_csv(campaign / "FEATURE_SHIFT.csv")
    concentration = pd.read_csv(campaign / "ERROR_CONCENTRATION.csv")
    upstream.to_csv(tables / "upstream_metrics.csv", index=False)
    shift.to_csv(tables / "feature_shift.csv", index=False)
    concentration.to_csv(tables / "error_concentration.csv", index=False)
    timing = json.loads((campaign / "TIMING_DIAGNOSTICS.json").read_text(encoding="utf-8"))
    time_one = timing["channels"]["channel1_train_minus_transfer_convention"]
    time_three = timing["channels"]["channel3_train_minus_transfer_convention"]
    text += [
        "## Diagnóstico de productores y cambio de dominio\n",
        "Predicciones reconstruidas desde las fases FP32 almacenadas, usando el "
        "decodificador H8 con límite ±60 s. Son diagnósticos posteriores al entrenamiento, "
        "no predicciones nativas exactas de cada productor ni nuevos baselines oficiales. "
        "La mediana de fases usa solo predicciones, sin elegir un experto mediante GT.\n",
        f"Convenciones temporales: la fracción del canal 1 con discrepancia mayor "
        f"que 1 µs es {time_one['fraction_above_one_microsecond_abs']:.4f}. En el canal 3, "
        f"la diferencia mediana es {1000 * time_three['quantiles_seconds']['median']:.3f} ms "
        f"y el p99 {1000 * time_three['quantiles_seconds']['p99']:.3f} ms. "
        "No se ha demostrado que esta diferencia explique el error de transferencia.\n",
        markdown(
            upstream[upstream.cohort.isin(["all", "positive_at_most_1", "positive_at_least_8"])][
                ["population", "method", "cohort", "n", "mae", "bias"]
            ]
        ),
        "\nFracción del error absoluto concentrada en cada intervalo. Se muestran las "
        "salidas fijas de tres cabezas:\n",
        markdown(
            concentration[
                concentration.method.isin(["H8_median3", "Direct_median3"])
                & ~concentration.cohort.str.startswith("family:")
                & (concentration.cohort != "all")
            ]
        ),
        "\nCaracterísticas con más observaciones fuera de los percentiles 0,5–99,5 "
        "de TRAIN40, calculados sin ponderación sobre las observaciones actuales. "
        "Se usa toda la población sensorial, incluso filas sin GT elegible. Esta "
        "fracción no es una probabilidad OOD calibrada ni demuestra causalidad:\n",
        markdown(
            shift.sort_values("outside_train_envelope_fraction", ascending=False)
            .groupby("population", sort=False)
            .head(6)[
                [
                    "population",
                    "feature",
                    "train_median",
                    "transfer_median",
                    "outside_train_envelope_fraction",
                ]
            ]
        ),
        "\n",
        "## Latencia\n",
        "![Desglose de latencia](figures/latency.png)\n",
        markdown(summary),
        "\nMediciones nuevas, en la misma pasada, sin los dos entrenamientos V13. "
        "CPU incluye lectura y representación; wrapper incluye transferencias, "
        "cómputo y sincronización. CUDA stream incluye huecos de envío desde el host. "
        "No comparar directamente con los tiempos históricos obtenidos bajo otras cargas. "
        "El benchmark de latencia usa consultas Dev32. FCWD conserva su preparación "
        "nativa para el replay de precisión; aquí no se mide una aceleración E2E de FCWD. "
        "El muestreo cubre ocho familias: una consulta por familia en modo independiente "
        "y cinco consecutivas por familia en modo cronológico, cada una repetida cinco "
        "veces. Las repeticiones no son secuencias independientes. La CPU y el escritorio "
        "no estuvieron aislados de toda actividad externa. "
        "La obtención de las ROI oráculo y la espera de sincronización RGB "
        "(hasta 1 ms posterior al ancla) no están incluidas en esos tiempos. "
        "El pico de memoria del CSV crudo corresponde al proceso con todos los modelos "
        "residentes, no a VRAM aislada de cada arquitectura.\n",
        "El baseline `h8_legacy_three` reproduce el adaptador histórico, que también "
        "construía la entrada de Garl aunque H8 no la consumiera. Parte del ahorro "
        "consiste en eliminar ese trabajo ajeno al modelo; no es una aceleración "
        "aislada de la red neuronal. Garl full conserva su adaptador sin caché "
        "persistente de eventos ni frames: sus tiempos describen esa implementación, "
        "no el límite de rendimiento de su arquitectura.\n",
        "Intervalos de la mejora de H8 con tres cabezas. El bootstrap remuestrea "
        "familias completas, conservando consultas y repeticiones emparejadas. "
        "Son intervalos descriptivos de esta muestra y este host:\n",
        markdown(
            latency_intervals[latency_intervals.candidate == "h8_fast_three"][
                [
                    "mode",
                    "families",
                    "reduction_percent",
                    "reduction_ci_low_percent",
                    "reduction_ci_high_percent",
                ]
            ]
        ),
        "\n",
    ]
    independent = summary[summary["mode"] == "independent_raw_cache_disabled"].set_index("system")
    old_ms = independent.loc["h8_legacy_three", "e2e_ms"]
    new_ms = independent.loc["h8_fast_three", "e2e_ms"]
    text.append(
        f"H8 tres cabezas: {old_ms:.3f} → {new_ms:.3f} ms por consulta en el modo "
        f"independiente ({100 * (1 - new_ms / old_ms):.2f} % menos; {old_ms / new_ms:.2f}×). "
        "La fila de una cabeza es otro sistema y no sustituye esta comparación.\n"
    )
    cached = summary[summary["mode"].str.contains("512MiB")].set_index("system")
    if not cached.empty:
        old_ms = cached.loc["h8_legacy_three", "e2e_ms"]
        new_ms = cached.loc["h8_fast_three", "e2e_ms"]
        text.append(
            f"Con retención de 512 MiB y consultas cronológicas, H8 tres cabezas: "
            f"{old_ms:.3f} → {new_ms:.3f} ms ({100 * (1 - new_ms / old_ms):.2f} % menos; "
            f"{old_ms / new_ms:.2f}×). Es una comparación con su baseline contemporáneo; "
            "no una resta entre campañas históricas ni una garantía de tiempo real.\n"
        )
        row = cached.loc["h8_fast_three"]
        text.append(
            f"En esa ruta optimizada, la preparación CPU ocupa "
            f"{100 * row['cpu_ms'] / row['e2e_ms']:.1f} % del total "
            f"({row['cpu_ms']:.1f} ms); el wrapper sincronizado, "
            f"{row['wrapper_ms']:.1f} ms. Las 24 ventanas originalmente construidas "
            "se reducen a 12 ventanas únicas, los productores procesan ocho "
            "observaciones reales y se reutilizan eventos crudos solapados. "
            "Los tres predictores TTC comparten productores: eliminar dos cabezas "
            "no elimina ese coste dominante.\n"
        )
    compiled_receipt = campaign / "compiled" / "COMPILED_PROBE.json"
    dense_profile = campaign / "DENSE_CPU_PROFILE.json"
    if dense_profile.exists():
        dense = json.loads(dense_profile.read_text(encoding="utf-8"))
        timings = dense["diagnostics"]
        text.append(
            f"Diagnóstico adicional de la consulta densa `{dense['query_id']}`: "
            f"lectura {timings['read_ms']:.1f} ms y preparación H8 posterior "
            f"{timings['h8_voxel_ms']:.1f} ms. Esta medición usa un thread CPU, "
            "caché inicialmente vacía y se hizo mientras ejecutaba el replay; "
            "no se mezcla con el benchmark ni sirve para comparar sistemas. "
            "El perfil conserva costes de lectura HDF5, recorte, proyección ROI "
            "y voxelización. Las primeras cinco consultas por familia no "
            "caracterizan todas las colas de las escenas de alta densidad.\n"
        )
    if compiled_receipt.exists():
        compiled = json.loads(compiled_receipt.read_text(encoding="utf-8"))
        text.append("### Ejecución compilada (wrapper, sin preparación CPU)\n")
        if compiled["status"] == "COMPLETE":
            timings = pd.read_csv(campaign / "compiled" / "COMPILED_LATENCY.csv")
            compiled_summary = (
                timings.groupby("system")
                .agg(
                    n=("wrapper_ms", "size"),
                    wrapper_ms=("wrapper_ms", "mean"),
                    p50_ms=("wrapper_ms", "median"),
                    p95_ms=("wrapper_ms", lambda x: x.quantile(0.95)),
                )
                .reset_index()
            )
            compiled_summary.to_csv(tables / "compiled_wrapper.csv", index=False)
            text += [
                markdown(compiled_summary),
                "\nLa compilación y sus llamadas de "
                "calentamiento figuran en el recibo separado. Estos tiempos no son E2E.\n",
            ]
        else:
            text.append(
                f"Estado: {compiled['status']}. "
                f"Motivo registrado: {compiled.get('error', '')[:500]}\n"
            )
    text.append("## Paridad y límites\n")
    impact = pd.read_csv(campaign / "NUMERICAL_METRIC_IMPACT.csv")
    impact.to_csv(tables / "numerical_metric_impact.csv", index=False)
    text += [
        "Impacto de ejecutar la salida fija mediana de tres cabezas por la ruta "
        "compacta, sobre las consultas con GT elegible. Se cuentan también los "
        "cambios de decisión de aviso (predicción positiva ≤1 s); la tolerancia "
        "numérica por sí sola no garantiza conservar una decisión discreta.\n",
        markdown(impact),
        "\n",
    ]
    for cohort in ("dev32", "fcwd"):
        admission = json.loads(
            (campaign / f"{cohort.upper()}_PREDICTIONS.json").read_text(encoding="utf-8")
        )
        text.append(
            f"- {cohort}: ejecución compacta admitida en todas las consultas: "
            f"{admission['compact_admitted_every_query']}; máxima diferencia "
            f"{admission['compact_max_abs_seconds']:.8f} s. Replay canónico frente "
            f"a CSV original: {admission['canonical_original_max_abs_seconds']}; "
            f"admisión {admission['canonical_original_admitted_every_query']}.\n"
        )
        text.append(
            f"  Cabezas Direct evaluadas en CPU con las características compactas: "
            f"admisión {admission['direct_compact_features_admitted']}; diferencia máxima "
            f"{admission['direct_compact_features_max_abs_seconds']:.8f} s. Ejecución "
            f"Direct completa en GPU, canónica y compacta: admisión "
            f"{admission['direct_gpu_admitted_every_query']}, diferencia máxima "
            f"{admission['direct_gpu_max_abs_seconds']:.8f} s frente al scoring CPU. "
            "Esta prueba no mide latencia aislada de Direct.\n"
        )
    text += [
        "\nLa optimización de entradas conserva los tensores de referencia en las pruebas "
        "y en las consultas reales de admisión. El replay completo verifica las "
        "predicciones, con sus diferencias numéricas explícitas.\n",
        "La cabeza nueva es un experimento, no una sustitución automática de H8. "
        "También cambia el batch de entrenamiento (256 frente a 128 histórico): "
        "Direct usa dimensión oculta 64 frente a 160 y LR constante 3e-4 frente "
        "al warmup y descenso coseno de H8. Por tanto, "
        "esta comparación no aísla causalmente el efecto de la pérdida. "
        "Si empeora MAE, sesgo, colas o avisos urgentes, ese fallo permanece visible. "
        "Aprender directamente en segundos elimina una singularidad de salida, pero "
        "no añade información que los productores no hayan conservado.\n",
        "Direct es un candidato de regresión TTC; no incorpora intervalos calibrados "
        "ni probabilidades de riesgo. La discrepancia entre semillas no constituye "
        "por sí sola una incertidumbre calibrada.\n",
        "Continúan pendientes una cohorte realmente ciega, el contrato oficial Garl "
        "completo, calibración espacial FCWD full y entrenamiento comparable de ambos "
        "sistemas. Las ROI son oráculo, las modalidades y duración de contexto difieren. "
        "No hay fundamento para afirmar que V12 bate a Garl en todo.\n",
        "Se conserva el piloto detenido en `pilot_full_raw` y la medición con 128 MiB. "
        "La ablation de 512 MiB mantiene su propio baseline y orden alternado. Las fuentes "
        "efectivas del benchmark inicial están archivadas en `source_archive/primary_cost`; "
        "los hashes iniciales de módulos no utilizados pueden corresponder a sus versiones "
        "anteriores al desarrollo del generador de informes y del replay.\n",
    ]
    official_path = campaign / "OFFICIAL_PROTOCOL_REVIEW.json"
    if official_path.exists():
        official = json.loads(official_path.read_text(encoding="utf-8"))
        text += [
            "### Diferencia respecto al benchmark oficial\n",
            "El contrato público separa train40 de test12 y conserva las etiquetas TTC "
            "de test12 privadas; recibe predicciones JSON. Dev32 y FCWD no son esa "
            "evaluación. [Dataset oficial](https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset).\n",
            f"La lista de entrenamiento del release local contiene "
            f"{official['local_release_config_train_sequences']} secuencias; V12 usa "
            f"{official['v12_train_sequences']}. El registro guarda las diferencias y "
            "sus hashes. Esto no demuestra que los pesos públicos se entrenaran con "
            "las 46: su ancestría y presupuesto no quedan igualados por una lista de "
            "configuración. La model card identifica los pesos full y los de sus ramas, "
            "sin resolver aquí esa equivalencia. "
            "[Model card](https://huggingface.co/NAIL-HNU/GarlTTC-model).\n",
        ]
    checks_path = campaign / "CPU_COMPLETION.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        if checks.get("status") == "COMPLETE":
            counts = checks["tests"]
            text += [
                "### Validación de la entrega\n",
                f"Pytest: {counts['tests']} casos, {counts['failures']} fallos, "
                f"{counts['errors']} errores y {counts['skipped']} omitidos. "
                "Estos son los casos V12 de esta revisión, separados de la suite "
                "RGB-PORT documentada por la auditoría V13. Los logs y el XML de "
                "Pytest están en el directorio de artefactos.\n",
                markdown(pd.DataFrame(checks["checks"])[["name", "exit_code"]]),
                "\n",
            ]
    budget = json.loads((campaign / "TRAINING_BUDGET.json").read_text(encoding="utf-8"))
    text.append(
        f"Tiempo registrado del ajuste de las tres cabezas: "
        f"{budget['elapsed_seconds'] / 60:.2f} min; el replay y benchmark se contabilizan "
        "aparte. Los logs y hashes están en `artifacts/ttc_revision_20261009`.\n"
    )
    resume_path = campaign / "V13_RESUME_REQUEST.json"
    if resume_path.exists():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        text.append(
            f"Ventana reservada desde la pausa de V13 hasta retirar nuestros marcadores: "
            f"{resume['elapsed_since_pause_seconds'] / 3600:.3f} h. Es tiempo de reloj, "
            "incluye preparación y espera; no representa horas de cómputo GPU activo.\n"
        )
    verified_resume = campaign / "V13_RESUME_VERIFIED.json"
    if verified_resume.exists():
        resume = json.loads(verified_resume.read_text(encoding="utf-8"))
        text.append(f"Estado verificado de reanudación V13: {resume['status']}.\n")
        if resume["status"] == "RESUMED_AND_VERIFIED":
            for fit_id, fit in resume["fits"].items():
                text.append(
                    f"- {fit_id}: checkpoint pausado en {fit['paused_updates']} "
                    f"actualizaciones; nuevo checkpoint observado en "
                    f"{fit['observed_updates']}; geometría {fit['geometry_precision']}.\n"
                )
            text.append(
                "\nLa preparación CUDA Graphs de C2F superó las comprobaciones "
                "de restauración exacta y consumió "
                f"{resume['c2f_prewarm']['optimizer_updates']} actualizaciones de optimizador.\n"
            )
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")
    atomic_json(
        destination / "REPORT_SOURCES.json",
        {
            "generator_sha256": digest(Path(__file__)),
            "tables": {p.name: digest(p) for p in tables.glob("*.csv")},
            "benchmark_sha256": digest(campaign / "BENCHMARK.json"),
            "cache512_benchmark_sha256": digest(cache_receipt) if cache_receipt.exists() else None,
            "report_sha256": digest(destination / "README.md"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("artifacts/ttc_revision_20261009"))
    parser.add_argument("--destination", type=Path, default=Path("docs/ttc_revision_20261009"))
    args = parser.parse_args()
    build(args.campaign, args.destination)
