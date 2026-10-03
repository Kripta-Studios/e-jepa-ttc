"""Publish the completed frozen-system evidence, keeping historical deliveries intact."""

from __future__ import annotations

# Checked JSON/Pandas receipts are intentionally dynamic at the I/O boundary.
# Report paragraphs deliberately remain single Markdown lines.
# ruff: noqa: ANN401, E501
import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from operational.simplex_t_post_campaign.contracts import checked, digest, read, save  # noqa: E402
from operational.simplex_t_post_campaign.run import (  # noqa: E402
    DOCS,
    EXTRACT,
    NIGHT,
    OUT,
    SOURCE_DOCS,
    Owner,
    admitted,
    guard,
    now,
    progress,
    table,
)


def reconcile_review() -> None:
    """Resolve CSV field aliases before judging folds/targets; preserve exact differences."""
    import numpy as np
    import pandas as pd

    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass

    inventory = read(OUT / "PREDICTION_INVENTORY.json")
    base = (
        pd.concat(
            [pd.read_parquet(row["path"]) for row in inventory["inputs"] if row["model"] == "H8@7"]
        )
        .sort_values("sample_token")
        .reset_index(drop=True)
    )
    review = read(OUT / "COMPARATOR_REVIEW.json")
    reader: Any = pd.read_csv
    for row in review["comparisons"]:
        f = (
            pd.concat(
                [reader(path, float_precision="round_trip") for path in row["evidence_paths"][1:]]
            )
            .sort_values("sample_token")
            .reset_index(drop=True)
        )
        row["alias_mapping"] = dict(fold="outer_fold", target_ttc_s="target_ttc")
        row["identical_outer_fold"] = bool(np.array_equal(f.fold, base.outer_fold))
        row["identical_target_ttc"] = bool(np.array_equal(f.target_ttc_s, base.target_ttc))
        row["max_target_difference"] = float(np.max(np.abs(f.target_ttc_s - base.target_ttc)))
        row["identical_macro_mass"] = bool(
            np.array_equal(
                strict_macro_mass(f.target_ttc_s.to_numpy(), f.sequence_id.to_numpy()),
                strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy()),
            )
        )
        row["target_difference_interpretation"] = (
            "CSV last-bit serialization differences; not evidence of different scientific population."
        )
        row["queries_and_folds_match_after_alias_resolution"] = bool(
            row["identical_query_tokens"]
            and row["identical_sequence_id"]
            and row["identical_track_id"]
            and row["identical_outer_fold"]
        )
    review["supersedes_unmapped_column_flags"] = "COMPARATOR_REVIEW.json"
    review["classification_unchanged"] = (
        "Producer genealogy/ROI/information contract remains insufficient; float rounding alone is not a scientific mismatch."
    )
    save(OUT / "COMPARATOR_REVIEW_RECONCILED.json", review)


def results() -> None:
    """Publish all timing blocks and retain the original accuracy criteria."""
    admitted()
    guard()
    import numpy as np
    import pandas as pd

    receipt = read(OUT / "PROFILE_RECEIPT.json")
    if receipt["status"] != "COMPLETE" or receipt["fragments"] != 540:
        raise ValueError("all three matched blocks required before final comparison")
    reconcile_review()
    cost = pd.read_csv(OUT / "MATCHED_HEAD_COST.csv")
    scores = read(OUT / "PREDICTION_INVENTORY.json")["scores"]
    scores.update(H1_SEED7=132.83161914277504, H8_SEED7=scores["H8@7"], H16_SEED7=scores["H16@7"])
    # H1 score comes from its physically published registry rather than a rounded prose number.
    registry = list(csv.DictReader((DOCS / "BASELINE_REGISTRY.csv").open(encoding="utf-8")))
    scores["H1_SEED7"] = float(next(r for r in registry if r["id"] == "TPR-D1-H1-C160@7")["MiD"])
    rows = []
    for label, group in cost.groupby("model", sort=False):
        p95 = group.p95_ms.to_numpy()
        rows.append(
            dict(
                model=label,
                MiD=scores[label],
                accuracy_scope="OLD_DEV_seed7_three_folds",
                p95_median_ms=float(np.median(p95)),
                p95_min_ms=float(p95.min()),
                p95_max_ms=float(p95.max()),
                p50_median_ms=float(group.p50_ms.median()),
                blocks=3,
                measurements=1500,
                cost_scope="TRAIN_fold0_seed7_prepared_input_head_only",
                system_cost_measured=False,
                pareto_point_estimate=not any(
                    scores[other.model] <= scores[label]
                    and other.p95_ms <= float(np.median(p95))
                    and (
                        scores[other.model] < scores[label] or other.p95_ms < float(np.median(p95))
                    )
                    for other in cost.groupby("model", as_index=False)
                    .p95_ms.median()
                    .itertuples(index=False)
                ),
            )
        )
    table("ACCURACY_COST_FRONTIER.csv", rows)
    screens = []
    for name in ("N2_RESULTS.json", "N3_RESULTS.json"):
        for key, row in read(DOCS / name)["comparisons"].items():
            c, r = row["candidate"], row["reference"]
            r = "H8_SEED7" if r == "H8@7" else r
            ratios = []
            for block in range(1, 4):
                a = cost[(cost.model == c) & (cost.block == block)].p95_ms.iloc[0]
                b = cost[(cost.model == r) & (cost.block == block)].p95_ms.iloc[0]
                ratios.append(float(1 - a / b))
            precision = row["engineering_screen"]["precision_screen_pass"]
            screens.append(
                dict(
                    contrast=key,
                    candidate=c,
                    reference=r,
                    precision_screen_pass=precision,
                    delta_MiD=row["paired_loss_delta"],
                    hierarchical_ci95=row["hierarchical_ci95"],
                    sequence_only_ci95=row["sequence_only"]["ci95"],
                    sign_error_delta=row["practical"]["weighted_sign_error_delta"],
                    crucial_delta=row["practical"]["crucial_bucket_mid_delta"],
                    p95_reduction_by_block=ratios,
                    all_blocks_p95_reduction_at_least_20_percent=all(x >= 0.2 for x in ratios),
                    latency_screen_by_block=[x >= 0.2 for x in ratios],
                    all_block_summary_is_descriptive=True,
                    qualifies_prepared_head_only=(
                        c != "FULL_C0" and precision and all(x >= 0.2 for x in ratios)
                    ),
                    full_c0_is_objective_control=c == "FULL_C0",
                    qualifies_system_substitution=False,
                    system_route_blocked=True,
                    changed_thresholds=False,
                    confirmatory=False,
                )
            )
    save(
        OUT / "FIXED_SCREEN_RECEIPT.json",
        dict(contrasts=screens, scope="exploratory_development_reuse", promotion=False),
    )
    # A static scientific figure shows ranges across all blocks, not selected fastest runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        center = row["p95_median_ms"]
        ax.errorbar(
            center,
            row["MiD"],
            xerr=[[center - row["p95_min_ms"]], [row["p95_max_ms"] - center]],
            fmt="o",
            capsize=3,
            label=row["model"],
        )
    ax.set_xlabel("Prepared head p95 ms: median and range of three matched blocks")
    ax.set_ylabel("OLD_DEV MiD, seed7, three folds (lower is better)")
    ax.set_title("Frozen SIMPLEX-T: head cost only; system cost not measured")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    SOURCE_DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(SOURCE_DOCS / "ACCURACY_HEAD_COST.svg")
    fig.savefig(SOURCE_DOCS / "ACCURACY_HEAD_COST.png", dpi=160)
    plt.close(fig)
    h16 = read(DOCS / "H16_RESULTS.json")
    decision = dict(
        study="FROZEN_SYSTEM_POST_20261003",
        stages=dict(
            P0="COMPLETE",
            P1="COMPLETE",
            P2="COMPLETE",
            P3="BLOCKED",
            P4="COMPLETE_WITH_SYSTEM_COST_LIMITATION",
        ),
        optimizer_updates=0,
        new_fits=0,
        historical_candidate="TPR-D1-H8-C160",
        candidate_replaced=False,
        H16=dict(
            status="HEAD_SEED_IMPROVEMENT_STABLE_SCENE_GENERALIZATION_UNCERTAIN",
            evidence=h16["comparisons"],
            promotion=False,
        ),
        reductions_and_aggregators="Original precision screens still fail; lower head latency cannot reverse those results.",
        full_route_dependency=read(OUT / "FULL_ROUTE_ADMISSION.json")["missing_dependency"],
        next_action="Complete frozen full-route profiling only after a live exclusive resource lease and verified producer/raw bindings; no training or sealed confirmation authorized.",
        application_latency_budget="not supplied; no invented acceptable system latency",
        scientific_limits=[
            "Nine independent sequences, not 27 seed-scenes",
            "Replicated heads, not all experts",
            "Past context informative; chronological mechanism not established",
            "No faster AEB/persistent tracking/calibrated uncertainty claim",
        ],
    )
    save(OUT / "NEXT_DECISION_POST_CAMPAIGN.json", decision)
    write_report(rows, screens)
    progress("P4", "COMPLETE", head_blocks=3, system_route="BLOCKED", optimizer_updates=0)


def write_report(rows: list[dict], screens: list[dict]) -> None:
    """Describe final evidence in Spanish with numerical scope and practical limits."""
    SOURCE_DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Estudio posterior SIMPLEX-T: modelos congelados",
        "",
        "P0, P1, P2 y P4 completados. P3 bloqueado por ausencia de un slot exclusivo explícito GPU/lectura pesada. No se ejecutaron nuevos fits ni updates de optimizador. T6, H8 registrado y los 24 endpoints de la campaña nocturna permanecen intactos.",
        "",
        "Se verificaron los hashes de los 24 endpoints (60.000 updates históricos guardados). Las 98.304 filas de 12 combinaciones modelo/seed conservan las 8.192 queries, nueve secuencias, targets y folds emparejados. La diferencia máxima de scores frente a las publicaciones físicas fue 1,9895196601282805e-13 MiD. Se exportaron 583 filas de contrastes y 90.079 filas de episodios/contrastes; los episodios son agrupaciones descriptivas por track/timestamp con salto fijo de 100 ms, no una demostración de tracking online.",
        "",
        "H16 mejora frente a H8 en las tres seeds de cabeza. Delta medio de pérdidas de las dos nuevas: −2,423702276 MiD; CI jerárquico95 [−5,244259455, +0,041091485]. Con las tres seeds: −2,542665966; CI [−5,213397889, −0,111240531]. Seed7 fue exploratoria ya observada. Las pérdidas se promedian por query; nunca se promedian TTC para crear un ensemble. Esto acredita estabilidad local de optimización de las cabezas, con incertidumbre entre escenas y desarrollo reutilizado.",
        "",
        "En OBneIVg4Cw los deltas H16−H8 son +4,799021788/+0,433603142/+3,279250808 para seeds7/13/23. Para las dos nuevas, el bucket positivo 0–3 s empeora +8,640189 MiD mientras los otros tres buckets mejoran; su delta secuencia es +1,856427. Es una localización descriptiva del fallo, no una causa demostrada. No se ha eliminado ninguna secuencia, retocado umbrales o elegido otro checkpoint. MODEL_DIAGNOSTICS.csv conserva signos, saturación, cap, cobertura, escapes y ganancias/daños por grupo. El hull completo de tres expertos es sólo diagnóstico para brazos reducidos; la cobertura de q10/q90 no demuestra calibración.",
        "",
        "La revisión acotada de C2F oficial V7 y Garl verifica queries/folds coincidentes tras resolver aliases fold/outer_fold y target_ttc_s/target_ttc. Las diferencias de targets son de última cifra CSV: 1,7763568394002505e-15 y 8,881784197001252e-16 s. No justifican declarar otra población. La autoridad completa de productores, ROI/modalidad/availability, TRAIN y selección sigue sin quedar ligada por esos manifiestos; ambos permanecen como evidencia insuficiente para proclamarlos ganadores comparables. Los flags preliminares sin aliases quedan explícitamente sustituidos por COMPARATOR_REVIEW_RECONCILED.json.",
        "",
        "Se midieron 13.500 inferencias batch1 en tres bloques, nueve cabezas, 500 medidas/modelo/bloque; 25 warmups/modelo/bloque y 540 fragmentos persistidos. Se conserva CPU FP32, cuatro threads, interop2 y emisión TTC canónica float64. El orden de modelos rota por fragmento, fijado antes de medir. Se publican todos los bloques, no la ronda más rápida. Los valores siguientes son mediana y rango de p95 entre bloques, no intervalos de confianza ni latencia del sistema completo.",
        "",
        "| Cabeza seed7 | MiD OLD_DEV | p95 mediano ms | Rango p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['MiD']:.6f} | {row['p95_median_ms']:.6f} | {row['p95_min_ms']:.6f}–{row['p95_max_ms']:.6f} |"
        )
    lines += [
        "",
        "Los costes corresponden exclusivamente a entradas TRAIN preparadas, fold0/seed7. No incluyen raw slice, ROI, contexto, expertos, detector, tracker ni decisión AEB. La carga del host, los intervalos entre llamadas y el backend pueden afectar los tiempos; esta ejecución registra backend/shapes/timer pero no aísla la causa de la discrepancia con los perfiles históricos. La precisión usa tres folds; el coste no equivale a medir esos tres sistemas en despliegue.",
        "",
        "Las tres reducciones y los dos agregadores siguen fallando el cribado original de precisión: límite superior CI95 del exceso <2 MiD, signo ≤+0,5 puntos porcentuales, crucial ≤+5 MiD y cobertura completa. El requisito de ≥20% de ahorro p95 sólo se examina en su scope de cabeza. No altera los CI, no demuestra equivalencia por un p-valor no significativo y no autoriza sustitución del sistema. FULL_C0 controla el cambio lambda_cost=0; su comparación con H8 tampoco promociona una nueva receta.",
        "",
        "Se implementó y probó una interfaz de tiempos segmentados condicionada a ROI externo, con sincronización explícita y rechazo de llamadas raw/productores sin admisión. Sus pruebas usan fixtures, no resultados eAP. La integración con productores reales y la medición integral quedan pendientes: falta el slot exclusivo del propietario Stage70–76; después deben validarse los bindings TRAIN raw, pesos de expertos y semántica de disponibilidad. No se obtuvo permiso de hardware o lectura pesada por observar procesos ausentes, ni se cambió Stage70–76.",
        "",
        "Decisión: mantener H8 como candidato histórico. H16 es el challenger de precisión local; el coste integral y la generalización independiente siguen por establecer. No añadir seeds, arquitecturas, encoders, LATENT, datasets o entrenamiento de rescate. El siguiente paso dentro de este plan es únicamente el perfilado congelado integral al disponer de la dependencia externa. Una futura confirmación en escenas independientes requeriría protocolo y autorización distintos.",
        "",
        "El bundle contiene predicciones canónicas de los 12 grupos modelo/seed, pesos e inputs preparados deduplicados para nueve cabezas de coste, tiempos por fragmento, recibos estadísticos históricos, código y pruebas. Permite regenerar scores, contrastes por estrato, cuantiles de tiempos archivados e inferencia de las nueve cabezas desde caché. No reconstruye eventos crudos, productores ni expertos; no puede reproducir físicamente tiempos de pared idénticos. La regeneración no equivale a reentrenar o a replicar todas las genealogías.",
        "",
        "Reanudación exacta local de P2, reutilizando fragmentos ya completos: `..\\e-jepa-ttc\\.venv\\Scripts\\python.exe -B operational/simplex_t_post_campaign/run.py profile`. P2 completo no requiere repetir medidas. P3 no tiene comando de ejecución real admitido todavía; usar sólo la interfaz verificada una vez concedido y validado el slot. Regeneración del bundle extraído: `python -B regenerate.py --root . --output REGENERATION.json --heads`.",
    ]
    episodes = read(OUT / "EPISODE_SCOPE.json")
    lines += [
        "",
        (
            f"Alcance temporal del diagnóstico: {episodes['episodes']} grupos de anotación, "
            f"{episodes['singleton_episodes']} con una sola query. Las anotaciones admitidas "
            "son mayoritariamente observaciones aisladas; este inventario no reconstruye "
            "trayectorias continuas ni tiempos de reacción. La regeneración verifica "
            "también estas particiones y todas las filas de diagnóstico exportadas."
        ),
        "",
        "![Precisión frente a coste de cabeza, con los tres bloques publicados](ACCURACY_HEAD_COST.png)",
    ]
    (SOURCE_DOCS / "INFORME_POST_CAMPANA_SIMPLEX_T.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        OUT / "NEXT_DECISION_POST_CAMPAIGN.json", SOURCE_DOCS / "NEXT_DECISION_POST_CAMPAIGN.json"
    )


def stage_bundle() -> None:
    """Deduplicate explicit sources in a new delivery; never rewrite the previous ZIP."""
    admitted()
    guard()
    bundle = OUT / "delivery/bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    copies: dict[str, Path] = {}
    index: dict[str, Any] = dict(predictions=[], profile=read(NIGHT / "PROFILE_INPUT_EXPORT.json"))
    for row in read(OUT / "PREDICTION_INVENTORY.json")["inputs"]:
        path = checked(Path(row["path"]), row["sha256"])
        name = "payloads/" + row["sha256"] + ".parquet"
        copies[name] = path
        index["predictions"].append(dict(**row, included_path=name))
    for row in index["profile"]["models"].values():
        for kind in ("weights", "inputs"):
            path = checked(NIGHT / row[kind + "_path"], row[kind + "_sha256"])
            name = "payloads/" + row[kind + "_sha256"] + ".npz"
            copies[name] = path
            row[kind + "_path"] = name
    for path in OUT.iterdir():
        if (
            path.is_file()
            and path.suffix in (".json", ".csv", ".xml", ".log")
            and path.name != "ACTIVE_OWNER.json"
        ):
            copies["results/" + path.name] = path
    for path in (OUT / "timings").glob("*.json"):
        copies["results/timings/" + path.name] = path
    for name in (
        "H16_RESULTS.json",
        "N2_RESULTS.json",
        "N3_RESULTS.json",
        "BASELINE_REGISTRY.csv",
        "COMPARABILITY.md",
        "HEAD_COST_REPLAY.csv",
        "COST_ACCURACY.csv",
    ):
        copies["historical_receipts/" + name] = DOCS / name
    for name in (
        "CHECKPOINT_JOURNAL_RECONCILIATION.json",
        "TECHNICAL_WORK.json",
        "FINAL_DELIVERY_COMPACT.json",
        "GITHUB_PUBLICATION.json",
    ):
        copies["historical_receipts/" + name] = NIGHT / name
    copies["historical_receipts/H16_ACCOUNTING.json"] = (
        ROOT / "artifacts/simplex_t/h16_replication_20261003/ACCOUNTING.json"
    )
    copies["historical_receipts/C0_ACCOUNTING.json"] = NIGHT / "execution/ACCOUNTING.json"
    copies["regenerate.py"] = ROOT / "operational/simplex_t_post_campaign/regenerate.py"
    copies["resource_guard.py"] = EXTRACT / "regenerate.py"
    for path in (ROOT / "src/e_jepa_ttc/simplex_t").glob("*.py"):
        copies["provenance/src/e_jepa_ttc/simplex_t/" + path.name] = path
    for name in ("__init__.py", "evaluation/__init__.py", "evaluation/stage63_65.py"):
        copies["provenance/src/e_jepa_ttc/" + name] = ROOT / "src/e_jepa_ttc" / name
    for path in (ROOT / "operational/simplex_t_cost_context").glob("*.py"):
        copies["provenance/operational/simplex_t_cost_context/" + path.name] = path
    for path in (ROOT / "operational/simplex_t_post_campaign").glob("*.py"):
        copies["provenance/operational/simplex_t_post_campaign/" + path.name] = path
    for path in SOURCE_DOCS.glob("*"):
        if path.is_file():
            copies[path.name] = path
    copies["PROPOSED_PLAN_SNAPSHOT.json"] = DOCS / "PLAN_CONTINUACION_POST_NOCTURNA.json"
    # Reserve incremental delivery bytes against the user's 10 GB artifact budget.
    occupied = sum(
        p.stat().st_size
        for parent in (NIGHT, OUT, ROOT / "artifacts/simplex_t/h16_replication_20261003")
        for p in parent.rglob("*")
        if p.is_file()
    )
    reservation = 3 * sum(p.stat().st_size for p in copies.values())
    if occupied + reservation > 10000000000:
        raise InterruptedError("artifact cap: delivery reservation exceeds 10 GB")
    save(
        OUT / "DELIVERY_RESERVATION.json",
        dict(existing_bytes=occupied, reserved_bytes=reservation, cap_bytes=10000000000),
    )
    for name, path in copies.items():
        dest = bundle / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(path):
            raise ValueError(f"staged byte conflict: {name}")
        if not dest.exists():
            shutil.copy2(path, dest)
    save(bundle / "INPUT_INDEX.json", index)
    save(
        OUT / "BUNDLE_STAGE.json",
        dict(
            status="STAGED_FOR_REGENERATION",
            root=str(bundle),
            members=len(copies) + 1,
            optimizer_updates=0,
        ),
    )


def pack() -> None:
    """Seal, verify CRC/SHA for every member and preserve an essential ZIP."""
    guard()
    bundle = OUT / "delivery/bundle"
    if read(OUT / "BUNDLE_REGENERATION.json")["status"] != "VERIFIED":
        raise ValueError("regeneration required before delivery")
    shutil.copy2(OUT / "BUNDLE_REGENERATION.json", bundle / "REGENERATION.json")
    manifest = dict(
        members={
            str(p.relative_to(bundle)).replace("\\", "/"): dict(
                bytes=p.stat().st_size, sha256=digest(p)
            )
            for p in sorted(bundle.rglob("*"))
            if p.is_file() and p.name != "CONTENT_MANIFEST.json"
        }
    )
    save(bundle / "CONTENT_MANIFEST.json", manifest)
    pin = digest(bundle / "CONTENT_MANIFEST.json")
    path = OUT / "delivery" / f"E_JEPA_TTC_POST_CAMPANA_{pin[:12]}.zip"
    if path.exists():
        raise ValueError("sealed delivery already exists; verify instead of overwriting")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                archive.write(p, str(p.relative_to(bundle)).replace("\\", "/"))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for name, row in manifest["members"].items():
            import hashlib

            data = archive.read(name)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("ZIP member hash failed")
    pin = digest(path)
    path.with_suffix(".zip.sha256").write_text(pin + "  " + path.name + "\n", encoding="ascii")
    save(
        OUT / "FINAL_DELIVERY.json",
        dict(
            status="DELIVERED_WITH_P3_EXTERNAL_BLOCK",
            zip=str(path),
            sha256=pin,
            bytes=path.stat().st_size,
            member_count=len(manifest["members"]) + 1,
            regeneration=read(OUT / "BUNDLE_REGENERATION.json"),
            optimizer_updates=0,
            observed_utc=now(),
        ),
    )
    shutil.copy2(OUT / "FINAL_DELIVERY.json", SOURCE_DOCS / "FINAL_DELIVERY.json")
    print(json.dumps(read(OUT / "FINAL_DELIVERY.json"), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("results", "stage", "pack"))
    args = parser.parse_args()
    with Owner():
        {"results": results, "stage": stage_bundle, "pack": pack}[args.task]()


if __name__ == "__main__":
    main()
