"""Regenerate the October 3–4 synthesis and verified release inventory; no training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/simplex_t_results_20261004"
TAG = "simplex-t-local-results-20261004"
URL = f"https://github.com/Kripta-Studios/e-jepa-ttc/releases/tag/{TAG}"


def sha(path: Path) -> str:
    """Hash without loading an archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Read frozen evidence and publish a reproducible descriptive synthesis."""
    OUT.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, str] = {}

    def read(name: str) -> Any:  # noqa: ANN401
        path = ROOT / name
        inputs[name] = sha(path)
        return json.loads(path.read_bytes())

    def rows(name: str) -> list[dict[str, str]]:
        path = ROOT / name
        inputs[name] = sha(path)
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def save(name: str, value: object) -> None:
        (OUT / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    night = read("docs/simplex_t_nocturnal_20261003/NEXT_DECISION_NOCTURNA.json")
    h16 = read("artifacts/simplex_t/h16_replication_20261003/analysis/RESULTS.json")
    families = [
        read(f"artifacts/simplex_t/nocturnal_20261003/execution/analysis/{n}/RESULTS.json")
        for n in ("N2", "N3")
    ]
    profile = rows("docs/simplex_t_post_campaign_20261003/ACCURACY_COST_FRONTIER.csv")
    registry = rows("docs/simplex_t_nocturnal_20261003/BASELINE_REGISTRY.csv")
    comparators = read("docs/simplex_t_post_campaign_20261003/COMPARATOR_REVIEW_RECONCILED.json")
    gpu_dir = "docs/simplex_t_shared_gpu_route_20261004/partial_1114"
    gpu = read(f"{gpu_dir}/DELIVERY_STATUS.json")
    costs = rows(f"{gpu_dir}/COST_ACCURACY_SHARED.csv")
    shares = rows(f"{gpu_dir}/STAGE_SHARES.csv")
    scores = {r["model"]: float(r["MiD"]) for r in profile}
    baseline = {r["id"]: r for r in registry}
    assert night["endpoints"] == 24 and night["scientific_saved_updates"] == 60000
    assert gpu["confirmed_fragments"] == 1114 and gpu["pending_fragments"] == 614
    contrasts = {
        name: {
            key: comparison[key]
            for key in (
                "candidate",
                "reference",
                "paired_loss_delta",
                "hierarchical_ci95",
                "sequence_only",
                "engineering_screen",
            )
        }
        for family in families
        for name, comparison in family["comparisons"].items()
    }
    for family in families:
        for model, value in family["scores"].items():
            label = "H8_SEED7" if model == "H8@7" else model
            assert abs(scores[label] - value) < 1e-10
    comparisons = h16["comparisons"]
    all_seeds, new_seeds = comparisons["all7_13_23"], comparisons["new13_23"]
    gain_pct = -100 * all_seeds["paired_loss_delta"] / all_seeds["h8_mid"]
    warm = {r["model"]: r for r in shares if r["mode"] == "warm_block2"}
    context_share = {
        model: float(warm[model]["raw_read_ms_fraction_of_measured_sum"])
        + float(warm[model]["roi_voxel_ms_fraction_of_measured_sum"])
        for model in ("H8_SEED7", "H16_SEED7")
    }
    summary = dict(
        dates=["2026-10-03", "2026-10-04"],
        release=URL,
        scientific_endpoints=24,
        scientific_saved_updates=60000,
        technical_updates=night["technical"]["confirmed_updates"],
        total_physical_updates_upper=night["total_physical_updates_upper"],
        h16=comparisons,
        contrasts=contrasts,
        prepared_head_cost=profile,
        shared_route_cost=costs,
        warm_block2_context_fraction=context_share,
        shared_route_confirmed=1114,
        shared_route_pending=614,
        matched_complete_queries=41,
        matched_producer_families=[0, 1],
        registered_candidate="TPR-D1-H8-C160",
        candidate_replaced=False,
        independent_sequences=9,
        confirmatory=False,
        ensemble=False,
        interpretation="head replication favorable; simplification screen negative; route partial",
        comparator_review=comparators,
        new_optimizer_updates_for_this_review=0,
    )
    save("RESULTS_SUMMARY.json", summary)

    assets = []
    receipts = [
        "artifacts/simplex_t/closure_20261002/essential_delivery/FINAL_DELIVERY.json",
        "docs/simplex_t_nocturnal_20261003/H16_FINAL_DELIVERY.json",
        "docs/simplex_t_nocturnal_20261003/FINAL_DELIVERY_NOCTURNA.json",
        "docs/simplex_t_post_campaign_20261003/FINAL_DELIVERY.json",
        "artifacts/simplex_t/shared_gpu_route_20261004/DELIVERY.json",
    ]
    for name in receipts:
        receipt = read(name)
        path = Path(receipt.get("archive", receipt.get("zip")))
        assert sha(path) == receipt["sha256"]
        assert path.stat().st_size == receipt.get("bytes", receipt.get("archive_bytes"))
        assets.append(
            dict(
                filename=path.name,
                path=path.relative_to(ROOT).as_posix(),
                bytes=path.stat().st_size,
                sha256=receipt["sha256"],
                receipt=name,
            )
        )
    readiness = ROOT / (
        "artifacts/simplex_t/route_readiness_20261004/E_JEPA_TTC_P3_READINESS_20261004.zip"
    )
    assert sha(readiness) == "c208373ed4fc24eb04c5342213ca4de12fdbacea8eaa6d08eca2d3b8c82c0127"
    assets.append(
        dict(
            filename=readiness.name,
            path=readiness.relative_to(ROOT).as_posix(),
            bytes=readiness.stat().st_size,
            sha256=sha(readiness),
        )
    )
    save("RELEASE_ASSETS.json", dict(tag=TAG, release=URL, archives=assets))
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{a['sha256']}  {a['filename']}\n" for a in assets),
        encoding="ascii",
        newline="\n",
    )

    lines = [
        "# SIMPLEX-T: resultados del 3 y 4 de octubre de 2026",
        "H8 conserva su identidad histórica. H16 mejora modestamente la precisión local; "
        "las simplificaciones no cumplen el cribado fijado. El trabajo de entrenamiento "
        "está completo; el perfilado de la ruta real está parcialmente bloqueado.",
        "Se completaron 24 cabezas hasta update 2.500 (60.000 updates científicos): "
        "seis réplicas H16, doce ajustes de dependencia de expertos y seis de agregación. "
        f"Se contabilizan además {night['technical']['confirmed_updates']} updates sintéticos "
        f"y una cota total física de {night['total_physical_updates_upper']} updates, incluidos "
        "los potencialmente repetidos o inciertos. T6 y sus 72 fits anteriores se reutilizan. "
        "Los análisis posteriores y esta síntesis no ejecutan optimizadores.",
        "MiD mide el error ponderado en PHASE17: **menor es mejor**. No son milisegundos "
        "de TTC ni de reacción. La evaluación usa 8.192 consultas de nueve secuencias OLD_DEV, "
        "con masas y folds registrados; es desarrollo reutilizado, no confirmación independiente.",
        "| Modelo | MiD OLD_DEV seed7 | p95 cabeza preparada, mediana de 3 bloques (ms) |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['model']} | {float(r['MiD']):.6f} | {float(r['p95_median_ms']):.3f} |"
        for r in profile
    ]
    lines[4:] = ["\n".join(lines[4:])]
    lines += [
        f"**H16 frente a H8.** Media de pérdidas de tres seeds: {all_seeds['h16_mid']:.6f} "
        f"frente a {all_seeds['h8_mid']:.6f}; mejora relativa {gain_pct:.2f} %. "
        f"Delta {all_seeds['paired_loss_delta']:.6f}; CI95 jerárquico "
        f"{all_seeds['hierarchical_ci95']}. Las tres seeds son favorables. "
        f"Las dos nuevas tienen delta {new_seeds['paired_loss_delta']:.6f}, CI95 "
        f"{new_seeds['hierarchical_ci95']}: aún incluye cero. Seed7 ya era exploratoria. "
        "Se replican cabezas, no expertos ni escenas; no se promedian TTC para crear un ensemble. "
        "La secuencia OBneIVg4Cw empeora en las tres seeds: la ganancia no es uniforme.",
        "**Quitar expertos.** FULL_C0 controla el cambio de objetivo auxiliar (lambda_cost=0). "
        "A5_ONLY, C2F_ONLY y A5_PAIR empeoran frente a ese control; ninguno satisface los "
        "límites prospectivos de precisión. Un intervalo que cruza cero no demuestra equivalencia. "
        "PAIR comparte el encoder A5; no cuenta como otro encoder independiente.",
        "| Contraste nuevo | Delta MiD | CI95 jerárquico | Cribado de precisión |",
        "|---|---:|---|---|",
    ]
    start = len(lines) - 2
    lines += [
        f"| {name} | {r['paired_loss_delta']:.6f} | {r['hierarchical_ci95']} | "
        f"{'Pasa' if r['engineering_screen']['precision_screen_pass'] else 'No pasa'} |"
        for name, r in contrasts.items()
    ]
    lines[start:] = ["\n".join(lines[start:])]
    c2f = baseline["SIMPLEX_CURRENT_C2F_NESTED"]
    ewma = baseline["EWMA_0P3S_H8"]
    lines += [
        "**Cambiar la GRU.** SET_AGE y SET_NOTIME cuestan aproximadamente la mitad como "
        "cabezas aisladas, pero sus intervalos no cumplen el margen registrado de +2 MiD. "
        "Son resultados exploratorios útiles, no sustitutos validados. Quitar tiempos explícitos "
        "no elimina necesariamente la información temporal que conservan los expertos.",
        f"**Por qué aporta algo frente a C2F.** El productor C2F compilado de esta campaña "
        f"obtiene {float(c2f['MiD']):.6f} MiD; la referencia EWMA registrada "
        f"{float(ewma['MiD']):.6f}. H8 seed7 obtiene {scores['H8_SEED7']:.6f}. "
        "C2F sigue siendo un componente útil: SIMPLEX-T estudia cómo combinar y refinar sus "
        "observaciones con contexto pasado. El C2F oficial V7 y Garl tienen consultas y folds "
        "coincidentes tras resolver aliases, pero falta ligar completamente sus productores, "
        "TRAIN y disponibilidad del ROI. No se proclama superioridad sobre todos los históricos. "
        "La revisión reconciliada sustituye los flags preliminares que atribuían diferencias "
        "de población a columnas distintas o redondeo del CSV.",
        "**Coste real y pendiente.** Se completaron 13.500 medidas emparejadas de nueve cabezas "
        "en CPU y se conservan 1.114 de 1.728 mediciones raw → contexto → expertos GPU → cabeza "
        "→ TTC. Hay 41 consultas con las 27 mediciones completas, de dos de las tres familias "
        "previstas; los 64 contextos pasaron previamente la admisión numérica. "
        f"En warm_block2, lectura y ROI/voxel suman {100 * context_share['H8_SEED7']:.1f} % "
        f"del tiempo H8 y {100 * context_share['H16_SEED7']:.1f} % del H16. "
        "La cabeza supone menos del 1 % en esos dos casos. Son fracciones de la suma de tiempos "
        "observados, con GPU compartida y ROI suministrado; no son p95 sumados ni latencia AEB. "
        "Faltan 614 mediciones porque dejó de estar accesible E:/eAP_dataset/data/train. "
        "La exclusividad GPU dejó de ser un requisito por autorización posterior; "
        "los documentos anteriores conservan su estado histórico.",
        "**Ingeniería y comprobaciones.** Se implementaron máscaras y anchors sin expertos "
        "excluidos, los dos agregadores, checkpoints completos, pruebas de reanudación, análisis "
        "y publicación por fragmentos, inferencia con productores congelados y tiempos por etapa. "
        "Se preservaron los fallos de recursos y el intento inicial con un runtime CUDA distinto; "
        "se restauró la configuración histórica antes de medir, sin ampliar tolerancias. "
        "El análisis posterior reconcilió 98.304 predicciones; el último bundle regeneró 398 "
        "entradas de cabeza y 27 filas de coste con diferencia cero. Los manifests verifican "
        "bytes y SHA-256; reproducir cachés no equivale a reconstruir datos crudos "
        "o entrenar expertos.",
        "**Qué significa.** H8 mejora sustancialmente OLD_DEV y el contexto pasado aporta "
        "información. H16 es un candidato posterior de precisión, con ganancia pequeña e "
        "incertidumbre entre escenas. Las reducciones probadas no justifican sustituir el sistema. "
        "El mecanismo cronológico sigue sin demostrarse; tampoco hay evidencia de AEB más rápida, "
        "tracking online persistente o incertidumbre calibrada. LATENT conserva su diagnóstico "
        "histórico negativo; no se corrigió ni reentrenó en esta campaña.",
        "**Continuación.** Primero recuperar E: y completar las 614 medidas ya autorizadas, "
        "conservando los fragmentos. La única propuesta posterior es optimizar la preparación "
        "retrospectiva raw/ROI con pesos congelados y paridad verificada, bajo un protocolo "
        "independiente. No se ejecuta esa propuesta ni se abren entrenamientos o holdouts.",
        f"[Descargar todos los bundles y sus hashes]({URL}). El bundle nocturno incluye "
        "también el ZIP H16, disponible por separado. T6 es la base histórica del 2 de octubre. "
        "Se publican los entregables finales y la última entrega parcial, no copias temporales "
        "redundantes. RELEASE_ASSETS.json inventaría el alcance; "
        "los raw externos no están incluidos.",
        "Fuentes: [campaña nocturna]"
        "(../simplex_t_nocturnal_20261003/INFORME_NOCTURNO_SIMPLEX_T.md), "
        "[diagnóstico y coste emparejado]"
        "(../simplex_t_post_campaign_20261003/INFORME_POST_CAMPANA_SIMPLEX_T.md), "
        "[ruta GPU parcial](../simplex_t_shared_gpu_route_20261004/partial_1114/"
        "INFORME_GPU_COMPARTIDA_SIMPLEX_T.md). "
        "RESULTS_SUMMARY.json conserva cifras completas e INPUT_HASHES.json sus fuentes. "
        "Regeneración local: `python -B operational/build_simplex_t_results_review.py`.",
    ]
    (OUT / "README.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    save("INPUT_HASHES.json", inputs)
    print(
        json.dumps(
            dict(
                archives=len(assets),
                bytes=sum(a["bytes"] for a in assets),
                output=str(OUT),
                optimizer_updates=0,
            )
        )
    )


if __name__ == "__main__":
    main()
