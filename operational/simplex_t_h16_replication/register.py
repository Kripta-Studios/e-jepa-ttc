"""Seal an independent six-fit protocol after successful zero-update preflight."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common import (
    ARM,
    CAMPAIGN,
    COMMIT_FLOOR,
    HISTORICAL_FREEZE,
    OUT,
    RESERVATION,
    ROOT,
    Resources,
    atomic_bytes,
    digest,
    ids,
    inventory,
    publish_json,
    record,
)

PROSE = """# Protocolo independiente H16 — 2026-10-03

Autorización: encargo explícito del usuario en esta sesión. El nombre solicitado
PROMPT_CODEX_REPLICAR_H16_20261003.md fue proporcionado inicialmente en el mensaje;
la ampliación verificada incorpora una copia en input_package/evidence/.
El mensaje explícito es la autoridad; el archivo no autoriza trabajo por sí solo.
T6 está cerrado y sus autorizaciones, freeze, 72 fits y H8 quedan intactos.

Sólo seis cabezas TPR-D1-H16-C160: seeds13/23 × folds0/1/2, CPU FP32,
batch128, 4 threads e interop2. Inicialización por seed; ningún warm start.
Se usa training.fit histórico byte a byte, AdamW wd1e-3/betas0.9,0.999/eps1e-8,
foreachFalse, warmup100/cosine2500, lr3e-4→3e-5, clip1, endpoint2500.
La autoridad completa son los hashes de código y las identidades seed7 verificadas.
El digest del nuevo PROTOCOL.json se usa en el campo freeze del nuevo checkpoint:
es un binding a este protocolo independiente, no una edición del freeze histórico.

Presupuesto: 15000 updates científicos guardados, cero probes con optimizer.
Un único escritor, ejecución secuencial, checkpoint completo cada100, RNG/sampler/
optimizador/scheduler determinista restaurados. Pausa real del primer fit tras100
para probar recuperación sin repetir trabajo; continúa desde101. Trabajo perdido
o incierto se contabiliza aparte y no autoriza fits nuevos. RAM disponible≥2GiB,
RSS árbol≤4GiB, disco libre-reserva1GiB≥10GB. Revisión operativa v2 previa a
cualquier update: reserva estimada de archivos1GiB dentro del techo propio2GiB,
Windows commit headroom≥1GiB. El preflight completo midió RSS<0.75GB; el
margen de compromiso se comprueba antes de cada update. Los tres límites del
usuario no se rebajan. Se preserva v1, que no produjo ningún checkpoint/update.
Sin modificación de precisión ni batch por recursos.

Antes de cualquier nueva inferencia OLD_DEV se sellan los seis checkpoints2500.
H8 seeds7/13/23 y H16 seed7 se reutilizan con hashes originales. Consultas, targets,
features, productores, PHASE17, masas y normalizadores idénticos; sólo H8/H16.
No nuevos expertos, datos, LATENT, sweeps, pretraining ni ensamble de TTC.

Análisis primario: H16-H8 de la misma seed, MiD macro por nueve secuencias,
bucket masses0.5/0.3/0.1/0.1. Seed7 exploratoria separada de13/23 nuevas.
Promediar pérdidas por query para el resumen de13/23 y de7/13/23 y aplicar masas.
Bootstrap jerárquico secuencia→track usando los draws sellados T2 (8192 aceptados),
con fragmentos256 y la rutina operativa ya validada. Contraste exacto por secuencias,
nueve unidades independientes, nunca27. Intervalos95 percentiles, sin reajuste.
Publicar cada seed, ambos resúmenes, diferencias por secuencia/fold y buckets,
signo, crucial0<TTC≤3, cobertura de cuantiles, cap, saturación y escapes.

Guardrails heredados aplicables: cobertura de queries completa8192, outputs finitos,
delta de error de signo ponderado≤0.005 y delta MiD crucial≤3 frente al H8 emparejado.
Son barreras descriptivas trasladadas a este contraste nuevo, no promoción histórica.
Umbrales de elegibilidad RISK17, H1 y soporte TRAIN H8 no se trasladan como prueba
de superioridad H16. Criterio nuevo de conclusión positiva local: delta medio<0,
las dos seeds nuevas con delta<0, ambos intervalos del resumen tres seeds con límite
superior<0 y guardrails en cada seed y ambos resúmenes. Si delta medio≥0: negativo
descriptivo; en los demás casos: incierto. No se optimiza usando estos resultados.
Nunca se sustituye automáticamente el candidato histórico H8 ni se abre Stage76.

Medir latencia sólo de cabeza con inputs cacheados, CPU FP32, batch1 y128 aparte,
warmup25 y500 mediciones si recursos, selección por hash de IDs TRAIN publicada.
Sin end-to-end: no se afirma latencia del sistema/ahorro de experto/tracking/AEB,
ni calibración de incertidumbre. Inputs exportados permiten inferencia de cabezas;
raw, TRAIN y pesos expertos no se prometen incluidos en el bundle esencial.
No push, holdouts, public/private/EvTTC/CodaBench ni actuaciones Stage70–76.
"""


def main() -> None:
    Resources().check()
    preflight = record(OUT / "PREFLIGHT.json")
    if preflight["status"] != "PREFLIGHT_VERIFIED_ZERO_UPDATES" or preflight["fits"] != ids():
        raise ValueError("six-fit verified preflight required")
    files = list(Path(__file__).parent.glob("*.py"))
    inputs = list(preflight["input_inventory"])
    inputs += inventory(files)
    draws = (
        ROOT
        / "artifacts/simplex_t/scientific_campaign/T6/CHECKPOINTED_WORK/analysis"
        / "analyses/T2/paired_uncertainty/HIERARCHICAL_DRAWS.jsonl"
    )
    inputs += inventory([draws, ROOT / "operational/simplex_t_closure/runtime.py"])
    window = ROOT / "artifacts/simplex_t/nocturnal_20261003/WINDOW_AUTHORIZATION.json"
    inputs += inventory([window])
    publish_json(
        OUT / "PROTOCOL.json",
        dict(
            schema="simplex_t_independent_h16_replication_v2_pre_update",
            supersedes_zero_update_protocol_sha256="60b6a45ec3748cd9fd098e17e933e5d85527f280c5d81a9b0a27cf4f71430a92",
            campaign=CAMPAIGN,
            arm=ARM,
            historical_candidate="TPR-D1-H8-C160",
            historical_freeze_sha256=HISTORICAL_FREEZE,
            fits=ids(),
            scientific_updates=15000,
            technical_optimizer_updates_authorized=0,
            endpoint2500=True,
            policy=dict(
                available_ram_floor=2 * 1024**3,
                tree_rss_ceiling=4 * 1024**3,
                free_disk_after_reservation_floor=10_000_000_000,
                own_reserved_bytes=RESERVATION,
                windows_commit_floor=COMMIT_FLOOR,
            ),
            launch=preflight["launch"],
            sources=preflight["sources"],
            historical_controls=preflight["historical_controls"],
            input_inventory=inputs,
            bootstrap_draws=str(draws),
            preflight_sha256=digest(OUT / "PREFLIGHT.json"),
            pre_execution_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            protocol_text=PROSE,
            no_push=True,
            holdouts_opened=False,
            scores_of_new_fits_read=False,
        ),
    )
    atomic_bytes(OUT / "PROTOCOL.md", PROSE.encode("utf-8"))
    pin = digest(OUT / "PROTOCOL.json")
    atomic_bytes(OUT / "PROTOCOL.sha256", (pin + "  PROTOCOL.json\n").encode())
    print("INDEPENDENT_PROTOCOL_SEALED " + pin, flush=True)


if __name__ == "__main__":
    main()
