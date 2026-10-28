# TRAIN40 — handoff del cierre solicitado, 28/10/2026

Se cierra esta ronda por petición del usuario. La publicación y la pausa del
goal quedan a cargo de root al terminar el cierre; este documento no afirma
que ya se haya publicado. El objetivo **50 Hz sostenidos y menos de 20 ms por
actualización completa** sigue sin lograrse. Primero corresponde un objeto,
después dos y sólo más adelante veinte. No se han reentrenado modelos ni
cambiado pesos, normalizadores o tolerancias.

Código final de perfil: **`f046b791`**, sobre la rama
`perf/train40-persistent-50hz`. La base recibida fue `2521de1a` y el handoff
anterior es [el de 27/10](../train40_runtime_resume_20261027/HANDOFF.md).
Los commits posteriores de documentación/evidencia no son resultados nuevos.
Los bindings conservan los hashes exactos de cada versión que sí se ejecutó.

## Resultado que debe conservar el siguiente swarm

El mejor replay con cobertura íntegra es
[`memory_cache_overlap_reverse1_250_v1`](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1/summary.json):
250/250 actualizaciones válidas con paridad, cero drops, errores de fuente y
acknowledgements fallidos; 722 archivos preservados. Ofertó cinco segundos.
No pasó el gate: **200 de 250 deadlines encontraron trabajo previo**, aunque
el máximo observado fuera uno y no hubiera drops. Ausencia de drops no
equivale a ausencia de acumulación.

Sus percentiles y todos los resultados negativos se regeneran en
[RESULTS_GENERATED.md](RESULTS_GENERATED.md); los hashes de las fuentes están
en [RESULTS_SOURCES.json](RESULTS_SOURCES.json). Para el replay anterior:

- Deadline→todos los TTC: p95 29,143155 ms; p99 32,414209 ms; máximo
  35,3326 ms; 201/250 respuestas alcanzaron o superaron 20 ms.
- RAW ya disponible→todos los TTC: p95 25,25299 ms; máximo 29,593 ms.
- Tramo de servicio: media 17,610386 ms; p95 21,426025 ms; máximo 25,9175 ms.
- Último TTC a 5,0036785 s; wall con cierre/join a 5,0493202 s, ambos desde el
  primer deadline. La fuente necesitó una cota declarada de 2.611.872.720 bytes
  bajo un presupuesto explícito de 3 GiB, no el default de 2 GiB.

Estos son relojes distintos: no se suman ni se convierten sus medias en
throughput sostenido. La fuente precargada simula llegadas RAW en CPU; no
acredita adquisición de cámara física ni disponibilidad real de anotaciones.
No hay ensayo sostenido de 60 s ni ensayo paced admitido con dos objetos.

## Perfil final y entorno

[DEFAULT_PROFILE.md](DEFAULT_PROFILE.md) documenta flags y controles. El servicio
usa PIL ROI-first, prefix/topk, query graphs, batches nativos, chunk RGB de un
objeto, ramas paralelas, solapamiento E/preparación RGB y transporte exacto
`reverse-summaries`. Reutilización de encodings E desactivada. Los productores
primitivos y oráculos conservan `original` como control.

El runner paced usa por defecto `memory`, primación sintética de todas las
historias nativas y caché de features RGB. Conserva tokens SHA-256 V1, sellados
causalmente dentro de la adquisición. **Crops OFF; uint8 OFF; V2/BLAKE3 sin
wiring; backend 5 ms opcional y fuera del runner paced.** La verificación
posterior sigue requiriendo `--verify-parity`. El perfil seleccionado tiene QA
CPU; no se ejecutó otro experimento GPU tras cambiar esos defaults.

Los bindings del mejor replay registran Windows 10 build 26200, CPython
3.11.15, Torch 2.11.0+cu128, CUDA 12.8 y NVIDIA GeForce RTX 5070 Ti Laptop GPU.
Precisión FP32 con TF32 desactivado, seed 7, threads Torch 2 y scheduler `sleep`
con deadlines absolutos basados en `perf_counter_ns`/QueryPerformanceCounter.
OMP/MKL se limitaron a 2 en QA. No introducir tests CPU concurrentes, hashing
de archivos ni compilaciones ajenas durante futuros ensayos de latencia GPU.

## Qué cambió y qué evidencia lo admite

1. Fuente RAW CPU precargada y tokens inmutables V1. La precarga sólo guarda RAW
   privado, sin tokens, hashes, crops ni features futuros. La llegada realiza
   copia, conversión y sellado causal dentro del reloj. El
   [companion RAW real](MEMORY_RAW_PARITY.md) comparó historia inicial de 600 ms
   y 50 updates contra el lector independiente de disco, incluyendo cada campo
   de eventos, pixel, tripleta, timestamp, caja y calibración.
2. Solapamiento E/preparación RGB con ownership y cleanup explícitos, y
   primación sintética de siete layouts. No usa observaciones futuras para
   preparar las formas. La primación evita capturas frías conocidas, sin
   sustituir la paridad real.
3. Transporte `reverse-summaries`: el diagnóstico preparado validó 462/462
   llamadas y features exactas, con 599 archivos preservados. El ahorro de ese
   tramo preparado no se atribuye al update completo. Los paced posteriores
   conservan su propia paridad y relojes.
4. Poblaciones de latencia separadas. `successful_all_object_ttc` requiere
   éxito y cobertura de todos los objetos; los acknowledgements fallidos se
   informan aparte. Los legacy mezclados quedan explícitamente etiquetados
   `ALL_RESPONSES_INCLUDING_FAILURES`. Drops no reciben latencia imputada.
5. Lifecycle v2: el feeder termina ofertas, espera drenaje o aborto y finaliza
   en su hilo. El cierre queda después de los últimos TTC, pero permanece en
   wall total. Un fallo de warmup antes de ofrecer datos ahora produce intervalo
   paced `null`, junto a wall real no negativo; no se reescribió el histórico.

## Resultados negativos y opciones no promovidas

- `native_log_sensitive_v1` y `native_match_sensitive_v1` fallan TTC en casos
  sensibles. No promover compilación/arithmetic distinta por latencia aislada.
  El reporter agrega todas las repeticiones; recibos antiguos `last-only` se
  reconocen como evidencia incompleta.
- `baseline1_50_v1`: sólo 13 TTC y 37 drops, con lectura/copia RAW costosa y una
  captura fría de aproximadamente 3,6 s. `memory1_50_v1`: 9 TTC, 40 acks y un
  drop. `memory_cache_overlap1_50_v1`: 27 TTC, 22 acks y un drop. No ocultar
  esos denominadores ni interpretar sus acks rápidos como TTC rápidos.
- [Crops companion](../../artifacts/train40_runtime_council_20261028/multiobject/crop_rgb_companion_v1/summary.json):
  18/18 repeticiones, 27 consultas por objeto, seis casos con una/dos ROI,
  MISS→feature HIT→clear sólo features+crop HIT; 623 archivos intactos. Verifica
  argumentos reales, deltas, features y TTC a anchor fijo. El sellado y clear
  forzado están fuera del tiempo interno, que no se presenta como latencia paced.
- [Crops paced250](../../artifacts/train40_runtime_council_20261028/paced/memory_crops_reverse1_250_v1/summary.json):
  FAILED, 115 TTC, 133 acks y dos drops. Los 115 TTC producidos pasan paridad,
  pero la cobertura global falla; 724 archivos intactos. Hubo 66 crop HIT,
  298 MISS/stores y cero bypass. No está demostrada causalidad de caché o GC;
  hace falta control con la misma fuente y versión tras importar core V2.
- BLAKE3 reduce el tiempo de copia+hash en su micro CPU; core V2 existe en
  `9e752ae4`, pero no está conectado a memory/immutable/paced y no tiene
  admisión TTC o de latencia completa. Ver [diseño](FROZEN_RGB_V2_DESIGN.md),
  [wiring propuesto](FROZEN_RGB_V2_WIRING_DESIGN.md) y
  [tabla CPU regenerable](FRAME_HASH_CPU_GENERATED.md).
- [GC trace v1](GC_TRACE_V1_PROPOSAL.md) queda congelado como propuesta con
  tests de callbacks simulados, sin importarlo en runtime ni runner. No fuerza
  colecciones, cambia umbrales o descuenta pausas; sólo correlacionaría intervalos.

## Diagnóstico final del backend 5 ms

[`subpacket_backend_sensitive_v1`](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v1/summary.json)
falló antes de comparar por presupuesto RAW de 256 MiB; el caso 0 requería
324.429.972 bytes. Se conserva el fallo y sus 626 archivos intactos.

[`subpacket_backend_sensitive_v2`](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/summary.json)
terminó COMPLETE con cap RAW de 512 MiB, mismas 49 fuentes de código y casos
0/137/553 de un objeto. Valida 93/93 fronteras (120,124,…,240), 558 buffers de
estado bitexact, 93 representaciones bitexact, 18 guardas parciales que rechazan
36 consultas, seis inputs canónicos y seis bundles TTC finales exactos
(18 componentes, max abs 0); 626 archivos preservados y RAW stats intactos.

Es una admisión numérica del backend y TTC en el anchor final, **sin medición
de latencia, fuente 200 Hz ni integración paced**. No demuestra ahorro de 3/4,
dos objetos o 50 Hz. Ver [backend](SUBPACKET_BACKEND.md) y
[revisión numérica](SUBPACKET_NUMERICS.md). No empezar por activar este backend
en producción: faltan diseño de releases/subbucket debt y medición completa.

## Reanudar de forma reproducible

Las rutas de pesos/datos son recursos externos; confirmar primero que existen
y verificar los seals. No subir RAW, checkpoints o .pt al repositorio. Ejemplo
para un **nuevo** run de control comparable, después de reanudar el objetivo:

```powershell
$py = '../e-jepa-ttc/.venv/Scripts/python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
& $py scripts/benchmark_train40_paced.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --reference artifacts/final_train40_transfer_20261027/final_train40/own `
  --base-manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --manifest artifacts/train40_runtime_council_20261028/measurement_audit_v1/causal_pilot_manifest.json `
  --output artifacts/train40_runtime_next/control1_250_v1 `
  --sequence CPLA-low --objects 1 --updates 250 --queue-capacity 2 `
  --raw-source memory --max-preloaded-source-bytes 3221225472 `
  --prime-native-histories --rgb-feature-cache --wait-strategy sleep `
  --transport-reuse reverse-summaries --overlap-event-with-rgb-preparation `
  --verify-parity
```

Para piloto de 50 slots, usar `--updates 50`, salida nueva y presupuesto
2 GiB si el preflight lo permite. Nunca reutilizar una carpeta de resultados.
Los argumentos del diagnóstico 5 ms y sus preflights quedan en binding y
[SUBPACKET_BACKEND.md](SUBPACKET_BACKEND.md). No ejecutar más GPU durante este
cierre. Consultar `--help` del checkout antes de añadir nuevos flags.

QA histórica: [recibo transport](../../artifacts/train40_runtime_council_20261028/qa/transport_reuse_integration_v1/QA_RECEIPT.json)
transcribe explícitamente la salida observada de **217 pass, 1 skip, 2 XPASS**;
los XPASS no se transforman en xfail ni se omiten. QA final de defaults:
[receipt](../../artifacts/train40_runtime_council_20261028/closure_profile_qa_v1/receipt.json),
202 pass, Ruff/format/Pyright y diff-check, con logs y SHA antes/después.
Root repitió 88 pruebas de defaults; la comprobación no constituye otro replay
GPU. El reporter tiene fixtures que conservan fallos tempranos, poblaciones
vacías, scopes preparados, crop forced clear y frontera intermedia fallida.

Comando stdlib de regeneración de tablas:

```powershell
& $py -I -B scripts/report_train40_resume.py `
  --source artifacts/train40_runtime_council_20261028 `
  --output docs/train40_runtime_council_20261028/RESULTS_GENERATED.md
```

## Evidencia portátil y siguientes pasos

`evidence/council_v1` permanece inmutable: 1218 archivos, 58.123.067 bytes,
manifest SHA `f41aae910797c43fa224786a4a615ae637d99bb582b6b4aeac217fc2f9818d6c`.
Es un corte histórico que decía objetivo activo; no modificarlo para reflejar
el cierre posterior. `evidence/council_final_delta_v1` añade recibos, código,
QA y docs actuales por SHA, referenciando bytes de esa base sin duplicarlos.
Su README/materializador verifica la base y reconstruye un árbol completo
en un destino nuevo. No contiene RAW, pesos, dependencias ni tensores .pt.

El siguiente swarm debe leer este documento, el perfil, tablas y protocolos,
comprobar estado de rama/seals y reservar GPU en exclusiva. Prioridad:

1. Repetir control de un objeto con fuente/código actuales y misma configuración,
   conservar cada slot, tres salidas TTC por objeto y todos los percentiles.
2. Si investiga picos, integrar GC trace opt-in con detach en finally y comparar
   brazos instrumentados equivalentes. No atribuir correlación a causalidad ni
   restar pausas; guardar overhead, drops del observador y eventos incompletos.
3. Evaluar wiring V2 sólo como opt-in con source bytes/paridad y presupuestos
   completos; no añadir SHA oculto a la ruta BLAKE3. Repetir companion real y TTC.
4. Si se integra 5 ms, mantener cap agregado de 20 ms, guardas de fase, último
   release RAW antes de copy/seal, deadline nominal y deuda por subbucket. No
   excluir tres ingests de cuatro para aparentar mejora.
5. Cuando un objeto supere todos los gates, ejecutar duración sostenida explícita.
   Ninguna de las 32 secuencias locales alcanza 60 s naturales; el
   [stress replay v1](STRESS_REPLAY_V1_PROPOSAL.md) es una propuesta sin runner,
   con transiciones/reset/prefill dentro de su frontera y timestamps originales.
6. Sólo después, repetir con dos objetos. No extrapolar desde dos ROI artificiales
   ni de throughput por objeto a veinte tracks reales simultáneos.

La pausa administrativa no significa que el objetivo se haya completado.
