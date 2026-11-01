# Dfull RGB con ranking: entrega local del 1 de noviembre de 2026

Se cierra la campaña ejecutada con el candidato `dfull-rgb-ranked-seed7-20261101`:
detector Dfull congelado y ranker ajustado exclusivamente en TRAIN. No requiere
volver a entrenar ni recuperar el volumen del pod para ejecutar inferencia.
Este cierre entrega un candidato de investigación; no cierra todos los gates
S0–S6 ni acredita producción, SOTA oficial o nueva precisión TTC.

## Decisión y resultados

La confirmación abarca los 23.958 frames de selección. La AP local pasa de
0,16474566 a 0,18502078 con el mismo conjunto de propuestas: +0,02027512 absoluto
y +12,31 % relativo. El ranker no modifica clase, geometría 3D, yaw ni velocidad.
La pequeña regresión en `other_vehicle` se conserva en el informe por clase.
La selección ya se utilizó para desarrollo y solo hay una semilla del detector.

Temporalidad y eventos no pasan el margen predefinido de +0,005 AP frente a los
controles. Sus ensayos cortos y resultados negativos quedan conservados; no se
incorporan al candidato. Esto no descarta otras recetas o entrenamientos mayores.
Las tablas completas y sus denominadores se regeneran en
[POST_DFULL_RESULTS.md](POST_DFULL_RESULTS.md).

## Artefactos y recuperación

El paquete local está en `D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7`.
Contiene el checkpoint seleccionado, ranker, productor histórico congelado,
runtime del ranker, CLI, versiones probadas, licencias y evidencias. Los datasets
y los pesos PHASE se mantienen fuera del paquete.

- Checkpoint SHA-256: `c58c13297fd15638d89364a25b88245cd8be669ab8bfdbcb741f03df0e9215b3`.
- Manifiesto del paquete SHA-256: `61fc1023d67cf63aec8409023ddbdc44d02041c7413c3c2f83361e69a9ab2fbc`.
- [Manifiesto publicado](evidence/release_candidate_20261101/release_manifest.json): 112 archivos.
- Dfull y estado de entrenamiento: `D:/PHASE_TTC_DFULL_20261031/campaign`.
- Ranking completo y eventos locales: `D:/PHASE_TTC_LOCAL_CONTINUATION_20261101`.
- Comprobaciones finales y replay: `D:/PHASE_TTC_RELEASE_20261101`.

La copia del candidato es suficiente para su inferencia. Para reanudar un
entrenamiento, conservar también los directorios de campaña originales: el
paquete de inferencia no sustituye los estados de optimizer, scheduler y RNG.
Estos artefactos están en el PC; no se afirma que exista una segunda copia
externa verificada durante la caída del centro de datos.

Desde la carpeta del paquete, usando Python 3.11 y PyTorch con CUDA compatible:

```text
python infer.py verify
python infer.py infer --data-root DATA_ROOT --tokens tokens.json --output ../replay
```

`tokens.json` admite una lista de tokens o `{"sample_tokens": [...]}` sin etiquetas.
Cada llamada acepta entre 1 y 128 frames. La salida debe ser una carpeta nueva
fuera del paquete. El lector requiere el manifiesto de medios/calibración eAP
y sus RGB TAR; no es todavía una API genérica para imágenes arbitrarias.
`predictions.json` conserva las detecciones con el nuevo ranking. Los scores
no son probabilidades calibradas ni estimaciones de riesgo TTC.

La reconstrucción desde los artefactos originales está implementada en
`scripts/package_detection_candidate.py`, con `--checkpoint`, `--ranker`,
`--evidence` y `--output`. El empaquetador comprueba los hashes históricos antes
de copiar. La CLI separa el detector y el ranker en procesos nuevos para respetar
la carga estricta del productor congelado.

## Verificación realizada

- Los 112 archivos del paquete pasan su verificación SHA-256.
- Inferencia CUDA desde fuera del repositorio: ocho frames y paridad exacta
  tanto del baseline como del ranking frente a la evaluación terminada.
- Revisión de 47 recibos de shards y sus 423 archivos: 23.958 frames completos.
- 99 pruebas aprobadas, sin fallos ni omisiones, sobre ranking, integridad del
  paquete, asociación y servicio multiobjeto.
- Las tablas se regeneran con igualdad de bytes respecto al informe publicado.

Los [recibos de verificación](evidence/release_candidate_20261101/verification.json),
la [paridad](evidence/release_candidate_20261101/replay_parity.json) y el
[resultado de pruebas](evidence/release_candidate_20261101/tests.xml) conservan
el alcance de cada comprobación. No se presenta esta revisión como una nueva
auditoría independiente.

## Integración real con PHASE

El replay de ocho frames alimenta el servicio multiobjeto con las detecciones
del paquete. En el último frame conserva seis detecciones, produce TTC firmado
y límites de fase nativos para cinco y registra `track_not_confirmed` para la
restante. El paquete de eventos se ingiere una vez; no se leen targets.

La ejecución GPU del tramo PHASE, incluida carga de modelos y lectura de medios,
tarda 2,488 s y alcanza 4,04 GiB asignados por Torch en este replay. Es una sola
comprobación funcional con arranque; no es un benchmark sostenido ni la latencia
total del detector más PHASE. Los q10/q90 permanecen en espacio de fase.
El runtime TRAIN40 ya estuvo expuesto a estas secuencias: no se mide ni se
atribuye nueva generalización TTC.

Se conserva el [replay completo](evidence/release_candidate_20261101/phase_gpu_replay.json)
y su [resumen](evidence/release_candidate_20261101/phase_summary.json).
Para reproducir desde el repositorio con los pesos PHASE originales:

```text
python scripts/replay_detection_phase_service.py --eap-root DATA_ROOT --predictions REPLAY/predictions.json --candidate-root PHASE_CANDIDATE --output phase_replay.json --device cuda --phase-bounds
```

## Gates que esta entrega no resuelve

| Área | Estado y evidencia necesaria |
|---|---|
| Detección RGB y ranking | Campaña y confirmación completas; candidato guardado. |
| Temporalidad y eventos | Screening terminado, promoción rechazada; no se declara convergencia. |
| Integración multiobjeto | Replay real con cinco TTC; sigue pendiente población real con más de veinte tracks maduros y latencia por carga. |
| Ocho clases | Persisten soporte raro limitado y AP casi nula de `other_vehicle`. |
| Confirmación estadística | Faltan varias semillas y auditoría final independiente de la selección expuesta. |
| Calibración operacional | Pendiente; el ranking no proporciona probabilidades calibradas. |
| eAP oficial | Acceso/verificación del scorer y reconciliación del release pendientes. |
| Calidad TTC conjunta independiente | Pendiente de productores, teachers, normalizadores y calibradores sin exposición a los grupos evaluados. |

La base para el último punto está en
[INDEPENDENT_PRODUCER_FOUNDATION.md](INDEPENDENT_PRODUCER_FOUNDATION.md) y
[INDEPENDENT_RAW_HISTORY.md](INDEPENDENT_RAW_HISTORY.md). Reentrenar solo un head
o reservar secuencias únicamente para el detector no resuelve la exposición de
TRAIN40. Esos entrenamientos no se declaran ejecutados con esta entrega.

No hace falta mantener el pod encendido para utilizar o conservar este candidato.
El cierre no incluye detener el pod ni eliminar su volumen.
