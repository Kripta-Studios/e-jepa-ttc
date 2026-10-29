# Resultados generados de la continuación de detección

> Archivo generado por `scripts/build_detection_sota_evidence.py`. No editar a mano.

Estas cifras se extraen de artefactos canónicos con hashes conservados en `evidence/current_v1/manifest.json`. Son evidencia local de desarrollo; no son una puntuación eAP oficial ni una afirmación SOTA.

## S0: evaluación v2 y controles

| Magnitud | Valor |
|---|---:|
| Frames | 256 |
| GT raw / elegible / ignorado | 1936 / 1035 / 901 |
| Predicciones input / evaluadas | 12800 / 11680 |
| Clases con soporte | 5 / 8 |
| AP101 desarrollo, clases con soporte | 0.019821723 |
| Control constante train-only | 0.003839391 |
| Control por permutación sin puntos fijos | 0.003654754 |
| Delta frente a constante | 0.015982332 |
| Delta frente a permutación | 0.016166969 |

La paridad del observador fue `PASS` con error máximo 0.0. El bootstrap usa 100 réplicas pareadas por secuencia, pero solo 2 clusters; sus intervalos son por ello débiles.

## S1: datos, admisión y ejecución

| Evidencia | Valor |
|---|---:|
| D4096 train frames | 4096 |
| D4096 selección frames | 1024 |
| GT evaluable v2 en D4096 train / selección | 14827 / 4639 |
| Targets entrenables legacy en D4096 train / selección | 13760 / 4275 |
| Smoke updates / exposiciones | 8 / 32 |
| Resume exacto en update | 4 |
| Inferencia independiente | PASS |
| R0 v2 | RUNNING_NO_SCIENTIFIC_RESULT_INCLUDED |

| Clase | GT evaluable v2 en train | Target entrenable legacy en train | GT evaluable v2 en selección |
|---|---:|---:|---:|
| bicycle | 173 | 162 | 26 |
| bus | 130 | 110 | 85 |
| car | 11191 | 10452 | 2524 |
| motorcycle | 749 | 693 | 963 |
| other_vehicle | 8 | 7 | 4 |
| pedestrian | 252 | 245 | 338 |
| tricycle | 105 | 97 | 80 |
| truck | 2219 | 1994 | 619 |

El R0 congelado usa el productor `fcad2299`; no se publica una métrica científica mientras sigue incompleto. El fallo v1 ocurrió antes de cualquier update y se conserva como recibo de fail-closed.

## S5: replay real hasta PHASE

Se ejecutaron 8 frames, con 17 detecciones finales, 7 tracks maduros no ambiguos y 119005 eventos ingeridos. `targets_read=false` y `optimizer_updates=0`. Los límites conservados son cuantiles nativos de fase, no un intervalo TTC continuo. Esta es evidencia de integración, no de precisión ni generalización TTC.

## Estado de gates

| Gate | Estado exacto en current_v1 |
|---|---|
| S0 | PARTIAL: local v2 diagnostics and controls closed; official scorer, broader independent population, and rare-class support remain open |
| S1 | RUNNING: D4096/data/smoke gates passed; R0 seed 7 v2 has no scientific result in this bundle |
| S2 | NOT_EVALUATED: code revision recorded; no frozen comparable result in current_v1 |
| S3 | NOT_STARTED: causal RGB history experiment absent |
| S4 | NOT_STARTED: event contribution matrix absent |
| S5 | PARTIAL_ENGINEERING: real label-free detection-to-PHASE replay executed; no TTC accuracy/generalization claim |
| S6 | NOT_STARTED: multi-seed confirmation, calibration, and final independent comparison absent |

## Regeneración

```powershell
python scripts/build_detection_sota_evidence.py
python scripts/build_detection_sota_evidence.py --check
python scripts/build_detection_sota_evidence.py --from-bundle --check
```
