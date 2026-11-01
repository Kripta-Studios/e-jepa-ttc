# Baselines: diagnóstico local expuesto

Targets CSV locales con geometría no verificada equivalente al scorer oficial. Los subconjuntos compartidos pueden ser más fáciles. Ninguna fila sustituye el score oficial de 434 consultas.

| Método | Predicción finita / solicitadas | Puntuadas | MAE micro (s) | Error relativo micro (%) | Error relativo macro (%) |
|---|---:|---:|---:|---:|---:|
| PHASE | 434/434 | 354 | 0.567491 | 21.591465 | 23.592364 |
| PHASE_event | 434/434 | 354 | 0.645562 | 28.056852 | 32.544051 |
| PHASE_rgb | 426/434 | 346 | 0.476165 | 19.174268 | 21.876781 |
| Garl_full | 424/434 | 346 | 0.684621 | 23.906856 | 24.332952 |
| Garl_visual | 426/434 | 346 | 0.585973 | 24.617587 | 27.099048 |
| Garl_event | 434/434 | 354 | 1.541326 | 64.916695 | 69.587562 |

## Comparación en consultas compartidas

| Comparador | Consultas | PHASE error relativo micro (%) | Comparador (%) |
|---|---:|---:|---:|
| PHASE_event | 354 | 21.591465 | 28.056852 |
| PHASE_rgb | 346 | 21.211483 | 19.174268 |
| Garl_full | 346 | 21.211483 | 23.906856 |
| Garl_visual | 346 | 21.211483 | 24.617587 |
| Garl_event | 354 | 21.591465 | 64.916695 |

PHASE: RGB+eventos, rama RGB o rama eventos del mismo ledger sellado. Garl: variantes full/visual/event con su preparación nativa declarada. Las diferencias de cámara, ROI, historia, endpoint y preparación siguen las guías `docs/evttc_competition.md` y `docs/garl_evttc_competition.md`.

No se ha ejecutado entrenamiento, interpolación de predicciones ni corrección del signo.
