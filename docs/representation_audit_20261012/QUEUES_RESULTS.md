# V17: resultados de las colas adicionales

Tablas regeneradas desde predicciones, con replay exacto de métricas. P es ajuste; H es cribado de productores; V es desarrollo interno expuesto. No son resultados oficiales ni SOTA.

| Cola | Rol | Fit | MiDc | MiDs | MiDl | MiDn | Overall | MAE s | Inválidas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B | H | U0-S13 | 157.154365 | 61.925801 | 57.819592 | 59.116506 | 108.848532 | 4.783695 | 0.000000 |
| B | H | U0-S23 | 154.404256 | 62.294991 | 58.485455 | 59.286578 | 107.667829 | 4.719205 | 0.000000 |
| B | H | U0-S7 | 156.585284 | 61.956289 | 57.782299 | 59.109314 | 108.568690 | 4.791682 | 0.000000 |
| B | H | U1-S13 | 154.181859 | 62.603356 | 57.698573 | 59.417408 | 107.583535 | 4.839212 | 0.000000 |
| B | H | U1-S23 | 152.011237 | 63.125056 | 58.306328 | 59.811143 | 106.754882 | 4.781916 | 0.000000 |
| B | H | U1-S7 | 153.600985 | 63.039157 | 57.703454 | 59.636219 | 107.446207 | 4.866122 | 0.000000 |
| B | P | U0-S13 | 138.341342 | 56.571154 | 48.833850 | 50.123093 | 96.037711 | 4.075520 | 0.000000 |
| B | P | U0-S23 | 135.907174 | 56.511427 | 49.299321 | 50.032019 | 94.840149 | 4.014424 | 0.000000 |
| B | P | U0-S7 | 138.257003 | 56.656686 | 48.721272 | 50.184731 | 96.016108 | 4.085028 | 0.000000 |
| B | P | U1-S13 | 136.545564 | 57.300183 | 48.736926 | 50.804867 | 95.417016 | 4.137537 | 0.000000 |
| B | P | U1-S23 | 134.174453 | 57.313144 | 49.377073 | 50.781562 | 94.297033 | 4.077940 | 0.000000 |
| B | P | U1-S7 | 135.978965 | 57.616030 | 48.823075 | 51.032403 | 95.259839 | 4.161638 | 0.000000 |
| D | V | D-S13 | 148.712487 | 58.781343 | 52.749089 | 61.109618 | 103.376517 | 2.639853 | 0.000000 |
| D | V | D-S23 | 149.672984 | 58.503604 | 52.146352 | 60.151553 | 103.617363 | 2.396993 | 0.000000 |
| D | V | D-S7 | 148.094343 | 56.165763 | 54.354440 | 62.788591 | 102.611203 | 2.605942 | 0.000000 |
| E | H | Q0-S7 | 149.649240 | 60.581932 | 59.357641 | 57.440928 | 104.679056 | 4.495739 | 0.000000 |
| E | H | Q1-S7 | NA | 277.121661 | 278.327699 | 240.952053 | NA | 9.164699 | 1.000000 |
| E | P | Q0-S7 | 134.516902 | 55.382971 | 50.133988 | 48.847362 | 93.771477 | 3.779105 | 0.000000 |
| E | P | Q1-S7 | NA | NA | NA | 250.812782 | NA | 9.056576 | 5.000000 |

NA mantiene ausente el overall cuando faltan predicciones MiD válidas. Las métricas parciales no sustituyen esa cobertura ni habilitan gates o bootstrap. Las filas inválidas permanecen en las tablas originales.

Las curvas comparan trayectorias dentro de cada objetivo; sus pérdidas absolutas no son una métrica común entre objetivos. El clipping cuenta normas previas al recorte >1. Los intervalos se remuestrean sobre grupos completos (2000 draws, seed12345), no ventanas independientes.

Los costes GPU incluyen entrenamiento, QA y evaluación registrados en GPU_BUDGET.json. No equivalen a latencia de inferencia desde eventos crudos. No se ejecuta evaluación externa para rescatar candidatos rechazados.
