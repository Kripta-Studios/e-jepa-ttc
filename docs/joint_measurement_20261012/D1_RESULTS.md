# D1: resultado del contraste de capacidad

Estado: **NO_CANDIDATE_D1**. No implica promoción del sistema ni SOTA.

Un único fit seed7 de 12.000 updates P duplica la anchura del predictor espacial y su MLP. La CNN nativa de 64 canales conserva su inicialización y se entrena realmente. El control PAIR-CORR-S7 se reutiliza con igual presupuesto de updates; su coste no se cuenta otra vez. Inputs, objetivo firmado, sampler y optimizador son los mismos.

Parámetros: 578,337 en el control y 1,693,185 en D1. El factor cambia la anchura de la torre y MLP; no se afirma que toda la arquitectura o el número de parámetros se hayan multiplicado exactamente por dos.

## Cohortes completas

| Población | Modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Signo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | A5-P2-FROZEN | 18422 | 112.285 | 157.267 | 68.257 | 66.231 | 65.520 | 5.637 | 5.694% |
| H | PAIR-CORR-S7 | 18422 | 103.510 | 115.556 | 70.968 | 88.421 | 156.002 | 5.496 | 13.804% |
| H | PAIR-CORR-WIDE-S7 | 18422 | 115.120 | 124.484 | 76.663 | 92.366 | 206.421 | 6.793 | 20.058% |
| P | A5-P2-FROZEN | 53890 | 100.874 | 141.813 | 62.952 | 55.295 | 55.522 | 4.751 | 3.333% |
| P | PAIR-CORR-S7 | 53890 | 75.736 | 65.709 | 58.176 | 78.532 | 175.753 | 5.827 | 17.575% |
| P | PAIR-CORR-WIDE-S7 | 53890 | 90.892 | 85.376 | 63.797 | 78.843 | 211.806 | 7.028 | 22.379% |

## Interpretación registrada

Cambio agregado P frente al control: +20.012%. Señal de capacidad según el criterio prospectivo: False. Este criterio requiere al menos 3% de mejora P y signo no peor. Describe el ajuste de entrenamiento; por sí solo no demuestra generalización.

La admisión H conserva A5-P2-FROZEN y las protecciones originales. Comprobaciones fallidas: bands, sign. La decisión completa incluye además las rutas de mejora o coste y se conserva en `SELECTION_FINAL.json`.

## Bootstrap y coste

- H/A5-P2-FROZEN: IC95 del cambio agregado D1 − referencia: [-6.812, 14.373].
- H/PAIR-CORR-S7: IC95 del cambio agregado D1 − referencia: [9.649, 13.688].
- P/A5-P2-FROZEN: IC95 del cambio agregado D1 − referencia: [-17.410, -2.634].
- P/PAIR-CORR-S7: IC95 del cambio agregado D1 − referencia: [13.723, 16.832].

Se remuestrean grupos completos, no ventanas IID. Son intervalos de una semilla y no estiman la variabilidad entre entrenamientos.

| Modelo | Mediana forward preparado, ms | p95, ms |
|---|---:|---:|
| A5-P2-FROZEN | 24.907 | 35.685 |
| PAIR-CORR-S7 | 8.366 | 13.887 |
| PAIR-CORR-WIDE-S7 | 9.252 | 27.440 |

Estos tiempos empiezan con tensores event preparados en GPU y terminan en TTC firmado. Excluyen lectura sensorial, preprocessing, transferencias y carga de pesos: **no son E2E**. Se conserva el orden emparejado y fijo.

El endpoint se comprueba sobre pesos, Adam, scheduler, encoder y 384.000 IDs de muestreo regenerados. Las predicciones completas tienen recálculo NumPy independiente. El export TorchScript incluye ejemplos CPU y paridad con el forward nativo; comienza en tensores P2, no en eventos crudos.

No se abre V para rescatar un fallo H. La familia de réplica seleccionada sigue siendo C1-fusión; D1 es un contraste adicional de capacidad y no una segunda búsqueda de semillas. Una señal P favorable y una admisión H son decisiones diferentes.

Regeneración: `python -B -m operational.joint_measurement.capacity_report`. Fuentes durables bajo `adaptive/D1_predictor_width`: `REGISTRATION.json`, `evaluation`, `evidence/COMPLETE.json`, `predictions`, `cost` y `exports`.

Errata de metadatos: el campo `decision.cost_scope` del JSON original heredó la palabra RGB del helper compartido. La medición registrada y ejecutada utiliza eventos P2, como indica este informe. Se conserva el JSON original y `evidence/COST_SCOPE_ERRATUM.json`; no cambia ninguna cifra.
