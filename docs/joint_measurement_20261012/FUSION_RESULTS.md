# V18: fusión C1 + R_CTX, semilla 7

Estado: señal de desarrollo replicable; **NO_SYSTEM_PROMOTION**. No hay evidencia oficial ni afirmación SOTA.

La medición espacial C1 se entrenó sobre P durante 12.000 updates. Su predictor independiente quedó rechazado por signo/banda negativa. Se reutilizan sus features espaciales nuevas (196 valores) como complemento de R_CTX (28 valores e historial RGB nativo), conservando el anchor firmado.

TRUE y ZERO tienen la misma capacidad, inicialización y receta: 2.500 updates cada uno, batch 128, AdamW y pérdida de fase emitida más 0,1 de fase raw. ZERO anula todo el bloque event, incluido el tiempo. Si falta RGB, ambos copian exactamente el TTC de C1. No se promedian TTC de signos opuestos.

R_CTX ya estaba entrenado en H. Estas cabezas usan H para ajuste secuencial supervisado; H no es OOF ni validación independiente. Los dos endpoints se congelaron antes de abrir V y antes de calcular sus diagnósticos H. V sigue siendo desarrollo interno expuesto.

## Resultados completos

| Población | Modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Error de signo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | FUSION-TRUE-S7 | 18422 | 53.773 | 80.708 | 27.039 | 26.591 | 26.485 | 1.114 | 1.091% |
| H | FUSION-ZERO-S7 | 18422 | 65.740 | 101.047 | 31.287 | 29.117 | 29.187 | 1.211 | 1.183% |
| H | R_CTX | 18422 | 66.630 | 102.426 | 32.262 | 28.259 | 29.124 | 1.245 | 1.189% |
| V | FUSION-TRUE-S7 | 16432 | 88.270 | 127.661 | 51.025 | 42.969 | 48.355 | 1.817 | 2.611% |
| V | FUSION-ZERO-S7 | 16432 | 92.638 | 136.989 | 50.225 | 42.182 | 48.580 | 1.842 | 2.684% |
| V | F_TRUE | 16432 | 90.650 | 131.501 | 52.740 | 42.788 | 47.986 | 2.003 | 2.550% |
| V | R_CTX | 16432 | 93.505 | 138.634 | 50.841 | 41.204 | 48.154 | 1.884 | 2.684% |

## Contrastes y decisión

- Frente a FUSION-ZERO-S7: agregado -4.72%, MiDc -6.81%; IC95 por grupos del cambio agregado [-7.402, -1.753]. Gate fallido: crucial_10pct.
- Frente a R_CTX: agregado -5.60%, MiDc -7.91%; IC95 por grupos del cambio agregado [-8.630, -2.410]. Gate fallido: crucial_10pct.
- Frente a F_TRUE: agregado -2.62%, MiDc -2.92%; IC95 por grupos del cambio agregado [-5.365, 0.577]. Gate fallido: crucial_10pct.

El intervalo contra F_TRUE incluye cero: una semilla no establece superioridad frente a ese sistema. Contra ZERO y R_CTX hay una señal agregada, pero falta la mejora crucial del 10% exigida para promoción. La banda larga empeora frente a R_CTX aunque permanece dentro de su salvaguarda. Todas las regresiones se conservan.

La señal controlada habilita la única familia de réplica: seeds 13 y 23, reentrenando encoder, lectura geométrica y las dos cabezas de cada semilla. Se mantienen A5/R_CTX históricos como inicialización/anchor compartidos; no es réplica del pretraining original. Son 34.000 updates científicos nuevos registrados antes de comenzar. Ningún resultado permite escoger la mejor semilla, reajustar la receta con V o promover C1 retrospectivamente.

## Verificación y alcance

- 18.422 filas H y 16.432 V por brazo, sin exclusión de consultas. En estas poblaciones existe RGB en todas las filas; el fallback se prueba también con máscaras sintéticas all/missing/mixed, no con fallos reales inventados.
- Cachés FP32 sin pérdida, relojes nativos separados, normalizadores ajustados sólo en H. Cada salida del extractor R_CTX se contrasta exactamente con el forward histórico; V coincide además con su CSV original admitido.
- Siete tablas recalculadas mediante NumPy independiente. Cinco contrastes con 2.000 resamples de grupos completos; no ventanas IID ni incertidumbre entre semillas disfrazada de bootstrap.
- 320.000 IDs de muestreo regenerados por cabeza; Adam y scheduler en 2.500, ocho tensores aprendidos por brazo y estado final del sampler exacto. La QA CUDA de reanudación contó 40 updates técnicos antes de los científicos.
- Dos exports TorchScript con normalizadores y fallback dinámico. Comienzan en features de productores: no contienen encoders ni preprocessing sensorial. Sus tiempos CPU son sólo de cabeza; el coste completo sigue pendiente.

Las fuentes durables son `fusion/REGISTRATION.json`, `ENDPOINTS.json`, `evaluation/RESULT.json`, `inputs/{H,V}`, `predictions/{H,V}`, `bootstrap/V` y `exports`. Los pesos, curvas y estados completos permanecen en `fusion/fits`.

Regeneración: `python -B -m operational.joint_measurement.fusion_report`.

## Comparador V16 conservado

Tras congelar la decisión de réplica se volvió a comprobar el estado real de V16: su candidato PHASE_ONLY y sus réplicas están terminados; conserva 6.762 predicciones test12 sin envío ni score oficial. No se repite ese trabajo.

En el contraste descriptivo de semilla 7 y cohorte V idéntica, PHASE_ONLY obtiene agregado 89.223 y la nueva fusión 88.270. El IC95 del cambio es [-3.337, 1.553] e incluye cero. La banda larga empeora de 40.113 a 42.969, superando también esa salvaguarda V18. No se ha demostrado superioridad sobre el candidato V16.

La comparación usa los CSV sin etiquetas sellados y sus checkpoints verificados; adjunta los targets V nativos sin modificar predicciones. No cambia los gates históricos de V16 ni la receta V18 ya registrada. La evidencia queda en `fusion/v16_compare`.
